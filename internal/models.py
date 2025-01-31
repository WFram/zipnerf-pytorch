import gin
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import train_utils
from internal import render
from internal import stepfun
from internal import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from nerfacc import OccGridEstimator, render_weight_from_density
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map
from tqdm import tqdm
from gridencoder import GridEncoder
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
try:
    from torch_scatter import segment_coo
except:
    pass

gin.config.external_configurable(math.safe_exp, module='math')


def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    num_prop_samples: int = 64  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = None  # The curve used for ray dists.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    distinct_prop: bool = True  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.
    power_lambda: float = -1.5
    std_scale: float = 0.5
    prop_desired_grid_size = [512, 2048]

    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config

        from extensions import Backend
        Backend.set_backend('dpcpp' if self.config.dpcpp_backend else 'cuda')
        self.backend = Backend.get_backend()
        self.generator = self.backend.get_generator()

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP(num_glo_features=self.num_glo_features,
                                num_glo_embeddings=self.num_glo_embeddings)
        if self.config.dpcpp_backend:
            self.generator = self.nerf_mlp.encoder.backend.get_generator()
        else:
            self.generator = None

        if self.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif not self.distinct_prop:
            self.prop_mlp = PropMLP()
        else:
            for i in range(self.num_levels - 1):
                self.register_module(f'prop_mlp_{i}', PropMLP(grid_disired_resolution=self.prop_desired_grid_size[i]))
        if self.num_glo_features > 0 and not config.zero_glo:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_vecs = nn.Embedding(self.num_glo_embeddings, self.num_glo_features)

        if self.learned_exposure_scaling:
            # Setup learned scaling factors for output colors.
            max_num_exposures = self.num_glo_embeddings
            # Initialize the learned scaling offsets at 0.
            self.exposure_scaling_offsets = nn.Embedding(max_num_exposures, 3)
            torch.nn.init.zeros_(self.exposure_scaling_offsets.weight)

        # TODO: set from config
        self.scene_radius = 5
        self.scene_aabb = torch.as_tensor([-self.scene_radius, -self.scene_radius, -self.scene_radius, self.scene_radius, self.scene_radius, self.scene_radius], dtype=torch.float32).cuda()
        occupancy_grid_res = [128, 128, 128]
        self.occupancy_grid_estimator = [OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=occupancy_grid_res[i_level],
            levels=1
        ).cuda() for i_level in range(self.num_levels)]
        self.global_occupancy_grid_update_step = 0
        self.render_step_size = [1.732 * 2 * self.scene_radius / self.num_prop_samples,
                                 1.732 * 2 * self.scene_radius / self.num_prop_samples,
                                 1.732 * 2 * self.scene_radius / self.num_nerf_samples]

    def forward(
            self,
            rand,
            batch,
            train_frac,
            compute_extras,
            zero_glo=True,
    ):
        """The mip-NeRF Model.

    Args:
      rand: random number generator (or None for deterministic output).
      batch: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
        device = batch['origins'].device
        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None

        ray_history = []
        renderings = []
        assert self.num_levels == 1
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            mlp = (self.get_submodule(
                f'prop_mlp_{i_level}') if self.distinct_prop else self.prop_mlp) if is_prop else self.nerf_mlp
            
            def occ_eval_fn(x):
                stds = torch.from_numpy(np.zeros_like(x[..., 0].cpu())).cuda()
                raw_density, _, _ = mlp.predict_density(x, stds, self.scene_radius, rand=rand)
                if mlp.use_fully_fused_mlp:
                    density = trunc_exp(raw_density + mlp.density_bias)
                else:
                    density = F.softplus(raw_density + mlp.density_bias)
                # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size) based on taylor series
                self.render_step_size[i_level] = 1.732 * 2 * self.scene_radius / num_samples
                return density[...,None] * self.render_step_size[i_level]

            # TODO update only for training
            self.occupancy_grid_estimator[i_level].update_every_n_steps(step=self.global_occupancy_grid_update_step, occ_eval_fn=occ_eval_fn)

            def sigma_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = batch['origins'][:, 0, 0][ray_indices] if len(batch['origins'].shape) == 4 else batch['origins'][ray_indices]
                t_dirs = batch['directions'][:, 0, 0][ray_indices] if len(batch['directions'].shape) == 4 else batch['directions'][ray_indices]
                positions = t_origins + t_dirs * (t_starts[..., None] + t_ends[..., None]) / 2.
                stds = torch.from_numpy(np.zeros_like(positions[..., 0].cpu())).cuda()
                raw_density, _, _ = mlp.predict_density(positions, stds, self.scene_radius, rand=rand)
                if mlp.use_fully_fused_mlp:
                    density = trunc_exp(raw_density + mlp.density_bias)
                else:
                    density = F.softplus(raw_density + mlp.density_bias)
                return density

            with torch.no_grad():
                ray_indices, t_starts, t_ends = self.occupancy_grid_estimator[i_level].sampling(
                    batch['origins'][:, 0, 0] if len(batch['origins'].shape) == 4 else batch['origins'],
                    batch['directions'][:, 0, 0] if len(batch['directions'].shape) == 4 else batch['directions'],
                    near_plane=torch.mean(batch['near']) if self.config.near_far_planes else 0.0,
                    far_plane=torch.mean(batch['far']) if self.config.near_far_planes else 1e10,
                    sigma_fn=sigma_fn,
                    render_step_size=self.render_step_size[i_level],
                    stratified=True,
                    cone_angle=0.0,
                    alpha_thre=0.0
                )
            ray_indices = ray_indices.long()
            t_origins = batch['origins'][:, 0, 0][ray_indices] if len(batch['origins'].shape) == 4 else batch['origins'][ray_indices]
            t_dirs = batch['directions'][:, 0, 0][ray_indices] if len(batch['directions'].shape) == 4 else batch['directions'][ray_indices]
            means, stds, ts = render.cast_rays_acc(t_starts, t_ends,
                                                   t_origins, t_dirs, 
                                                   batch['radii'][:, 0, 0][ray_indices] if len(batch['radii'].shape) == 4 else batch['radii'][ray_indices],
                                                   rand, std_scale=self.std_scale)
            midpoints = (t_starts + t_ends)[..., None] / 2.0
            intervals = (t_ends - t_starts)[..., None]

            ray_results = mlp(
                rand,
                means, stds,
                self.scene_radius,
                viewdirs=t_dirs if self.use_viewdirs else None,
                imageplane=batch.get('imageplane'),
                glo_vec=None if is_prop else glo_vec,
                exposure=batch.get('exposure_values'),
            )
            if self.config.gradient_scaling:
                ray_results['rgb'], ray_results['density'] = train_utils.GradientScaler.apply(
                    ray_results['rgb'], ray_results['density'], ts.mean(dim=-1))

            n_rays = batch['origins'].shape[0]
            weights, _, _ = render_weight_from_density(t_starts, t_ends, ray_results['density'], ray_indices=ray_indices, n_rays=n_rays)

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif rand is None:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.bg_intensity_range[0]
                maxval = self.bg_intensity_range[1]
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval

            # RawNeRF exposure logic.
            if batch.get('exposure_idx') is not None:
                # Scale output colors by the exposure.
                ray_results['rgb'] *= batch['exposure_values'][..., None, :]
                if self.learned_exposure_scaling:
                    exposure_idx = batch['exposure_idx'][..., 0]
                    # Force scaling offset to always be zero when exposure_idx is 0.
                    # This constraint fixes a reference point for the scene's brightness.
                    mask = exposure_idx > 0
                    # Scaling is parameterized as an offset from 1.
                    scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx.long())
                    ray_results['rgb'] *= scaling[..., None, :]

            # Render each ray.
            rendering = render.volumetric_rendering_acc(
                rgb=ray_results['rgb'],
                weights=weights,
                ray_indices=ray_indices,
                midpoints=midpoints,
                n_rays=n_rays,
                background_color=torch.from_numpy(np.full((3,), bg_rgbs, dtype=np.float32)).cuda()
            )

            # TODO: return hash decay loss
            # if self.training:
            #     # Compute the hash decay loss for this level.
            #     idx = mlp.encoder.idx
            #     param = mlp.encoder.embeddings
            #     if self.config.dpcpp_backend:
            #         ray_results['loss_hash_decay'] = (param ** 2).mean()
            #     else:
            #         loss_hash_decay = segment_coo(param ** 2,
            #                                       idx,
            #                                       torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
            #                                       reduce='mean'
            #                                       ).mean()
            #         ray_results['loss_hash_decay'] = loss_hash_decay

            renderings.append(rendering)
            ray_results['weights'] = weights.clone()
            ray_results['midpoints'] = midpoints.clone()
            ray_results['intervals'] = intervals.clone()
            ray_results['ray_indices'] = ray_indices.clone()
            ray_history.append(ray_results)

        self.global_occupancy_grid_update_step +=1
        
        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history


def get_rank():
    import os
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def load_omega_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def omega_config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve) 


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


class MLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: int = 128  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 128  # The width of the second part of MLP.
    skip_layer_dir: int = 1000  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    scale_featurization: bool = False
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21
    net_width_glo: int = 128  # The width of the second part of MLP.
    net_depth_glo: int = 2  # The width of the second part of MLP.
    use_fully_fused_mlp: bool = False  # Usage of fully-fused MLP.
    ffmlp_config: str = ''  # Path to YAML file with fully-fused MLP config.

    def __init__(self, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')

        if not self.use_fully_fused_mlp:
            # Precompute and define viewdir or refdir encoding function.
            if self.use_directional_enc:
                self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
                dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
            else:

                def dir_enc_fn(direction, _):
                    return coord.pos_enc(
                        direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

                self.dir_enc_fn = dir_enc_fn
                dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
            self.grid_num_levels = int(
                np.log(self.grid_disired_resolution / self.grid_base_resolution) / np.log(self.grid_level_interval)) + 1
            self.encoder = GridEncoder(input_dim=3,
                                    num_levels=self.grid_num_levels,
                                    level_dim=self.grid_level_dim,
                                    base_resolution=self.grid_base_resolution,
                                    desired_resolution=self.grid_disired_resolution,
                                    log2_hashmap_size=self.grid_log2_hashmap_size,
                                    gridtype='hash',
                                    align_corners=False)
            last_dim = self.encoder.output_dim
            if self.scale_featurization:
                last_dim += self.encoder.num_levels
        
            self.density_layer = nn.Sequential(nn.Linear(last_dim, 64),
                                               nn.ReLU(),
                                               nn.Linear(64,
                                                         1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
            last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
            if self.enable_pred_normals:
                self.normal_layer = nn.Linear(last_dim, 3)

            if not self.disable_rgb:
                if self.use_diffuse_color:
                    self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)

                if self.use_specular_tint:
                    self.specular_layer = nn.Linear(last_dim, 3)

                if self.enable_pred_roughness:
                    self.roughness_layer = nn.Linear(last_dim, 1)

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    last_dim_rgb = self.bottleneck_width
                else:
                    last_dim_rgb = 0

                last_dim_rgb += dim_dir_enc

                if self.use_n_dot_v:
                    last_dim_rgb += 1

                if self.num_glo_features > 0:
                    last_dim_glo = self.num_glo_features
                    for i in range(self.net_depth_glo - 1):
                        self.register_module(f"lin_glo_{i}", nn.Linear(last_dim_glo, self.net_width_glo))
                        last_dim_glo = self.net_width_glo
                    self.register_module(f"lin_glo_{self.net_depth_glo - 1}",
                                         nn.Linear(last_dim_glo, self.bottleneck_width * 2))

                input_dim_rgb = last_dim_rgb
                for i in range(self.net_depth_viewdirs):
                    lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                    torch.nn.init.kaiming_uniform_(lin.weight)
                    self.register_module(f"lin_second_stage_{i}", lin)
                    last_dim_rgb = self.net_width_viewdirs
                    if i == self.skip_layer_dir:
                        last_dim_rgb += input_dim_rgb
                self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)
        else:
            ffmlp_config = load_omega_config(self.ffmlp_config)
            density_config = ffmlp_config.get('density')
            rgb_config = ffmlp_config.get('rgb')
            
            from dataclasses import dataclass
            feature_encoding_config = density_config.get('xyz_encoding_config')
            @dataclass
            class XyzEncodingParams():
                num_levels: int
                level_dim: int
                log2_hashmap_size: int
                base_resolution: int
                per_level_scale: int
                grid_sizes: torch.Tensor = torch.from_numpy(np.array([], dtype=np.int32))
            self.xyz_encoding_params = XyzEncodingParams(feature_encoding_config.get('n_levels', 6),
                                                         feature_encoding_config.get('n_features_per_level'),
                                                         feature_encoding_config.get('log2_hashmap_size'),
                                                         feature_encoding_config.get('base_resolution'),
                                                         feature_encoding_config.get('per_level_scale'))
            
            # self.xyz_encoding_params.num_levels = int(
            #     np.log(self.grid_disired_resolution / self.xyz_encoding_params.base_resolution) / np.log(self.grid_level_interval)) + 1
            # feature_encoding_config['n_levels'] = self.xyz_encoding_params.num_levels

            self.xyz_encoding_params.num_levels = feature_encoding_config['n_levels']
            density_mlp_config = density_config.get('mlp_network_config')
            assert self.bottleneck_width > 0
            last_dim = self.xyz_encoding_params.num_levels * self.xyz_encoding_params.level_dim
            resolutions = [int(np.ceil(self.xyz_encoding_params.base_resolution * self.xyz_encoding_params.per_level_scale ** (i // self.xyz_encoding_params.level_dim))) + 1 \
                           for i in range(self.xyz_encoding_params.num_levels * self.xyz_encoding_params.level_dim)]
            self.xyz_encoding_params.grid_sizes = torch.from_numpy(np.array(resolutions, dtype=np.int32)).cuda()
            
            with torch.cuda.device(get_rank()):
                self.feature_encoding_n_input = 3
                self.encoder = tcnn.Encoding(self.feature_encoding_n_input, omega_config_to_primitive(feature_encoding_config))
                if self.scale_featurization:
                    last_dim += self.xyz_encoding_params.num_levels
                self.density_layer_n_input_dims = last_dim
                self.density_layer_n_output_dims = 1 if self.disable_rgb else self.bottleneck_width
                self.density_layer = tcnn.Network(self.density_layer_n_input_dims, self.density_layer_n_output_dims, omega_config_to_primitive(density_mlp_config))
                if not self.disable_rgb:
                    rgb_mlp_config = rgb_config.get('mlp_network_config')
                    dir_encoding_config = rgb_config.get('dir_encoding_config')
                    dir_encoding_n_input = 3
                    self.dir_enc_fn = tcnn.Encoding(dir_encoding_n_input, omega_config_to_primitive(dir_encoding_config))
                    dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3).cuda()).shape[-1]
                    last_dim_rgb = self.bottleneck_width + dim_dir_enc
                    self.rgb_layer_n_input_dims = last_dim_rgb
                    self.rgb_layer_n_output_dims = self.num_rgb_channels
                    self.rgb_layer = tcnn.Network(self.rgb_layer_n_input_dims, self.rgb_layer_n_output_dims, omega_config_to_primitive(rgb_mlp_config))

    def predict_density(self, means, stds, scene_radius, rand=False, no_warp=False):
        """Helper function to output density."""
        # Encode input positions
        if not self.use_fully_fused_mlp and self.warp_fn is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound
        else:
            means, stds = coord.contract_to_unisphere(means, stds, scene_radius)
        if not self.use_fully_fused_mlp:
            if len(means.shape) == 2:
                means = means[:, None]
                stds = stds[:, None]
            features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
            weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * self.encoder.grid_sizes ** 2))
            features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
            if self.scale_featurization:
                with torch.no_grad():
                    vl2mean = segment_coo((self.encoder.embeddings ** 2).sum(-1),
                                        self.encoder.idx,
                                        torch.zeros(self.grid_num_levels, device=weights.device),
                                        self.grid_num_levels,
                                        reduce='mean'
                                        )
                featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoder.init_std ** 2 + vl2mean).sqrt()
                features = torch.cat([features, featurized_w], dim=-1)
            x = self.density_layer(features)
            raw_density = x[..., 0]  # Hardcoded to a single channel.
            # Add noise to regularize the density predictions if needed.
            if rand and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn_like(raw_density)
            return raw_density, x, means.mean(dim=-2)
        else:
            features = self.encoder(means.view(-1, self.feature_encoding_n_input)).view(*means.shape[:-1], self.density_layer_n_input_dims).float()
            if len(features.shape) > 2:
                # weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * self.xyz_encoding_params.grid_sizes ** 2))
                weights = torch.ones_like(features)
                features = (features * weights).mean(dim=-2)
            x = self.density_layer(features.view(-1, self.density_layer_n_input_dims)).view(*features.shape[:-1], self.density_layer_n_output_dims).float()
            raw_density = x[..., 0]  # Hardcoded to a single channel.
            # Add noise to regularize the density predictions if needed.
            if rand and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn_like(raw_density)
            return raw_density, x, means.mean(dim=-2)

    def forward(self,
                rand,
                means, stds,
                scene_radius,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None,
                no_warp=False):
        """Evaluate the MLP.

    Args:
      rand: if random .
      means: [..., n, 3], coordinate means.
      stds: [..., n], coordinate stds.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        if self.disable_density_normals:
            raw_density, x, means_contract = self.predict_density(means, stds, scene_radius, rand=rand, no_warp=no_warp)
            raw_grad_density = None
            normals = None
        else:
            with torch.enable_grad():
                means.requires_grad_(True)
                raw_density, x, means_contract = self.predict_density(means, stds, scene_radius, rand=rand, no_warp=no_warp)
                d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
                raw_grad_density = torch.autograd.grad(
                    outputs=raw_density,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
            raw_grad_density = raw_grad_density.mean(-2)
            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -ref_utils.l2_normalize(raw_grad_density)

        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        if self.use_fully_fused_mlp:
            density = trunc_exp(raw_density + self.density_bias)
        else:
            density = F.softplus(raw_density + self.density_bias)

        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = (F.softplus(raw_roughness + self.roughness_bias))

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    # Append GLO vector if used.
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f"lin_glo_{i}")(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift

                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                dir_enc = None
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    if not self.use_fully_fused_mlp:
                        dir_enc = self.dir_enc_fn(viewdirs, roughness)
                    else:
                        viewdirs_scaled = (viewdirs + 1.) / 2.
                        dir_enc = self.dir_enc_fn(viewdirs_scaled.view(-1, 3))

                if not self.use_fully_fused_mlp:    




                    x = torch.cat([bottleneck.view(-1, bottleneck.shape[-1]), dir_enc], dim=-1)
                    inputs = x
                    for i in range(self.net_depth_viewdirs):
                        x = self.get_submodule(f"lin_second_stage_{i}")(x)
                        x = F.relu(x)
                        if i == self.skip_layer_dir:
                            x = torch.cat([x, inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            if self.use_fully_fused_mlp:
                network_inp = torch.cat([bottleneck.view(-1, bottleneck.shape[-1]), dir_enc], dim=-1)
                rgb = torch.sigmoid(self.rgb_premultiplier *
                                    self.rgb_layer(network_inp).view(*bottleneck.shape[:-1], self.rgb_layer_n_output_dims).float() +
                                    self.rgb_bias)
            else:
                rgb = torch.sigmoid(self.rgb_premultiplier *
                                    self.rgb_layer(x) +
                                    self.rgb_bias)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)


        return dict(
            coord=means_contract,
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


@torch.no_grad()
def render_image(model,
                 batch,
                 rand,
                 train_frac,
                 config,
                 verbose=True,
                 return_weights=False):
    """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rand, batch) -> pytree.
    batch: a `Rays` pytree, the rays to be rendered.
    rand: if random
    config: A Config class.

  Returns:
    rgb: rendered color image.
    disp: rendered disparity image.
    acc: rendered accumulated weights per pixel.
  """
    model.eval()

    height, width = batch['origins'].shape[:2]
    num_rays = height * width
    batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}

    chunks = []
    idx0s = tqdm(range(0, num_rays, config.render_chunk_size),
                 desc="Rendering chunk", leave=False)

    for i_chunk, idx0 in enumerate(idx0s):
        chunk_batch = tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], batch)

        chunk_renderings, ray_history = model(rand,
                                              chunk_batch,
                                              train_frac=train_frac,
                                              compute_extras=False,
                                              zero_glo=True)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]

        if return_weights:
            chunk_rendering['weights'] = ray_history[-1]['weights']
            chunk_rendering['coord'] = ray_history[-1]['coord']
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = {}
    for k in chunks[0].keys():
        if isinstance(chunks[0][k], list):
            rendering[k] = []
            for i in range(len(chunks[0][k])):
                rendering[k].append(torch.cat([item[k][i] for item in chunks]))
        else:
            rendering[k] = torch.cat([item[k] for item in chunks])

    for k, z in rendering.items():
        if not k.startswith('ray_'):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith('ray_')]
    if keys:
        num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(num_rays)
        ray_idx = ray_idx[:config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]
    model.train()
    return rendering

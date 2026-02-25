import os
import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.causal_model_DS import CausalWanModelDS

DEFAULT_WAN_MODEL_NAME = "Wan2.1-T2V-14B"


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = DEFAULT_WAN_MODEL_NAME, base_dir: str = "wan_models") -> None:
        super().__init__()

        model_dir = os.path.join(base_dir, model_name)
        t5_ckpt_path = os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_dir = os.path.join(model_dir, "google", "umt5-xxl")

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(t5_ckpt_path, map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=f"{tokenizer_dir}/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name: str = DEFAULT_WAN_MODEL_NAME, base_dir: str = "wan_models"):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        model_dir = os.path.join(base_dir, model_name)
        self.model = _video_vae(
            pretrained_path=os.path.join(model_dir, "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name=DEFAULT_WAN_MODEL_NAME,
            timestep_shift=8.0,
            is_causal=False,
            is_ds_only=False,
            local_attn_size=-1,
            sink_size=0,
            budget=16,
            recent=4,
            st_enable=True,
            st_target_budget=None,
            st_grid_size=(4, 2, 2),
            st_pool_size=1024,
            st_lambda_reg=0.5,
            st_epsilon=1e-5,
            st_recent_window_frames=4,
            st_max_query_tokens=2048,
            st_keep_sinks=True,
            **kwargs,
    ):
        super().__init__()
        # Backward-compatible aliases for existing configs/CLI overrides.
        if "Budget" in kwargs and kwargs["Budget"] is not None:
            budget = kwargs.pop("Budget")
        if "Recent" in kwargs and kwargs["Recent"] is not None:
            recent = kwargs.pop("Recent")

        if "ST_enable" in kwargs and kwargs["ST_enable"] is not None:
            st_enable = kwargs.pop("ST_enable")
        if "ST_target_budget" in kwargs and kwargs["ST_target_budget"] is not None:
            st_target_budget = kwargs.pop("ST_target_budget")
        if "ST_grid_size" in kwargs and kwargs["ST_grid_size"] is not None:
            st_grid_size = kwargs.pop("ST_grid_size")
        if "ST_pool_size" in kwargs and kwargs["ST_pool_size"] is not None:
            st_pool_size = kwargs.pop("ST_pool_size")
        if "ST_lambda_reg" in kwargs and kwargs["ST_lambda_reg"] is not None:
            st_lambda_reg = kwargs.pop("ST_lambda_reg")
        if "ST_epsilon" in kwargs and kwargs["ST_epsilon"] is not None:
            st_epsilon = kwargs.pop("ST_epsilon")
        if "ST_recent_window_frames" in kwargs and kwargs["ST_recent_window_frames"] is not None:
            st_recent_window_frames = kwargs.pop("ST_recent_window_frames")
        if "ST_max_query_tokens" in kwargs and kwargs["ST_max_query_tokens"] is not None:
            st_max_query_tokens = kwargs.pop("ST_max_query_tokens")
        if "ST_keep_sinks" in kwargs and kwargs["ST_keep_sinks"] is not None:
            st_keep_sinks = kwargs.pop("ST_keep_sinks")

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected WanDiffusionWrapper kwargs: {unknown}")

        if st_target_budget is None or int(st_target_budget) <= 0:
            st_target_budget = 1560 * int(budget)

        # ST budget is in tokens, but the KV cache has a hard capacity determined by local attention.
        # If you request a larger budget than the cache, ST compression will try to keep more tokens
        # than can be written back into `kv_cache["k"]` / `kv_cache["v"]`.
        if is_causal:
            if local_attn_size != -1:
                kv_capacity_tokens = int(local_attn_size) * 1560
            else:
                kv_capacity_tokens = 32760
            if int(st_target_budget) > kv_capacity_tokens:
                print(
                    f"[ST] Capping ST_target_budget from {int(st_target_budget)} to {kv_capacity_tokens} "
                    f"(KV cache capacity; local_attn_size={local_attn_size})."
                )
                st_target_budget = kv_capacity_tokens

        if is_causal:
            if is_ds_only:
                self.model = CausalWanModelDS.from_pretrained(
                    f"wan_models/{model_name}/",
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    ST_enable=st_enable,
                    ST_target_budget=st_target_budget,
                    ST_grid_size=st_grid_size,
                    ST_pool_size=st_pool_size,
                    ST_lambda_reg=st_lambda_reg,
                    ST_epsilon=st_epsilon,
                    ST_recent_window_frames=st_recent_window_frames,
                    ST_max_query_tokens=st_max_query_tokens,
                    ST_keep_sinks=st_keep_sinks,
                )
            else: 
                self.model = CausalWanModel.from_pretrained(
                    f"wan_models/{model_name}/",
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    PC_capacity=1560*budget,
                    PC_window=1560*recent,
                    ST_enable=st_enable,
                    ST_target_budget=st_target_budget,
                    ST_grid_size=st_grid_size,
                    ST_pool_size=st_pool_size,
                    ST_lambda_reg=st_lambda_reg,
                    ST_epsilon=st_epsilon,
                    ST_recent_window_frames=st_recent_window_frames,
                    ST_max_query_tokens=st_max_query_tokens,
                    ST_keep_sinks=st_keep_sinks,
                )
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

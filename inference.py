import argparse
import torch
import os
import time
import json
from datetime import datetime, timezone
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument(
    "--model_name",
    type=str,
    default=None,
    help="Override Wan model folder name under wan_models/ (e.g. Wan2.1-T2V-14B or Wan2.1-T2V-1.3B)",
)
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=126,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=1356145, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--is_ds_only", dest="is_ds_only", default=0, type=int,
                    help="Whether to use DS only mode")
parser.add_argument("--Budget", type=int, default=16, help="Budget for PC")
parser.add_argument("--Recent", type=int, default=4, help="Recent for PC")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Propagate CLI flags into model kwargs (used by WanDiffusionWrapper)
if not hasattr(config, "model_kwargs") or config.model_kwargs is None:
    config.model_kwargs = OmegaConf.create({})
config.model_kwargs["is_ds_only"] = bool(args.is_ds_only)
config.model_kwargs["budget"] = int(args.Budget)
config.model_kwargs["recent"] = int(args.Recent)
if args.model_name:
    config.model_kwargs["model_name"] = args.model_name
    # Keep legacy/other callsites consistent: some parts of the codebase use `real_name`.
    config["real_name"] = args.model_name

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt_key = 'generator' if not args.use_ema else 'generator_ema'
    if ckpt_key not in state_dict:
        raise KeyError(f"Checkpoint is missing key '{ckpt_key}'. Available keys: {list(state_dict.keys())}")

    # Detect common mismatch (Self-Forcing checkpoints are often Wan2.1-T2V-1.3B).
    ckpt_sd = state_dict[ckpt_key]
    model_sd = pipeline.generator.state_dict()
    probe_key = "model.patch_embedding.weight"
    if probe_key in ckpt_sd and probe_key in model_sd:
        ckpt_shape = tuple(ckpt_sd[probe_key].shape)
        model_shape = tuple(model_sd[probe_key].shape)
        if ckpt_shape != model_shape:
            raise RuntimeError(
                "Checkpoint backbone does not match the instantiated Wan model.\n"
                f"- checkpoint {probe_key}: {ckpt_shape}\n"
                f"- model      {probe_key}: {model_shape}\n\n"
                "Fix options:\n"
                "1) To run 14B weights, omit --checkpoint_path.\n"
                "2) To run a 1.3B Self-Forcing checkpoint, pass --model_name Wan2.1-T2V-1.3B "
                "and ensure wan_models/Wan2.1-T2V-1.3B exists.\n"
            )

    pipeline.generator.load_state_dict(ckpt_sd)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate 81 frames
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer_s = time.perf_counter() - t0

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        t1 = time.perf_counter()
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
        t_write_s = time.perf_counter() - t1

        # Throughput logging (rank 0 only)
        if local_rank == 0:
            gen_frames = int(num_generated_frames)
            fps_infer = gen_frames / max(t_infer_s, 1e-9)
            fps_total = gen_frames / max(t_infer_s + t_write_s, 1e-9)

            perf = getattr(pipeline, "last_perf", None)
            fps_denoise = None
            fps_denoise_post_roll = None
            fps_forward_denoise = None
            fps_forward_denoise_post_roll = None
            fps_forward_total = None
            fps_forward_total_post_roll = None
            if isinstance(perf, dict):
                denoise_s = perf.get("denoise_seconds_total", None)
                denoise_post_s = perf.get("denoise_seconds_post_roll", None)
                fwd_denoise_s = perf.get("forward_denoise_seconds_total", None)
                fwd_denoise_post_s = perf.get("forward_denoise_seconds_post_roll", None)
                fwd_ctx_s = perf.get("forward_context_seconds_total", None)
                fwd_ctx_post_s = perf.get("forward_context_seconds_post_roll", None)
                frames_total_perf = perf.get("frames_total", None)
                frames_post_perf = perf.get("frames_post_roll", None)
                if isinstance(denoise_s, (int, float)) and denoise_s > 0 and isinstance(frames_total_perf, int):
                    fps_denoise = frames_total_perf / denoise_s
                if isinstance(denoise_post_s, (int, float)) and denoise_post_s > 0 and isinstance(frames_post_perf, int):
                    fps_denoise_post_roll = frames_post_perf / denoise_post_s
                if isinstance(fwd_denoise_s, (int, float)) and fwd_denoise_s > 0 and isinstance(frames_total_perf, int):
                    fps_forward_denoise = frames_total_perf / fwd_denoise_s
                if isinstance(fwd_denoise_post_s, (int, float)) and fwd_denoise_post_s > 0 and isinstance(frames_post_perf, int):
                    fps_forward_denoise_post_roll = frames_post_perf / fwd_denoise_post_s
                if isinstance(fwd_denoise_s, (int, float)) and isinstance(fwd_ctx_s, (int, float)) and (fwd_denoise_s + fwd_ctx_s) > 0 and isinstance(frames_total_perf, int):
                    fps_forward_total = frames_total_perf / (fwd_denoise_s + fwd_ctx_s)
                if isinstance(fwd_denoise_post_s, (int, float)) and isinstance(fwd_ctx_post_s, (int, float)) and (fwd_denoise_post_s + fwd_ctx_post_s) > 0 and isinstance(frames_post_perf, int):
                    fps_forward_total_post_roll = frames_post_perf / (fwd_denoise_post_s + fwd_ctx_post_s)

            # Best-effort readback of key knobs for experiment tracking
            mk = dict(getattr(config, "model_kwargs", {}) or {})
            record = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": args.config_path,
                "output_folder": args.output_folder,
                "model_name": getattr(config, "real_name", None),
                "is_ds_only": bool(args.is_ds_only),
                "budget": int(args.Budget),
                "recent": int(args.Recent),
                "seed": int(args.seed),
                "num_samples": int(args.num_samples),
                "num_output_frames_arg": int(args.num_output_frames),
                "generated_frames": gen_frames,
                "infer_seconds": _safe_float(t_infer_s),
                "write_seconds": _safe_float(t_write_s),
                "fps_infer": _safe_float(fps_infer),
                "fps_total": _safe_float(fps_total),
                "fps_denoise": _safe_float(fps_denoise) if fps_denoise is not None else None,
                "fps_denoise_post_roll": _safe_float(fps_denoise_post_roll) if fps_denoise_post_roll is not None else None,
                "fps_forward_denoise": _safe_float(fps_forward_denoise) if fps_forward_denoise is not None else None,
                "fps_forward_denoise_post_roll": _safe_float(fps_forward_denoise_post_roll) if fps_forward_denoise_post_roll is not None else None,
                "fps_forward_total": _safe_float(fps_forward_total) if fps_forward_total is not None else None,
                "fps_forward_total_post_roll": _safe_float(fps_forward_total_post_roll) if fps_forward_total_post_roll is not None else None,
                # common knobs if present
                "local_attn_size": mk.get("local_attn_size", None),
                "sink_size": mk.get("sink_size", None),
                "st_enable": mk.get("st_enable", mk.get("ST_enable", None)),
                "st_recent_window_frames": mk.get("st_recent_window_frames", mk.get("ST_recent_window_frames", None)),
                "st_pool_size": mk.get("st_pool_size", mk.get("ST_pool_size", None)),
                "st_max_query_tokens": mk.get("st_max_query_tokens", mk.get("ST_max_query_tokens", None)),
                "prompt_preview": prompt[:200],
            }
            metrics_path = os.path.join(args.output_folder, "throughput.jsonl")
            _append_jsonl(metrics_path, record)
            print(
                f"[THROUGHPUT] frames={gen_frames} infer_s={t_infer_s:.3f} write_s={t_write_s:.3f} "
                f"fps_infer={fps_infer:.3f} fps_total={fps_total:.3f} "
                + (f"fps_denoise={fps_denoise:.3f} " if fps_denoise is not None else "")
                + (f"fps_denoise_post_roll={fps_denoise_post_roll:.3f} " if fps_denoise_post_roll is not None else "")
                + (f"fps_forward_denoise={fps_forward_denoise:.3f} " if fps_forward_denoise is not None else "")
                + (f"fps_forward_denoise_post_roll={fps_forward_denoise_post_roll:.3f} " if fps_forward_denoise_post_roll is not None else "")
                + (f"fps_forward_total={fps_forward_total:.3f} " if fps_forward_total is not None else "")
                + (f"fps_forward_total_post_roll={fps_forward_total_post_roll:.3f} " if fps_forward_total_post_roll is not None else "")
                + f"-> {metrics_path}"
            )

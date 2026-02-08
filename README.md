<p align="center">
<h1 align="center">Deep Forcing</h1>
<h3 align="center">Training-Free Long Video Generation with Deep Sink and Participative Compression</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://yj-142150.github.io/jungyi/">Jung Yi</a><sup></sup>
    ·
    <a href="https://scholar.google.com/citations?user=7cyLEQ0AAAAJ&hl=en/"> Wooseok Jang</a><sup></sup>
    ·
    <a href="">Paul Hyunbin Cho</a><sup></sup>
    ·
    <a href="https://nam-jisu.github.io/">Jisu Nam</a><sup></sup>
    ·
    <a href="https://yoon-heez.github.io/">Heeji Yoon</a><sup></sup>
    ·
    <a href="https://cvlab.kaist.ac.kr/">Seungryong Kim</a><sup></sup><br>
    <sup></sup>KAIST AI 
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2512.05081">Paper</a> | <a href="https://cvlab-kaist.github.io/DeepForcing/">Website</a></h3>
</p>

<span style="font-size: 18px; font-weight: 800;">
  New! Check Deep Forcing on Interactive Prompting, World Models & Causal Forcing at:
  <a href="https://cvlab-kaist.github.io/DeepForcing/">https://cvlab-kaist.github.io/DeepForcing/</a>
</span>


---

Deep Forcing is a training-free framework that enables long-video generation in autoregressive video diffusion models by combining Deep Sink and Participative Compression. Deep Forcing achieves more than 12× length extrapolation (5s → 60s+) without fine-tuning. 

---

## Highlights
- **Deep Sink** maintains a substantially enlarged attention sink (~50% of cache), with temporal RoPE adjustment, ensuring temporal coherence between sink tokens and current frames.

- **Participative Compression** selectively prunes redundant tokens by computing attention scores from recent frames, retains only the top-C most contextually relevant tokens while evicting redundant and degraded tokens.

## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 3090, A6000, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. It is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.

* **Currently demo.py is not supported for Deep Forcing. Stay tuned.**

## CLI Inference
### Deep Sink Only Inference
Example inference script:
```
bash DS_inference.sh
```
```
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/self_forcing_dmd/self_forcing_dmd_sink14.yaml \
    --output_folder ./output/DS \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path ./prompts/MovieGenVideoBench_txt/line_0010.txt \
    --use_ema \
    --is_ds_only 1
```
Note: 
* Sink size 10–14 is recommended for Deep Sink–only inference (configs: self_forcing_dmd_sink10.yaml–self_forcing_dmd_sink14.yaml).

### Deep Sink & Participative Compression
Example inference script:
```
bash DS_PC_inference.sh
```
```
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config_path configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml \
    --output_folder ./output/DS_PC \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path ./prompts/MovieGenVideoBench_txt/line_0043.txt \
    --use_ema
```

## Acknowledgements
This codebase is built on top of the open-source implementation of [Self Forcing](https://github.com/guandeh17/Self-Forcing) by [Xun Huang](https://www.xunhuang.me/).

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{yi2025deep,
  title={Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression},
  author={Yi, Jung and Jang, Wooseok and Cho, Paul Hyunbin and Nam, Jisu and Yoon, Heeji and Kim, Seungryong},
  journal={arXiv preprint arXiv:2512.05081},
  year={2025}
}
```

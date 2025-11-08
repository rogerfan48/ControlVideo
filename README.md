# TRACE paper Evaluation - ControlVideo

## Purpose of This Fork
This repository is a fork maintained by the TRACE paper team for experimental comparison in our work:
**"TRACE - Temporal Rectification of Attention for Cross-object Editing"** (paper forthcoming).
We use this fork to benchmark ControlVideo against our TRACE model using a unified dataset + scripts.

### Added / Modified In This Fork (RTX 5090 Hopper Compatibility)
Target stack: **RTX 5090 (Hopper / sm_120) + CUDA 12.8/13.0 + PyTorch 2.8.0**.
Original upstream was tied to CUDA 11.6 + torch 1.13, which breaks on Hopper. This fork:
* Upgrades core libs: PyTorch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0
* Aligns high‑level libs: diffusers 0.35.2, transformers 4.57.1, controlnet-aux 0.0.10, xformers 0.0.32.post2
* Replaces legacy attention (`baddbmm` + softmax) with fused PyTorch `scaled_dot_product_attention` to prevent >100GiB allocations & flash kernel crash
* Keeps xformers installed but disables its flash attention fast path on Hopper; SDPA memory‑efficient backend is used instead
* Provides reproducible environment via `environment.yml` (DO NOT use legacy `requirements.txt`)
* Adds bilingual fix docs: `RTX5090_HOPPER_FIX_zh-TW.md`, `RTX5090_HOPPER_FIX_en.md`
* Supplies batch script `run.sh` for one‑command generation of all evaluation videos

### Dataset & Assets
* Evaluation mp4 inputs under `assets/<category>/`
* Generated outputs under `outputs/<category>/<video_name>/`

### Reproducible Environment (Conda)
Use the lock‑style `environment.yml` containing the exact working Hopper stack.
```bash
git clone git@github.com:rogerfan48/ControlVideo.git
cd ControlVideo
conda env create -f environment.yml
conda activate controlvideo

python - <<'PY'
import torch, diffusers, transformers, xformers
print('torch', torch.__version__)
print('diffusers', diffusers.__version__)
print('transformers', transformers.__version__)
print('xformers', getattr(xformers,'__version__','n/a'))
print('cuda?', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
```
Expected: torch 2.8.0 / diffusers 0.35.2 / transformers 4.57.1 / xformers 0.0.32.post2.

### Checkpoint Acquisition
Download required weights (≈53GB total): Stable Diffusion v1.5 + ControlNet depth/canny/openpose + RIFE optical flow.
```bash
mkdir -p checkpoints && cd checkpoints

huggingface-cli download runwayml/stable-diffusion-v1-5 \
  --local-dir stable-diffusion-v1-5 --local-dir-use-symlinks False
huggingface-cli download lllyasviel/sd-controlnet-depth \
  --local-dir sd-controlnet-depth --local-dir-use-symlinks False
huggingface-cli download lllyasviel/sd-controlnet-canny \
  --local-dir sd-controlnet-canny --local-dir-use-symlinks False
huggingface-cli download lllyasviel/sd-controlnet-openpose \
  --local-dir sd-controlnet-openpose --local-dir-use-symlinks False
wget https://github.com/megvii-research/ECCV2022-RIFE/releases/download/v1.0/flownet.pkl
cd ..
```
Directory essentials:
```
checkpoints/
  stable-diffusion-v1-5/{unet,vae,text_encoder,tokenizer,scheduler,...}
  sd-controlnet-depth/diffusion_pytorch_model.safetensors
  sd-controlnet-canny/diffusion_pytorch_model.safetensors
  sd-controlnet-openpose/diffusion_pytorch_model.safetensors
  flownet.pkl
```

### One-Command Batch Generation
```bash
conda activate controlvideo
./run.sh
```
Generates all evaluation outputs into `outputs/`.

### Single Example Run
```bash
conda activate controlvideo
python inference.py \
  --prompt "A blue bowl and a yellow bowl are moving on the table." \
  --condition depth_midas \
  --video_path assets/o2o/3ball.mp4 \
  --output_path outputs/o2o/3ball \
  --video_length 15 \
  --width 512 --height 512 \
  --version v10 --seed 42
```
Result video: `outputs/o2o/3ball/*.mp4`.

### Batch Script (`run.sh`) Overview
* Categories: o2o / o2p / p2p
* Common config: width=512 height=512 seed=42 version=v10
* Per‑video override (e.g. `n_u_c_s` uses 12 frames)
To add a new video: place mp4 in `assets/<cat>/`, append a block mirroring existing entries.

### Legacy Requirements Warning
`requirements.txt` belongs to old CUDA 11.6 stack (torch 1.13.1). **Do not use** for Hopper. Prefer `environment.yml`. Minimal custom install set:
```

### Troubleshooting / Technical Notes
* Hopper flash attention crash → solved by SDPA fallback (see docs linked below)
* Memory pressure → solved by removing explicit score tensor allocation; fused kernel scales better
* To re‑enable xformers on non‑Hopper GPUs you can later add an env toggle (see docs)

Then add: accelerate, einops, omegaconf, opencv-python, imageio, moviepy, decord, pandas, scikit-image, tqdm, ftfy, timm, tensorboard, wandb, addict, easydict.

### Troubleshooting Quick Table
| Symptom | Fix |
|---------|-----|
| `sm_120 not compatible` | Recreate env using environment.yml (old torch used) |
| Flash attention CUDA invalid argument | Fork disables flash path; ensure you didn't re‑enable it |
| OOM in attention | Verify SDPA patch (no giant `baddbmm` tensor) |
| `cached_download` import error | Upgrade diffusers to 0.35.2 |
| HuggingFace hub version conflict | `pip install 'huggingface-hub<1.0,>=0.34.0'` |

### Optional Next Steps
* Add env toggle: `CONTROLVIDEO_USE_XFORMERS=1` for non‑Hopper GPUs
* Migrate to `torch.nn.attention.sdpa_kernel()` (remove deprecation warning)
* Add lightweight benchmarking script (VRAM + step time)

### Further Reading
* Detailed fix: `RTX5090_HOPPER_FIX.md`

---
# ControlVideo

Official pytorch implementation of "ControlVideo: Training-free Controllable Text-to-Video Generation"

[![arXiv](https://img.shields.io/badge/arXiv-2305.13077-b31b1b.svg)](https://arxiv.org/abs/2305.13077)
[![Project](https://img.shields.io/badge/Project-Website-orange)](https://controlvideov1.github.io/)
[![HuggingFace demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Yabo/ControlVideo)
[![Replicate](https://replicate.com/cjwbw/controlvideo/badge)](https://replicate.com/cjwbw/controlvideo) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=YBYBZhang/ControlVideo)

<p align="center">
<img src="assets/overview.png" width="1080px"/> 
<br>
<em>ControlVideo adapts ControlNet to the video counterpart without any finetuning, aiming to directly inherit its high-quality and consistent generation </em>
</p>

## News
* [07/16/2023] Add [HuggingFace demo](https://huggingface.co/spaces/Yabo/ControlVideo)!
* [07/11/2023] Support [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) based version! 
* [05/28/2023] Thank [chenxwh](https://github.com/chenxwh), add a [Replicate demo](https://replicate.com/cjwbw/controlvideo)!
* [05/25/2023] Code [ControlVideo](https://github.com/YBYBZhang/ControlVideo/) released!
* [05/23/2023] Paper [ControlVideo](https://arxiv.org/abs/2305.13077) released!

## Setup

### 1. Download Weights
All pre-trained weights are downloaded to `checkpoints/` directory, including the pre-trained weights of [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), ControlNet 1.0 conditioned on [canny edges](https://huggingface.co/lllyasviel/sd-controlnet-canny), [depth maps](https://huggingface.co/lllyasviel/sd-controlnet-depth), [human poses](https://huggingface.co/lllyasviel/sd-controlnet-openpose), and ControlNet 1.1 in [here](https://huggingface.co/lllyasviel). 
The `flownet.pkl` is the weights of [RIFE](https://github.com/megvii-research/ECCV2022-RIFE).
The final file tree likes:

```none
checkpoints
├── stable-diffusion-v1-5
├── sd-controlnet-canny
├── sd-controlnet-depth
├── sd-controlnet-openpose
├── ...
├── flownet.pkl
```
### 2. Requirements

```shell
conda create -n controlvideo python=3.10
conda activate controlvideo
pip install -r requirements.txt
```
Note: `xformers` is recommended to save memory and running time. `controlnet-aux` is updated to version 0.0.6.

## Inference

To perform text-to-video generation, just run this command in `inference.sh`:
```bash
python inference.py \
    --prompt "A striking mallard floats effortlessly on the sparkling pond." \
    --condition "depth" \
    --video_path "data/mallard-water.mp4" \
    --output_path "outputs/" \
    --video_length 15 \
    --smoother_steps 19 20 \
    --width 512 \
    --height 512 \
    --frame_rate 2 \
    --version v10 \
    # --is_long_video
```
where `--video_length` is the length of synthesized video, `--condition` represents the type of structure sequence,
`--smoother_steps` determines at which timesteps to perform smoothing, `--version` selects the version of ControlNet (e.g., `v10` or `v11`), and `--is_long_video` denotes whether to enable efficient long-video synthesis.

## Visualizations

### ControlVideo on depth maps

<table class="center">
<tr>
  <td width=30% align="center"><img src="assets/depth/A_charming_flamingo_gracefully_wanders_in_the_calm_and_serene_water,_its_delicate_neck_curving_into_an_elegant_shape..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/depth/A_striking_mallard_floats_effortlessly_on_the_sparkling_pond..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/depth/A_gigantic_yellow_jeep_slowly_turns_on_a_wide,_smooth_road_in_the_city..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A charming flamingo gracefully wanders in the calm and serene water, its delicate neck curving into an elegant shape."</td>
  <td width=30% align="center">"A striking mallard floats effortlessly on the sparkling pond."</td>
  <td width=30% align="center">"A gigantic yellow jeep slowly turns on a wide, smooth road in the city."</td>
</tr>
 <tr>
	<td width=30% align="center"><img src="assets/depth/A_sleek_boat_glides_effortlessly_through_the_shimmering_river,_van_gogh_style..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/depth/A_majestic_sailing_boat_cruises_along_the_vast,_azure_sea..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/depth/A_contented_cow_ambles_across_the_dewy,_verdant_pasture..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A sleek boat glides effortlessly through the shimmering river, van gogh style."</td>
  <td width=30% align="center">"A majestic sailing boat cruises along the vast, azure sea."</td>
  <td width=30% align="center">"A contented cow ambles across the dewy, verdant pasture."</td>
</tr>
</table>

### ControlVideo on canny edges

<table class="center">
<tr>
  <td width=30% align="center"><img src="assets/canny/A_young_man_riding_a_sleek,_black_motorbike_through_the_winding_mountain_roads..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/canny/A_white_swan_moving_on_the_lake,_cartoon_style..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/canny/A_dusty_old_jeep_was_making_its_way_down_the_winding_forest_road,_creaking_and_groaning_with_each_bump_and_turn..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A young man riding a sleek, black motorbike through the winding mountain roads."</td>
  <td width=30% align="center">"A white swan movingon the lake, cartoon style."</td>
  <td width=30% align="center">"A dusty old jeep was making its way down the winding forest road, creaking and groaning with each bump and turn."</td>
</tr>
 <tr>
  <td width=30% align="center"><img src="assets/canny/A_shiny_red_jeep_smoothly_turns_on_a_narrow,_winding_road_in_the_mountains..gif" raw=true></td>
  <td width=30% align="center"><img src="assets/canny/A_majestic_camel_gracefully_strides_across_the_scorching_desert_sands..gif" raw=true></td>
	<td width=30% align="center"><img src="assets/canny/A_fit_man_is_leisurely_hiking_through_a_lush_and_verdant_forest..gif" raw=true></td>
</tr>
<tr>
  <td width=30% align="center">"A shiny red jeep smoothly turns on a narrow, winding road in the mountains."</td>
  <td width=30% align="center">"A majestic camel gracefully strides across the scorching desert sands."</td>
  <td width=30% align="center">"A fit man is leisurely hiking through a lush and verdant forest."</td>
</tr>
</table>


### ControlVideo on human poses

<table class="center">
<tr>
  <td width=25% align="center"><img src="assets/pose/James_bond_moonwalk_on_the_beach,_animation_style.gif" raw=true></td>
  <td width=25% align="center"><img src="assets/pose/Goku_in_a_mountain_range,_surreal_style..gif" raw=true></td>
	<td width=25% align="center"><img src="assets/pose/Hulk_is_jumping_on_the_street,_cartoon_style.gif" raw=true></td>
  <td width=25% align="center"><img src="assets/pose/A_robot_dances_on_a_road,_animation_style.gif" raw=true></td>
</tr>
<tr>
  <td width=25% align="center">"James bond moonwalk on the beach, animation style."</td>
  <td width=25% align="center">"Goku in a mountain range, surreal style."</td>
  <td width=25% align="center">"Hulk is jumping on the street, cartoon style."</td>
  <td width=25% align="center">"A robot dances on a road, animation style."</td>
</tr></table>

### Long video generation

<table class="center">
<tr>
  <td width=60% align="center"><img src="assets/long/A_steamship_on_the_ocean,_at_sunset,_sketch_style.gif" raw=true></td>
	<td width=40% align="center"><img src="assets/long/Hulk_is_dancing_on_the_beach,_cartoon_style.gif" raw=true></td>
</tr>
<tr>
  <td width=60% align="center">"A steamship on the ocean, at sunset, sketch style."</td>
  <td width=40% align="center">"Hulk is dancing on the beach, cartoon style."</td>
</tr>
</table>

## Citation
If you make use of our work, please cite our paper.
```bibtex
@article{zhang2023controlvideo,
  title={ControlVideo: Training-free Controllable Text-to-Video Generation},
  author={Zhang, Yabo and Wei, Yuxiang and Jiang, Dongsheng and Zhang, Xiaopeng and Zuo, Wangmeng and Tian, Qi},
  journal={arXiv preprint arXiv:2305.13077},
  year={2023}
}
```

## Acknowledgement
This work repository borrows heavily from [Diffusers](https://github.com/huggingface/diffusers), [ControlNet](https://github.com/lllyasviel/ControlNet), [Tune-A-Video](https://github.com/showlab/Tune-A-Video), and [RIFE](https://github.com/megvii-research/ECCV2022-RIFE).
The code of HuggingFace demo borrows from [fffiloni/ControlVideo](https://huggingface.co/spaces/fffiloni/ControlVideo).
Thanks for their contributions!

There are also many interesting works on video generation: [Tune-A-Video](https://github.com/showlab/Tune-A-Video), [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero), [Follow-Your-Pose](https://github.com/mayuelala/FollowYourPose), [Control-A-Video](https://github.com/Weifeng-Chen/control-a-video), et al.

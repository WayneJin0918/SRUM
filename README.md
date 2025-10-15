<p align="center">
  <img src="assets/srum_log_2.png" alt="SRUM" width="220"/>
</p>

<p align="center">
  <a href="https://waynejin0918.github.io/srum_web/">
    <img
      src="https://img.shields.io/badge/SRUM-Website-blue"
      alt="SRUM Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2510.12784">
    <img
      src="https://img.shields.io/badge/SRUM-Paper-red"
      alt="SRUM Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/Wayne-King/SRUM_BAGEL_7B_MoT">
    <img 
        src="https://img.shields.io/badge/SRUM-HF%20Model-yellow" 
        alt="SRUM Model"
    />
  </a>
  <a href="https://huggingface.co/datasets/Wayne-King/SRUM_6k_CompBench_Train">
    <img 
        src="https://img.shields.io/badge/SRUM-HF%20Datasets-yellow" 
        alt="SRUM Data"
    />
  </a>
  <a href="https://www.modelscope.cn/models/SOTAowner/SRUM_7B_BAGEL">
    <img 
        src="https://img.shields.io/badge/SRUM-MS%20Model-8A2BE2" 
        alt="SRUM Model"
    />
  </a>
  <a href="https://www.modelscope.cn/datasets/SOTAowner/SRUM_image_t2i_comp_6k">
    <img 
        src="https://img.shields.io/badge/SRUM-MS%20Datasets-8A2BE2" 
        alt="SRUM Model"
    />
  </a>

</p>

# SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models
> [Weiyang Jin*](https://github.com/WayneJin0918), [Yuwei Niu*](https://purshow.github.io/), Jiaqi Liao, [Chengqi Duan](https://scholar.google.com/citations?user=r9qb4ZwAAAAJ&hl=en), Aoxue Li, [Shenghua Gao](https://scholar.google.com/citations?user=fe-1v0MAAAAJ&hl=en), [Xihui Liu :email: ](https://xh-liu.github.io/)
>
> contact: xihuiliu@hku.hk
> 
> We present **SRUM**, a post-training reward fine-tuning method based on Unified Multimodal Models (UMMs) leverages UMMs' inherent understanding capabilities to boost their generative abilities, bridging the gaps in performance caused by conflicts during the previous training phase. SRUM demonstrates exceptional generalization across both common positions and world knowledge..
The figure below showcases SRUM's qualitative performance compared with SFT and Base Model.

<!-- ## 🧠 Method
BAGEL adopts a Mixture-of-Transformer-Experts (MoT) architecture to maximize the model’s capacity to learn from richly diverse multimodal information. Following the same principle of capacity maximization, it utilizes two separate encoders to capture pixel-level and semantic-level features of an image. The overall framework follows a Next Group of Token Prediction paradigm, where the model is trained to predict the next group of language or visual tokens as a compression target.

BAGEL scales MoT’s capacity through Pre-training, Continued Training, and Supervised Finetuning on trillions of interleaved multimodal tokens spanning language, image, video, and web data. It surpasses open models on standard understanding and generation benchmarks and demonstrates advanced in-context multimodal abilities like free-form image editing, future frame prediction, 3D manipulation, world navigation, and sequential reasoning.

<p align="center"><img src="assets/arch.png" width="95%"></p>


## 🌱 Emerging Properties
<p align="center"><img src="assets/emerging_curves.png" width="95%"></p>

As we scale up BAGEL’s pretraining with more multimodal tokens, we observe consistent performance gains across understanding, generation, and editing tasks. Different capabilities emerge at distinct training stages—multimodal understanding and generation appear early, followed by basic editing, while complex, intelligent editing emerges later. This staged progression suggests an emergent pattern, where advanced multimodal reasoning builds on well-formed foundational skills. Ablation studies further show that combining VAE and ViT features significantly improves intelligent editing, underscoring the importance of visual-semantic context in enabling complex multimodal reasoning and further supporting its role in the emergence of advanced capabilities. -->

## 📢 News

We sincerely thank all contributors from the open community for their valuable support.

- **Nov. 15, 2025:** We released the official [website](https://waynejin0918.github.io/srum_web/), [model](https://huggingface.co/Wayne-King/SRUM_BAGEL_7B_MoT), and [report](https://arxiv.org/abs/2510.12784) for SRUM. And please upvote for our [huggingface daily paper](https://huggingface.co/papers/2510.12784)


## 📮 Notice
<!-- **Call for Bad Cases:** If you have encountered any cases where the model performs poorly, we would greatly appreciate it if you could share them in the [issue#11](https://github.com/ByteDance-Seed/Bagel/issues/11) or [Discord](https://discord.gg/Z836xxzy). -->
Follow the Bagel's original settings, you should focus:

**About Inference Hyperparameters:**
- **`cfg_text_scale`:** Controls how strongly the model follows the text prompt. `1.0` disables text guidance. Typical range: `4.0–8.0`.
- **`cfg_image_scale`:** Controls how much the model preserves input image details. `1.0` disables image guidance. Typical range: `1.0–2.0`.
- **`cfg_interval`:** Fraction of denoising steps where CFG is applied. Later steps can skip CFG to reduce computation. Typical: `[0.4, 1.0]`.
- **`timestep_shift`:** Shifts the distribution of denoising steps. Higher values allocate more steps at the start (affects layout); lower values allocate more at the end (improves details).
- **`num_timesteps`:** Total denoising steps. Typical: `50`.
- **`cfg_renorm_min`:** Minimum value for CFG-Renorm. `1.0` disables renorm. Typical: `0`.
- **`cfg_renorm_type`:** CFG-Renorm method:  
  - `global`: Normalize over all tokens and channels (default for T2I).
  - `channel`: Normalize across channels for each token.
  - `text_channel`: Like `channel`, but only applies to text condition (good for editing, may cause blur).
- **If edited images appear blurry, try `global` CFG-Renorm, decrease `cfg_renorm_min` or decrease `cfg_scale`.**


## 🔥 Quick Start

1️⃣  Set up environment
```bash
git clone https://github.com/WayneJin0918/SRUM
cd SRUM
conda env create -f environment.yaml
conda activate SRUM
pip install -r requirements.txt
```
if flash attention is hard to pip, please follow:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Or you can follow the settings of Bagel

2️⃣  Download Bagel pretrained or our SRUM checkpoint
```python
#bagel
from huggingface_hub import snapshot_download

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

```

```python
#SRUM
from huggingface_hub import snapshot_download

save_dir = "models/SRUM_BAGEL_7B_MoT"
repo_id = "Wayne-King/SRUM_BAGEL_7B_MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

```
<!-- 3️⃣ Use Gradio WebUI to start playing with BAGEL!
```bash
# For 32GB+ VRAM GPU or multi GPUs.
python app.py
```

```bash
# For 12~32GB VRAM GPU, recommend using NF4 quantization. And use Chinese interface.
python app.py --mode 2 --zh
```

```bash
# For 22~32GB VRAM GPU, not recommended to use INT8 quantization.
python app.py  --mode 3
``` -->

## 🔥 Train & Eval

### Train

1️⃣  Data preparation

Use `srum_data_infer/compt2i.sh` for images inference in multi-gpus. Please change the output file address `--output_dir` as `./your_images_address`

```bash
bash srum_data_infer/compt2i.sh
```
Then you will get the image folder `./your_images_address` and next use `srum_data_infer/vlm.sh` for scoring. generally, `--image_dir` in bash file should same as `./your_address`. 

Before using vlm inference, you should download the SAM weights under SRUM

```bash
wget https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth
```

```bash
bash srum_data_infer/vlm.sh
```
Now, you have jsonl file `your_vlm_output.jsonl` and image folder `./your_images_address`, add these into `data/dataset_info.py`.

```python
        'comp_data': {
            'jsonl_path': './your_vlm_output.jsonl',
            # Replace 'image_base_dir' with the 'image_dirs' dictionary
            'image_dirs': {
                # Key 'good' for the ground-truth images
                'good': './your_images_address',
                # Key 'bad' for the input images that have rewards same as good one
                'bad': './your_images_address' 
            },
            'num_total_samples': 5911, # total number of samples in dataset
        },

```

Or you can directly use our [HF training data](https://huggingface.co/datasets/Wayne-King/SRUM_6k_CompBench_Train) in huggingface or [MS training data](https://huggingface.co/datasets/Wayne-King/SRUM_6k_CompBench_Train) in modelscape.

2️⃣  Starting training

Down the base model. Then, add yaml file: `scripts/data/rft_comp.yaml`.

```python
regional_reward:
  dataset_names:
  - comp_data
  image_transform_args:
    image_stride: 256
    max_image_size: 1024
    min_image_size: 512
  num_used_data: # The sum should be larger that NUM_GPUS x NUM_WORKERS
  - 8
  weight: 1

```

```bash
bash scripts/train_reg_comp.sh
```

Please do not forger to change the `PYTHONPATH` to your root SRUM path like `/mnt/SRUM`. If you are not using 8 GPUs in one node, please change the `--num_shard` to your number of GPUs.

And we highly recommand max of `--save_every` is `--total_steps` minus one.

3️⃣  Trans to hf weights

```bash
bash tool/trans2hf.sh 
```

If you want use generated data to SFT the base model, please use `tool/trans2parquet.py` to change images and jsons into parquet.

You can replace the variables in the script with your own before running. 
See [TRAIN](TRAIN.md) for more details.

### Eval
Bagel provide the scripts for evaluating VLM, T2I and Editing benchmarks. 
Please See [EVAL](EVAL.md) for more details.

And if you want eval on T2I-CompBench, referring using file in `SRUM/CompBench_eval`, it is easy to start. We highly recommend use `Qwen2.5-VL-72B-Instruct` for evaluation, but you also can use `Qwen2.5-VL-32B-Instruct` for instead when having not enough memory, the conclusions and overall scores are similar.

Then, run the following command:

```shell
bash CompBench_eval/comp_eval_infer.sh
```

the image output will be saved in `BASE_OUTPUT_DIR`.

```shell
bash CompBench_eval/qwen_eval.sh
```

the score output will be saved in `OUTPUT_DIR`.


## 📊 Benchmarks

### 1. Composition

| T2I Model | 3d spatial | Color | Complex | Nonspatial | Numeracy | Shape | Spatial | Texture | Overall |
|-------|-----------|-------|---------|------------|----------|-------|---------|---------|---------|
| FLUX.1-dev | 76.39 | 90.63 | 83.51 | 87.47 | 75.30 | 80.20 | 84.23 | 87.07 | 83.10 |
| FLUX.1-schnell | 79.38 | 84.53 | 81.96 | 85.55 | 72.82 | 82.20 | 85.49 | 86.38 | 82.29 |
| SD-3-medium | 77.83 | 91.63 | 84.73 | 86.12 | 72.80 | 83.72 | 88.20 | 89.03 | 84.26 |
| SD-xl-base-1 | 72.25 | 77.75 | 75.00 | 85.28 | 57.14 | 72.18 | 77.08 | 78.38 | 74.38 |

| Unified Model | 3d spatial | Color | Complex | Nonspatial | Numeracy | Shape | Spatial | Texture | Overall |
|-------|-----------|-------|---------|------------|----------|-------|---------|---------|---------|
| Janus-Pro | 76.17 | 84.25 | 80.28 | 80.47 | 56.43 | 65.14 | 79.67 | 69.67 | 74.01 |
| Show-o2 | 88.61 | 87.73 | 87.88 | 85.91 | 69.74 | 73.99 | 86.60 | 82.17 | 82.83 |
| BLIP3o | 81.73 | 89.92 | 85.55 | 84.78 | 71.67 | 83.75 | 92.47 | 87.45 | 84.66 |
| OmniGen2 | 82.21 | 92.22 | 86.87 | 88.51 | 72.00 | 83.95 | 90.07 | 90.88 | 85.84 |
| Bagel | 77.98 | 89.30 | 83.32 | 85.03 | 70.40 | 81.94 | 81.52 | 87.93 | 82.18 |
| Bagel (CoT) | 84.66 | 88.85 | 86.10 | 85.64 | 75.36 | 84.33 | 82.71 | 88.07 | 84.46 |
| BLIP3o+SRUM | 83.78↑ | 90.22↑ | 86.57↑ | 85.10↑ | 74.52↑ | 85.44↑ | 93.88↑ | 86.52↓ | 85.75↑ |
| Bagel+SRUM | 83.10↑ | 92.90↑ | 88.69↑ | 88.47↑ | 78.52↑ | 84.23↑ | 86.92↑ | 89.57↑ | 86.55↑ |
| Bagel+SRUM (CoT) 🏆| 88.60↑ | 92.90↑ | 91.31↑ | 90.48↑ | 80.12↑ | 84.47↑ | 89.93↑ | 89.15↑ | 88.37↑ |

### 2. Reasoning-informed


| **Model** | **Entity** | **Idiom** | **Scientific** | **Textual Image** | **Average** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bagel | 49.70 | 34.46 | 47.52 | 43.59 | 43.82 |
| Bagel+SFT | 50.53 | 39.43 | 47.45 | 44.08 | 45.37 |
| Bagel+SRUM | **52.85** | **40.51** | **47.83** | **45.83** | **46.75** |

*Performance comparison of Bagel models across four categories and their average scores. **Bold values** indicate the best performance in each column.*

## ✍️ Citation

```bibtex
@article{deng2025bagel,
  title   = {SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models},
  author  = {Jin, Weiyang and Niu, Yuwei and Liao, Jiaqi and Duan, Chengqi and Li, Aoxue and Gao, Shenghua and Liu, Xihui},
  journal = {arXiv preprint arXiv:2510.12784},
  year    = {2025}
}
```

## 📜 License
SRUM is licensed under the Apache 2.0.

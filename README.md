
# 🎤 MultiAPI-Spoof: Multi-API Audio Anti-Spoofing Dataset & Nes2Net-LA

[🔗 Project Code](https://github.com/XuepingZhang/MultiAPI-Spoof) 
[📥 Dataset Download on Google Drive](https://drive.google.com/file/d/1d1MCt6ZYKv90XZic4_KvOD67RDoi2m5-/view?usp=sharing)
[📂 Dataset Details & Audio Examples](https://xuepingzhang.github.io/MultiAPI-Spoof-Dataset/)
[💾 Pretrained Models (XLSR-Nes2Net-LA)](https://huggingface.co/XuepingZhang/XLSR-Nes2Net-LA/)

---

## 🌟 Overview

This repository contains code **MultiAPI Spoof**, a multi-API audio anti-spoofing dataset, and the **Nes2Net-LA** model, a **local-attention enhanced anti-spoofing network**.  

Existing speech anti-spoofing datasets mostly rely on a limited set of public TTS/VC models, creating a gap from real-world scenarios where commercial systems use diverse, proprietary APIs. **MultiAPI Spoof** addresses this by including **~230 hours** of synthetic speech from **30 APIs**, covering commercial TTS services, open-source models, and online TTS platforms.  

Two tasks are provided:  
1. **🎯 Anti-Spoofing Detection:** Classify bona fide vs. spoofed audio.  
2. **🕵️‍♂️ API Tracing:** Identify which API generated a given spoofed sample.  

**Nes2Net-LA** improves Nes2Net-X by adding **local attention**, boosting local context modeling and fine-grained spoofing feature extraction.

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/XuepingZhang/MultiAPI-Spoof.git
cd MultiAPI-Spoof

# Create conda environment
conda create -n multiapi-spoof python=3.10
conda activate multiapi-spoof

#install fairseq
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./

# Install dependencies
pip install -r requirements.txt
````


---

## 🚀 Quick Start

**Pretrained XLSR**

The pretrained model XLSR can be found at this [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

### 🟢 Anti-Spoofing Training

```bash
cd job/xlsr2_nes2net_ATT
sbatch sh/multi_train.slurm
```

### 🟢 API Tracing Training

```bash
cd job/api_tracing
sbatch sh/multi_train.slurm
```

### 🟢 Inference / Evaluation

```bash
cd job/xlsr2_nes2net_ATT
sbatch sh/multi_eval.slurm
```

---

## 📊 Experimental Results

### Anti-Spoofing Performance

| Model           | ITW                  | MultiAPI Spoof       | AI4T                 |
| --------------- |----------------------|----------------------|----------------------|
| XLSR+Nes2Net-LA | 1.42 / 0.020 / 0.021 | 0.56 / 0.008 / 0.008 | 5.64 / 0.051 / 0.077 | 

*Metrics: EER / minDCF / actDCF*

### API Tracing Performance

| Class                 | Precision | Recall | F1    |
| --------------------- | --------- | ------ | ----- |
| Seen APIs (A0–A20)    | 0.950     | 0.923  | 0.936 |
| Unseen APIs (A24–A29) | 0.972     | 0.520  | 0.678 |
| Overall               | 0.770     | 0.917  | 0.782 |

---

## 📖 Citation

If you use this code or dataset, please cite:

```

```


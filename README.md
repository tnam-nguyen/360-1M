# 360-1M

## Overview
This repository code for downloading the 360-1M dataset, processing the videos, and training the ODIN model. The metadata with URLs for all videos can be found at the following link:

[Metadata with Video URLs](https://huggingface.co/datasets/mwallingford/360-1M/tree/main) 

---

## Installation Guide

### Environment Setup
a. Create a new Conda environment:
   ```
   conda create -n ODIN python=3.9
   conda activate ODIN
```
b. Clone the repository:

```bash
cd ODIN
pip install -r requirements.txt
```
c. Install additional dependencies:

```bash
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
```
```bash
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
Processing Videos
Install MAST3R and yt-dlp
```
a. Clone the MAST3R repository:
```bash
git clone --recursive https://github.com/naver/mast3r
cd mast3r
```
b. Install MAST3R dependencies:

```bash
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
For detailed installation instructions, visit the MAST3R repository.
```

c. Install yt-dlp
```bash
pip install yt-dlp
```


## Data Download and Preprocessing
The videos can be downloaded using the provided script:

```bash
python Downloads/Download_GCP.py --path 360-1M.parquet
```
The total size of all videos at max resolution is about 200 TB. Currently we supply a script for downloading videos at scale to GCP. We will soon release code for downloading a smaller, high-quality subset to facilitate those who want to work with a smaller version of 360-1M.

### Extracting Frames
To extract frames from videos, use the video_to_frames.py script:


```bash
python video_to_frames.py --path /path/to/videos --out /path/to/frames
```

Extracting Pairwise Poses
Once frames are extracted, pairwise poses can be calculated using:

```bash
python extract_poses.py --path /path/to/frames
```

## Training
Download Stable Diffusion Checkpoint
Download the image-conditioned Stable Diffusion checkpoint released by Lambda Labs:

```bash
wget https://cv.cs.columbia.edu/zero123/assets/sd-image-conditioned-v2.ckpt
```
Start Training
Run the training script:

```bash
python main.py \
    -t \
    --base configs/sd-ODIN-finetune-c_concat-256.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from sd-image-conditioned-v2.ckpt
```
### Coming Soon
- Smaller, high quality subset for easier downloading.
- Model checkpoints and evaluation code. 

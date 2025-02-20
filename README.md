
## [Website](https://mattwallingford.github.io/ODIN/) | [HuggingFace](https://huggingface.co/datasets/mwallingford/360-1M/tree/main) | [Paper](https://openreview.net/pdf?id=otxOtsWCMb) | 

**360-1M** is a large-scale 360° video dataset consisting of over 1 million videos for training video and 3D foundation models. This repository contains the following:
1. Links to the videos URLs for download from YouTube. We also provide a smaller 24k filtered subset for experimentation.
2. Metadata for each video including category, resolution, and views. 
3. Code for downloading the videos locally and to Google Cloud Platform (recommended).
4. Code for filtering, processing, and obtaining camera pose for the videos.
5. Code for training the novel view synthesis model, [ODIN](https://openreview.net/pdf?id=otxOtsWCMb).
   
| **Reference Image**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/nyc-256x256.png" width="256" alt="NYC Reference" /><br>**Generated Scene Trajectory**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/nyc4.gif" width="256" alt="NYC Demo" /> | **Reference Image**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/livingroom-256x256.jpg" width="256" alt="Living Room Reference" /><br>**Generated Scene Trajectory**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/living_room_zoom.gif" width="256" alt="Living Room Demo" /> | **Reference Image**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/picnic-256x256.png" width="256" alt="Picnic Reference" /><br>**Generated Scene Trajectory**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/picnic2.gif" width="256" alt="Picnic Demo" /> |
| --- | --- | --- |


## Setting up a Python environment using Conda
```
conda create -n 360-1M python=3.10 ffmpeg
conda activate 360-1M
pip install -r requirements.txt
```
## Downloading Videos
Metadata and video URLs can be downloaded from here: [Metadata with Video URLs](https://huggingface.co/datasets/mwallingford/360-1M/tree/main) .
The filtered subset which is around 5 TB in size can be found here: [Filtered Subset](https://huggingface.co/datasets/mwallingford/360-1M/blob/main/Filtered_24k.parquet)




To download the high quality subset:
```bash
python DownloadVideos/download_local.py --in_path Filtered_24k.parquet --out_dir /path/to/videos
```

The total size of all videos at max resolution is about 200 TB. We recommend downloading to a cloud platform due to bandwidth limitations and provide a script for use with GCP.

```bash
python DownloadVideos/Download_GCP.py --path 360-1M.parquet
```
---

## Installation Guide for Video Processing And Training

### Extracting Frames
To extract frames from videos, use the video_to_frames.py script:

```bash
python video_to_frames.py /path/to/videos  /path/to/frames 
```


### Generating Camera Poses using mast3r

#### 

```
1 - Clone the repository
cd VideoProcessing
mkdir third_party
cd third_party
git clone --recursive https://github.com/naver/mast3r

2 - Download model checkpoint
cd mast3r
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/

3 - Go back to the top directory
cd ../../../
```

Then, run
```
python extract_poses.py  --root-folder /path/to/frames --output-root /path/to/pose_out
```



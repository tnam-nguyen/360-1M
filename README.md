
## [Website](https://mattwallingford.github.io/ODIN/) | [HuggingFace](https://huggingface.co/datasets/mwallingford/360-1M/tree/main) | [Paper](https://openreview.net/pdf?id=otxOtsWCMb) | 

**360-1M** is a large-scale 360° video dataset consisting of over 1 million videos for training video and 3D foundation models. This repository contains the following:
1. Links to the videos URLs for download from YouTube. We also provide a smaller 24k filtered subset for experimentation.
2. Metadata for each video including category, resolution, and views. 
3. Code for downloading the videos locally and to Google Cloud Platform (recommended).
4. Code for filtering, processing, and obtaining camera pose for the videos.
5. Code for training the novel view synthesis model, [ODIN](https://openreview.net/pdf?id=otxOtsWCMb).
   
| **Reference Image**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/nyc-256x256.png" width="256" alt="NYC Reference" /><br>**Generated Scene Trajectory**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/nyc4.gif" width="256" alt="NYC Demo" /> | **Reference Image**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/livingroom-256x256.jpg" width="256" alt="Living Room Reference" /><br>**Generated Scene Trajectory**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/living_room_zoom.gif" width="256" alt="Living Room Demo" /> | **Reference Image**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/picnic-256x256.png" width="256" alt="Picnic Reference" /><br>**Generated Scene Trajectory**<br><img src="https://raw.githubusercontent.com/MattWallingford/ODIN/main/picnic2.gif" width="256" alt="Picnic Demo" /> |
| --- | --- | --- |


## Setting up a Conda environment with FFMPEG
```
conda create -n 360-1M python=3.10 ffmpeg
conda activate 360-1M
```
## Downloading Videos
Metadata and video URLs can be downloaded from here: [Metadata with Video URLs](https://huggingface.co/datasets/mwallingford/360-1M/tree/main) .
The filtered subset which is around 5 TB in size can be found here: [Filtered Subset](https://huggingface.co/datasets/mwallingford/360-1M/blob/main/Filtered_24k.parquet)

To download the videos we recommend using the yt-dlp package. To run our download scripts you'll also need pandas and pyarrow to parse the metadata parquet:
```bash
#Install packages for downloading videos
pip install -r requirements.txt
```

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
python video_to_frames_eqr.py /path/to/videos  /path/to/frames 
```

To extract frames up to a maximum duration (in seconds), with a specific FPS and resolution:

```bash
python video_to_frames_eqr.py /path/to/videos  /path/to/frames --fps 30 --max_duration 10 --max_height 1024
```

### Extracting EQR Poses using SphereSFM

#### Build SphereSFM -- A modified version of COLMAP with spherical images support

```
### Tested on Ubuntu 22.04 WSL 2 -- CUDA-supported build ###

#1 - Install dependencies
sudo apt-get install -y \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev \
    libceres-dev libboost-all-dev

#2 - Install CUDA Toolkit and CUDA Compiler
sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

#3 - Set up GCC/G++ 10 as the compiler (for Ubuntu 22.04)
sudo apt-get install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10

#4 - Clone and Set Up SphereSfM
git clone https://github.com/json87/SphereSfM.git
cd SphereSfM
mkdir build
cd build

#5 - Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

#6 - Start CMake, replace XX with your GPU compute capability from the previous step
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=XX

#7 - Start the build
ninja
sudo ninja install
```



After building SphereSFM, we can get image poses from the extracted EQR by using:

```bash
python extract_poses_eqr.py  /path/to/frames /path/to/colmap_out
```
The camera poses of all frames of the scene should be in `/path/to/colmap_out/scene_name/sparse/0/cameras.bin`

To visualize the COLMAP output for each scene
```
colmap gui --database_path ./path/to/colmap_out/scene_name/database.db --image_path  ./path/to/frames/scene_name/ --import_path ./path/to/colmap_out/scene_name/sparse/0
```



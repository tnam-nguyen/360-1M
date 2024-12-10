import os
import subprocess
# import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from google.cloud import storage
import psutil
import json
import time
from pathlib import Path


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = #


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def read_jsonl(path):
    lines = []
    with open(path, 'r') as jsonl_file:
        for line in jsonl_file:
            lines.append(json.loads(line))
    
    return lines


def upload_blob(bucket_name, source_file_name, destination_blob_name, max_retries=5):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    for retry in range(max_retries):
        try:
            blob.upload_from_filename(source_file_name)
            return
        except Exception as e: 
            if retry < max_retries - 1:
                print(f"Retrying uploading blob {destination_blob_name} to bucket {bucket_name}...")
                time.sleep(5)


def list_blobs_by_prefix(bucket_name, prefix):
    """Lists all the blobs in the bucket under a specified directory (prefix)."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name) 
    # Note: The delimiter argument ensures that only blobs in the 'directory' are listed
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    return blobs


def download_video_parallel(
    video_ids, 
    output_folder="./mp4_videos_full/", 
    max_workers=5,
    root_blob_dir="videos",
    bucket_name="360-videos",
    **kwargs
    ):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_id in video_ids:
            output_file = os.path.join(output_folder, f"{video_id}.mp4")
            futures.append(executor.submit(
                download_video, 
                video_id, 
                output_file, 
                root_blob_dir, 
                bucket_name,
                **kwargs
            ))
        for future in tqdm(futures, total=len(video_ids), desc="Downloading videos"):
            future.result()


def download_video(
    video_id: str,
    output_file: str = None,
    root_blob_dir: str = "videos",
    bucket_name: str = "360-videos",
    max_height: int = 600,
    include_audio: bool = False,
    start_time: int = None,
    end_time: int = None,
    fps: int = None,
    quiet: bool = True,
    check_output: bool = False,
    surpress_output: bool = True,

) -> None:
    # if output_file is None:
    #     output_file = os.path.join("./mp4_videos_full/", f"{video_id}.mp4")
    assert output_file is not None, "output_file must be provided."
    # Check if the output file already exists, and skip the download if it does
    if os.path.exists(output_file):
        print(f"Video {video_id} already exists at {output_file}. Skipping download.")
        return
    dirname = os.path.dirname(output_file)
    if dirname != "":
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Construct the format string based on whether audio is to be included
    # if include_audio:
    #     format_str = (
    #         f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"
    #     )
    # else:
    #     format_str = f"bestvideo[height<={max_height}][ext=mp4]/best[ext=mp4]"
    # # Add download sections if start_time and end_time are provided
    # if start_time is not None and end_time is not None:
    #     # need https for start_time and end_time
    #     format_str = f"bestvideo[height<={max_height}][protocol*=https]"
    format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"
    # Base command
    command = [
        "yt-dlp",
        f"https://youtube.com/watch?v={video_id}",
        "-f", format_str,
        "-o", output_file,
        "--user-agent", "",
        "--force-keyframes-at-cuts",
    ]
    # Add download sections if start_time and end_time are provided
    if start_time is not None and end_time is not None:
        command.append("--download-sections")
        command.append(f"*{start_time}-{end_time}")
        # add -f bv*[protocol*=https]
        # command.append("-f")
        # command.append("bv*[protocol*=https]")
    if fps is not None:
        command.append("--downloader-args")
        command.append(f"ffmpeg:-filter:v fps={fps} -vcodec h264 -f mp4")
    # Suppress output if quiet is True
    if quiet:
        command.append("-q")
    # Run the command
    output_destination = subprocess.DEVNULL if surpress_output else None
    subprocess.run(
        command,
        check=check_output,
        stdout=output_destination,
        stderr=output_destination,
    )

    blob_name = f"{root_blob_dir}/{video_id}.mp4"
    # upload the video to the bucket and remove the video file
    upload_blob(bucket_name, output_file, blob_name)

    try: 
        os.remove(output_file)
    except Exception as e:
        print(f"Could not remove {output_file}. Error: {e}")



if __name__ == "__main__":
    # Example usage
    #video_ids = ["video_id_1", "video_id_2", "video_id_3"]  # Add your video IDs here
    num_cpu_cores = psutil.cpu_count(logical=False)
    # df = list(pd.read_parquet('~/video_list.snappy.parquet')['id'])
    root_blob_dir = os.environ.get("ROOT_BLOB_DIR")
    assert root_blob_dir is not None, "ROOT_BLOB_DIR not set."
    bucket_name = os.environ.get("BUCKET_NAME", "360-videos")
    yt_ids_fpath = os.environ.get("YT_IDS_FPATH")  # path to the jsonl file containing the youtube ids 

    print("ROOT_BLOB_DIR:", root_blob_dir)
    print("BUCKET_NAME:", bucket_name)
    print("YT_IDS_FPATH:", yt_ids_fpath)

    n_shards = os.environ.get("N_SHARDS")
    if n_shards is not None: 
        n_shards = int(n_shards)
        if n_shards == 1:
            shard_id = 0
        else:
            shard_id = int(os.environ.get("SHARD_ID"))
            if shard_id is None:
                raise ValueError(f"N_SHARDS = {n_shards} and SHARD_ID not set.")
    else:
        n_shards = 1
        shard_id = 0

    print(f"N_SHARDS: {n_shards}, SHARD_ID: {shard_id}")

    download_blob(bucket_name, yt_ids_fpath, "/app/yt_ids.jsonl")
    yt_ids_fpath = "/app/yt_ids.jsonl"

    try:
        yt_ids = read_jsonl(yt_ids_fpath)
        yt_ids = [el["id"] for el in yt_ids]

        # Remove already downloaded videos from the list
        downloaded_videos = list_blobs_by_prefix(bucket_name="360-videos", prefix=f"{root_blob_dir}/")
        downloaded_yt_ids = [Path(blob.name).stem for blob in downloaded_videos]
        yt_ids = list(set(yt_ids) - set(downloaded_yt_ids))
        yt_ids = yt_ids[shard_id::n_shards]
    except:
        raise RuntimeError(f"Error in reading {yt_ids_fpath}. Are you sure it's a jsonl file?")

    os.makedirs("/app/videos/", exist_ok=True)

    download_video_parallel(
        yt_ids,
        max_height=1000,
        include_audio=False,
        max_workers=num_cpu_cores * 1,
        output_folder="/app/videos/",
        bucket_name=bucket_name,
        root_blob_dir=root_blob_dir,
    )

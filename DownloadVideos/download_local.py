import os
import subprocess
import pandas as pd
import argparse

def download_video(
    video_id: str,
    output_file: str,
    max_height: int = 1000,
    include_audio: bool = False,
    start_time: int = None,
    end_time: int = None,
    fps: int = None,
    quiet: bool = True,
    surpress_output: bool = True,
):
    if os.path.exists(output_file):
        print(f"Video {video_id} already exists at {output_file}. Skipping download.")
        return

    dirname = os.path.dirname(output_file)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)


    format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"

    command = [
        "yt-dlp",
        f"https://youtube.com/watch?v={video_id}",
        "-f", format_str,
        "-o", output_file,
        "--user-agent", ""
    ]

    if start_time is not None and end_time is not None:
        command.append("--download-sections")
        command.append(f"*{start_time}-{end_time}")

    if fps is not None:
        command.append("--downloader-args")
        command.append(f"ffmpeg:-filter:v fps={fps} -vcodec h264 -f mp4")


    if quiet:
        command.append("-q")


    output_destination = subprocess.DEVNULL if surpress_output else None
    subprocess.run(
        command,
        stdout=output_destination,
        stderr=output_destination,
    )

    print(f"Downloaded {video_id} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos from a parquet file.")
    parser.add_argument('--in_path', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--start_time', type=int, help='Start time of the video (in seconds)', default=None)
    parser.add_argument('--end_time', type=int, help="End time of the video (in seconds)", default = None)
    parser.add_argument('--fps', type=int, help="FPS of the video", default=None)
    parser.add_argument('--start_idx', type=int, help="start index")
    parser.add_argument('--end_idx', type=int, help="end index")
    # parser.add_argument("--max_duration", type=int, help="Maximum duration (in seconds) to dowload for each video.", default=None)
    
    
    args = parser.parse_args()

    parquet_path = args.in_path
    output_folder = args.out_dir
    
    df = pd.read_parquet(parquet_path)
   
    yt_ids = df["id"].unique().tolist()
    if args.start_idx is not None and args.end_idx is not None:
        yt_ids = yt_ids[args.start_idx:args.end_idx]
    os.makedirs(output_folder, exist_ok=True)

    for idx, video_id in enumerate(yt_ids, start=1):
        if args.start_time is not None and args.end_time is not None:
            assert args.end_time > args.start_time
        if args.fps is not None:
            assert args.fps > 0
        output_file = os.path.join(output_folder, f"{video_id}.mp4")
        print(f"Downloading {idx}/{len(yt_ids)}: {video_id}")
        download_video(video_id, output_file=output_file, start_time=args.start_time, end_time=args.end_time, fps = args.fps)

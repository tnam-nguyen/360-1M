import os
import subprocess
import argparse

def extract_frames(input_dir, output_dir, fps, max_duration, max_height):
    """Extract frames from MP4 videos in input_dir and save them as PNGs in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_dir, filename)
            video_name = os.path.splitext(filename)[0]
            video_output_dir = os.path.join(output_dir, video_name)
            
           
            os.makedirs(video_output_dir, exist_ok=True)
            
            ffmpeg_command = [
                "ffmpeg",
                "-i", input_path,       # Input video
                "-vf", f"fps={fps}",  # Set frame rate
            ]

            if max_height is not None:
                ffmpeg_command[-1] += f",scale={max_height*2}:{max_height}"
            if max_duration is not None:
                ffmpeg_command.extend(["-t", str(max_duration)])  # Limit extraction to max_duration seconds
            
            ffmpeg_command.append(os.path.join(video_output_dir, "%04d.png"))  # Output frame pattern
            print(f"Processing {filename}...")
            subprocess.run(ffmpeg_command, check=True)
    
    print("Frame extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EQR frames from MP4 videos using FFmpeg.")
    parser.add_argument("input_dir", help="Path to the directory containing MP4 videos.")
    parser.add_argument("output_dir", help="Path to the directory where extracted frames will be stored.")
    parser.add_argument("--fps", type=int, help="Frames per second for extracted frames.", default=30)
    parser.add_argument("--max_duration", type=int, help="Maximum duration (in seconds) to extract frames from each video.", default=None)
    parser.add_argument("--max_height", type=int, default=None, help="Maximum height for resampling the video. If None, keep the original resolution.")
    
    args = parser.parse_args()
    extract_frames(args.input_dir, args.output_dir, args.fps, args.max_duration,args.max_height)
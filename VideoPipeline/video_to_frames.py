from moviepy.editor import VideoFileClip
from PIL import Image
import os
import numpy as np
from equilib import Equi2Pers

def parse_video(video_path, output_folder):

    # Load the video clip
    video_clip = VideoFileClip(video_path)
    
    # Initialize Equi2Pers
    equi2pers = Equi2Pers(
        height=512,
        width=512,
        fov_x=72,
        mode="bilinear",
    )
    
    # Extract frames at 1 frame every 3 seconds
    fps = 1 / 1  # Desired frames per second
    duration = video_clip.duration
    num_frames = int(duration * fps)
    
    # Calculate timestamps to extract frames at 1 frame every 3 seconds
    timestamps = np.linspace(0, duration, num_frames, endpoint=False)
    
    for i, timestamp in enumerate(timestamps):
        print(f"Processing frame {i} of video {video_path}")
        # Convert frame to the required format (RGB)
    
        # Rotate frame to capture 4 images per frame (yaw rotation at 90 degrees)
        for j in range(5):
            output_path = os.path.join(output_folder, f"frame_{i}_{j}.png")
            if not os.path.exists(output_path):
                frame = video_clip.get_frame(timestamp)
                equi_img = np.transpose(frame, (2, 0, 1))
                # Calculate rotation angles
                yaw = 2 * np.pi / 5 * j  # Rotate yaw at 90 degrees per image
                
                # Obtain perspective image
                pers_img = equi2pers(
                    equi=equi_img,
                    rots={'pitch':0, 'roll':0, 'yaw': yaw},  # Rotate yaw
                )
                
                # Convert to PIL image
                pers_img_pil = Image.fromarray(np.uint8(np.transpose(pers_img, (1, 2, 0))))  # Un-transpose
                
                # Save the frame as an image file if it doesn't exist already
                output_path = os.path.join(output_folder, f"frame_{i}_{j}.png")
                pers_img_pil.save(output_path)
            else:
                print(f"Skipping frame {i}_{j}.png as it already exists.")
        
    # Close the video clip
    video_clip.close()

def process_videos(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])
            if os.path.exists(video_output_folder):
                print(f"Video already processed: {video_output_folder}")
            else:
                os.makedirs(video_output_folder, exist_ok=True)
                parse_video(video_path, video_output_folder)

from moviepy import VideoFileClip
from PIL import Image
import os
import numpy as np
from equilib import Equi2Pers
import torch
import cv2
import argparse
from tqdm import tqdm



def imwrite(filepath, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, bgr)

def parse_video(video_path, output_folder, use_gpu= False):

    # Load the video clip
    video_clip = VideoFileClip(video_path)
    
    # Initialize Equi2Pers
    equi2pers = Equi2Pers(
        height=512,
        width=512,
        fov_x=72,
        mode="bilinear",
    )

    output_folder_pers = os.path.join(output_folder, "pers")
    output_folder_eqrs = os.path.join(output_folder, "eqrs")
    os.makedirs(output_folder_eqrs, exist_ok=True)
    os.makedirs(output_folder_pers, exist_ok=True)
    
    # Extract frames at 1 frame every 1 seconds
    
    fps = 1 / 1  # Desired frames per second
    duration = video_clip.duration
    num_frames = int(duration * fps)
    frame_iter = video_clip.iter_frames(fps = fps) ## Might be faster to retreive the frames rather than manual seeking
   
    
    for i, frame in enumerate(tqdm(frame_iter, video_path, total=num_frames)):
        # print(f"Processing frame {i} of video {video_path}")
        # Convert frame to the required format (RGB)
        equi_img = frame
        output_path = os.path.join(output_folder_eqrs, f"frame_{i}.png")
        imwrite(output_path,  np.uint8(equi_img))
        # equi_img_pil.save(output_path)        
        
        equi_img = np.transpose(equi_img, (2, 0, 1))
        if use_gpu:
            equi_img = torch.from_numpy(equi_img).cuda()


        # Rotate frame to capture 4 images per frame (yaw rotation at 90 degrees)
        for j in range(5):
            output_path = os.path.join(output_folder_pers, f"frame_{i}_{j}.png")
            if not os.path.exists(output_path):
               
                # Calculate rotation angles
                yaw = 2 * np.pi / 5 * j  # Rotate yaw at 90 degrees per image
                
                # Obtain perspective image
                pers_img = equi2pers(
                    equi=equi_img,
                    rots={'pitch':0, 'roll':0, 'yaw': yaw},  # Rotate yaw
                )
                
                if use_gpu:
                    pers_img = pers_img.cpu().numpy()
                
                # Convert to PIL image
                pers_img_array = np.uint8(np.transpose(pers_img, (1, 2, 0)))  # Un-transpose
                
                # Save the frame as an image file if it doesn't exist already
                output_path = os.path.join(output_folder_pers, f"frame_{i}_{j}.png")
                imwrite(output_path, pers_img_array)
            else:
                print(f"Skipping frame {i}_{j}.png as it already exists.")
        
    # Close the video clip
    video_clip.close()

def process_videos(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    ## Check if CUDA is availabe
    use_gpu = torch.cuda.is_available()
    print(f"Use CUDA: {use_gpu}")

    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])
            if os.path.exists(video_output_folder):
                print(f"Video already processed: {video_output_folder}")
            else:
                os.makedirs(video_output_folder, exist_ok=True)
                parse_video(video_path, video_output_folder, use_gpu)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract EQR frames from MP4 videos using FFmpeg.")
    parser.add_argument("input_dir", help="Path to the directory containing MP4 videos.")
    parser.add_argument("output_dir", help="Path to the directory where extracted frames will be stored.")
    args = parser.parse_args()

    process_videos(args.input_dir, args.output_dir)
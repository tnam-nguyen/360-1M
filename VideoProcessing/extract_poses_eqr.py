import os
import subprocess
import argparse
import numpy as np
import pandas as pd
from utils.read_write_model import read_images_binary

def extract_colmap_pose_to_txt_file(output_file_path, colmap_model):
    """
    Extracts camera poses from a COLMAP binary model and saves them to a text file in CSV format.

    This function reads the `images.bin` file from a COLMAP reconstruction, extracts the image names,
    quaternions (qw, qx, qy, qz), and translation vectors (tx, ty, tz), and saves the data to a CSV file.

    Args:
        output_file_path (str): The file path where the extracted camera poses will be saved as a CSV.
        colmap_model (str): The path to the COLMAP binary model (`images.bin`).

    Returns:
        None: The function writes the output directly to a CSV file.

    Example:
        extract_colmap_pose_to_txt_file("output/poses.csv", "colmap_model/sparse/0/images.bin")

    The output CSV file will have the following format:

        image_name,qw,qx,qy,qz,tx,ty,tz
        image1.jpg,0.98,0.01,-0.02,0.03,1.2,-0.5,3.1
        image2.jpg,0.97,0.02,-0.01,0.05,2.1,1.3,-0.7
        ...

    """
    # Read images.bin
    images = read_images_binary(colmap_model)

    # Extract data into a dictionary
    data = {
        "image_name": [image.name for image in images.values()],
        "qw": [image.qvec[0] for image in images.values()],
        "qx": [image.qvec[1] for image in images.values()],
        "qy": [image.qvec[2] for image in images.values()],
        "qz": [image.qvec[3] for image in images.values()],
        "tx": [image.tvec[0] for image in images.values()],
        "ty": [image.tvec[1] for image in images.values()],
        "tz": [image.tvec[2] for image in images.values()],
    }

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by="image_name")

    # Save to file
    df.to_csv(output_file_path, index=False)

def run_colmap(scene_path, output_path, run_dense_construction):
    database_path = os.path.join(output_path, "database.db")
    sparse_path = os.path.join(output_path, "sparse")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(sparse_path, exist_ok=True)
    
    # Feature extraction
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", scene_path,
        "--ImageReader.camera_model", "SPHERE",
        "--ImageReader.camera_params", "1,2048,1024",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "25000"
    ])
    
    # # Spatial matching
    subprocess.run([
        "colmap", "spatial_matcher",
        "--database_path", database_path,
        "--SiftMatching.max_error", "4",
        "--SiftMatching.min_num_inliers", "50",
        "--SpatialMatching.is_gps", "0",
        "--SpatialMatching.max_distance", "50",
        "--SiftMatching.max_num_matches", "25000"
    ])
    
    # # Sparse reconstruction
    subprocess.run([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", scene_path,
        "--output_path", sparse_path,
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.sphere_camera", "1"
    ])

    ## Extract camera poses from the sparse reconstruction
    colmap_model_path = os.path.join(sparse_path, "0", "images.bin")
    if os.path.exists(colmap_model_path):
        extract_colmap_pose_to_txt_file(os.path.join(output_path, "poses.txt"), colmap_model_path)
    else:
        print(f"COLMAP fails to generate poses for scene {scene_path}")

    if not run_dense_construction:
        return
    
    # Run dense reconstruction
    subprocess.run([
        "colmap", "sphere_cubic_reprojecer",
        "--image_path", scene_path,
        "--input_path", os.path.join(sparse_path, "0"),
        "--output_path", os.path.join(output_path, "sparse-cubic")
    ])

    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", os.path.join(output_path, "sparse-cubic"),
        "--input_path", os.path.join(output_path, "sparse-cubic", "sparse"),
        "--output_path", os.path.join(output_path, "dense"),
        "--output_type", "COLMAP"
    ])

    subprocess.run([
        "colmap", "patch_match_stereo",
        "--workspace_path", os.path.join(output_path, "dense"),
        "--PatchMatchStereo.geom_consistency", "false",
        "--PatchMatchStereo.depth_min", "0.001",
        "--PatchMatchStereo.depth_max", "1000.001",
    ])


    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", os.path.join(output_path, "dense"),
        "--output_path", os.path.join(output_path, "dense","fused.ply"), 
        "--input_type", "photometric",
         "--StereoFusion.min_num_pixels", "2",  # Default = 3, lowering allows more points
        "--StereoFusion.max_reproj_error", "5.0",  # Increase to tolerate motion artifacts
        "--StereoFusion.max_depth_error", "1.0"  # Default = 0.01, increasing allows more depth variation
    ])


def main():
    parser = argparse.ArgumentParser(description="Run COLMAP pose estimation on EQR video frames")
    parser.add_argument("input_dir", help="Path to the input directory containing video frames on all scenes.")
    parser.add_argument("output_dir", help="Path to the output directory contatining  COLMAP sparse reconstruction of each scene")
    parser.add_argument("--dense_reconstruction", help="Whether to perform dense reconstruction", action='store_true')
    args = parser.parse_args()
    
    for scene in os.listdir(args.input_dir):
        scene_path = os.path.join(args.input_dir, scene)
        output_path = os.path.join(args.output_dir, scene)
        if os.path.isdir(scene_path):
            print(f"Processing scene: {scene}")
            run_colmap(scene_path, output_path, args.dense_reconstruction)
    
if __name__ == "__main__":
    main()

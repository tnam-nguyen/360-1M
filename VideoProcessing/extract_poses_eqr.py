import os
import subprocess
import argparse


def run_colmap(scene_path, output_path):
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
        "--ImageReader.single_camera", "1"
    ])
    
    # Spatial matching
    subprocess.run([
        "colmap", "spatial_matcher",
        "--database_path", database_path,
        "--SiftMatching.max_error", "4",
        "--SiftMatching.min_num_inliers", "50",
        "--SpatialMatching.is_gps", "0",
        "--SpatialMatching.max_distance", "50"
    ])
    
    # Sparse reconstruction
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

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP pose estimation on EQR video frames")
    parser.add_argument("input_dir", help="Path to the input directory containing video frames on all scenes.")
    parser.add_argument("output_dir", help="Path to the output directory contatining  COLMAP sparse reconstruction of each scene")
    args = parser.parse_args()
    
    for scene in os.listdir(args.input_dir):
        scene_path = os.path.join(args.input_dir, scene)
        output_path = os.path.join(args.output_dir, scene)
        if os.path.isdir(scene_path):
            print(f"Processing scene: {scene}")
            run_colmap(scene_path, output_path)
    
if __name__ == "__main__":
    main()

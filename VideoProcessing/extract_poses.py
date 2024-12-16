import os
import cv2
import pickle
import pandas as pd
import torch
import argparse
import numpy as np
import shutil
import tempfile
from tqdm import tqdm
import sys
import math
sys.path.append(".")

from third_party.mast3r.mast3r.model import AsymmetricMASt3R
from third_party.mast3r.mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from third_party.mast3r.dust3r.dust3r.inference import inference
from third_party.mast3r.dust3r.dust3r.utils.image import load_images

from config_util import LazyConfig, instantiate


def make_pairs(
    imgs, scene_graph="swin", iscyclic=True, winsize=3, symmetrize=True, prefilter=None
):
    pairs = []
    if scene_graph == "complete":  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("swin"):
        pairsid = set()
        for i in range(len(imgs)):
            for j in range(1, winsize + 1):
                idx = i + j
                if iscyclic:
                    idx = idx % len(imgs)  # explicit loop closure
                if idx >= len(imgs):
                    continue
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            print(i, j)
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith("logwin"):
        offsets = [2**i for i in range(winsize)]
        pairsid = set()
        for i in range(len(imgs)):
            ixs_l = [i - off for off in offsets]
            ixs_r = [i + off for off in offsets]
            for j in ixs_l + ixs_r:
                if iscyclic:
                    j = j % len(imgs)  # Explicit loop closure
                if j < 0 or j >= len(imgs) or j == i:
                    continue
                pairsid.add((i, j) if i < j else (j, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]

    # now, remove edges
    if isinstance(prefilter, str) and prefilter.startswith("seq"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith("cyc"):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def sel(x, kept):
    if isinstance(x, dict):
        return {k: sel(v, kept) for k, v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x[kept]
    if isinstance(x, (tuple, list)):
        return type(x)([x[k] for k in kept])


def _filter_edges_seq(edges, seq_dis_thr, cyclic=False):
    # number of images
    n = max(max(e) for e in edges) + 1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def filter_pairs_seq(pairs, seq_dis_thr, cyclic=False):
    edges = [(img1["idx"], img2["idx"]) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def filter_edges_seq(view1, view2, pred1, pred2, seq_dis_thr, cyclic=False):
    edges = [(int(i), int(j)) for i, j in zip(view1["idx"], view2["idx"])]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    print(
        f">> Filtering edges more than {seq_dis_thr} frames apart: kept {len(kept)}/{len(edges)} edges"
    )
    return sel(view1, kept), sel(view2, kept), sel(pred1, kept), sel(pred2, kept)


def save_video_frames(folder_name, outpath):
    # Create a folder named after the video (without file extension)
    os.makedirs(outpath, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(folder_name)
    frame_count = 0

    # Loop through each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save each frame as a PNG file in the created folder
        frame_path = os.path.join(outpath, f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()


def delete_folder(folder_name):
    # Derive the folder name from the video name

    # Delete the folder and all its contents
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print(f"Folder '{folder_name}' has been deleted.")
    else:
        print(f"Folder '{folder_name}' does not exist.")


# Function to process frames and save outputs
def process_frames(video_name, input_root, output_root, args):
    device = torch.device("cuda")

    master_ckpt_path = os.environ.get(
        "MAST3R_CHECKPOINT",
        "/checkpoint/dream/vkramanuj/models/mast3r"
    )

    model_name = os.path.join(
        master_ckpt_path,
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    )

    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    out_path = os.path.join(output_root, video_name)
    os.makedirs(out_path, exist_ok=True)

    folder_name = os.path.join(
        input_root, video_name
    )  # os.path.splitext(video_basename)[0]

    lr1 = args.lr1
    niter1 = args.niter1
    lr2 = args.lr2
    niter2 = args.niter2
    optim_level = "refine"
    shared_intrinsics = True
    matching_conf_thr = 1

    frames_per_batch = 180
    overlap = 1

    num_frames = len(os.listdir(folder_name))
    img_paths_suffix = np.sort(os.listdir(folder_name))
    img_paths = [os.path.join(folder_name, i) for i in img_paths_suffix]
    iters = math.ceil((num_frames - overlap) / (frames_per_batch - overlap))
    for j in range(iters):
        start_frame = j * (frames_per_batch - overlap)
        end_frame = (j + 1) * (frames_per_batch - overlap) + overlap
        filelist = img_paths[start_frame:end_frame]
        images = load_images(filelist, size=512)
        pairs = make_pairs(images)
        
        cache_root_dir = os.environ.get("MAST3R_CACHE", "/tmp")
        cache_dir = tempfile.mkdtemp(suffix="_cache", dir=cache_root_dir)
        scene = sparse_global_alignment(
            filelist,
            pairs,
            cache_dir,
            model,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            device=device,
            opt_depth="depth" in optim_level,
            shared_intrinsics=shared_intrinsics,
            matching_conf_thr=matching_conf_thr,
        )

        focals = scene.get_focals().cpu().numpy()
        poses = scene.get_im_poses().cpu().numpy()
        file_name = f"pose_focal{start_frame}_{end_frame}.pkl"
        output_path = os.path.join(out_path, file_name)
        data = {"poses": poses, "focals": focals}

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Temporary cache directory removed from: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video frames using specified GPU."
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
    )
    parser.add_argument("--niter1", type=int, default=300)
    parser.add_argument("--niter2", type=int, default=100)
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--lr2", type=float, default=0.014)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    for video_name in os.listdir(args.root_folder):
        process_frames(video_name, args.root_folder, args.output_root, args)

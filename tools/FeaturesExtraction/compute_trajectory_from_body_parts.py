import glob
import json
import os
import argparse
import numpy as np


def compute_features_trajectories(args):
    for root, directories, filenames in os.walk(os.path.join(args.poses_base_dir, args.input_dir)):
        for directory in directories:
            video_dir = os.path.join(root, directory)
            print(video_dir)
            frames = sorted(glob.glob(video_dir + '/*.json'))
            if len(frames) > 0:
                features_video = []
                for x in range(0, len(frames), args.stride):
                    if x + args.number_frames < len(frames):
                        features = np.zeros(shape=(15, args.number_frames, 2))
                        prev_body_parts = None
                        for y in range(x, x + args.number_frames + 1):
                            body_parts = read_body_parts_file(frames[y])
                            if prev_body_parts is None:
                                prev_body_parts = body_parts
                            else:
                                diffs = compute_displacement(prev_body_parts, body_parts)
                                idx = (y - 1) - x
                                for a in range(15):
                                    features[a, idx] = diffs[a]

                        # Here normalize
                        for y in range(15):
                            for k in range(2):
                                s = np.sum(abs(features[y, :, k]))
                                if s != 0:
                                    features[y, :, k] = features[y, :, k] / s

                        features_video.append(features.flatten())

                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                file = os.path.join(features_dir, video_name + '_trajectories_features.json')
                np.savetxt(file, np.asarray(features_video), delimiter=',', fmt='%.7f')


def compute_displacement(prev_body_parts, body_parts):
    diffs = np.zeros(shape=(15, 2))
    for x in range(15):
        x1, y1 = get_max_prob(prev_body_parts[x])
        x2, y2 = get_max_prob(body_parts[x])
        diffs[x, :] = (x2 - x1, y2 - y1)

    return diffs


def read_body_parts_file(key_points_file):
    body_parts_int = {}

    # Read json pose points
    with open(key_points_file) as f:
        data = json.load(f)

    body_parts = data['part_candidates'][0]
    if len(body_parts) > 0:

        for key, value in body_parts.items():
            body_parts_int[int(key)] = [item for item in value]

    return body_parts_int


def get_max_prob(body_part):
    m = 0
    x = 0
    y = 0
    for p in range(0, len(body_part), 3):
        if body_part[p + 2] > m:
            m = float(body_part[p + 2])
            x = int(body_part[p])
            y = int(body_part[p + 1])

    return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory features from OpenPose points to Human Action Recognition"
    )

    parser.add_argument("--poses_base_dir", type=str,
                        # default='/home/murilo/dataset/KTH',
                        default='/home/murilo/dataset/Weizmann',
                        help="Name of directory where input points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='2DPoses',
                        help="Name of directory to output computed features.")

    parser.add_argument("--output_dir", type=str,
                        default='Trajectories_from_2DPoses',
                        help="Name of directory to output computed features.")

    parser.add_argument("--number_frames", type=int,
                        default=20,
                        help="Number of frames to consider to extract trajectories features.")

    parser.add_argument("--stride", type=int,
                        default=10,
                        help="Stride to compute features from the frames.")

    args = parser.parse_args()

    print(args)
    compute_features_trajectories(args)


if __name__ == "__main__":
    main()

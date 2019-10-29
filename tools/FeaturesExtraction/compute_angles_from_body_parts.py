import glob
from Enumerators import BodyStraight
import json
import os
import argparse
import math
import numpy as np

POSE_BODY_25_PAIRS_RENDER_GPU = \
    [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11,
     8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0, 16, 16, 18, 14,
     19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]

POSE_BODY_14_PAIRS = \
    [BodyStraight.Upper_body.value, BodyStraight.Left_shoulder.value,
     BodyStraight.Upper_body.value, BodyStraight.Right_shoulder.value,
     BodyStraight.Upper_body.value, BodyStraight.Left_hip.value,
     BodyStraight.Upper_body.value, BodyStraight.Right_hip.value,
     BodyStraight.Left_shoulder, BodyStraight.Neck.value,
     BodyStraight.Right_shoulder, BodyStraight.Neck.value,
     BodyStraight.Left_forearm.value, BodyStraight.Left_arm.value,
     BodyStraight.Left_arm.value, BodyStraight.Left_shoulder.value,
     BodyStraight.Right_forearm.value, BodyStraight.Right_arm.value,
     BodyStraight.Right_arm.value, BodyStraight.Right_shoulder.value,
     BodyStraight.Left_thigh.value, BodyStraight.Left_hip.value,
     BodyStraight.Left_thigh.value, BodyStraight.Left_leg.value,
     BodyStraight.Right_thigh.value, BodyStraight.Right_hip.value,
     BodyStraight.Right_thigh.value, BodyStraight.Right_leg.value]


def compute_angles_from_body_parts(args):
    print(os.path.join(args.poses_base_dir, args.input_dir))
    for root, directories, filenames in os.walk(os.path.join(args.poses_base_dir, args.input_dir)):
        for directory in directories:
            video_dir = os.path.join(root, directory)
            print(video_dir)
            ctd_frame = 0
            frames = sorted(glob.glob(video_dir + '/*.json'))
            if len(frames) > 0:
                features = np.zeros(shape=(len(frames), 14))
                for frame in frames:
                    print(frame)
                    body_parts = read_body_parts_file(frame)
                    if len(body_parts) > 0:  # and sum([len(v) for k, v in body_parts.items()]) > 69:
                        diffs = compute_angles_differences(body_parts, rad=True)
                        features[ctd_frame, :] = diffs
                        ctd_frame += 1

                #features = features[~np.all(features==0, axis=1)]
                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                file = os.path.join(features_dir, video_name + '_diffs_full.json')
                np.savetxt(file, np.asarray(features), delimiter=',')


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


def compute_angles_differences(body_parts, rad=True):
    features = np.zeros(14)
    for x in range(0, 14, 1):
        coords = []
        part_found = True
        for y in range(0, 2):
            i = POSE_BODY_14_PAIRS[x * 2 + y]
            # print(BodyStraight(i))
            x1, y1, x2, y2, _, _, _ = return_body_points_coord(i, body_parts)
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                part_found = False
                break
            else:
                coords.append([x1, y1, x2, y2])

        x_base = 0
        y_base = 0
        v1_idx = 0
        v2_idx = 0
        if part_found:
            if (coords[0][0] == coords[1][0] and
                    coords[0][1] == coords[1][1]):
                x_base = coords[0][0]
                y_base = coords[0][1]
                v1_idx = 2
                v2_idx = 2
            elif (coords[0][0] == coords[1][2] and
                  coords[0][1] == coords[1][3]):
                x_base = coords[0][0]
                y_base = coords[0][1]
                v1_idx = 2
                v2_idx = 0
            elif (coords[0][2] == coords[1][0] and
                  coords[0][3] == coords[1][1]):
                x_base = coords[0][2]
                y_base = coords[0][3]
                v1_idx = 0
                v2_idx = 2
            elif (coords[0][2] == coords[1][1] and
                  coords[0][3] == coords[1][3]):
                x_base = coords[0][2]
                y_base = coords[0][3]
                v1_idx = 0
                v2_idx = 0

            if x_base > 0 and y_base > 0:
                v1 = [coords[0][v1_idx] - x_base, coords[0][v1_idx + 1] - y_base, 0]
                v2 = [coords[1][v2_idx] - x_base, coords[1][v2_idx + 1] - y_base, 0]

                # print(x_base, y_base, v1_idx, v2_idx)
                # print(v1,v2)
                if np.count_nonzero(v1) and np.count_nonzero(v2):
                    a = angle(v1, v2, rad)
                    # print('Diff: %.2f\n\n'%a)
                    features[x] = a
            else:
                print('test')

    return features


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2, rad=True):
    a = math.acos(np.round(dotproduct(v1, v2) / (length(v1) * length(v2)), decimals=7))
    if not rad:
        a = (180 * a) / np.pi
    return a


def return_body_points_coord(i, body_parts):
    x1 = y1 = x2 = y2 = x = color_id = id1 = id2 = 0
    if i == BodyStraight.Neck.value:  # 1 => 0 Neck
        x = 13
    elif i == BodyStraight.Upper_body.value:  # 1 => 8 Upper body
        x = 0
    elif i == BodyStraight.Right_arm.value:  # 2 => 3 Right Arm
        x = 3
    elif i == BodyStraight.Right_forearm.value:  # 3 => 4 Right Forearm
        x = 4
    elif i == BodyStraight.Left_arm.value:  # 5 => 6 Left Arm
        x = 5
    elif i == BodyStraight.Left_forearm.value:  # 6 => 7 Left Forearm
        x = 6
    elif i == BodyStraight.Right_thigh.value:  # 9 => 10 Right Thigh
        x = 8
    elif i == BodyStraight.Right_leg.value:  # 10 => 11 Right Leg
        x = 9
    elif i == BodyStraight.Left_thigh.value:  # 12 => 13 Left Thigh
        x = 11
    elif i == BodyStraight.Left_leg.value:  # 13 => 14 Left Leg
        x = 12
    elif i == BodyStraight.Right_hip.value:  # 8 => 9 Right Hip
        x = 7
    elif i == BodyStraight.Left_hip.value:  # 8 => 12 Left Hip
        x = 10
    elif i == BodyStraight.Right_shoulder.value:  # 1 => 2 Right Shoulder
        x = 1
    elif i == BodyStraight.Left_shoulder.value:  # 1 => 5 Left Shoulder
        x = 2

    x = x * 2
    if (len(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]]) > 0 and len(
            body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]]) > 0):
        x1, y1 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]])
        x2, y2 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]])
        color_id = POSE_BODY_25_PAIRS_RENDER_GPU[x + 1] * 3
        id1 = POSE_BODY_25_PAIRS_RENDER_GPU[x]
        id2 = POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]

    return x1, y1, x2, y2, color_id, id1, id2


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
        description="Compute features from Hough Points to Human Action Recognition"
    )

    parser.add_argument("--poses_base_dir", type=str,
                        default='/home/murilo/dataset/Weizmann',
                        help="Name of directory where input points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='2DPoses',
                        help="Name of directory to output computed features.")

    parser.add_argument("--output_dir", type=str,
                        default='Angles_from_2DPoses',
                        help="Name of directory to output computed features.")

    args = parser.parse_args()

    print(args)

    compute_angles_from_body_parts(args)


if __name__ == "__main__":
    main()

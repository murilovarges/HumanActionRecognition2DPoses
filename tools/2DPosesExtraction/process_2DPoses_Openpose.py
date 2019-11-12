# Description..: Reads OpenPose json files with body parts and considere only files with person
# Date.........: 21/03/2019
# Author.......: Murilo Varges da Silva

import glob
import os
import argparse
import json
from shutil import copyfile

datasets = ['test', 'training', 'validation']
actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
actions = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']


def perform_frames_analysis(args):
    print(args)
    base_dir = os.path.join(args.poses_base_dir, args.input_dir)
    print(base_dir)
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if not dirnames:
            print(dirpath)
            ctd_frame = 0
            frames = sorted(glob.glob(dirpath + os.sep + '*.json'))
            for frame in frames:
                print(frame)
                body_parts = read_body_parts_file(frame)
                if len(body_parts) > 0 and sum([len(v) for k, v in body_parts.items()]) > 69:

                    frame_name = frame.replace(args.input_dir, args.output_dir)
                    path, file = os.path.split(frame_name)
                    if not os.path.exists(path):
                        os.makedirs(path)

                    file = file.split('_')
                    file[-2] = '%012d' % ctd_frame
                    file_name = '_'
                    file_name = file_name.join(file)
                    frame_name = os.path.join(path, file_name)
                    copyfile(frame, frame_name)

                    ctd_frame += 1


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


def main():
    parser = argparse.ArgumentParser(
        description="Process 2D poses to consider only files with pose detected"
    )

    parser.add_argument("--poses_base_dir", type=str,
                        default='/home/murilo/dataset/KTH',
                        help="Name of directory where input points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='2D_Poses',
                        help="Name of directory to output computed features.")

    parser.add_argument("--output_dir", type=str,
                        default='2D_Poses_Person',
                        help="Name of directory to output computed features.")

    args = parser.parse_args()

    print(args)

    perform_frames_analysis(args)


if __name__ == "__main__":
    main()
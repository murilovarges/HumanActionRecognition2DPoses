"""
Description..: From videos extract 2D Poses using OpenPose and write in json files
Date.........: 21/03/2019
Author.......: Murilo Varges da Silva
"""
import argparse
import glob
import os


def extract_2d_poses(args):
    os.chdir(args.open_pose_base_dir)
    for file in glob.glob(os.path.join(args.videos_base_dir, "**/*.avi"), recursive=True):
        base_video_name = os.path.splitext(os.path.basename(file))[0]
        print(file)

        poses_path = os.path.join(args.poses_base_dir,os.path.basename(os.path.dirname(file)), base_video_name)
        print(poses_path)
        if not os.path.exists(poses_path):
            os.makedirs(poses_path)

        os.system(
            "build/examples/openpose/openpose.bin --display 0 --video " + file
            + " --write_json " + poses_path + " --part_candidates --disable_blending --render_pose 0")


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from Hough Points to Human Action Recognition"
    )

    parser.add_argument("--videos_base_dir", type=str,
                        default='/home/murilo/dataset/KTH/VideosTrainValidationTest',
                        help="Name of directory where videos are located.")

    parser.add_argument("--open_pose_base_dir", type=str,
                        default='/home/murilo/openpose',
                        help="Name of directory where Openpose is located.")

    parser.add_argument("--poses_base_dir", type=str,
                        default='/home/murilo/dataset/KTH/2DPoses',
                        help="Name of directory to output computed 2D poses.")

    args = parser.parse_args()

    print(args)

    extract_2d_poses(args)


if __name__ == "__main__":
    main()

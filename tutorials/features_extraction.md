# Tutorial 2: Features extraction

This tutorial will help you, step-by-step, how to extract features (Angles and Trajectories) from 2D Poses.

Before proceeding make sure that you have already extract or download 2D poses, see [2D Poses Extraction](tutorials/2DPoses_extraction.md) for more information.

Experiments were performed in two public dataset [KTH](http://www.nada.kth.se/cvap/actions/) and [Weizmann](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html).


In this example, we will extract features from 2D poses by running the following scripts.

## Computing Angles Features

```
python tools/Cartesian_Points/compute_angles_from_body_parts.py \
--points_base_dir=/home/murilo/dataset/KTH \
--input_dir=2DPoses\
--output_dir=Angles_from_2d_poses
```

## Computing Trajectiories Features

```
python tools/Cartesian_Points/compute_trajectory_from_body_parts.py \
--points_base_dir=/home/murilo/dataset/KTH \
--input_dir=2DPoses \
--output_dir=Trajectories_from_2d_poses \
--number_frames=20 --stride=5
```
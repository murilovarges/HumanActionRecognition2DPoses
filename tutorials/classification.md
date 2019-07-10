# Tutorial 3: Human Action Classification

This tutorial will help you, step-by-step, how to perform classification of human action ustion extracted features (Angles and Trajectories) from 2D Poses.

Before proceeding make sure that you have already extracted features from 2D poses, see [Features Extraction](tutorials/features_extraction.md) for more information.

Experiments were performed in two public dataset [KTH](http://www.nada.kth.se/cvap/actions/) and [Weizmann](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html).


In this example, we will perform classificarion using extracted features from 2D poses by running the following scripts.

## Classification Using Only Angles Features
```
python /home/murilo/PycharmProjects/VideosClassification/BOW/MainHumanActionClassification.py \
--test_name=KTH-LOOCV-FV-K20-ANGLES-LSVM \
--base_path=/home/murilo/dataset/KTH/Angles_from_2d_poses \
--label_path=/home/murilo/dataset/KTH/class_names.txt
```

## Classification Using Only Trajectories Features
```
python /home/murilo/PycharmProjects/VideosClassification/BOW/MainHumanActionClassification.py \
--test_name=KTH-LOOCV-FV-K20-TRAJECTORY-LSVM \
--base_path=/home/murilo/dataset/KTH/Trajectories_from_2d_poses \
--label_path=/home/murilo/dataset/KTH/class_names.txt
```


## Classification Using Fusion of Angles and Trajectories Features
```
python /home/murilo/PycharmProjects/VideosClassification/BOW/MainHumanActionClassification.py \
--test_name=KTH-LOOCV-FV-K20-FUSION-LSVM \
--base_path=/home/murilo/dataset/KTH/Angles_from_2d_poses \
--base_path2=/home/murilo/dataset/KTH/Trajectories_from_2d_poses \
--label_path=/home/murilo/dataset/KTH/class_names.txt
```
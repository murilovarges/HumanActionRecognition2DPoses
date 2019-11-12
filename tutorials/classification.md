# Tutorial 3: Human Action Classification

This tutorial will help you, step-by-step, how to perform classification of human actions using extracted features (Angles and Trajectories) from 2D Poses.

Before proceeding make sure that you have already extracted features from 2D poses, see [Features Extraction](features_extraction.md) for more information.

Experiments were performed in two public datasets [KTH](http://www.nada.kth.se/cvap/actions/) and [Weizmann](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html). The experiments were performed using Leave-One-Out-Cross-Validation (LOOCV).


In this example, we will perform classification using extracted features from 2D poses by running the following scripts.

After running these scripts the results will be printed in screen (Accuracy and Confusion Matrix) as well as the results will be written in the results directory (Accuracy, Confusion Matrix and the result of classification for each video).

**Note:** Since in the process of creating the vocabulary of poses using the Gaussian Mixture Model (GMM) there is the parameter "random_state" that is used as a seed by the random number generator, the results may differ slightly from the results presented in the article.

## Classification Using Only Angles Features
KTH dataset example:
```
python tools/Classification/MainHumanActionClassification.py \
--test_name=KTH-BOP-FV-K20-ANGLES-LSVM \
--base_path=/home/murilo/dataset/KTH/Angles_from_2DPoses \
--label_path=/home/murilo/dataset/KTH/class_names.txt
```
Weizmann dataset example:
```
python tools/Classification/MainHumanActionClassification.py \
--test_name=Weizmann-BOP-FV-K20-ANGLES-LSVM \
--base_path=/home/murilo/dataset/Weizmann/Angles_from_2DPoses \
--label_path=/home/murilo/dataset/Weizmann/class_names.txt
```
## Classification Using Only Trajectories Features
KTH dataset example:
```
python tools/Classification/MainHumanActionClassification.py \
--test_name=KTH-BOP-FV-K20-TRAJECTORY-LSVM \
--base_path=/home/murilo/dataset/KTH/Trajectories_from_2DPoses \
--label_path=/home/murilo/dataset/KTH/class_names.txt
```
Weizmann dataset example:
```
python tools/Classification/MainHumanActionClassification.py \
--test_name=KTH-BOP-FV-K20-TRAJECTORY-LSVM \
--base_path=/home/murilo/dataset/KTH/Trajectories_from_2DPoses \
--label_path=/home/murilo/dataset/KTH/class_names.txt
```


## Classification Using Fusion of Angles and Trajectories Features
KTH dataset example:
```
python tools/Classification/MainHumanActionClassification.py \
--test_name=KTH-BOP-FV-K20-FUSION-LSVM \
--base_path=/home/murilo/dataset/KTH/Angles_from_2DPoses \
--base_path2=/home/murilo/dataset/KTH/Trajectories_from_2DPoses \
--label_path=/home/murilo/dataset/KTH/class_names.txt
```

Weizmann dataset example:
```
python tools/Classification/MainHumanActionClassification.py \
--test_name=Weizmann-BOP-FV-K20-FUSION-LSVM \
--base_path=/home/murilo/dataset/Weizmann/Angles_from_2DPoses \
--base_path2=/home/murilo/dataset/Weizmann/Trajectories_from_2DPoses \
--label_path=/home/murilo/dataset/Weizmann/class_names.txt
```
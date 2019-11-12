# Tutorial 4: Features Visualization

This tutorial will help you, how to visualize features distribution using t-Distributed Stochastic Neighbor Embedding (t-SNE) and Principal Component Analysis (PCA)

Before proceeding make sure that you have already extracted features from 2D poses, see [Features Extraction](features_extraction.md) for more information.


In this example, we will visualize features distribution, after run these scripts the images with visualization will be generated in output image directory. 

KTH dataset example using Angles and Trajectories features:
```
python tools/Visualization/tsne.py \
--base_path=/home/murilo/dataset/KTH/Angles_from_2DPoses \
--base_path2=/home/murilo/dataset/KTH/Trajectories_from_2DPoses \
--label_path=/home/murilo/dataset/KTH/class_names.txt \
--output_features=tools/Visualization/Temp/KTH/data.csv \
--output_image=tools/Visualization/Temp/KTH \
--number_classes=6 --mark_size=100 --alpha=0.6
 
```
Weizmann dataset example using Angles and Trajectories features:
```
python tools/Visualization/tsne.py \
--base_path=/home/murilo/dataset/Weizmann/Angles_from_2DPoses \
--base_path2=/home/murilo/dataset/Weizmann/Trajectories_from_2DPoses \
--label_path=/home/murilo/dataset/Weizmann/class_names.txt \
--output_features=tools/Visualization/Temp/Weizmann/data.csv \
--output_image=tools/Visualization/Temp/Weizmann \
--number_classes=10 --mark_size=300 --alpha=0.6
```


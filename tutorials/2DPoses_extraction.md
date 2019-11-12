# Tutorial 1: 2D Poses Extraction

This tutorial will help you, step-by-step, how to extract 2D Poses using Openpose framework.

Before proceeding make sure that you have already installed the Openpose framework. If you do not want to install the Openpose framework we have the **2D poses** already extracted available for the KTH and Weizmman databases, in this case you can download and skip the 2D pose extraction step.

Links to download 2D poses:

- :link: [KTH 2D poses - All video frames (About 200mb)](https://1drv.ms/u/s!Ah7VRrgHtdpMioVejcyyNKfYAwLdeg?e=w2qhxV) 
- :link: [KTH 2D poses - Only frames with person (About 130mb)](https://1drv.ms/u/s!Ah7VRrgHtdpMio8FUjhBk85Be-L-uQ?e=VJBWDB)
- :link: [Weizmann 2D poses - All video frames (About   5mb)](https://1drv.ms/u/s!Ah7VRrgHtdpMioVfbsxNLZQx84noTg?e=GKziIl)   

Experiments were performed in two public dataset [KTH](http://www.nada.kth.se/cvap/actions/) and [Weizmann](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html).

## Preparing data

In this example, we assume that you would to extract 2D poses from the videos as following (e.g. KTH dataset):

/your_path/kth_dataset/boxing/person02_boxing_d1_uncomp.avi
/your_path/kth_dataset/boxing/person02_boxing_d2_uncomp.avi<br>
...<br>
/your_path/kth_dataset/handclapping/person02_handclapping_d1_uncomp.avi
/your_path/kth_dataset/handclapping/person02_handclapping_d2_uncomp.avi<br>
...<br>
/your_path/kth_dataset/handwaving/person02_handwaving_d1_uncomp.avi
/your_path/kth_dataset/handwaving/person02_handwaving_d2_uncomp.avi<br>
...<br>
/your_path/kth_dataset/jogging/person02_jogging_d1_uncomp.avi
/your_path/kth_dataset/jogging/person02_jogging_d2_uncomp.avi<br>
...<br>
/your_path/kth_dataset/running/person02_running_d1_uncomp.avi
/your_path/kth_dataset/running/person02_running_d2_uncomp.avi<br>
...<br>
/your_path/kth_dataset/walking/person02_walking_d1_uncomp.avi
/your_path/kth_dataset/walking/person02_walking_d2_uncomp.avi<br>
...


## Extracting 2D Poses using Openpose

In this example, we will extract 2D poses for the above-mentioned **KTH dataset** videos.


```
python tools/2DPosesExtraction/extract_2DPoses_Openpose.py \
--videos_base_dir=/home/murilo/dataset/KTH/VideosTrainValidationTest \
--open_pose_base_dir=/home/murilo/openpose \
--poses_base_dir=/home/murilo/dataset/KTH/2DPoses
```


In this example, we will extract 2D poses for **Weizmann dataset** videos .

```
python tools/2DPosesExtraction/extract_2DPoses_Openpose.py \
--videos_base_dir=/home/murilo/dataset/Weizmann/Videos \
--open_pose_base_dir=/home/murilo/openpose \
--poses_base_dir=/home/murilo/dataset/Weizmann/2DPoses
```

## Processing 2D Poses using Openpose

In this example, since the KTH dataset contains frames without people, we will process 2D poses to consider only files where at least a person was detected.

```
python tools/2DPosesExtraction/process_2DPoses_Openpose.py \
--poses_base_dir=/home/murilo/dataset/KTH \
--input_dir=2DPoses \
--output_dir=2DPoses_Person
```

## Next
As next step follow the link:
[Features Extraction](features_extraction.md)


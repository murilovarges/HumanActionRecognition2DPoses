# Human Action Recognition using 2D Poses

[[Research Group Page](http://recogna.tech/)] [[Paper Link](HumanActionRecognitionUsing2DPoses.pdf)]

<h5>This repository presents a new method based spatiotemporal features to perform Human Action Recognition from videos.</h5> 

Ilustration of architeture of proposed method:
![architeture_har](https://raw.githubusercontent.com/murilovarges/HumanActionRecognition2DPoses/master/architeture.png)

## Abstract

The advances in video capture, storage and sharing technologies have caused a high demand in techniques for automatic recognition of humans actions. Among the main applications, we can highlight surveillance in public places, detection of falls in the elderly, no-checkout-required stores (Amazon Go), self-driving car, inappropriate content posted on the Internet, etc. The automatic recognition of human actions in videos is a challenging task because in order to obtain a good result one has to work with spatial information (e.g., shapes found in a single frame) and temporal information (e.g., movements found across frames). In this work, we present a simple methodology for describing human actions in videos that use extracted data from 2-Dimensional poses. The experimental results show that the proposed technique can encode spatial and temporal information, obtaining competitive accuracy rates compared to state-of-the-art methods.


If you find this work helpful for your research, please cite our following paper:

M. Varges and A. N. Marana. **Human Action Recognition using 2D Poses.** 8th Brazilian Conference on Intelligent Systems (BRACIS), 2019.

```
@inproceedings{har2dposes_bracis2019,
    title = {Human Action Recognition using 2D Poses},
    author = {Murilo Varges da Silva and Aparecido Nilceu Marana.},
    booktitle = {2019 8th Brazilian Conference on Intelligent Systems (BRACIS)},
    year = {2019},
    keywords={Human action recognition; Surveillance systems; Spatio-temporal features; Videos sequences},
}
```
If you have any question or feedback about the code, please contact: murilo.varges@gmail.com.

## Requirements

To run this project you will need to install the follows softwares:
* [OpenPose Framework](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (Optional)
* Python version 3.6 or higher

This project requires the following dependencies:

```
matplotlib==2.2.2
numpy==1.15.0
pandas==0.22.0
seaborn==0.9.0
Tools==0.1.9
scikit_learn==0.21.3
```

Install the requirements with command:
```
pip install -r requirements.txt
```
## Tutorials
We provide some basic tutorials for you to get familar with the code and tools.
1. [2D Poses Extraction](tutorials/2DPoses_extraction.md)
2. [Features Extraction](tutorials/features_extraction.md)
3. [Human Action Recognition](tutorials/classification.md)
4. [Features Embedding Visualization](tutorials/visualization.md)


## License
This project is Apache 2.0 licensed, as found in the LICENSE file.

### Acknowledgements
We thank NVIDIA Corporation for the donation of the GPU used in this study. This study was financed in part by CAPES - Brazil (Finance Code 001).


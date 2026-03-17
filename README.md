# Unsupervised Light Field Salient Object Detection Enhanced by Segment Anything Model and Depth Quality-aware Learning

## Data Preparation
The dataset can be downloaded from the following link: Lytro Illum ([LINK](https://github.com/pencilzhang/MAC-light-field-saliency-net)), HFUT ([LINK](https://github.com/pencilzhang/MAC-light-field-saliency-net)), and DUTLF-V2 ([LINK](https://github.com/DUT-IIAU-OIP-Lab/DUTLF-V2)). 

## Environmental Setups
- python 3.8
- pytorch 2.0.1
- numpy 1.24.4
- torchvision 0.15.2
- opencv-python 4.8.0.76
- scipy 1.10.1

## Usage
### Train
First download Segment Anything Model ([SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)). Run `pseudo_label.py` to generate pseudo_labels with SAM. Then run `train.py` to train the network.
### Test
Put the trained model in the appropriate folder. Run `test.py` to generate the final prediction map.

## Visual Results


<img width="761" height="460" alt="fig5" src="https://github.com/user-attachments/assets/c0deea02-3e68-4e0c-81f0-9587bce04ed0" />



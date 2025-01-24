# Distributed Image Compression with Conditional Diffusion Models
## Requirements
`Python 3.9` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt
```

## Dataset
The datasets used for experiments are KITTI Stereo and Cityscape.

For KITTI Stereo you can download the necessary image pairs from [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow_multiview.zip) and [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip). After obtaining `data_stereo_flow_multiview.zip` and `data_scene_flow_multiview.zip`, run the following commands:
```bash
unzip data_stereo_flow_multiview.zip # KITTI 2012
mkdir data_stereo_flow_multiview
mv training data_stereo_flow_multiview
mv testing data_stereo_flow_multiview

unzip data_scene_flow_multiview.zip # KITTI 2015
mkdir data_scene_flow_multiview
mv training data_scene_flow_multiview
mv testing data_scene_flow_multiview
```

For Cityscape you can download the image pairs from [here](https://www.cityscapes-dataset.com/downloads/). After downloading `leftImg8bit_trainvaltest.zip` and `rightImg8bit_trainvaltest.zip`, run the following commands:
```bash
mkdir cityscape_dataset
unzip leftImg8bit_trainvaltest.zip
mv leftImg8bit cityscape_dataset
unzip rightImg8bit_trainvaltest.zip
mv rightImg8bit cityscape_dataset
```
## Getting Started

Our code utilizes the Accelerate library to simplify multi-card execution. To use the code, follow the steps below:

```bash
accelerate launch train.py
```
## Acknowledgment
Our work is inspired by and implemented based on the following projects. We deeply appreciate their outstanding open-source contributions:
- [CDC_compression](https://github.com/buggyyang/CDC_compression)
- [NDIC](https://github.com/ipc-lab/NDIC)


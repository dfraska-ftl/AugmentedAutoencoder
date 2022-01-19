## Augmented Autoencoders  

### Implicit 3D Orientation Learning for 6D Object Detection from RGB Images   
Martin Sundermeyer, Zoltan-Csaba Marton, Maximilian Durner, Manuel Brucker, Rudolph Triebel  
Best Paper Award, ECCV 2018.    

[paper](https://arxiv.org/pdf/1902.01275), [supplement](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-01231-1_43/MediaObjects/474211_1_En_43_MOESM1_ESM.pdf), [oral](https://www.youtube.com/watch?v=jgb2eNNlPq4)

### Citation
If you find Augmented Autoencoders useful for your research, please consider citing:  
```
@InProceedings{Sundermeyer_2018_ECCV,
author = {Sundermeyer, Martin and Marton, Zoltan-Csaba and Durner, Maximilian and Brucker, Manuel and Triebel, Rudolph},
title = {Implicit 3D Orientation Learning for 6D Object Detection from RGB Images},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

### Multi-path Learning for Object Pose Estimation Across Domains
Martin Sundermeyer, Maximilian Durner, En Yen Puang, Zoltan-Csaba Marton, Narunas Vaskevicius, Kai O. Arras, Rudolph Triebel  
CVPR 2020  
The code of this work can be found [here](https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath)

## Overview

We propose a real-time RGB-based pipeline for object detection and 6D pose estimation. Our novel 3D orientation estimation is based on a variant of the Denoising Autoencoder that is trained on simulated views of a 3D model using Domain Randomization. This so-called Augmented Autoencoder has several advantages over existing methods: It does not require real, pose-annotated training data, generalizes to various test sensors and inherently handles object and view symmetries.  

<p align="center">
<img src='docs/pipeline_with_scene_vertical_ext.jpeg' width='600'>
<p>

1) Train the Augmented Autoencoder(s) using only a 3D model to predict 3D Object Orientations from RGB image crops
2) For full RGB-based 6D pose estimation, also train a 2D Object Detector (e.g. https://github.com/fizyr/keras-retinanet)
3) Optionally, use our standard depth-based ICP to refine the 6D Pose

## Requirements: Hardware
### For Training
Nvidia GPU with >4GB memory (or adjust the batch size)  
RAM >8GB  
Duration depending on Configuration and Hardware: ~3h per Object

## Requirements: Software

Python 3.6+  
CUDA 11.0: https://developer.nvidia.com/cuda-11.0-download-archive  
cuDNN: https://developer.nvidia.com/rdp/cudnn-download  
VOC training set (mirror): https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar  

### Linux
#### GLFW
```bash
sudo apt-get install libglfw3-dev libglfw3  
```

#### Assimp
```bash
sudo apt-get install libassimp-dev  
```
**Continued in the the All Platforms section below.**

### Windows
#### GLFW
1) Download from https://www.glfw.org/  
Tested with Version 3.3.3 (https://github.com/glfw/glfw/releases/download/3.3.3/glfw-3.3.3.bin.WIN64.zip)  
2) Unzip to a permanent location  
3) Set the GLFW_ROOT environment variable to the location where it was unzipped  
4) Install cython & wheel  
   ```cmd
   pip install --user cython wheel
   ```
5) Install the pip package  
   ```cmd
   pip install --user cyglfw3
   ```
   If you get an error about missing Visual Studio build tools, you may need to install them through the Visual Studio installer.  
6) Copy the file glfw3.dll from lib-vc2012 into the Python Lib\site-packages\cyglfw3 where cyglfw is installed.  
7) Add the lib-vc2012 directory to the PATH environment variable  

#### Assimp
1) Download & Install the package from https://github.com/assimp/assimp/releases/tag/v4.1.0/  
2) Add the &lt;Program Files&gt;\Assimp\bin\x64 directory to the PATH environment variable  

#### Troubleshooting
If you run into the error "Unable to load numpy_formathandler accelerator from OpenGL_accelerate", replace numpy with numpy‑1.19.5+mkl‑cp38‑cp38‑win_amd64.whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy (run "pip install" on the whl). Match the value for cp38 to your Python version - i.e. cp38 is for Python 3.8.

**Continued in the the All Platforms section below.**

### All Platforms

#### OpenCV >= 3.1
1) See https://pypi.org/project/opencv-python/ for GPU acceleration instructions.  
To install the CPU-only version:  
```bash
pip install --user opencv-python
```
2a) Latest versions of dependencies:  
```bash
pip install --user --pre --upgrade PyOpenGL PyOpenGL_accelerate
pip install --user tensorflow cython cyglfw3 pyassimp==3.3 imgaug progressbar tf_agents
```
2b) Exact versions of dependencies:
```bash
pip install --user tensorflow==2.4.1 PyOpenGL==3.1.5 PyOpenGL_accelerate==3.1.5 cython==0.29.19 cyglfw3==3.1.0.2 pyassimp==3.3 imgaug==0.4.0 progressbar==2.5 tf_agents==0.7.1
```

### Headless Rendering
Please note that we use the GLFW context as default which does not support headless rendering. To allow for both, onscreen rendering & headless rendering on a remote server, set the context to EGL: 
```
export PYOPENGL_PLATFORM='egl'
```
In order to make the EGL context work, you might need to change PyOpenGL like [here](https://github.com/mcfletch/pyopengl/issues/27)

## Preparatory Steps

1) Pip installation
```bash
pip install --user .
```

2) Set Workspace path
### Linux:
Consider adding this to your bash profile
```bash
export AE_WORKSPACE_PATH=/path/to/autoencoder_ws  
```
### Windows:
   Edit your system environment variables to add the AE_WORKSPACE_PATH variable which points to the location where you want to store training results and configurations for AugmentedAutoencoder

3) Create Workspace, Init Workspace (if installed locally, make sure .local/bin/ is in your PATH)
```bash
mkdir $AE_WORKSPACE_PATH
cd $AE_WORKSPACE_PATH
ae_init_workspace
```
   
## Train an Augmented Autoencoder

1) Create the training config file. Insert the paths to your 3D model and background images.
```bash
mkdir $AE_WORKSPACE_PATH/cfg/exp_group
cp $AE_WORKSPACE_PATH/cfg/train_template.cfg $AE_WORKSPACE_PATH/cfg/exp_group/my_autoencoder.cfg
gedit $AE_WORKSPACE_PATH/cfg/exp_group/my_autoencoder.cfg
```

2) Generate and check training data. The object views should be strongly augmented but identifiable.

(Press *ESC* to close the window.)
```bash
ae_train exp_group/my_autoencoder -d
```
This command does not start training and should be run on a PC with a display connected.  

Output:
![](docs/training_images_29999.png)

3) Train the model
(See the [Headless Rendering](#headless-rendering) section if you want to train directly on a server without display)

```bash
ae_train exp_group/my_autoencoder
```

```bash
$AE_WORKSPACE_PATH/experiments/exp_group/my_autoencoder/train_figures/training_images_29999.png  
```
Middle part should show reconstructions of the input object (if all black, set higher bootstrap_ratio / auxilliary_mask in training config)  

4) Create the embedding
```bash
ae_embed exp_group/my_autoencoder
```

## Testing

### Augmented Autoencoder only

have a look at /auto_pose/test/   

*Feed one or more object crops from disk into AAE and predict 3D Orientation*
```bash
python aae_image.py exp_group/my_autoencoder -f /path/to/image/file/or/folder
```

*The same with a webcam input stream*
```bash
python aae_webcam.py exp_group/my_autoencoder
```

### Multi-object RGB-based 6D Object Detection from a Webcam stream

*Option 1: Train a RetinaNet Model from https://github.com/fizyr/keras-retinanet*

adapt $AE_WORKSPACE_PATH/eval_cfg/aae_retina_webcam.cfg

```bash
python auto_pose/test/aae_retina_webcam_pose.py -test_config aae_retina_webcam.cfg -vis
```

*Option 2: Using the Google Detection API with Fixes*

Train a 2D detector following https://github.com/naisy/train_ssd_mobilenet  
adapt /auto_pose/test/googledet_utils/googledet_config.yml  

```bash
python auto_pose/test/aae_googledet_webcam_multi.py exp_group/my_autoencoder exp_group/my_autoencoder2 exp_group/my_autoencoder3
```


## Evaluate a model

*For the evaluation you will also need*
https://github.com/thodan/sixd_toolkit + our extensions, see sixd_toolkit_extension/help.txt  

*Create the evaluation config file*
```bash
mkdir $AE_WORKSPACE_PATH/cfg_eval/eval_group
cp $AE_WORKSPACE_PATH/cfg_eval/eval_template.cfg $AE_WORKSPACE_PATH/cfg_eval/eval_group/eval_my_autoencoder.cfg
gedit $AE_WORKSPACE_PATH/cfg_eval/eval_group/eval_my_autoencoder.cfg
```

### Evaluate and visualize 6D pose estimation of AAE with ground truth bounding boxes

Set estimate_bbs=False in the evaluation config  

```bash
ae_eval exp_group/my_autoencoder name_of_evaluation --eval_cfg eval_group/eval_my_autoencoder.cfg
e.g.
ae_eval tless_nobn/obj5 eval_name --eval_cfg tless/5.cfg
```

### Evaluate 6D Object Detection with a 2D Object Detector

Set estimate_bbs=True in the evaluation config  

*Generate a training dataset for T-Less using detection_utils/generate_sixd_train.py*
```bash
python detection_utils/generate_sixd_train.py
```

Train https://github.com/fizyr/keras-retinanet or https://github.com/balancap/SSD-Tensorflow

```bash
ae_eval exp_group/my_autoencoder name_of_evaluation --eval_cfg eval_group/eval_my_autoencoder.cfg
e.g.
ae_eval tless_nobn/obj5 eval_name --eval_cfg tless/5.cfg
```


# Config file parameters
```yaml
[Paths]
# Path to the model file. All formats supported by assimp should work. Tested with ply files.
MODEL_PATH: /path/to/my_3d_model.ply
# Path to some background image folder. Should contain a * as a placeholder for the image name.
BACKGROUND_IMAGES_GLOB: /path/to/VOCdevkit/VOC2012/JPEGImages/*.jpg

[Dataset]
#cad or reconst (with texture)
MODEL: reconst
# Height of the AE input layer
H: 128
# Width of the AE input layer
W: 128
# Channels of the AE input layer (default BGR)
C: 3
# Distance from Camera to the object in mm for synthetic training images
RADIUS: 700
# Dimensions of the renderered image, it will be cropped and rescaled to H, W later.
RENDER_DIMS: (720, 540)
# Camera matrix used for rendering and optionally for estimating depth from RGB
K: [1075.65, 0, 720/2, 0, 1073.90, 540/2, 0, 0, 1]
# Vertex scale. Vertices need to be scaled to mm
VERTEX_SCALE: 1
# Antialiasing factor used for rendering
ANTIALIASING: 8
# Padding rendered object images and potentially bounding box detections 
PAD_FACTOR: 1.2
# Near plane
CLIP_NEAR: 10
# Far plane
CLIP_FAR: 10000
# Number of training images rendered uniformly at random from SO(3)
NOOF_TRAINING_IMGS: 10000
# Number of background images that simulate clutter
NOOF_BG_IMGS: 10000

[Augmentation]
# Using real object masks for occlusion (not really necessary)
REALISTIC_OCCLUSION: False
# Maximum relative translational offset of input views, sampled uniformly  
MAX_REL_OFFSET: 0.20
# Random augmentations at random strengths from imgaug library
CODE: Sequential([
    #Sometimes(0.5, PerspectiveTransform(0.05)),
    #Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
    Sometimes(0.5, Affine(scale=(1.0, 1.2))),
    Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
    Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
    Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
    Sometimes(0.3, Invert(0.2, per_channel=True)),
    Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
    Sometimes(0.5, Multiply((0.6, 1.4))),
    Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))
    ], random_order=False)

[Embedding]
# for every rotation save rendered bounding box diagonal for projective distance estimation
EMBED_BB: True
# minimum number of equidistant views rendered from a view-sphere
MIN_N_VIEWS: 2562
# for each view generate a number of in-plane rotations to cover full SO(3)
NUM_CYCLO: 36

[Network]
# additionally reconstruct segmentation mask, helps when AAE decodes pure blackness
AUXILIARY_MASK: False
# Variational Autoencoder, factor in front of KL-Divergence loss
VARIATIONAL: 0
# Reconstruction error metric
LOSS: L2
# Only evaluate 1/BOOTSTRAP_RATIO of the pixels with highest errors, produces sharper edges
BOOTSTRAP_RATIO: 4
# regularize norm of latent variables
NORM_REGULARIZE: 0
# size of the latent space
LATENT_SPACE_SIZE: 128
# number of filters in every Conv layer (decoder mirrored)
NUM_FILTER: [128, 256, 512, 512]
# stride for encoder layers, nearest neighbor upsampling for decoder layers
STRIDES: [2, 2, 2, 2]
# filter size encoder
KERNEL_SIZE_ENCODER: 5
# filter size decoder
KERNEL_SIZE_DECODER: 5


[Training]
OPTIMIZER: Adam
NUM_ITER: 30000
BATCH_SIZE: 64
LEARNING_RATE: 1e-4
SAVE_INTERVAL: 5000

[Queue]
# number of threads for producing augmented training data (online)
NUM_THREADS: 10
# preprocessing queue size in number of batches
QUEUE_SIZE: 50

# CS230-Final-Project:Arbitrary-Style-Transfer 
This file contains the code we implement/modified for our cs230 final project: Arbitrary Image Style Transfer (with AdaIN/CORAL/Histogram Matching Layer).

## Description
We implement our code based on https://github.com/elleryqueenhomels/arbitrary_style_transfer.git The code we wrote/modified is as following:
1) In the utils_clean.py, we add a filter function to clean out WikiArt images that are too large to train or has zero pixel;
2) In the adaptive_instance_norm_modv2.py, we wrote the matching function for three other matching method besides AdaIN layer, which are Correlation Alignment layer, Histogram Matching layer and AdaClip layer.
3) In the train_monitor.py, we modified the original code to print the loss after every step for better monitoring the training process. 

## Prerequisties
1) Pre-trained VGG19 normalised network
2) Microsoft COCO dataset
3) WikiArt dataset

## Trained Model
Our trained models have been submitted through gradescope, which are trained with AdaIN layer with style weights equal to 1.5/2.0/2.5, learning_rate equals to 10^(-3)/10^(-4)/10^(-5) and CORAL layer with style weight equals to 2.0, learning_rate equals to 10^(-4).

## Manual
(Modified based on https://github.com/elleryqueenhomels/arbitrary_style_transfer.git)
The main file main_lossgif.py is a demo, which has already contained training procedure and inferring procedure (inferring means generating stylized images).
You can switch these two procedures by changing the flag IS_TRAINING.
By default,
1) The content images lie in the folder "./images/content/".
2) The style images lie in the folder "./images/style/".
3) The weights file of the pre-trained VGG-19 lies in the current working directory.
4) The MS-COCO images dataset for training lies in the folder "../MS_COCO/".
5) The WikiArt images dataset for training lies in the folder "../WikiArt/". 
6) The checkpoint files of trained models lie in the folder "./models/models.xx.xx/." (You should unzip the model.zip file before inferring and change this path to the folder that holds the trained weights for the model you want to train/check. For example, folder "./models/models.Adain.2.0.10^-3/." holds the ckpts for our trained model with AadIN layer and weights = 2.0, learning_rate = 10^-4.)
7) After inferring procedure, the stylized images will be generated and output to the folder "./outputs/"

For inferring, you should make sure 1), 2), 3) and 6) are prepared correctly.
Of course, you can organize all the files and folders as you want, and what you need to do is just modifying related parameters in the main_lossfig.py file.

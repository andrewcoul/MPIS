![tamu_texas_a-m-university-logo](https://user-images.githubusercontent.com/36116977/170396505-c2bc0ac1-d5bf-4485-93d3-6e637a671024.png)
![UT-Logo](https://user-images.githubusercontent.com/36116977/170396567-618abb90-fdfb-45ec-ba47-5549ad950cd0.png)

# MPIS - Material Phase Image Segmentation
## Created by Andrew Coulson and William Avery with the use of Pytorch-UNet architecture by milesial at https://github.com/milesial/Pytorch-UNet

# Project Overview
The purpose of this project was to develop a neural network to aid in the characterization of material microstructures.  

The functional goal is to reach a point in which a researcher can give the model MicroCT scans of various common materials/application and get a segmented image stack in return.  

Semantic Image segmentation is the process of classifying different parts of an image into several predefined categories.  

![Capture](https://user-images.githubusercontent.com/36116977/170401304-ebd647d4-552b-47f2-8119-4ac40ce83f40.PNG)  
*Segmentation of objects in image of bike race*

![Capture](https://user-images.githubusercontent.com/36116977/170401954-182e15cb-3a11-4196-a0bc-66ebb491558a.PNG)  
*Segmentation of glass fibers in fiber reinforced bentonite*

The process is done using a “convolutional neural network“. These CNNs are a specific type of neural network in which the input undergoes a “convolution” where it is down sampled in resolution, but up sampled in depth.  

Image segmentation has the unique requirement of not only identifying what something in an image is, but also where in the image it is located. The network architecture we used to overcome this pitfall of CNNs is called “UNet”  

UNet performs some number of convolutional operations which decrease the input resolution and increase the depth, and then deconvolutes the input to determine the relative locations of the segmented portions of the image. The output of such an operation is a set of masks predicting the location of the desired classes.

![68747470733a2f2f692e696d6775722e636f6d2f6a6544567071462e706e67](https://user-images.githubusercontent.com/36116977/170402392-d4126d05-ca39-4628-b724-f747b0949ede.png)

Example Application: Finding the location of the glass fibers in a sample of fiber-reinforced bentonite. In this example, a researcher would input the images from a MicroCT scan of the sample to the model and would get a set of images predicting where the fibers are in return.  

Training data should be created using an image processing software such as Fiji.
The user should take the images they wish to create a model for and create segmented image masks showing the location of the desired objects.

Once the user has created the masks, or “ground truths”, they can train a model that can be used for all similar applications in the future. That is, if a user commonly need to identify and characterize cracks in concrete, they only need to train the model on one set of scans. It is possible to achieve better results by training on more images.

MPIS is a machine learning application utilizing a UNet architecture to perform semantic image segmentation on material microstructures. Each unique application needs to be trained, however once a model is created for that application it can be used for all similar applications in perpetuity.





# Instructions for Use

## Setup Github
1) Set up a github account at https://github.com/
2) Download and install Guthub Desktop at https://desktop.github.com/ and login to the desktop client
3) Go to file -> Clone Repository -> URL and enter this website URL into the box
4) Choose a location on your device where you want the repository to be downloaded
5) You are good to go. You can pull future updates to the program through the check origin and pull button on the desktop client.

## Setup Weights and Biases
1) Set up a github account at https://wandb.ai/
2) Navigate to your newly created profile and create a new project under the desired team or individual
3) Navigate to wandb settings and copy your api key
4) In the command line, type the command **wandb login**
5) Use your wandb username and api key as your password to login

## Setup Python Virtual Environment
1) Open the windows command line
2) Navigate to the location of the repository by using the command: **cd C://...**
3) Create a python virtual environment for the project using the command: **python -m venv mpis** 
4) Activate the virtual environment you just created with the command: **mpis\Scripts\activate.bat**
5) Install dependencies using the command: **pip install -r requirements.txt**

   **NOTE: Only steps 1, 2, and 4 need to be done after the initial setup**
   
## Setup Files
1) Place the real images in **./data/imgs** and place the image masks in **./data/masks**
2) After model is trained and predictions have been made, relocate the masks and predictions to the corresponding files in **./data/PREDICT/**

## Train Model
1) Once you have activated the python virtual environment, you can use the following command to train the model: **python train.py**
2) The following flags can be added to the command to change certain paramaters within the program:  
        -e: specify the number of epochs to train for      
        -v: percentage of the total dataset which will be used as validation    
        -b: batch size which will be used to update weights  
        --scale: scale factor applied to data for training  
        --pretrained: enable pretrained encoder specified in program  
3) Once the model training is complete, the desired model checkpoint can be retrieved from **./checkpoints**

## Predict and Check Accuracy
1) Change the proper parameters in **test.py** to use the correct encoder and model name
2) In your python virtual environment run the command **python test.py**
3) The output will include prediction images in **./data/PREDICT/preds/** and a dice score in the terminal
4) For future models, the dice score is unnecessary and only the predictcions are relevant

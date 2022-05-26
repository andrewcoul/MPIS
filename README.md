![tamu_texas_a-m-university-logo](https://user-images.githubusercontent.com/36116977/170396505-c2bc0ac1-d5bf-4485-93d3-6e637a671024.png)
![UT-Logo](https://user-images.githubusercontent.com/36116977/170396567-618abb90-fdfb-45ec-ba47-5549ad950cd0.png)

# MPIS - Material Phase Image Segmentation
## Created with the use of Pytorch-UNet architecture by milesial at https://github.com/milesial/Pytorch-UNet

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

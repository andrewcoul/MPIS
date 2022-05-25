# MPIS - Material Phase Image Segmentation
## Created with the use of Pytorch-UNet architecture by milesial at https://github.com/milesial/Pytorch-UNet

# Instructions for Use

## Setup Github
1) Set up a github account at https://github.com/
2) Download and install Guthub Desktop at https://desktop.github.com/ and login to the desktop client
3) Go to file -> Clone Repository -> URL and enter this website URL into the box
4) Choose a location on your device where you want the repository to be downloaded
5) You are good to go. You can pull future updates to the program through the check origin and pull button on the desktop client.

## Setup Python Virtual Environment
1) Open the windows command line
2) Navigate to the location of the repository by using the command: **cd C://...**
3) Create a python virtual environment for the project using the command: **python -m venv mpis** 
4) Activate the virtual environment you just created with the command: **mpis\Scripts\activate.bat**
5) Install dependencies using the command: **pip install -r requirements.txt**

   **NOTE: Only steps 1, 2, and 4 need to be done after the initial setup**
   
## Setup Files
1) Place the real images in **./data/imgs** and place the image masks in **./data/masks**
2) After model is trained and predictions have been made, relocate the masks and 

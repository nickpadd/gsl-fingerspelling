# gsl-fingerspelling

# Dataset
A Dataset has been created from scratch for 6 letters (Β, Γ, Η, Θ, Ζ, Φ). Two signers captured frames for 2 days and 3 sessions per day (morning / evening / night). Final dataset consists of 5400 rgb photos (224x224), 5400 grayscale photos (28x28) and the corresponding csv files (a python script file has been used to convert grayscale to pixels). 

### How to capture frames
- Run script capture_frames.py
- click space each time you want to capture a frame
- click esc for saving frames and exit

Below we describe the conditions of each session per signer:

## Data/pixels (user_1)
- morning S1 csv files refer to photos taken at 11am, on white rib backround
- morning S2 csv files refer to photos taken at 9am, on black dark backround
- evening S1 csv files refer to photos taken at 2pm, on light blue background
- evening S2 csv files refer to photos taken at 5pm, on natural light (outdoor)
- night S1 csv files refer to photos taken at 1am on light blue backround under lighting
- night S2 csv files refer to photos taken at 9pm on yellow backround under lighting

## Data/pixels (user_2)
- morning S1 csv files refer to photos taken at 12am, on mixed white/light purple background
- morning S2 csv files refer to photos taken at 11am, on light/yellow rib background 
- evening S1 csv files refer to photos taken at 6pm, on light gray background
- evening S2 csv files refer to photos taken at 7pm, on mixed white/light purple background
- night S1 csv files refer to photos taken at 9am on light purple background
- night S2 csv files refer to photos taken at 10pm on light gray background

### In terms of training process, the dataset has been splitted in 3 parts, train (70%) / test (10%) / validation (20%)
Each part contains unique sessions, in order to avoid overfitting during training process.



# Architectures used:

## CNN Model
A CNN with 3 layers (1->16->32) has been implemented. Each convolutional layer consists of
- Conv2d(d_in, d_out, kernel=3, stride=1, padding=1)
- BatchNorm2d(d_out)
- ReLU(inplace=True)
- Dropout(d)
- MaxPool2d(kernel=2, stride=2)

**Final CNN model ran with the fllowing hyperparameters:**  
Lr = 0.001  
batch_size = 1024  
n_epochs = 100  
Patience = 15  
Dropout = 0.0  
No L2-Regularization  


## VGG Model
We followed the method of Transfer Learning. Initially we freezed all layers and replaced the final decision layer (fc8) with one corresponding to 6 class output. However, we noticed that best results come with unfreezing 3 classification layers.

**Final VGG model ran with the fllowing hyperparameters:**

Lr = 0.001  
batch_size = 64  
n_epochs = 100  
Patience = 20  
Dropout = 0.2  


## How to execute CNN:

- install requirements_cnn.txt
- run: python3 MyCNN_batch.py
  ** script loads csv files from Final_Dataset_CSV folder
- output: plots, configuration file and final model will be saved in folder Saved_models

  
## How to execute VGG:

- install requirements_vgg.txt
- run: python3 VGG-Transfer.py  
  ** script loads rgb frames from drive
- output: plots, configuration file and final model will be saved in drive

drive link: https://drive.google.com/drive/folders/1xK0gFjICOMl83dOvY1jLtAkxo69UiWmF?usp=drive_link



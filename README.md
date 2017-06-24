# Lunar crater counting using Convolutional Networks

This repo contains the essential pieces of code to train a convnet to recognize craters on the moon. 

# Getting the Data
Create a folder on your home directory to house all your data. By default this repo contains the folder 'dataset' for this purpose. This will correspond to the ‘dir’ variable in moon_unet_s256_rings_public.py

From /scratch/r/rein/silburt/ on scinet, get the following numpy files, and copy them to the following folders within the ‘dir’ directory:

*File*				*Folder within ‘dir’*
train_data.npy   	->	Train_rings/
train_target.npy   	-> 	Train_rings/
dev_data.npy		->	Dev_rings/
dev_target.npy		->	Dev_rings
test_data.npy		->	Test_rings/
test_target.npy		->	Test_rings/
custom_loss_csvs.npy	->	Dev_rings_for_loss/
custom_loss_images.npy	->	Dev_rings_for_loss/

# Running the Code
All the code is contained within moon_unet_s256_rings_public.py. All the main parameters that you might want to change is at the bottom of the script, under #Arguments, Run#. These parameters have explanations given. 

In addition, if you want to iterate over parameters (i.e. run a grid search), look for:
########## Parameters to Iterate Over ########## 
in the run_cross_validation_create_models function. I’ve given a simple example of how to do this in the code. As it currently stands, it will save models for every set of parameters you iterate over. Look for model.save() within the train_and_test_model() function and make sure that the name assigned to each model is unique so that models wont get overwritten as you iterate.
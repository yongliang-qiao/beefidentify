# beefidentify
deep learning based beef cattle identification

Mani steps:
1. Extracting features
    Extract jpegs of each frame for each video using python 2_extract_files.py ;
    extract features from the images with the CNN by running extract_features.py;

2. model training
   CNN model: train_cnn.py
   Sequence based model: train.py
   
 3. model configure
 
  All model setting in models.py. When training model, chaning the corresponding setting in  train.py
  
  Model name and sequence length can be changed. 
 

# End to end pipeline for face mask detection



# Dataset:
Contains images of people
Includes
Classes:
with_mask
without_mask
mask_weared_incorrect



# Done
-  Getting the dataset from kaggle
- Downloaded the dataset and found out the dataset is already annotated so will be moving to the next task which is to divide the data into train test and validation split



# not yet done 

- Data Set Loading 
- Load and parse annotation files (Pascal VOC XML → TensorFlow format)
- Resize images uniformly (e.g., 224x224 or 416x416)
- Encode bounding boxes and class labels
- Data set cleanup
- Split into train/val/test
- Data preprocessing 
- Selecting the model for data and image detection for this i will try to do it with the help of yolo 
- Try to train model and get some good acuracy score from that 
- After the model training is complete i will be making a streamlit/ flask web application so that i can get results based on my data that i will be putting to the pipleine 
- after the task complete i will be making a end to end pipeline from this 




## Path 
Use pre-trained as feature extractor
Add custom regression head for
Add classification head for
Train using a : (MSE for boxes + Categorical CrossEntropy for class)


## Report 
at IoU=0.5
per class
Overlay predicted bounding boxes on 10 random test images
Deploy as a using:
OpenCV + TensorFlow SavedModel
Streamlit-based web app
Quantize with (TFLite)





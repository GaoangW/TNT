# TC_CNN
# Training 2D Tracking
1. Prepare the Data. <br />
Ground truth tracking file: follow the format of MOT (https://motchallenge.net/). <br />
The frame index and object index are from 1 (not 0) for both tracking ground truth and video frames. <br />
2. Convert MOT fromat to UA-Detrac format. <br />
TC_CNN/General/MOT_to_UA_Detrac.m <br />
3. Crop the ground truth detection into individual bounding box images. <br />
TC_CNN/General/crop_UA_Detrac.m <br />
4. Train the triplet appearance model based on FaceNet using the cropped data. <br />
See https://github.com/davidsandberg/facenet. <br />
All the useful scource code are in TC_CNN/src/. <br />
5. Train 2D tracking. <br />
Set directory paths in TC_CNN/train_cnn_trajectory_2d.py before the definition of all the functions. <br />
Change the sample probability (sample_prob) according to your data density. The number of element in sample_prob is the number of your input Mat files. <br />
Set the learning rate (lr) to 1e-3 at the beginning. Every 2000 steps, decrease lr by 10 times until it reaches 1e-5. <br />
The output model will be stored in save_dir. <br />

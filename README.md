# TNT
# 2D Tracking Training
1. Prepare the Data. <br />
Ground truth tracking file: follow the format of MOT (https://motchallenge.net/). <br />
The frame index and object index are from 1 (not 0) for both tracking ground truth and video frames. <br />
2. Convert MOT fromat to UA-Detrac format. <br />
TNT/General/MOT_to_UA_Detrac.m <br />
3. Crop the ground truth detection into individual bounding box images. <br />
TNT/General/crop_UA_Detrac.m <br />
4. Train the triplet appearance model based on FaceNet using the cropped data. <br />
See https://github.com/davidsandberg/facenet. <br />
All the useful scource code are in TNT/src/. <br />
5. Train 2D tracking. <br />
Set directory paths in TNT/train_cnn_trajectory_2d.py before the definition of all the functions. <br />
Change the sample probability (sample_prob) according to your data density. The number of element in sample_prob is the number of your input Mat files. <br />
Set the learning rate (lr) to 1e-3 at the beginning. Every 2000 steps, decrease lr by 10 times until it reaches 1e-5. <br />
The output model will be stored in save_dir. <br />
# 2D Tracking Testing
1. Prepare the detection data. <br />
follow the format of MOT (https://motchallenge.net/). <br />
The frame index and object index are from 1 (not 0) for both tracking ground truth and video frames. <br />
2. Set your data and model paths correctly on the top of TNT/tracklet_utils_3c.py. <br />
3. Set the file_len to be the string length of your input frame name before the extension. <br />
4. Adjust the tracking parameters in track_struct['track_params'] of TNT/tracklet_utils_3c.py in the function TC_tracker(). <br />

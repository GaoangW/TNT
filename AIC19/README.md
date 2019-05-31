# TNT
Environment setting is follows. <br />
Operating system: windows, need to change (from scipy.io import loadmat in TNT/train_cnn_trajectory_2d.py) to (h5py) in Linux. <br />
Python: 3.5.4 <br />
tensorflow: 1.4.0 <br />
cuda: 8.0 <br />
cudnn: 5.1.10 <br />
opencv: 3.2.0 <br />
Other packages: numpy, pickle, sklearn, scipy, matplotlib, PIL. <br />

# 2D Tracking Testing
1. Prepare the detection data. <br />
follow the format of MOT (https://motchallenge.net/). <br />
The frame index and object index are from 1 (not 0) for both tracking ground truth and video frames. <br />
2. Set your data and model paths correctly on the top of TNT/AIC19/tracklet_utils_3c.py. <br />
The model can be downloaded from https://drive.google.com/open?id=1mbJf08hJY0qXV2ZnMBHq-Bzw8U3Br54l <br />
3. Set the file_len to be the string length of your input frame name before the extension. <br />
4. Adjust the tracking parameters in track_struct['track_params'] of TNT/AIC19/tracklet_utils_3c.py in the function TC_tracker(). <br />
5. Run python TNT/AIC19/TC_tracker.py. <br />
6. Run post_deep_match.py for post processing if necessary.

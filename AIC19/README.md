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
The model can be downloaded from https://drive.google.com/drive/folders/1UJHoCz1P9rINqjHJP7ozGRnW5_rX2IXl?usp=sharing. <br />
3. Set the file_len to be the string length of your input frame name before the extension. <br />
4. Adjust the tracking parameters in track_struct['track_params'] of TNT/AIC19/tracklet_utils_3c.py in the function TC_tracker(). <br />
5. Run python TNT/AIC19/TC_tracker.py. <br />
6. Run post_deep_match.py for post processing if necessary.
7. The SCT results are saved in the txt_result folder.
8. Run get_GPS.m to obtain the GPS location of the tracking results.
9. The final outputs are saved in the txt_GPS_new folder.

# SCT results
The SCT results are saved in TNT/AIC19/txt_GPS_new.

# Citation
Use this bibtex to cite this repository:
```
@inproceedings{wang2019exploit,
  title={Exploit the connectivity: Multi-object tracking with trackletnet},
  author={Wang, Gaoang and Wang, Yizhou and Zhang, Haotian and Gu, Renshu and Hwang, Jenq-Neng},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={482--490},
  year={2019},
  organization={ACM}
}
```

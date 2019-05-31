
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import cv2
import pickle

from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import svm
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from PIL import Image
import seq_nn_3d_v2
import tracklet_utils_3c
#import post_deep_match
import track_lib



track_struct = tracklet_utils_3c.TC_tracker()
'''
#cam_list = ['c020','c021','c023','c020','c025','c027','c029','c035']
cam_list = ['c029']
for n in range(0,len(cam_list)):
    seq_name = cam_list[n]
    img_name = cam_list[n]
    sub_seq_name = ''
    file_len = 6
    ROI_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/s5_MOG2_result/s5_MOG2_result/test_S05_'+cam_list[n]+'/frame/'
    det_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/aic19-track1-mtmc/test/s05/'+cam_list[n]+'/det/det_ssd512.txt'
    img_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/test_img/'+seq_name
    crop_det_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/crop_det/'+seq_name+sub_seq_name
    tracking_img_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/tracking_img/'+seq_name+sub_seq_name
    tracking_video_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/tracking_video/'+seq_name+sub_seq_name+'.avi'
    appear_mat_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/appear_mat/'+seq_name+'.obj'
    txt_result_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/txt_result/'+seq_name+sub_seq_name+'.txt'
    track_struct_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019/track_struct/'+seq_name+sub_seq_name+'.obj'
    
    track_struct, img_size = tracklet_utils_3c_fun.TC_tracker(seq_name, img_name, sub_seq_name, file_len, ROI_path, det_path, img_folder, crop_det_folder, tracking_img_folder, tracking_video_path, appear_mat_path, txt_result_path, track_struct_path)
    
    #post_deep_match_fun.post_deep_match_fun(seq_name,img_name,img_size)
    #import pdb; pdb.set_trace()
'''

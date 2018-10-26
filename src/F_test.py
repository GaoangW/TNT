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
import time
from functools import wraps

from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy import misc
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

import track_lib


seq_name = 'MOT17-14-FRCNN'
img_name = 'MOT17-14'
sub_seq_name = ''
det_path = 'D:/Data/MOT/MOT17Labels/test/'+seq_name+'/det/det.txt'
img_folder = 'D:/Data/MOT/MOT17Det/test/'+img_name+sub_seq_name+'/img1'
crop_det_folder = 'D:/Data/MOT/crop_det/'+seq_name+sub_seq_name
triplet_model = 'D:/Data/UA-Detrac/UA_Detrac_model/MOT'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/cnn_MOT/model.ckpt'
seq_model = 'D:/Data/UA-Detrac/MOT_2d/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/semi_train_model/model.ckpt'
tracking_img_folder = 'D:/Data/MOT/tracking_img/'+seq_name+sub_seq_name
tracking_video_path = 'D:/Data/MOT/tracking_video/'+seq_name+sub_seq_name+'.avi'
svm_model_path = 'D:/Data/MOT/MOT17_train_det_crop/cnn_svm_MOT17.pkl'
rand_forest_model_path = 'D:/Data/MOT/MOT17_train_det_crop/rand_forest_MOT17_FRCNN.pkl'

save_fea_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'.obj'
save_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_label.obj'
save_remove_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_remove_set.obj'
save_all_fea_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all.obj'
save_all_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label.obj'

save_all_label_path1 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label0.obj'
save_all_label_path2 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label1.obj'
save_all_label_path3 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label2.obj'
save_all_label_path4 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label3.obj'

txt_result_path = 'D:/Data/MOT/txt_result/'+seq_name+sub_seq_name+'.txt'
track_struct_path = 'D:/Data/MOT/track_struct/'+seq_name+sub_seq_name+'.obj'

models = []
model_info = []
inliers1 = []
inliers2 = []

geometry_folder = 'D:/Data/MOT/geometry_info'
geo_model_path = geometry_folder+'/MOT17-14-FRCNN_models.obj'
geo_model_info_path = geometry_folder+'/MOT17-14-FRCNN_model_info.obj'
geo_inlier1_path = geometry_folder+'/MOT17-14-FRCNN_inliers1.obj'
geo_inlier2_path = geometry_folder+'/MOT17-14-FRCNN_inliers2.obj'

global track_struct
track_struct = pickle.load(open(track_struct_path,'rb'))

global remove_set
remove_set = []


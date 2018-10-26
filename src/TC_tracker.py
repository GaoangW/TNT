
# coding: utf-8

# In[1]:


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
from PIL import Image
import seq_nn
import tracklet_utils


# In[ ]:


'''
det_path = 'D:/Data/UA-Detrac/CompACT-test/CompACT/MVI_39051_Det_CompACT.txt'
img_folder = 'D:/Data/UA-Detrac/DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_39051'
crop_det_folder = 'D:/Data/UA-Detrac/crop_det/MVI_39051'
triplet_model = 'D:/Data/UA-Detrac/UA_Detrac_model/20180627-211315'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
seq_model = 'D:/Data/UA-Detrac/fine_tune_model/model.ckpt'
tracking_img_folder = 'D:/Data/UA-Detrac/tracking_img/MVI_39051'
tracking_video_path = 'D:/Data/UA-Detrac/tracking_video/MVI_39051.avi'
save_fea_path = 'D:/Data/UA-Detrac/save_fea_mat/MVI_39051.obj'
save_label_path = 'D:/Data/UA-Detrac/save_fea_mat/MVI_39051_label.obj'
'''

'''
det_path = 'D:/Data/KITTI/data_tracking_det_2_regionlets/testing/det_02/0000.txt'
img_folder = 'D:/Data/KITTI/data_tracking_image_2/testing/image_02/0000'
crop_det_folder = 'D:/Data/KITTI/crop_det/0000'
triplet_model = 'D:/Data/UA-Detrac/UA_Detrac_model/20180627-211315'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
seq_model = 'D:/Data/UA-Detrac/cnn_appear_reducemean_7700steps/model.ckpt'
tracking_img_folder = 'D:/Data/KITTI/tracking_img/0000'
tracking_video_path = 'D:/Data/KITTI/tracking_video/0000.avi'
save_fea_path = 'D:/Data/KITTI/save_fea_mat/0000.obj'
save_label_path = 'D:/Data/KITTI/save_fea_mat/0000_label.obj'
max_length = 64
feature_size = 4+512
batch_size = 64
num_classes = 2
'''


# In[ ]:


'''
track_set = np.array([[4,9,1],
                      [1,16,0],
                      [16,20,0],
                      [20,33,0],
                      [33,38,0],
                      [38,44,0],
                      [44,48,0],
                      [8,11,0],
                      [12,21,0],
                      [14,23,0],
                      [10,30,0],
                      [9,25,0],
                      [25,28,0],
                      [28,36,0],
                      [36,39,0],
                      [39,47,0],
                      [41,43,1],
                      [52,56,0],
                      [52,63,1],
                      [63,76,0],
                      [73,77,0],
                      [73,78,1],
                      [78,93,0],
                      [90,93,1],
                      [96,100,0],
                      [86,98,0],
                      [75,97,0],
                      [84,99,0],
                      [104,105,0],
                      [95,105,1],
                      [105,108,1],
                      [113,117,0],
                      [122,130,0],
                      [130,135,0],
                      [127,130,1],
                      [128,145,0],
                      [131,139,0],
                      [139,150,0],
                      [138,155,0],
                      [148,160,0],
                      [114,134,0],
                      [134,140,0],
                      [140,147,0],
                      [147,152,0],
                      [137,152,1],
                      [157,162,1]])
'''
track_struct = tracklet_utils.TC_tracker()


# In[ ]:





# In[ ]:





# In[ ]:


'''
track_set = np.array([[8,9,0],
                     [5,9,1],
                     [13,14,0],
                     [13,19,1],
                     [24,26,0],
                     [24,25,1],
                     [34,39,0],
                     [33,36,0],
                     [33,35,1],
                     [45,47,0],
                     [45,48,1],
                     [47,50,1],
                     [49,51,1],
                     [48,52,0],
                     [60,62,0],
                     [43,45,0],
                     [54,57,1],
                     [57,59,1],
                     [32,39,0],
                     [39,41,1],
                     [42,45,1]])
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





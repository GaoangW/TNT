    
/*
 * Copyright ©2019 Gaoang Wang.  All rights reserved.  Permission is
 * hereby granted for academic use.  No other use, copying, distribution, or modification
 * is permitted without prior written consent. Copyrights for
 * third-party components of this work must be honored.  Instructors
 * interested in reusing these course materials should contact the
 * author.
 */


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
import seq_nn_3d

seq_name = '2011_09_26_drive_0091_sync'
img_name = '2011_09_26_drive_0091_sync'
sub_seq_name = ''
#det_path = 'D:/Data/KITTI/'+seq_name+'/disturb_gt_2.txt'
det_path = 'D:/Data/KITTI/dets.txt'
img_folder = 'D:/Data/KITTI/'+img_name+sub_seq_name+'/image_02/data'
crop_det_folder = 'D:/Data/KITTI/crop_det/'+seq_name+sub_seq_name
triplet_model = 'D:/Data/UA-Detrac/UA_Detrac_model/KITTI_model'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
seq_model = 'D:/Data/UA-Detrac/KITTI_model/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/final_model/model.ckpt'
tracking_img_folder = 'D:/Data/KITTI/tracking_img/'+seq_name+sub_seq_name
tracking_video_path = 'D:/Data/KITTI/tracking_video/'+seq_name+sub_seq_name+'.avi'

save_fea_path = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'.obj'
save_label_path = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_label.obj'
save_remove_path = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_remove_set.obj'
save_all_fea_path = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_all.obj'
save_all_label_path = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_all_label.obj'

save_all_label_path1 = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_all_label0.obj'
save_all_label_path2 = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_all_label1.obj'
save_all_label_path3 = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_all_label2.obj'
save_all_label_path4 = 'D:/Data/KITTI/save_fea_mat/'+seq_name+sub_seq_name+'_all_label3.obj'

txt_result_path = 'D:/Data/KITTI/txt_result/'+seq_name+sub_seq_name+'.txt'
track_struct_path = 'D:/Data/KITTI/track_struct/'+seq_name+sub_seq_name+'.obj'


max_length = 64
feature_size = 4+512
batch_size = 64
num_classes = 2


track_set = []
remove_set = []


#track_set = pickle.load(open(save_label_path,'rb'))
#remove_set = pickle.load(open(save_remove_path,'rb'))


#save_fea_mat = np.zeros((len(track_set),feature_size,max_length,2))


global all_fea_mat
global all_fea_label
all_fea_mat = np.zeros((10000,feature_size,max_length,3))
all_fea_label = np.zeros((10000,4))


def linear_pred(y):
    if len(y)==1:
        return y
    else:
        x = np.array(range(0,len(y)))
        slope, intercept, _, _, _ = stats.linregress(x,y)
        return slope*len(y)+intercept
    
def linear_pred_v2(tr_t, tr_y, ts_t):
    ts_y = np.ones(len(ts_t))
    if len(tr_t)==1:
        ts_y = ts_y*tr_y
    else:
        slope, intercept, _, _, _ = stats.linregress(tr_t,tr_y)
        ts_y = slope*ts_t+intercept
    return ts_y

def file_name(num, length):
    cnt = 1
    temp = num
    while 1:
        temp = int(temp/10)
        if temp>0:
            cnt = cnt+1
        else:
            break
    num_len = cnt
    for n in range(length-num_len): 
        if n==0:
            out_str = '0'
        else:
            out_str = out_str+'0'
    if length-num_len>0:
        return out_str+str(num)
    else:
        return str(num)
    
#bbox = [x, y, w, h]
def get_IOU(bbox1, bbox2): 
    area1 = bbox1[2]*bbox1[3] 
    area2 = bbox2[2]*bbox2[3] 
    x1 = max(bbox1[0], bbox2[0]) 
    y1 = max(bbox1[1], bbox2[1]) 
    x2 = min(bbox1[0]+bbox1[2]-1, bbox2[0]+bbox2[2]-1) 
    y2 = min(bbox1[1]+bbox1[3]-1, bbox2[1]+bbox2[3]-1)

    #import pdb; pdb.set_trace()
    overlap_area = max(0, (x2-x1+1))*max(0, (y2-y1+1))
    ratio = overlap_area/(area1+area2-overlap_area)
    return ratio

def get_overlap(bbox1, bbox2): 
    num1 = bbox1.shape[0] 
    num2 = bbox2.shape[0] 
    overlap_mat = np.zeros((num1, num2)) 
    for n in range(num1): 
        for m in range(num2):

            #import pdb; pdb.set_trace()
            overlap_mat[n,m] = get_IOU(bbox1[n,:], bbox2[m,:])

    return overlap_mat

def load_detection(file_name, dataset):

    # M=[fr_id (from 1), x, y, w, h, det_score]
    if dataset=='Underwater':
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        M = np.zeros((f.shape[0], 6))
        M[:,0] = f[:,0]+1
        M[:,1:5] = f[:,1:5]
        M[:,5] = f[:,5]
        M[:,3] = M[:,3]-M[:,1]+1
        M[:,4] = M[:,4]-M[:,2]+1
        return M
    if dataset=='UA-Detrac':
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        M = np.zeros((f.shape[0], 6))
        M[:,0] = f[:,0]
        M[:,1:6] = f[:,2:7]
        #import pdb; pdb.set_trace()
        return M
    if dataset=='KITTI':
        f = np.loadtxt(det_path,delimiter=' ',dtype='str')
        mask = np.zeros((len(f),1))
        for n in range(len(f)):
            if f[n][2]=='Car' or f[n][2]=='Van':
                mask[n,0] = 1
        num = int(np.sum(mask))
        M = np.zeros((num, 6))
        cnt = 0
        for n in range(len(f)):
            if mask[n,0]==1:
                M[cnt,0] = int(float(f[n][0]))+1
                M[cnt,1] = int(float(f[n][6]))
                M[cnt,2] = int(float(f[n][7]))
                M[cnt,3] = int(float(f[n][8]))-int(float(f[n][6]))+1
                M[cnt,4] = int(float(f[n][9]))-int(float(f[n][7]))+1
                M[cnt,5] = float(f[n][17])
                cnt = cnt+1
                
        #import pdb; pdb.set_trace()
        return M
    if dataset=='MOT':
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        M = np.zeros((f.shape[0], 6))
        M[:,0] = f[:,0]
        M[:,1:6] = f[:,2:7]
        #import pdb; pdb.set_trace()
        return M
    if dataset=='KITTI_3d':
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        mask = np.zeros((len(f),1))
        for n in range(len(f)):
            # only for pedestrian
            if f[n][7]==4 or f[n][7]==5 or f[n][7]==6:
                mask[n,0] = 1
        num = int(np.sum(mask))
        
        M = np.zeros((num, 10))
        cnt = 0
        for n in range(len(f)):
            if mask[n,0]==1:
                M[cnt,0] = int(float(f[n][0]))+1
                M[cnt,1] = int(float(f[n][2]))
                M[cnt,2] = int(float(f[n][3]))
                M[cnt,3] = int(float(f[n][4]))
                M[cnt,4] = int(float(f[n][5]))
                M[cnt,5] = 1.0
                M[cnt,6] = float(f[n][8])
                M[cnt,7] = float(f[n][9])
                M[cnt,8] = float(f[n][10])
                M[cnt,9] = float(f[n][11])
                cnt = cnt+1
            #import pdb; pdb.set_trace()
        return M
    
    if dataset=='KITTI_3d_2':
        f = np.loadtxt(file_name, dtype=str, delimiter=',')
        f = np.array(f)
        mask = np.zeros((len(f),1))
        for n in range(len(f)):
            # only for pedestrian
            if f[n][11]=="Pedestrian" or f[n][11]=="Cyclist":
                mask[n,0] = 1
        num = int(np.sum(mask))
        
        M = np.zeros((num, 10))
        cnt = 0
        for n in range(len(f)):
            if mask[n,0]==1:
                M[cnt,0] = int(float(f[n][0]))+1
                M[cnt,1] = int(float(f[n][1]))
                M[cnt,2] = int(float(f[n][2]))
                M[cnt,3] = int(float(f[n][3]))
                M[cnt,4] = int(float(f[n][4]))
                M[cnt,5] = float(f[n][10])/100.0
                M[cnt,6] = float(f[n][5])/723.0
                M[cnt,7] = float(f[n][7])/723.0
                M[cnt,8] = float(f[n][8])/723.0
                M[cnt,9] = float(f[n][9])/723.0
                cnt = cnt+1
            #import pdb; pdb.set_trace()
        return M
        
def bbox_associate(overlap_mat, IOU_thresh): 
    idx1 = [] 
    idx2 = [] 
    while 1: 
        idx = np.unravel_index(np.argmax(overlap_mat, axis=None), overlap_mat.shape) 
        if overlap_mat[idx]<IOU_thresh: 
            break 
        else: 
            idx1.append(idx[0]) 
            idx2.append(idx[1]) 
            overlap_mat[idx[0],:] = 0 
            overlap_mat[:,idx[1]] = 0

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    return idx1, idx2

def merge_bbox(bbox, IOU_thresh, det_score): 
    num = bbox.shape[0] 
    cand_idx = np.ones((num,1)) 
    for n1 in range(num-1): 
        for n2 in range(n1+1,num): 
            if cand_idx[n1,0]==0 or cand_idx[n2,0]==0: 
                continue

            #import pdb; pdb.set_trace()
            a = np.zeros((1,4))
            b = np.zeros((1,4))
            a[0,:] = bbox[n1,:]
            b[0,:] = bbox[n2,:]
            r = get_overlap(a, b)[0,0]
            s1 = det_score[n1]
            s2 = det_score[n2]
            if r>IOU_thresh:
                if s1>s2:
                    cand_idx[n2] = 0
                else:
                    cand_idx[n1] = 0
    idx = np.where(cand_idx==1)[0]
    new_bbox = bbox[idx,:]
    return idx, new_bbox

def preprocessing(tracklet_mat, len_thresh): 
    new_tracklet_mat = tracklet_mat 
    N_tracklet = new_tracklet_mat['xmin_mat'].shape[0] 
    remove_idx = []
    for n in range(N_tracklet): 
        idx = np.where(new_tracklet_mat['xmin_mat'][n,:]!=-1)[0] 
        max_det_score = np.max(new_tracklet_mat['det_score_mat'][n,idx])
        if len(idx)<len_thresh and max_det_score<0.2: 
            remove_idx.append(n) 
            
    new_tracklet_mat['xmin_mat'] = np.delete(new_tracklet_mat['xmin_mat'], remove_idx, 0) 
    new_tracklet_mat['ymin_mat'] = np.delete(new_tracklet_mat['ymin_mat'], remove_idx, 0) 
    new_tracklet_mat['xmax_mat'] = np.delete(new_tracklet_mat['xmax_mat'], remove_idx, 0) 
    new_tracklet_mat['ymax_mat'] = np.delete(new_tracklet_mat['ymax_mat'], remove_idx, 0) 
    new_tracklet_mat['det_score_mat'] = np.delete(new_tracklet_mat['det_score_mat'], remove_idx, 0) 
    new_tracklet_mat['x_3d_mat'] = np.delete(new_tracklet_mat['x_3d_mat'], remove_idx, 0) 
    new_tracklet_mat['y_3d_mat'] = np.delete(new_tracklet_mat['y_3d_mat'], remove_idx, 0) 
    new_tracklet_mat['w_3d_mat'] = np.delete(new_tracklet_mat['w_3d_mat'], remove_idx, 0) 
    new_tracklet_mat['h_3d_mat'] = np.delete(new_tracklet_mat['h_3d_mat'], remove_idx, 0) 
    return new_tracklet_mat

#M = [fr_idx, x, y, w, h, score]
def forward_tracking(track_id1, track_id2, bbox1, bbox2, bbox_3d_1, bbox_3d_2, det_score1, det_score2, mean_color1, mean_color2, 
                     fr_idx2, track_params, tracklet_mat, max_id): 
    color_thresh = track_params['color_thresh']
    num_fr = track_params['num_fr']
    linear_pred_thresh = track_params['linear_pred_thresh']
    if len(bbox1)>0: 
        num1 = bbox1.shape[0] 
    else: 
        num1 = 0 
    if len(bbox2)>0: 
        num2 = bbox2.shape[0] 
    else: 
        num2 = 0

    new_track_id1 = track_id1
    new_tracklet_mat = tracklet_mat
    if fr_idx2==2 and num1>0:
        new_track_id1 = list(range(1,num1+1))
        '''
        new_tracklet_mat['xmin_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['ymin_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['xmax_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['ymax_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['det_score_mat'] = -np.ones((num1, num_fr))
        '''
        new_tracklet_mat['xmin_mat'][0:num1,0] = bbox1[:,0]
        new_tracklet_mat['ymin_mat'][0:num1,0] = bbox1[:,1]
        new_tracklet_mat['xmax_mat'][0:num1,0] = bbox1[:,0]+bbox1[:,2]-1
        new_tracklet_mat['ymax_mat'][0:num1,0] = bbox1[:,1]+bbox1[:,3]-1
        new_tracklet_mat['det_score_mat'][0:num1,0] = det_score1        
        new_tracklet_mat['x_3d_mat'][0:num1,0] = bbox_3d_1[:,0]
        new_tracklet_mat['y_3d_mat'][0:num1,0] = bbox_3d_1[:,1]
        new_tracklet_mat['w_3d_mat'][0:num1,0] = bbox_3d_1[:,2]
        new_tracklet_mat['h_3d_mat'][0:num1,0] = bbox_3d_1[:,3]
        max_id = num1
        
    if len(bbox1)==0 and len(bbox2)!=0:
        idx1 = []
        idx2 = []
    elif len(bbox1)!=0 and len(bbox2)==0:
        idx1 = []
        idx2 = []    
    elif len(bbox1)==0 and len(bbox2)==0:
        idx1 = []
        idx2 = [] 
    elif len(bbox1)!=0 and len(bbox2)!=0:
        # pred bbox1
        pred_bbox1 = np.zeros((len(bbox1),4))
        for k in range(len(bbox1)):
            temp_track_id = new_track_id1[k]-1
            t_idx = np.where(new_tracklet_mat['xmin_mat'][temp_track_id,:]!=-1)[0]
            t_min = np.min(t_idx)
            if t_min<fr_idx2-2-linear_pred_thresh:
                t_min = fr_idx2-2-linear_pred_thresh
            xx = (new_tracklet_mat['xmin_mat'][temp_track_id,int(t_min):fr_idx2-1]
                  +new_tracklet_mat['xmax_mat'][temp_track_id,int(t_min):fr_idx2-1])/2
            yy = (new_tracklet_mat['ymin_mat'][temp_track_id,int(t_min):fr_idx2-1]
                  +new_tracklet_mat['ymax_mat'][temp_track_id,int(t_min):fr_idx2-1])/2
            ww = (new_tracklet_mat['xmax_mat'][temp_track_id,int(t_min):fr_idx2-1]
                  -new_tracklet_mat['xmin_mat'][temp_track_id,int(t_min):fr_idx2-1])+1
            hh = (new_tracklet_mat['ymax_mat'][temp_track_id,int(t_min):fr_idx2-1]
                  -new_tracklet_mat['ymin_mat'][temp_track_id,int(t_min):fr_idx2-1])+1
            if len(xx)==0:
                import pdb; pdb.set_trace()
            pred_x = linear_pred(xx)
            pred_y = linear_pred(yy)
            pred_w = linear_pred(ww)
            pred_h = linear_pred(hh)
            pred_bbox1[k,2] = max(pred_w,1)
            pred_bbox1[k,3] = max(pred_h,1)
            pred_bbox1[k,0] = pred_x-pred_w/2
            pred_bbox1[k,1] = pred_y-pred_h/2
            
        #import pdb; pdb.set_trace()
        overlap_mat = get_overlap(pred_bbox1, bbox2)
        # color dist
        color_dist = np.zeros((len(bbox1),len(bbox2)))
        for n1 in range(len(bbox1)):
            for n2 in range(len(bbox2)):
                color_dist[n1,n2] = np.max(np.absolute(mean_color1[n1,:]-mean_color2[n2,:]))
        overlap_mat[color_dist>color_thresh] = 0    
        idx1, idx2 = bbox_associate(overlap_mat, track_params['IOU_thresh'])

    if len(idx1)==0 and num2>0:
        new_track_id2 = list(np.array(range(1,num2+1))+max_id)
        '''
        new_tracklet_mat['xmin_mat'] = \
            np.append(new_tracklet_mat['xmin_mat'], -np.ones((num2,num_fr)), axis=0)
        new_tracklet_mat['ymin_mat'] = \
            np.append(new_tracklet_mat['ymin_mat'], -np.ones((num2,num_fr)), axis=0)
        new_tracklet_mat['xmax_mat'] = \
            np.append(new_tracklet_mat['xmax_mat'], -np.ones((num2,num_fr)), axis=0)
        new_tracklet_mat['ymax_mat'] = \
            np.append(new_tracklet_mat['ymax_mat'], -np.ones((num2,num_fr)), axis=0)
        new_tracklet_mat['det_score_mat'] = \
            np.append(new_tracklet_mat['det_score_mat'], -np.ones((num2,num_fr)), axis=0)
        '''
        max_id = max_id+num2
        new_tracklet_mat['xmin_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,0]
        new_tracklet_mat['ymin_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,1]
        new_tracklet_mat['xmax_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,0]+bbox2[:,2]-1
        new_tracklet_mat['ymax_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,1]+bbox2[:,3]-1
        new_tracklet_mat['det_score_mat'][max_id-num2:max_id,fr_idx2-1] = det_score2
        new_tracklet_mat['x_3d_mat'][max_id-num2:max_id,fr_idx2-1] = bbox_3d_2[:,0]
        new_tracklet_mat['y_3d_mat'][max_id-num2:max_id,fr_idx2-1] = bbox_3d_2[:,1]
        new_tracklet_mat['w_3d_mat'][max_id-num2:max_id,fr_idx2-1] = bbox_3d_2[:,2]
        new_tracklet_mat['h_3d_mat'][max_id-num2:max_id,fr_idx2-1] = bbox_3d_2[:,3]
    elif len(idx1)>0:
        new_track_id2 = []
        for n in range(num2):
            #import pdb; pdb.set_trace()
            temp_idx = np.where(idx2==n)[0]
            if len(temp_idx)==0:
                max_id = max_id+1
                new_track_id2.append(max_id)
                '''
                new_tracklet_mat['xmin_mat'] = \
                    np.append(new_tracklet_mat['xmin_mat'], -np.ones((1,num_fr)), axis=0)
                new_tracklet_mat['ymin_mat'] = \
                    np.append(new_tracklet_mat['ymin_mat'], -np.ones((1,num_fr)), axis=0)
                new_tracklet_mat['xmax_mat'] = \
                    np.append(new_tracklet_mat['xmax_mat'], -np.ones((1,num_fr)), axis=0)
                new_tracklet_mat['ymax_mat'] = \
                    np.append(new_tracklet_mat['ymax_mat'], -np.ones((1,num_fr)), axis=0)
                new_tracklet_mat['det_score_mat'] = \
                    np.append(new_tracklet_mat['det_score_mat'], -np.ones((1,num_fr)), axis=0)
                '''
                #import pdb; pdb.set_trace()
                new_tracklet_mat['xmin_mat'][max_id-1,fr_idx2-1] = bbox2[n,0]
                new_tracklet_mat['ymin_mat'][max_id-1,fr_idx2-1] = bbox2[n,1]
                new_tracklet_mat['xmax_mat'][max_id-1,fr_idx2-1] = bbox2[n,0]+bbox2[n,2]-1
                new_tracklet_mat['ymax_mat'][max_id-1,fr_idx2-1] = bbox2[n,1]+bbox2[n,3]-1
                new_tracklet_mat['det_score_mat'][max_id-1,fr_idx2-1] = det_score2[n]
                new_tracklet_mat['x_3d_mat'][max_id-1,fr_idx2-1] = bbox_3d_2[n,0]
                new_tracklet_mat['y_3d_mat'][max_id-1,fr_idx2-1] = bbox_3d_2[n,1]
                new_tracklet_mat['w_3d_mat'][max_id-1,fr_idx2-1] = bbox_3d_2[n,2]
                new_tracklet_mat['h_3d_mat'][max_id-1,fr_idx2-1] = bbox_3d_2[n,3]
            else:
                temp_idx = temp_idx[0]
                new_track_id2.append(new_track_id1[idx1[temp_idx]])
                new_tracklet_mat['xmin_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox2[n,0]
                new_tracklet_mat['ymin_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox2[n,1]
                new_tracklet_mat['xmax_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox2[n,0]+bbox2[n,2]-1
                new_tracklet_mat['ymax_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox2[n,1]+bbox2[n,3]-1
                new_tracklet_mat['det_score_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = det_score2[n]
                new_tracklet_mat['x_3d_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox_3d_2[n,0]
                new_tracklet_mat['y_3d_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox_3d_2[n,1]
                new_tracklet_mat['w_3d_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox_3d_2[n,2]
                new_tracklet_mat['h_3d_mat'] \
                    [new_track_id1[idx1[temp_idx]]-1,fr_idx2-1] = bbox_3d_2[n,3]
    else:
        new_track_id2 = []

    #import pdb; pdb.set_trace()
    new_max_id = max_id
    return new_tracklet_mat, new_track_id1, new_track_id2, new_max_id

def init_clustering(): 
    
    global track_struct
    
    N_tracklet = track_struct['tracklet_mat']['xmin_mat'].shape[0]

    # track interval
    track_struct['tracklet_mat']['track_interval'] = np.zeros((N_tracklet, 2))

    # track cluster
    track_struct['tracklet_mat']['track_cluster'] = []

    # track class
    track_struct['tracklet_mat']['track_class'] = np.arange(N_tracklet, dtype=int)

    # time cluster
    track_struct['tracklet_mat']['time_cluster'] = []
    for n in range(track_struct['track_params']['num_time_cluster']):
        track_struct['tracklet_mat']['time_cluster'].append([])
    
    track_struct['tracklet_mat']['track_cluster_t_idx'] = []
    for n in range(N_tracklet):
        idx = np.where(track_struct['tracklet_mat']['xmin_mat'][n,:]!=-1)[0]
        track_struct['tracklet_mat']['track_interval'][n,0] = np.min(idx)
        track_struct['tracklet_mat']['track_interval'][n,1] = np.max(idx)
        track_struct['tracklet_mat']['track_cluster'].append([n])
        
        if n in remove_set:
            track_struct['tracklet_mat']['track_cluster_t_idx'].append([-1])
        else:
            min_time_cluster_idx = int(np.floor(max(track_struct['tracklet_mat']['track_interval'][n,0]
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
            max_time_cluster_idx = int(np.floor(min(track_struct['tracklet_mat']['track_interval'][n,1]
                                            +track_struct['track_params']['t_dist_thresh']+5,
                                                    track_struct['tracklet_mat']['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
            track_struct['tracklet_mat']['track_cluster_t_idx'].append(list(range(min_time_cluster_idx,max_time_cluster_idx+1)))
            for k in range(min_time_cluster_idx,max_time_cluster_idx+1):
                track_struct['tracklet_mat']['time_cluster'][k].append(n)

    # get center position of each detection location
    mask = track_struct['tracklet_mat']['xmin_mat']==-1
    track_struct['tracklet_mat']['center_x'] = \
        (track_struct['tracklet_mat']['xmin_mat']+track_struct['tracklet_mat']['xmax_mat'])/2
    track_struct['tracklet_mat']['center_y'] = \
        (track_struct['tracklet_mat']['ymin_mat']+track_struct['tracklet_mat']['ymax_mat'])/2
    track_struct['tracklet_mat']['w'] = \
        track_struct['tracklet_mat']['xmax_mat']-track_struct['tracklet_mat']['xmin_mat']+1
    track_struct['tracklet_mat']['h'] = \
        track_struct['tracklet_mat']['ymax_mat']-track_struct['tracklet_mat']['ymin_mat']+1
    track_struct['tracklet_mat']['center_x'][mask] = -1
    track_struct['tracklet_mat']['center_y'][mask] = -1
    track_struct['tracklet_mat']['w'][mask] = -1
    track_struct['tracklet_mat']['h'][mask] = -1

    # neighbor_track_idx and conflict_track_idx
    track_struct['tracklet_mat']['neighbor_track_idx'] = []
    track_struct['tracklet_mat']['conflict_track_idx'] = []
    for n in range(N_tracklet):
        track_struct['tracklet_mat']['neighbor_track_idx'].append([])
        track_struct['tracklet_mat']['conflict_track_idx'].append([])
    for n in range(N_tracklet-1):
        for m in range(n+1, N_tracklet):
            t_min1 = track_struct['tracklet_mat']['track_interval'][n,0]
            t_max1 = track_struct['tracklet_mat']['track_interval'][n,1]
            t_min2 = track_struct['tracklet_mat']['track_interval'][m,0]
            t_max2 = track_struct['tracklet_mat']['track_interval'][m,1]
            overlap_len = min(t_max2,t_max1)-max(t_min1,t_min2)+1
            overlap_r = overlap_len/(t_max1-t_min1+1+t_max2-t_min2+1-overlap_len)
            if overlap_len>0 and overlap_r>track_struct['track_params']['track_overlap_thresh']:
                track_struct['tracklet_mat']['conflict_track_idx'][n].append(m)
                track_struct['tracklet_mat']['conflict_track_idx'][m].append(n)
                continue
            if overlap_len>0 and overlap_r<=track_struct['track_params']['track_overlap_thresh']:
                # check the search region
                t1 = int(max(t_min1,t_min2))
                t2 = int(min(t_max2,t_max1))
                if (t_min1<=t_min2 and t_max1>=t_max2) or (t_min1>=t_min2 and t_max1<=t_max2) or overlap_len>4:
                    track_struct['tracklet_mat']['conflict_track_idx'][n].append(m)
                    track_struct['tracklet_mat']['conflict_track_idx'][m].append(n)
                    continue

                cand_t = np.array(range(t1,t2+1))
                dist_x = abs(track_struct['tracklet_mat']['center_x'][n,cand_t] \
                         -track_struct['tracklet_mat']['center_x'][m,cand_t])
                dist_y = abs(track_struct['tracklet_mat']['center_y'][n,cand_t] \
                         -track_struct['tracklet_mat']['center_y'][m,cand_t])
                w1 = track_struct['tracklet_mat']['w'][n,cand_t]
                h1 = track_struct['tracklet_mat']['h'][n,cand_t]
                w2 = track_struct['tracklet_mat']['w'][m,cand_t]
                h2 = track_struct['tracklet_mat']['h'][m,cand_t]
                
                min_len = np.min([np.min(w1),np.min(h1),np.min(w2),np.min(h2)])
                min_dist_x1 = np.min(dist_x/min_len)
                min_dist_y1 = np.min(dist_y/min_len)
                min_dist_x2 = np.min(dist_x/min_len)
                min_dist_y2 = np.min(dist_y/min_len)
                if min_dist_x1<track_struct['track_params']['search_radius'] \
                    and min_dist_y1<track_struct['track_params']['search_radius'] \
                    and min_dist_x2<track_struct['track_params']['search_radius'] \
                    and min_dist_y2<track_struct['track_params']['search_radius']:
                    track_struct['tracklet_mat']['neighbor_track_idx'][n].append(m)
                    track_struct['tracklet_mat']['neighbor_track_idx'][m].append(n)

            if overlap_len<=0 and min(abs(t_min1-t_max2),abs(t_min2-t_max1)) \
                <track_struct['track_params']['t_dist_thresh']:
                if t_min1>=t_max2:
                    t1 = int(t_min1)
                    t2 = int(t_max2)
                else:
                    t1 = int(t_max1)
                    t2 = int(t_min2)
                
                #***********************************
                tr_t1 = np.array(range(int(t_min1),int(t_max1+1)))
                tr_x1 = track_struct['tracklet_mat']['center_x'][n,int(t_min1):int(t_max1+1)]
                tr_y1 = track_struct['tracklet_mat']['center_y'][n,int(t_min1):int(t_max1+1)]
                if len(tr_t1)>10:
                    if t_min1>=t_max2:
                        tr_t1 = tr_t1[0:10]
                        tr_x1 = tr_x1[0:10]
                        tr_y1 = tr_y1[0:10]
                    else:
                        tr_t1 = tr_t1[-10:]
                        tr_x1 = tr_x1[-10:]
                        tr_y1 = tr_y1[-10:]         
                ts_x1 = linear_pred_v2(tr_t1, tr_x1, np.array([t2]))
                ts_y1 = linear_pred_v2(tr_t1, tr_y1, np.array([t2]))
                dist_x1 = abs(ts_x1[0]-track_struct['tracklet_mat']['center_x'][m,t2])
                dist_y1 = abs(ts_y1[0]-track_struct['tracklet_mat']['center_y'][m,t2])
                
                tr_t2 = np.array(range(int(t_min2),int(t_max2+1)))
                tr_x2 = track_struct['tracklet_mat']['center_x'][m,int(t_min2):int(t_max2+1)]
                tr_y2 = track_struct['tracklet_mat']['center_y'][m,int(t_min2):int(t_max2+1)]
                if len(tr_t2)>10:
                    if t_min2>t_max1:
                        tr_t2 = tr_t2[0:10]
                        tr_x2 = tr_x2[0:10]
                        tr_y2 = tr_y2[0:10]
                    else:
                        tr_t2 = tr_t2[-10:]
                        tr_x2 = tr_x2[-10:]
                        tr_y2 = tr_y2[-10:]   
                        
                ts_x2 = linear_pred_v2(tr_t2, tr_x2, np.array([t1]))
                ts_y2 = linear_pred_v2(tr_t2, tr_y2, np.array([t1]))
                dist_x2 = abs(ts_x2[0]-track_struct['tracklet_mat']['center_x'][n,t1])
                dist_y2 = abs(ts_y2[0]-track_struct['tracklet_mat']['center_y'][n,t1])
                
                dist_x = min(dist_x1, dist_x2)
                dist_y = min(dist_y1, dist_y2)
                #***********************************
                
                #import pdb; pdb.set_trace()
                '''
                dist_x = abs(track_struct['tracklet_mat']['center_x'][n,t1] \
                         -track_struct['tracklet_mat']['center_x'][m,t2])
                dist_y = abs(track_struct['tracklet_mat']['center_y'][n,t1] \
                         -track_struct['tracklet_mat']['center_y'][m,t2])
                '''
                
                w1 = track_struct['tracklet_mat']['w'][n,t1]
                h1 = track_struct['tracklet_mat']['h'][n,t1]
                w2 = track_struct['tracklet_mat']['w'][m,t2]
                h2 = track_struct['tracklet_mat']['h'][m,t2]
                
                min_len = np.min([np.min(w1),np.min(h1),np.min(w2),np.min(h2)])
                min_dist_x1 = dist_x/min_len
                min_dist_y1 = dist_y/min_len
                min_dist_x2 = dist_x/min_len
                min_dist_y2 = dist_y/min_len
                
                if min_dist_x1<track_struct['track_params']['search_radius'] \
                    and min_dist_y1<track_struct['track_params']['search_radius'] \
                    and min_dist_x2<track_struct['track_params']['search_radius'] \
                    and min_dist_y2<track_struct['track_params']['search_radius']:
                    track_struct['tracklet_mat']['neighbor_track_idx'][n].append(m)
                    track_struct['tracklet_mat']['neighbor_track_idx'][m].append(n)

    # update neighbor idx
    #***********************************
    
    for n in range(len(track_set)):
        temp_label = track_set[n,2]
        if temp_label==1:
            if abs(track_struct['tracklet_mat']['track_interval'][track_set[n,0],1]-
                  track_struct['tracklet_mat']['track_interval'][track_set[n,1],0])>60:
                continue
                
            if track_set[n,0] not in track_struct['tracklet_mat']['neighbor_track_idx'][track_set[n,1]]:
                track_struct['tracklet_mat']['neighbor_track_idx'][track_set[n,1]].append(track_set[n,0])
                track_struct['tracklet_mat']['neighbor_track_idx'][track_set[n,0]].append(track_set[n,1])
            if track_set[n,0] in track_struct['tracklet_mat']['conflict_track_idx'][track_set[n,1]]:
                track_struct['tracklet_mat']['conflict_track_idx'][track_set[n,1]].remove(track_set[n,0])
                track_struct['tracklet_mat']['conflict_track_idx'][track_set[n,0]].remove(track_set[n,1])
        
        else:
            if track_set[n,0] in track_struct['tracklet_mat']['neighbor_track_idx'][track_set[n,1]]:
                track_struct['tracklet_mat']['neighbor_track_idx'][track_set[n,1]].remove(track_set[n,0])
                track_struct['tracklet_mat']['neighbor_track_idx'][track_set[n,0]].remove(track_set[n,1])
            if track_set[n,0] not in track_struct['tracklet_mat']['conflict_track_idx'][track_set[n,1]]:
                track_struct['tracklet_mat']['conflict_track_idx'][track_set[n,1]].append(track_set[n,0])
                track_struct['tracklet_mat']['conflict_track_idx'][track_set[n,0]].append(track_set[n,1])
        
                
    # cluster cost
    track_struct['tracklet_mat']['cluster_cost'] = []
    for n in range(N_tracklet):
        track_struct['tracklet_mat']['cluster_cost'].append(0)

    # save all comb cost for two tracklets
    # comb_track_cost [track_id1, track_id2, cost]
    # track_struct['tracklet_mat']['comb_track_cost'] = []

    # save feature mat for training
    '''
    if len(track_struct['tracklet_mat']['track_set'])>0:
        track_struct['tracklet_mat']['save_fea_mat'] = np.zeros((len(track_struct['tracklet_mat']['track_set']), feature_size, max_length, 2))
    else:
        track_struct['tracklet_mat']['save_fea_mat'] = []
    '''
    return 

def comb_cost(tracklet_set, feature_size, max_length, img_size, sess, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv): 
    #comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #comb_track_cost = np.array(tracklet_mat['comb_track_cost'].copy()) 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy() 
    #track_set = tracklet_mat['track_set'].copy()

    global track_struct
    global all_fea_mat
    global all_fea_label
    #import pdb; pdb.set_trace()
    tracklet_mat = track_struct['tracklet_mat']
    loc_scales = track_struct['track_params']['loc_scales']
    
    temp_sum = np.sum(all_fea_mat[:,4,:,1], axis=1)
    if len(np.where(temp_sum!=0)[0])==0:
        fea_id = 0
    else:
        fea_id = int(np.max(np.where(temp_sum!=0)[0]))+1
    
    
    #print(fea_id)
    #import pdb; pdb.set_trace()
    # cnn classifier
    N_tracklet = len(tracklet_set)
    track_interval = tracklet_mat['track_interval']
    sort_idx = np.argsort(track_interval[np.array(tracklet_set),1])
    cost = 0
    if len(sort_idx)<=1:
        return cost


    remove_ids = []
    #comb_fea_mat = np.zeros((len(sort_idx)-1,feature_size,max_length,2))
    #comb_fea_label = np.zeros((len(sort_idx)-1,4))
    
    comb_fea_mat = np.zeros((int(len(sort_idx)*(len(sort_idx)-1)/2),feature_size,max_length,3))
    comb_fea_label = np.zeros((int(len(sort_idx)*(len(sort_idx)-1)/2),4))
    
    temp_cost_list = []
    #print(len(comb_track_cost))
    cnt = -1
    for n in range(0, len(sort_idx)-1):
        for kk in range(n+1,len(sort_idx)):
            cnt = int(cnt+1)
            track_id1 = tracklet_set[sort_idx[n]]
            track_id2 = tracklet_set[sort_idx[kk]]
            if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                remove_ids.append(cnt)
                continue
                
            if tracklet_mat['comb_track_cost_mask'][track_id1,track_id2]==1:
                cost = cost+tracklet_mat['comb_track_cost'][track_id1,track_id2]
                remove_ids.append(cnt)
                continue
            
            comb_fea_label[cnt,0] = track_id1 
            comb_fea_label[cnt,1] = track_id2 

        #if track_id1==32 and track_id2==46:
        #    import pdb; pdb.set_trace()
            '''
            start_time = time.time()
            if len(comb_track_cost)>0:
                search_idx = np.where(np.logical_and(comb_track_cost[:,0]==track_id1, comb_track_cost[:,1]==track_id2))
                if len(search_idx[0])>0:
                    remove_ids.append(n)
                    #import pdb; pdb.set_trace()
                    cost = cost+comb_track_cost[search_idx[0][0],2]
                    elapsed_time = time.time() - start_time
                    print(elapsed_time)
                    continue
            '''
            temp_cost_list.append([track_id1,track_id2])
        
        
        # t starts from 0
        #import pdb; pdb.set_trace()
            t1_min = int(track_interval[track_id1,0])
            t1_max = int(track_interval[track_id1,1])
            t2_min = int(track_interval[track_id2,0])
            t2_max = int(track_interval[track_id2,1])
            t_min = int(min(t1_min,t2_min))
            t_max = int(max(t1_max,t2_max))

            if t_max-t_min+1<=max_length:
                comb_fea_mat[cnt,:,t1_min-t_min:t1_max-t_min+1,1] = 1
                comb_fea_mat[cnt,0,t1_min-t_min:t1_max-t_min+1,0] = tracklet_mat['x_3d_mat'][track_id1,t1_min:t1_max+1]/loc_scales[0]
                
                comb_fea_mat[cnt,1,t1_min-t_min:t1_max-t_min+1,0] = tracklet_mat['y_3d_mat'][track_id1,t1_min:t1_max+1]/loc_scales[1]
                
                comb_fea_mat[cnt,2,t1_min-t_min:t1_max-t_min+1,0] = tracklet_mat['w_3d_mat'][track_id1,t1_min:t1_max+1]/loc_scales[2]
                
                comb_fea_mat[cnt,3,t1_min-t_min:t1_max-t_min+1,0] = tracklet_mat['h_3d_mat'][track_id1,t1_min:t1_max+1]/loc_scales[3]
                
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id1+1)[0]

                if comb_fea_mat[cnt,4:,t1_min-t_min:t1_max-t_min+1,0].shape[1]!=np.transpose(tracklet_mat['appearance_fea_mat'] \
                                                                                       [cand_idx,2:]).shape[1]:
                    import pdb; pdb.set_trace()
                comb_fea_mat[cnt,4:,t1_min-t_min:t1_max-t_min+1,0] = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])

                comb_fea_mat[cnt,:,t2_min-t_min:t2_max-t_min+1,2] = 1

                comb_fea_mat[cnt,0,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['x_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[0]
                
                comb_fea_mat[cnt,1,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['y_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[1]
                
                comb_fea_mat[cnt,2,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['w_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[2]
                
                comb_fea_mat[cnt,3,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['h_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[3]
                
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id2+1)[0]
                if comb_fea_mat[cnt,4:,t2_min-t_min:t2_max-t_min+1,0].shape[1]!=np.transpose(tracklet_mat['appearance_fea_mat'] \
                                                                                       [cand_idx,2:]).shape[1]:
                    import pdb; pdb.set_trace()
                
                comb_fea_mat[cnt,4:,t2_min-t_min:t2_max-t_min+1,0] = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])
            else:
                t_len1 = t1_max-t1_min+1
                t_len2 = t2_max-t2_min+1
                t_len_min = min(t_len1,t_len2)
                mid_t = int(0.5*(t1_max+t2_min))
                if mid_t-t1_min+1>=0.5*max_length and t2_max-mid_t+1<=0.5*max_length:
                    t2_end = t2_max
                    t1_start = t2_end-max_length+1
                #t1_start = mid_t-int(0.5*max_length)+1
                #t2_end = t1_start+max_length-1
                elif mid_t-t1_min+1<=0.5*max_length and t2_max-mid_t+1>=0.5*max_length:
                    t1_start = t1_min
                    t2_end = t1_start+max_length-1
                else: # mid_t-t1_min+1>=0.5*max_length and t2_max-mid_t+1>=0.5*max_length:
                    t1_start = mid_t-int(0.5*max_length)+1
                    t2_end = t1_start+max_length-1

                comb_fea_mat[cnt,:,0:t1_max-t1_start+1,1] = 1
                if comb_fea_mat[cnt,0,0:t1_max-t1_start+1,0].shape[0] \
                    !=tracklet_mat['xmax_mat'][track_id1,t1_start:t1_max+1].shape[0]:
                    import pdb; pdb.set_trace()
                comb_fea_mat[cnt,0,0:t1_max-t1_start+1,0] = tracklet_mat['x_3d_mat'][track_id1,t1_start:t1_max+1]/loc_scales[0]

                comb_fea_mat[cnt,1,0:t1_max-t1_start+1,0] = tracklet_mat['y_3d_mat'][track_id1,t1_start:t1_max+1]/loc_scales[1]
                
                comb_fea_mat[cnt,2,0:t1_max-t1_start+1,0] = tracklet_mat['w_3d_mat'][track_id1,t1_start:t1_max+1]/loc_scales[2]
                
                comb_fea_mat[cnt,3,0:t1_max-t1_start+1,0] = tracklet_mat['h_3d_mat'][track_id1,t1_start:t1_max+1]/loc_scales[3]
                
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id1+1)[0]
                cand_idx = cand_idx[t1_start-t1_min:]
                comb_fea_mat[cnt,4:,0:t1_max-t1_start+1,0] = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])

                comb_fea_mat[cnt,:,t2_min-t1_start:t2_end-t1_start+1,2] = 1
                if comb_fea_mat[cnt,0,t2_min-t1_start:t2_end-t1_start+1,0].shape[0] \
                    !=tracklet_mat['xmin_mat'][track_id2,t2_min:t2_end+1].shape[0]:
                    import pdb; pdb.set_trace()
                comb_fea_mat[cnt,0,t2_min-t1_start:t2_end-t1_start+1,0] = \
                    tracklet_mat['x_3d_mat'][track_id2,t2_min:t2_end+1]/loc_scales[0]
                comb_fea_mat[cnt,1,t2_min-t1_start:t2_end-t1_start+1,0] = \
                    tracklet_mat['y_3d_mat'][track_id2,t2_min:t2_end+1]/loc_scales[1]
                comb_fea_mat[cnt,2,t2_min-t1_start:t2_end-t1_start+1,0] = \
                    tracklet_mat['w_3d_mat'][track_id2,t2_min:t2_end+1]/loc_scales[2]
                comb_fea_mat[cnt,3,t2_min-t1_start:t2_end-t1_start+1,0] = \
                    tracklet_mat['h_3d_mat'][track_id2,t2_min:t2_end+1]/loc_scales[3]
                    
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id2+1)[0]
                #import pdb; pdb.set_trace()
                cand_idx = cand_idx[0:t2_end-t2_min+1]
                comb_fea_mat[cnt,4:,t2_min-t1_start:t2_end-t1_start+1,0] \
                    = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])
                
        #if track_id1==34 and track_id2==39:
        #    import pdb; pdb.set_trace()

        # remove overlap detections
            t_overlap = np.where(comb_fea_mat[cnt,0,:,1]+comb_fea_mat[cnt,0,:,2]>1)
            if len(t_overlap)>0:
                t_overlap = t_overlap[0]
                comb_fea_mat[cnt,:,t_overlap,:] = 0

            
            if len(track_set)>0:
                search_idx = np.where(np.logical_and(track_set[:,0]==track_id1, track_set[:,1]==track_id2))
                if len(search_idx[0])>0:
                    #save_fea_mat[search_idx[0][0],:,:,:] = comb_fea_mat[n,:,:,:]
                    if track_set[search_idx[0][0],2]==1:
                        comb_fea_label[cnt,2] = 1
                    else:
                        comb_fea_label[cnt,3] = 1
            
    
    if len(remove_ids)>0:
        comb_fea_mat = np.delete(comb_fea_mat, np.array(remove_ids), axis=0)
        comb_fea_label = np.delete(comb_fea_label, np.array(remove_ids), axis=0)
        
    if len(comb_fea_mat)>0:
        max_batch_size = 16
        num_batch = int(np.ceil(comb_fea_mat.shape[0]/max_batch_size))
        pred_y = np.zeros((comb_fea_mat.shape[0],2))
        for n in range(num_batch):
            if n!=num_batch-1:
                batch_size = max_batch_size
            else:
                batch_size = int(comb_fea_mat.shape[0]-(num_batch-1)*max_batch_size)
                
            batch_size = comb_fea_mat.shape[0]
            x = np.zeros((batch_size,1,max_length,1))
            y = np.zeros((batch_size,1,max_length,1))
            w = np.zeros((batch_size,1,max_length,1))
            h = np.zeros((batch_size,1,max_length,1))
            ap = np.zeros((batch_size,feature_size-4,max_length,1))
            mask_1 = np.zeros((batch_size,1,max_length,2))
            mask_2 = np.zeros((batch_size,feature_size-4,max_length,2))
            x[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,0,:,0]
            y[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,1,:,0]
            w[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,2,:,0]
            h[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,3,:,0]
            ap[:,:,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,4:,:,0]
            mask_1[:,0,:,:] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,0,:,1:]
            mask_2[:,:,:,:] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,4:,:,1:]
            pred_y[n*max_batch_size:n*max_batch_size+batch_size,:] = sess.run(y_conv, feed_dict={batch_X_x: x,
                                     batch_X_y: y,
                                     batch_X_w: w,
                                     batch_X_h: h,
                                     batch_X_a: ap,
                                     batch_mask_1: mask_1,
                                     batch_mask_2: mask_2,
                                     batch_Y: np.zeros((batch_size,2)), 
                                     keep_prob: 1.0})
        
        for n in range(len(pred_y)):
            if np.sum(comb_fea_label[n,2:4])>0:
                continue
            if pred_y[n,0]>pred_y[n,1]:
                comb_fea_label[n,2] = 1
            else:
                comb_fea_label[n,3] = 1

        if comb_fea_mat.shape[0]!=len(pred_y):
            import pdb; pdb.set_trace()
            
        
        all_fea_mat[fea_id:fea_id+len(pred_y),:,:,:] = comb_fea_mat
        all_fea_label[fea_id:fea_id+len(pred_y),:] = comb_fea_label
        
        #if len(np.where(np.logical_and(comb_fea_label[:,0]==428,comb_fea_label[:,1]==435))[0])>0:
        #    import pdb; pdb.set_trace()
        #print(comb_fea_label)
        
        cost = cost+np.sum(pred_y[:,1]-pred_y[:,0])
        #import pdb; pdb.set_trace()
        
        if pred_y.shape[0]!=len(temp_cost_list):
            import pdb; pdb.set_trace()
        for n in range(pred_y.shape[0]):
            #import pdb; pdb.set_trace()
            '''
            if tracklet_mat['comb_track_cost_mask'].shape[0]<=temp_cost_list[n][0] \
                or tracklet_mat['comb_track_cost_mask'].shape[1]<=temp_cost_list[n][1]:
                    import pdb; pdb.set_trace()
            '''
            tracklet_mat['comb_track_cost_mask'][temp_cost_list[n][0],temp_cost_list[n][1]] = 1
            tracklet_mat['comb_track_cost'][temp_cost_list[n][0],temp_cost_list[n][1]] = pred_y[n,1]-pred_y[n,0]
            
        #comb_track_cost_list = comb_track_cost_list+temp_cost_list
        #print(np.sum(tracklet_mat['comb_track_cost_mask']))
    return cost

def get_split_cost(track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                   batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    #comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    global track_struct
    
        
    tracklet_mat = track_struct['tracklet_mat']
    new_cluster_cost = np.zeros((2,1))
    if len(tracklet_mat['track_cluster'][track_id])<2:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    track_interval = tracklet_mat['track_interval'].copy()
    change_cluster_idx = [len(tracklet_mat['track_cluster']), tracklet_mat['track_class'][track_id]]
    new_cluster_set = []
    new_cluster_set.append([track_id])
    remain_tracks = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    remain_tracks.remove(track_id)
    new_cluster_set.append(remain_tracks)

    # get cost
    if len(remain_tracks)>1:
        sort_idx = np.argsort(track_interval[np.array(new_cluster_set[1]),1])
        for n in range(0, len(sort_idx)-1):
            track_id1 = new_cluster_set[1][sort_idx[n]]
            track_id2 = new_cluster_set[1][sort_idx[n+1]]
            #if track_id1==42:
            #    import pdb; pdb.set_trace()
            if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                diff_cost = float("inf")
                new_cluster_cost = []
                new_cluster_set = []
                change_cluster_idx = []
                return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

        #*********************************
        new_cluster_cost[1,0] = comb_cost(remain_tracks, feature_size, 
                                                                         max_length, 
                                                            img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                            batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                            batch_Y, keep_prob, y_conv)
        #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
        
    # cross cost
    comb_cluster = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    sort_idx = np.argsort(track_interval[np.array(comb_cluster),1])
    cross_cost = np.zeros((2,1))
    '''
    for n in range(0, len(sort_idx)-1):
        track_id1 = comb_cluster[sort_idx[n]]
        track_id2 = comb_cluster[sort_idx[n+1]]
        if (track_id1 in new_cluster_set[0] and track_id2 in new_cluster_set[1]) \
            or (track_id1 in new_cluster_set[1] and track_id2 in new_cluster_set[0]):
            if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                continue
            cross_cost[1,0] = cross_cost[1,0]+comb_cost([track_id1,track_id2], feature_size, max_length, 
                                                            img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                            batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                            batch_Y, keep_prob, y_conv)
    '''
    cost = np.sum(new_cluster_cost)-cross_cost[1,0]
    prev_cost = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]]-cross_cost[0,0]
    diff_cost = cost-prev_cost

    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

def get_assign_cost(track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, 
                    batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    #comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    global track_struct
    tracklet_mat = track_struct['tracklet_mat']
    #import pdb; pdb.set_trace()
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    new_cluster_cost = np.zeros((2,1))
    new_cluster_set = []
    new_cluster_set.append(cluster1.copy())
    new_cluster_set[0].remove(track_id)
    track_interval = tracklet_mat['track_interval'].copy()
    # get cost
    if len(new_cluster_set[0])>1:

        sort_idx = np.argsort(track_interval[np.array(new_cluster_set[0]),1])
        for n in range(0, len(sort_idx)-1):
            track_id1 = new_cluster_set[0][sort_idx[n]]
            track_id2 = new_cluster_set[0][sort_idx[n+1]]
            if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                diff_cost = float("inf")
                new_cluster_cost = []
                new_cluster_set = []
                change_cluster_idx = []
                return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

        new_cluster_cost[0,0] = comb_cost(new_cluster_set[0], feature_size, 
                                                                         max_length, 
                                                            img_size, sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                                                            batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, 
                                                            y_conv)
        #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()

    track_class = track_struct['tracklet_mat']['track_class'][track_id]
    t_cluster_idx = track_struct['tracklet_mat']['track_cluster_t_idx'][track_class]
    
    NN_cluster = len(tracklet_mat['track_cluster'])
    temp_new_cluster_cost = float("inf")*np.ones((NN_cluster,1))
    prev_cost_vec = np.zeros((NN_cluster,1))
    cross_cost_vec = np.zeros((NN_cluster,2))

    for nn in range(len(t_cluster_idx)):
        N_cluster = len(track_struct['tracklet_mat']['time_cluster'][t_cluster_idx[nn]])
        for mm in range(N_cluster):
            n = track_struct['tracklet_mat']['time_cluster'][t_cluster_idx[nn]][mm]
            # the original cluster
            if tracklet_mat['track_class'][track_id]==n:
                continue

            # check neighbor and conflict track
            cluster2 = tracklet_mat['track_cluster'][n]
            neighbor_flag = 1
            conflict_flag = 0
            #remove_flag = 0
            temp_cluster_set = cluster2.copy()
            temp_cluster_set.append(track_id)
            sort_idx = np.argsort(track_interval[np.array(temp_cluster_set),1])
            for m in range(0, len(sort_idx)-1):
                track_id1 = temp_cluster_set[sort_idx[m]]
                track_id2 = temp_cluster_set[sort_idx[m+1]]
                #if cluster2[m] in remove_set:
                #    remove_flag = 1
                #    break
                if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                    neighbor_flag = 0
                    break
                if track_id1 in tracklet_mat['conflict_track_idx'][track_id2]:
                    conflict_flag = 1
                    break
            if neighbor_flag==0 or conflict_flag==1:# or remove_flag==1:
                continue

            # get cost
            temp_set = cluster2.copy()
            temp_set.append(track_id)
            temp_new_cluster_cost[n,0] = comb_cost(temp_set, feature_size, 
                                                                              max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
        #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()

        #import pdb; pdb.set_trace()
            prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
                +tracklet_mat['cluster_cost'][n]
            '''    
            # cross cost
            comb_cluster = cluster1+cluster2
            sort_idx = np.argsort(track_interval[np.array(comb_cluster),1])
            for m in range(0, len(sort_idx)-1):
                track_id1 = comb_cluster[sort_idx[m]]
                track_id2 = comb_cluster[sort_idx[m+1]]
                if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                    continue
                if (track_id1 in cluster1 and track_id2 in cluster2) or (track_id1 in cluster2 and track_id2 in cluster1):
                    cross_cost_vec[n,0] = cross_cost_vec[n,0]+comb_cost([track_id1,track_id2], feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
                if (track_id1 in new_cluster_set[0] and track_id2 in temp_set) or \
                    (track_id1 in temp_set and track_id2 in new_cluster_set[0]):
                    cross_cost_vec[n,1] = cross_cost_vec[n,1]+comb_cost([track_id1,track_id2], feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
            '''        
                

    cost_vec = temp_new_cluster_cost[:,0]+new_cluster_cost[0,0]-cross_cost_vec[:,1]
    prev_cost_vec = prev_cost_vec[:,0]-cross_cost_vec[:,0]
    
    diff_cost_vec = cost_vec-prev_cost_vec
    min_idx = np.argmin(diff_cost_vec)
    cost = cost_vec[min_idx]
    if cost==float("inf"):
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    diff_cost = diff_cost_vec[min_idx]
    new_cluster_cost[1,0] = temp_new_cluster_cost[min_idx,0]
    change_cluster_idx = [tracklet_mat['track_class'][track_id],min_idx]
    temp_set = tracklet_mat['track_cluster'][min_idx].copy()
    temp_set.append(track_id)
    new_cluster_set.append(temp_set)

    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

def get_merge_cost(track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                   batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    #comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    global track_struct
    tracklet_mat = track_struct['tracklet_mat']
    track_interval = tracklet_mat['track_interval'].copy()
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    if len(cluster1)==1:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    track_class = track_struct['tracklet_mat']['track_class'][track_id]
    t_cluster_idx = track_struct['tracklet_mat']['track_cluster_t_idx'][track_class]
    
    NN_cluster = len(tracklet_mat['track_cluster'])
    new_cluster_cost_vec = float("inf")*np.ones((NN_cluster,1))
    prev_cost_vec = np.zeros((NN_cluster,1))
    cross_cost_vec = np.zeros((NN_cluster,2))

    for nn in range(len(t_cluster_idx)):
        N_cluster = len(track_struct['tracklet_mat']['time_cluster'][t_cluster_idx[nn]])
        
        for mm in range(N_cluster):
            n = track_struct['tracklet_mat']['time_cluster'][t_cluster_idx[nn]][mm]
        
            # the original cluster
            if tracklet_mat['track_class'][track_id]==n:
                continue

            # check neighbor and conflict track
            cluster2 = tracklet_mat['track_cluster'][n].copy()
            if len(cluster2)<=1:
                continue
        
            neighbor_flag = 1
            conflict_flag = 0
            #remove_flag = 0
            temp_cluster_set = cluster1+cluster2
            sort_idx = np.argsort(track_interval[np.array(temp_cluster_set),1])
            for m in range(0, len(sort_idx)-1):
                track_id1 = temp_cluster_set[sort_idx[m]]
                track_id2 = temp_cluster_set[sort_idx[m+1]]

                if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                    neighbor_flag = 0
                    break
                if track_id1 in tracklet_mat['conflict_track_idx'][track_id2]:
                    conflict_flag = 1
                    break

            if neighbor_flag==0 or conflict_flag==1:# or remove_flag==1:
                continue

            
            # get cost
            new_cluster_cost_vec[n,0] = comb_cost(cluster1+cluster2, feature_size, 
                                                                max_length, img_size, sess, batch_X_x, batch_X_y, 
                                                                batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
                                                                batch_mask_2, batch_Y, keep_prob, y_conv)
            #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
            #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()

            prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
                +tracklet_mat['cluster_cost'][n]
                
            '''
            # cross cost
            comb_cluster = cluster1+cluster2
            sort_idx = np.argsort(track_interval[np.array(comb_cluster),1])
            for m in range(0, len(sort_idx)-1):
                track_id1 = comb_cluster[sort_idx[m]]
                track_id2 = comb_cluster[sort_idx[m+1]]
                if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                    continue
                if (track_id1 in cluster1 and track_id2 in cluster2) or (track_id1 in cluster2 and track_id2 in cluster1):
                    cross_cost_vec[n,0] = cross_cost_vec[n,0]+comb_cost([track_id1,track_id2], feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
            '''    
    prev_cost_vec = prev_cost_vec[:,0]-cross_cost_vec[:,0]
    diff_cost_vec = new_cluster_cost_vec[:,0]-prev_cost_vec
    min_idx = np.argmin(diff_cost_vec)
    cost = new_cluster_cost_vec[min_idx,0]
    if cost==float("inf"):
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    diff_cost = diff_cost_vec[min_idx]
    new_cluster_cost = np.zeros((2,1))
    new_cluster_cost[0,0] = cost
    change_cluster_idx = [tracklet_mat['track_class'][track_id], min_idx]
    new_cluster_set = []
    temp_set = cluster1.copy()
    temp_set = temp_set+tracklet_mat['track_cluster'][min_idx]
    new_cluster_set.append(temp_set)
    new_cluster_set.append([])
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

def get_switch_cost(track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                    batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    #comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    global track_struct
    tracklet_mat = track_struct['tracklet_mat']
    track_interval = tracklet_mat['track_interval'].copy()
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    S1 = []
    S2 = []
    for k in range(len(cluster1)):
        temp_id = cluster1[k]
        if tracklet_mat['track_interval'][temp_id,1]<=tracklet_mat['track_interval'][track_id,1]:
            S1.append(temp_id)
        else:
            S2.append(temp_id)
    if len(S1)==0 or len(S2)==0:
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    track_class = track_struct['tracklet_mat']['track_class'][track_id]
    t_cluster_idx = track_struct['tracklet_mat']['track_cluster_t_idx'][track_class]
    
    NN_cluster = len(tracklet_mat['track_cluster'])
    cost_vec = float("inf")*np.ones((NN_cluster,1))
    prev_cost_vec = np.zeros((NN_cluster,1))
    new_cluster_cost_vec1 = float("inf")*np.ones((NN_cluster,1))
    new_cluster_cost_vec2 = float("inf")*np.ones((NN_cluster,1))
    cross_cost_vec = np.zeros((NN_cluster,2))
    
    track_id_set = []
    for n in range(NN_cluster):
        track_id_set.append([])

    for nn in range(len(t_cluster_idx)):
        N_cluster = len(track_struct['tracklet_mat']['time_cluster'][t_cluster_idx[nn]])  
        
        for mm in range(N_cluster):
            n = track_struct['tracklet_mat']['time_cluster'][t_cluster_idx[nn]][mm]
            
            # the original cluster
            if tracklet_mat['track_class'][track_id]==n:
                continue

            # switch availability check
            S3 = []
            S4 = []
            #remove_flag = 0
            cluster2 = tracklet_mat['track_cluster'][n].copy()
            for k in range(len(cluster2)):
                temp_id = cluster2[k]
                #if temp_id in remove_set:
                #    remove_flag = 1
                #    break
                if tracklet_mat['track_interval'][temp_id,1]<=tracklet_mat['track_interval'][track_id,1]:
                    S3.append(temp_id)
                else:
                    #********************************************
                    if tracklet_mat['track_interval'][temp_id,1] >=tracklet_mat['track_interval'][track_id,1] \
                        and tracklet_mat['track_interval'][temp_id,0] <=tracklet_mat['track_interval'][track_id,1]:
                        if tracklet_mat['track_interval'][temp_id,1] -tracklet_mat['track_interval'][track_id,1] \
                            >tracklet_mat['track_interval'][track_id,1]-tracklet_mat['track_interval'][temp_id,0]:
                            S4.append(temp_id)
                        else: 
                            S3.append(temp_id)
                    else:
                        S4.append(temp_id)
        
            #if remove_flag==1:
            #    continue
            
            neighbor_flag1 = 1
            conflict_flag1 = 0
            if len(S3)==0:
                neighbor_flag1 = 1
                conflict_flag1 = 0
            else:
                temp_cluster_set = S3+S2
                sort_idx = np.argsort(track_interval[np.array(temp_cluster_set),1])
                for k in range(0,len(sort_idx)-1):
                    track_id1 = temp_cluster_set[sort_idx[k]]
                    track_id2 = temp_cluster_set[sort_idx[k+1]]
                    if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                        neighbor_flag1 = 0
                        break
                    if track_id1 in tracklet_mat['conflict_track_idx'][track_id2]:
                        conflict_flag1 = 1
                        break
                    

            neighbor_flag2 = 1
            conflict_flag2 = 0
            if len(S4)==0:
                neighbor_flag2 = 1
                conflict_flag2 = 0
            else:
                temp_cluster_set = S4+S1
                sort_idx = np.argsort(track_interval[np.array(temp_cluster_set),1])
                for k in range(0,len(sort_idx)-1):
                    track_id1 = temp_cluster_set[sort_idx[k]]
                    track_id2 = temp_cluster_set[sort_idx[k+1]]
                    if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                        neighbor_flag2 = 0
                        break
                    if track_id1 in tracklet_mat['conflict_track_idx'][track_id2]:
                        conflict_flag2 = 1
                        break

            if neighbor_flag1==0 or conflict_flag1==1 or neighbor_flag2==0 or conflict_flag2==1:
                continue

            
                
            # get cost
            S_1 = S1+S4
            S_2 = S2+S3
            
            #if (428 in S_1 and 435 in S_1) or (428 in S_2 and 435 in S_2):
            #    import pdb; pdb.set_trace()
                
            new_cluster_cost_vec1[n,0] = comb_cost(S_1, feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
            #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
            #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
            new_cluster_cost_vec2[n,0] = comb_cost(S_2, feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
            #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
            #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
            cost_vec[n,0] = new_cluster_cost_vec1[n,0]+new_cluster_cost_vec2[n,0]

            track_id_set[n].append(S_1.copy())
            track_id_set[n].append(S_2.copy())
            prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
                        +tracklet_mat['cluster_cost'][n]
                
            '''
            # cross cost
            comb_cluster = S_1+S_2
            sort_idx = np.argsort(track_interval[np.array(comb_cluster),1])
            for m in range(0, len(sort_idx)-1):
                track_id1 = comb_cluster[sort_idx[m]]
                track_id2 = comb_cluster[sort_idx[m+1]]
                if track_id1 not in tracklet_mat['neighbor_track_idx'][track_id2]:
                    continue
                if (track_id1 in cluster1 and track_id2 in cluster2) or (track_id1 in cluster2 and track_id2 in cluster1):
                    cross_cost_vec[n,0] = cross_cost_vec[n,0]+comb_cost([track_id1,track_id2], feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
                if (track_id1 in S_1 and track_id2 in S_2) or (track_id1 in S_2 and track_id2 in S_1):
                    cross_cost_vec[n,1] = cross_cost_vec[n,1]+comb_cost([track_id1,track_id2], feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
            '''    

    cost_vec = cost_vec[:,0]-cross_cost_vec[:,1]
    prev_cost_vec = prev_cost_vec[:,0]-cross_cost_vec[:,0]
    diff_cost_vec = cost_vec-prev_cost_vec
    min_idx = np.argmin(diff_cost_vec)
    cost = cost_vec[min_idx]
    if cost==float("inf"):
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    diff_cost = diff_cost_vec[min_idx]
    new_cluster_cost = np.zeros((2,1))
    new_cluster_cost[0,0] = new_cluster_cost_vec1[min_idx,0]
    new_cluster_cost[1,0] = new_cluster_cost_vec2[min_idx,0]

    change_cluster_idx = [tracklet_mat['track_class'][track_id], min_idx]
    new_cluster_set = []
    new_cluster_set.append(track_id_set[min_idx][0])
    new_cluster_set.append(track_id_set[min_idx][1])
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

def get_break_cost(track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                   batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    #comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()
    '''
    cost = float("inf")
    diff_cost = float("inf")
    new_cluster_cost = []
    new_cluster_set = []
    change_cluster_idx = []
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx
    '''
    
    global track_struct
    tracklet_mat = track_struct['tracklet_mat']
    
    new_cluster_cost = np.zeros((2,1))
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    if len(cluster1)<=2:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    # get cost
    after_ids = []
    for n in range(len(cluster1)):
        if tracklet_mat['track_interval'][cluster1[n],1]>tracklet_mat['track_interval'][track_id,1]:
            after_ids.append(cluster1[n])

    if len(after_ids)==0:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    before_ids = list(set(cluster1)-set(after_ids))
    if len(before_ids)<=1:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    change_cluster_idx = [len(tracklet_mat['track_cluster']), tracklet_mat['track_class'][track_id]]
    new_cluster_set = []
    new_cluster_set.append(before_ids)
    remain_tracks = after_ids
    new_cluster_set.append(remain_tracks)
    new_cluster_cost[0,0] = comb_cost(new_cluster_set[0], feature_size, 
                                                                     max_length, 
                                                        img_size, sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                                                        batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, 
                                                        y_conv)
    #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
    #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
    new_cluster_cost[1,0] = comb_cost(new_cluster_set[1], feature_size, 
                                                                     max_length, 
                                                        img_size, sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                                                        batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, 
                                                        y_conv)
    #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
    #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
    cost = np.sum(new_cluster_cost)
    diff_cost = cost-tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]]
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

def update_tracklet_mat(tracklet_mat):  
    final_tracklet_mat = tracklet_mat.copy() 
    track_interval = tracklet_mat['track_interval'] 
    num_cluster = len(tracklet_mat['track_cluster'])
    final_tracklet_mat['track_id_mat'] = -1*np.ones((num_cluster,tracklet_mat['xmin_mat'].shape[1]))
    
    final_xmin_mat = -1*np.ones((num_cluster, final_tracklet_mat['xmin_mat'].shape[1]))
    final_ymin_mat = -1*np.ones((num_cluster, final_tracklet_mat['ymin_mat'].shape[1]))
    final_xmax_mat = -1*np.ones((num_cluster, final_tracklet_mat['xmax_mat'].shape[1]))
    final_ymax_mat = -1*np.ones((num_cluster, final_tracklet_mat['ymax_mat'].shape[1]))
    final_det_score_mat = -1*np.ones((num_cluster, final_tracklet_mat['det_score_mat'].shape[1]))
    final_tracklet_mat['xmin_mat'] = final_xmin_mat.copy()
    final_tracklet_mat['ymin_mat'] = final_ymin_mat.copy()
    final_tracklet_mat['xmax_mat'] = final_xmax_mat.copy()
    final_tracklet_mat['ymax_mat'] = final_ymax_mat.copy()
    final_tracklet_mat['det_score_mat'] = final_det_score_mat.copy()

    for n in range(num_cluster):
        for k in range(len(final_tracklet_mat['track_cluster'][n])):
            temp_id = final_tracklet_mat['track_cluster'][n][k]
            #import pdb; pdb.set_trace()
            final_tracklet_mat['track_id_mat'][n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] = temp_id
            final_xmin_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = tracklet_mat['xmin_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            final_ymin_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = tracklet_mat['ymin_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            final_xmax_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = tracklet_mat['xmax_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            final_ymax_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = tracklet_mat['ymax_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            final_det_score_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = tracklet_mat['det_score_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]

    det_xmin_mat = final_xmin_mat.copy()
    det_ymin_mat = final_ymin_mat.copy()
    det_xmax_mat = final_xmax_mat.copy()
    det_ymax_mat = final_ymax_mat.copy()
    det_det_score_mat = final_det_score_mat.copy()
    
    window_size = 2
    for n in range(num_cluster):
        det_idx = np.where(final_xmin_mat[n,:]!=-1)[0]
        t_min = np.min(det_idx)
        t_max = np.max(det_idx)
        miss_idx = np.where(final_xmin_mat[n,t_min:t_max+1]==-1)[0]
        if len(miss_idx)==0:
            continue
        miss_idx = miss_idx+t_min
        final_xmin_mat[n,miss_idx] = np.interp(miss_idx, det_idx, final_xmin_mat[n,det_idx])
        
        final_ymin_mat[n,miss_idx] = np.interp(miss_idx, det_idx, final_ymin_mat[n,det_idx])
        
        final_xmax_mat[n,miss_idx] = np.interp(miss_idx, det_idx, final_xmax_mat[n,det_idx])
        
        final_ymax_mat[n,miss_idx] = np.interp(miss_idx, det_idx, final_ymax_mat[n,det_idx])
        
        final_det_score_mat[n,miss_idx] = np.interp(miss_idx, det_idx, final_det_score_mat[n,det_idx])
    
    
    '''
    # merge two trajectories if they overlap 
    bbox_overlap_thresh = 0.7
    time_overlap_tresh = 5
    det_overlap_thresh = 0.1
    bbox_overlap_mat = np.zeros((num_cluster,num_cluster))
    for n in range(num_cluster-1):
        for m in range(n+1,num_cluster):
            cand_t = np.where(np.logical_and(final_xmin_mat[n,:]!=-1, final_xmin_mat[m,:]!=-1))[0]
            if len(cand_t)<time_overlap_tresh:
                continue
                
            avg_IOU = np.zeros((len(cand_t),1))
            for k in range(len(cand_t)):
                bbox1 = [final_xmin_mat[n,cand_t[k]],final_ymin_mat[n,cand_t[k]],
                         final_xmax_mat[n,cand_t[k]]-final_xmin_mat[n,cand_t[k]]+1,
                        final_ymax_mat[n,cand_t[k]]-final_ymin_mat[n,cand_t[k]]+1]
                bbox2 = [final_xmin_mat[m,cand_t[k]],final_ymin_mat[m,cand_t[k]],
                         final_xmax_mat[m,cand_t[k]]-final_xmin_mat[m,cand_t[k]]+1,
                        final_ymax_mat[m,cand_t[k]]-final_ymin_mat[m,cand_t[k]]+1]
                avg_IOU[k,0] = get_IOU(bbox1,bbox2)
            bbox_overlap_mat[n,m] = np.mean(avg_IOU)
            if bbox_overlap_mat[n,m]<bbox_overlap_thresh:
                continue
            
            t1_min = np.min(np.where(final_xmin_mat[n,:]!=-1)[0])
            t1_max = np.max(np.where(final_xmin_mat[n,:]!=-1)[0])
            t2_min = np.min(np.where(final_xmin_mat[m,:]!=-1)[0])
            t2_max = np.max(np.where(final_xmin_mat[m,:]!=-1)[0])
            union_len = max(t2_max,t1_max)-min(t1_min,t2_min)+1
            cand_t2 = np.where(np.logical_and(det_xmin_mat[n,:]!=-1, det_xmin_mat[m,:]!=-1))[0]
            if len(cand_t2)/union_len>det_overlap_thresh:
                continue
            
            final_tracklet_mat['track_id_mat'][n,int(t2_min):int(t2_max)+1] = \
                final_tracklet_mat['track_id_mat'][m,int(t2_min):int(t2_max)+1]
            final_xmin_mat[n,int(t2_min):int(t2_max)+1] = final_xmin_mat[m,int(t2_min):int(t2_max)+1]
            final_ymin_mat[n,int(t2_min):int(t2_max)+1] = final_ymin_mat[m,int(t2_min):int(t2_max)+1]
            final_xmax_mat[n,int(t2_min):int(t2_max)+1] = final_xmax_mat[m,int(t2_min):int(t2_max)+1]
            final_ymax_mat[n,int(t2_min):int(t2_max)+1] = final_ymax_mat[m,int(t2_min):int(t2_max)+1]
            final_det_score_mat[n,int(t2_min):int(t2_max)+1] = final_det_score_mat[m,int(t2_min):int(t2_max)+1]
            
            final_tracklet_mat['track_id_mat'][m,int(t2_min):int(t2_max)+1] = -1
            final_xmin_mat[m,:] = -1
            final_ymin_mat[m,:] = -1
            final_xmax_mat[m,:] = -1
            final_ymax_mat[m,:] = -1
            final_det_score_mat[m,:] = -1
            
    '''
    final_tracklet_mat['xmin_mat'] = final_xmin_mat
    final_tracklet_mat['ymin_mat'] = final_ymin_mat
    final_tracklet_mat['xmax_mat'] = final_xmax_mat
    final_tracklet_mat['ymax_mat'] = final_ymax_mat
    final_tracklet_mat['det_score_mat'] = final_det_score_mat
       
       
    
    # moving average
    for n in range(num_cluster):  
        cand_t = np.where(final_xmin_mat[n,:]!=-1)[0]
        if len(cand_t)==0:
            continue
        t1 = int(np.min(cand_t))
        t2 = int(np.max(cand_t))
        for k in range(t1,t2+1):
            t_start = max(k-window_size,t1)
            t_end = min(k+window_size,t2)
            final_tracklet_mat['xmin_mat'][n,k] = np.sum(final_xmin_mat[n,t_start:t_end+1])/(t_end-t_start+1)
            final_tracklet_mat['ymin_mat'][n,k] = np.sum(final_ymin_mat[n,t_start:t_end+1])/(t_end-t_start+1)
            final_tracklet_mat['xmax_mat'][n,k] = np.sum(final_xmax_mat[n,t_start:t_end+1])/(t_end-t_start+1)
            final_tracklet_mat['ymax_mat'][n,k] = np.sum(final_ymax_mat[n,t_start:t_end+1])/(t_end-t_start+1)
            final_tracklet_mat['det_score_mat'][n,k] = np.sum(final_det_score_mat[n,t_start:t_end+1])/(t_end-t_start+1)
    
            
    return final_tracklet_mat

def post_processing(): 
    
    global track_struct
    tracklet_mat = track_struct['tracklet_mat']
    track_params = track_struct['track_params']
    new_tracklet_mat = tracklet_mat.copy()
    #import pdb; pdb.set_trace()
    
    # update track cluster
    N_cluster = len(tracklet_mat["track_cluster"])
    remove_idx = []
    for n in range(N_cluster):
        if len(tracklet_mat["track_cluster"][n])==0:
            remove_idx.append(n)
            continue
        if tracklet_mat["track_cluster"][n][0] in remove_set:
            remove_idx.append(n)
            continue
            
        temp_track_intervals = tracklet_mat["track_interval"][np.array(tracklet_mat["track_cluster"][n]),:]
        start_fr = np.min(temp_track_intervals[:,0])
        end_fr = np.max(temp_track_intervals[:,1])
        num_frs = end_fr-start_fr+1
        if num_frs<track_params['const_fr_thresh']:
            remove_idx.append(n)

    new_tracklet_mat['track_cluster'] = list(np.delete(new_tracklet_mat['track_cluster'], remove_idx))
    new_tracklet_mat['cluster_cost'] = list(np.delete(new_tracklet_mat['cluster_cost'], remove_idx))

    # update track class
    N_tracklet = new_tracklet_mat['xmin_mat'].shape[0]
    new_tracklet_mat['track_class'] = -1*np.ones((N_tracklet,1),dtype=int)
    for n in range(len(new_tracklet_mat['track_cluster'])):
        for k in range(len(new_tracklet_mat['track_cluster'][n])):
            track_id = new_tracklet_mat['track_cluster'][n][k]
            new_tracklet_mat['track_class'][track_id,0] = n

    #import pdb; pdb.set_trace()
    # assign tracklet
    track_struct['final_tracklet_mat'] = update_tracklet_mat(new_tracklet_mat.copy())
    #import pdb; pdb.set_trace()
    return

def tracklet_clustering(sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, 
                        batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    
    global track_struct
    if 'track_interval' not in track_struct['tracklet_mat']: 
        init_clustering() 
        
    #import pdb; pdb.set_trace()    
    track_interval = track_struct['tracklet_mat']['track_interval'] 
    N_tracklet = track_interval.shape[0] 
    change_flag = 0 
    img_size = track_struct['track_params']['img_size']
    
    # sort tracklet in ascending order
    sort_idx = np.argsort(track_interval[:,1])
    for n in range(N_tracklet):
        print(n)
        track_id = sort_idx[n]
        track_class = track_struct['tracklet_mat']['track_class'][track_id]
        t_cluster_idx = track_struct['tracklet_mat']['track_cluster_t_idx'][track_class]
        
        #if n>600:
        #    import pdb; pdb.set_trace()
            
        # remove_set
        if t_cluster_idx[0]==-1:
            continue
            
        #if track_struct['tracklet_mat']['track_class'][track_id]<0:
        #    continue

        #if track_id in remove_set:
        #    continue
            
        diff_cost = np.zeros((5,1))
        new_C = [] # new cost
        new_set = []
        change_idx = []

        #cluster_cost = track_struct['tracklet_mat']['cluster_cost']
        #track_class = track_struct['tracklet_mat']['track_class']

        # get split cost
        #import pdb; pdb.set_trace()
        diff_cost[0,0],temp_new_C,temp_new_set,temp_change_idx \
            = get_split_cost(track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        #track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get assign cost
        #import pdb; pdb.set_trace()
        diff_cost[1,0],temp_new_C,temp_new_set,temp_change_idx \
            = get_assign_cost(track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        #track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get merge cost
        diff_cost[2,0],temp_new_C,temp_new_set,temp_change_idx \
            = get_merge_cost(track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        #track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get switch cost
        diff_cost[3,0],temp_new_C,temp_new_set,temp_change_idx \
            = get_switch_cost(track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        #track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get break cost
        diff_cost[4,0],temp_new_C,temp_new_set,temp_change_idx \
            = get_break_cost(track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        #track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # update cluster
        min_idx = np.argmin(diff_cost[:,0])
        min_cost = diff_cost[min_idx,0]
        if min_cost>=0:
            continue

        change_flag = 1
        #if track_id==251:
        #    import pdb; pdb.set_trace()
            
        #****************
        #import pdb; pdb.set_trace()
        print(min_idx)
        print(new_set)
        new_t_idx = []
        if len(new_set[min_idx][0])==0:
            new_t_idx.append([-1])
        else:
            t_min_array = np.zeros((len(new_set[min_idx][0]),1))
            t_max_array = np.zeros((len(new_set[min_idx][0]),1))
            for m in range(len(new_set[min_idx][0])):
                t_min_array[m,0] = track_struct['tracklet_mat']['track_interval'][new_set[min_idx][0][m],0]
                t_max_array[m,0] = track_struct['tracklet_mat']['track_interval'][new_set[min_idx][0][m],1]
                                   
            min_time_cluster_idx = int(np.floor(max(np.min(t_min_array)
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
            max_time_cluster_idx = int(np.floor(min(np.max(t_max_array)
                                            +track_struct['track_params']['t_dist_thresh']+5,
                                                    track_struct['tracklet_mat']['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
            new_t_idx.append(list(range(min_time_cluster_idx,max_time_cluster_idx+1)))
                           
        if len(new_set[min_idx][1])==0:
            new_t_idx.append([-1])
        else:
            t_min_array = np.zeros((len(new_set[min_idx][1]),1))
            t_max_array = np.zeros((len(new_set[min_idx][1]),1))
            for m in range(len(new_set[min_idx][1])):
                t_min_array[m,0] = track_struct['tracklet_mat']['track_interval'][new_set[min_idx][1][m],0]
                t_max_array[m,0] = track_struct['tracklet_mat']['track_interval'][new_set[min_idx][1][m],1]
                                   
            min_time_cluster_idx = int(np.floor(max(np.min(t_min_array)
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
            max_time_cluster_idx = int(np.floor(min(np.max(t_max_array)
                                            +track_struct['track_params']['t_dist_thresh']+5,
                                                    track_struct['tracklet_mat']['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
            new_t_idx.append(list(range(min_time_cluster_idx,max_time_cluster_idx+1)))
                                   
        if change_idx[min_idx][0]>=len(track_struct['tracklet_mat']['track_cluster']):
            for m in range(len(track_struct['tracklet_mat']['track_cluster']),change_idx[min_idx][0]):
                track_struct['tracklet_mat']['track_cluster'].append([])
                track_struct['tracklet_mat']['track_cluster_t_idx'].append([-1])
            track_struct['tracklet_mat']['track_cluster'].append(new_set[min_idx][0])   
            track_struct['tracklet_mat']['track_cluster_t_idx'].append(new_t_idx[0])
        else:
            track_struct['tracklet_mat']['track_cluster'][change_idx[min_idx][0]] = new_set[min_idx][0]
            track_struct['tracklet_mat']['track_cluster_t_idx'][change_idx[min_idx][0]] = new_t_idx[0]

        if change_idx[min_idx][1]>=len(track_struct['tracklet_mat']['track_cluster']):
            for m in range(len(track_struct['tracklet_mat']['track_cluster']),change_idx[min_idx][1]):
                track_struct['tracklet_mat']['track_cluster'].append([])
                track_struct['tracklet_mat']['track_cluster_t_idx'].append([-1])
            track_struct['tracklet_mat']['track_cluster'].append(new_set[min_idx][1])  
            track_struct['tracklet_mat']['track_cluster_t_idx'].append(new_t_idx[1])
        else:
            track_struct['tracklet_mat']['track_cluster'][change_idx[min_idx][1]] = new_set[min_idx][1]
            track_struct['tracklet_mat']['track_cluster_t_idx'][change_idx[min_idx][1]] = new_t_idx[1]
        
        for m in range(track_struct['track_params']['num_time_cluster']):
            #import pdb; pdb.set_trace()
            if change_idx[min_idx][0] in track_struct['tracklet_mat']['time_cluster'][m]:
                track_struct['tracklet_mat']['time_cluster'][m].remove(change_idx[min_idx][0])                   
            if change_idx[min_idx][1] in track_struct['tracklet_mat']['time_cluster'][m]:
                track_struct['tracklet_mat']['time_cluster'][m].remove(change_idx[min_idx][1])
                                   
        for m in range(track_struct['track_params']['num_time_cluster']):
            if m in new_t_idx[0]:
                track_struct['tracklet_mat']['time_cluster'][m].append(change_idx[min_idx][0])                   
            if m in new_t_idx[1]:
                track_struct['tracklet_mat']['time_cluster'][m].append(change_idx[min_idx][1])
                                   
        if change_idx[min_idx][0]>=len(track_struct['tracklet_mat']['cluster_cost']):
            for m in range(len(track_struct['tracklet_mat']['cluster_cost']),change_idx[min_idx][0]):
                track_struct['tracklet_mat']['cluster_cost'].append(0)
            track_struct['tracklet_mat']['cluster_cost'].append(new_C[min_idx][0])
        else:
            track_struct['tracklet_mat']['cluster_cost'][change_idx[min_idx][0]] = new_C[min_idx][0]

        if change_idx[min_idx][1]>=len(track_struct['tracklet_mat']['cluster_cost']):
            for m in range(len(track_struct['tracklet_mat']['cluster_cost']),change_idx[min_idx][1]):
                track_struct['tracklet_mat']['cluster_cost'].append([])
            track_struct['tracklet_mat']['cluster_cost'].append(new_C[min_idx][1])  
        else:
            track_struct['tracklet_mat']['cluster_cost'][change_idx[min_idx][1]] = new_C[min_idx][1]

        for k in range(len(track_struct['tracklet_mat']['track_cluster'][change_idx[min_idx][0]])):
            track_struct['tracklet_mat']['track_class'][track_struct['tracklet_mat'] \
                                                    ['track_cluster'][change_idx[min_idx][0]][k]] = change_idx[min_idx][0]

        for k in range(len(track_struct['tracklet_mat']['track_cluster'][change_idx[min_idx][1]])):
            track_struct['tracklet_mat']['track_class'][track_struct['tracklet_mat'] \
                                                    ['track_cluster'][change_idx[min_idx][1]][k]] = change_idx[min_idx][1]
        #import pdb; pdb.set_trace()
    return change_flag

def crop_det(tracklet_mat, crop_size, img_folder, crop_det_folder, flag): 
    
    if not os.path.isdir(crop_det_folder): 
        os.makedirs(crop_det_folder) 
        
    N_tracklet = tracklet_mat['xmin_mat'].shape[0] 
    T = tracklet_mat['xmin_mat'].shape[1] 
    img_list = os.listdir(img_folder) 
    cnt = 0 
    #import pdb; pdb.set_trace()
    for n in range(T): 
        fr_id = n
        track_ids = np.where(tracklet_mat['xmax_mat'][:,n]!=-1) 
        if len(track_ids)==0: 
            continue 
        track_ids = track_ids[0] 
        img_name = file_name(fr_id,10)+'.png'
        #import pdb; pdb.set_trace()
        if img_name in img_list:
            img_path = img_folder+'/'+img_name 
            img = misc.imread(img_path) 
            img_size = img.shape
        else:
            continue

        for m in range(len(track_ids)): 
            if flag==0: 
                xmin = int(max(0,tracklet_mat['xmin_mat'][track_ids[m],n])) 
                xmax = int(min(img.shape[1]-1,tracklet_mat['xmax_mat'][track_ids[m],n])) 
                ymin = int(max(0,tracklet_mat['ymin_mat'][track_ids[m],n])) 
                ymax = int(min(img.shape[0]-1,tracklet_mat['ymax_mat'][track_ids[m],n])) 
                img_patch = img[ymin:ymax,xmin:xmax,:] 
                img_patch = misc.imresize(img_patch, size=[crop_size,crop_size]) 
                class_name = file_name(track_ids[m]+1,4) 
                patch_name = class_name+'_'+file_name(fr_id,4)+'.png' 
                save_path = crop_det_folder+'/'+class_name 
                if not os.path.isdir(save_path): 
                    os.makedirs(save_path) 
                save_path = save_path+'/'+patch_name

                #import pdb; pdb.set_trace()
                misc.imsave(save_path, img_patch)
            cnt = cnt+1
    return cnt, img_size

def feature_encode(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, 
                   batch_size_placeholder, control_placeholder, embeddings, labels, image_paths, 
                   batch_size, distance_metric):

    # Run forward pass to calculate embeddings
    #print('Runnning forward pass on LFW images')

    use_flipped_images = False
    use_fixed_image_standardization = False
    use_random_rotate = False
    use_radnom_crop = False
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(image_paths)  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)

    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    if use_random_rotate:
        control_array += facenet.RANDOM_ROTATE
    if use_radnom_crop:
        control_array += facenet.RANDOM_CROP

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, 
                      labels_placeholder: labels_array, control_placeholder: control_array})

    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    #import pdb; pdb.set_trace()
    #np.savetxt("emb_array.csv", emb_array, delimiter=",")
    return emb_array

def feature_extract(feature_size, num_patch, max_length, patch_folder, triplet_model): 
    f_image_size = 160 
    distance_metric = 0 
    with tf.Graph().as_default():

        with tf.Session() as sess:

            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            nrof_preprocess_threads = 4
            image_size = (f_image_size, f_image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, 
                                                         labels_placeholder, control_placeholder], 
                                                        name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, 
                                                                 nrof_preprocess_threads, batch_size_placeholder)

            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(triplet_model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            fea_mat = np.zeros((num_patch,feature_size-4+2))
            tracklet_list = os.listdir(patch_folder)
            N_tracklet = len(tracklet_list)
            cnt = 0
            for n in range(N_tracklet):
                tracklet_folder = patch_folder+'/'+tracklet_list[n]
                patch_list = os.listdir(tracklet_folder)

                # get patch list, track_id and fr_id, starts from 1
                prev_cnt = cnt
                for m in range(len(patch_list)):
                    # track_id
                    fea_mat[cnt,0] = n+1
                    # fr_id
                    fea_mat[cnt,1] = int(patch_list[m][-8:-4])
                    cnt = cnt+1
                    patch_list[m] = tracklet_folder+'/'+patch_list[m]


                #print(n)
                lfw_batch_size = len(patch_list)     
                emb_array = feature_encode(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, 
                                    phase_train_placeholder,batch_size_placeholder, control_placeholder, 
                                    embeddings, label_batch, patch_list, lfw_batch_size, distance_metric)
                fea_mat[prev_cnt:prev_cnt+lfw_batch_size,2:] = np.copy(emb_array)
    return fea_mat

def color_table(num): 
    digit = '0123456789ABCDEF' 
    table = [] 
    for n in range(num): 
        select_idx = np.random.randint(16, size=6) 
        for k in range(6): 
            if k==0: 
                temp_color = digit[select_idx[k]] 
            else: 
                temp_color = temp_color+digit[select_idx[k]] 
        table.append(temp_color) 
    return table

def draw_result(img_folder, save_folder): 
    #track_struct = pickle.load(open(track_struct_path,'rb'))
    
    global track_struct
    tracklet_mat = track_struct['final_tracklet_mat']
    img_list = os.listdir(img_folder) 
    table = color_table(len(tracklet_mat['track_cluster'])) 
    #import pdb; pdb.set_trace()
    for n in range(track_struct['final_tracklet_mat']['xmin_mat'].shape[1]): 
        fr_id = n
        img_name = file_name(fr_id,10)+'.png'
        if img_name not in img_list:
            continue
        img_path = img_folder+'/'+img_name
        img = misc.imread(img_path)

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(img)

        # Create Rectangle patches
        
        
        for k in range(tracklet_mat['xmin_mat'].shape[0]):
            #
            track_id = int(tracklet_mat['track_id_mat'][k,n])
            
            '''
            if track_id==-1:
                track_class = -1
            else:
                track_class = int(tracklet_mat['track_class'][track_id,0])
            '''
            
            if tracklet_mat['xmin_mat'][k,n]!=-1:
                xmin = tracklet_mat['xmin_mat'][k,n]
                ymin = tracklet_mat['ymin_mat'][k,n]
                xmax = tracklet_mat['xmax_mat'][k,n]
                ymax = tracklet_mat['ymax_mat'][k,n]
                w = xmax-xmin
                h = ymax-ymin
                rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='#'+table[k], facecolor='none')
                img_text = plt.text(xmin,ymin,str(k)+'_'+str(track_id),fontsize=6,color='#'+table[k])
                # Add the patch to the Axes
                ax.add_patch(rect)

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder+'/'+img_name
        plt.savefig(save_path,bbox_inches='tight',dpi=400)
        
        plt.clf()
        plt.close('all')
        #plt.show()
        #import pdb; pdb.set_trace()
    return
        
def convert_frames_to_video(pathIn,pathOut,fps): 
    frame_array = [] 
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]

    #for sorting the file names properly
    #files.sort(key = lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        
        if i==0:
            size = (width,height)
        img = cv2.resize(img,size)
        #print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
        
def wrt_txt(tracklet_mat):
    num_det = np.sum(tracklet_mat['xmin_mat']!=-1)
    f = np.zeros((num_det, 9), dtype=int)
    cnt = 0
    for n in range(tracklet_mat['xmin_mat'].shape[1]):
        for m in range(tracklet_mat['xmin_mat'].shape[0]):
            if tracklet_mat['xmin_mat'][m,n]==-1:
                continue
            f[cnt,0] = n+1
            f[cnt,1] = m+1
            f[cnt,2] = tracklet_mat['xmin_mat'][m,n]
            f[cnt,3] = tracklet_mat['ymin_mat'][m,n]
            f[cnt,4] = tracklet_mat['xmax_mat'][m,n]-tracklet_mat['xmin_mat'][m,n]+1
            f[cnt,5] = tracklet_mat['ymax_mat'][m,n]-tracklet_mat['ymin_mat'][m,n]+1
            f[cnt,6] = -1
            f[cnt,7] = -1
            f[cnt,8] = -1
            cnt = cnt+1
    np.savetxt(txt_result_path, f, delimiter=',',fmt='%d')

def time_cluster_check():
    
    global track_struct
    tracklet_mat = track_struct['tracklet_mat']
    N_cluster = len(tracklet_mat['track_cluster'])
    err_flag = 0
    #import pdb; pdb.set_trace()
    for n in range(N_cluster):
        if len(tracklet_mat['track_cluster'][n])==0:
            if tracklet_mat['track_cluster_t_idx'][n][0]!=-1:
                err_flag = 1
                import pdb; pdb.set_trace()
                return err_flag
        elif tracklet_mat['track_cluster'][n][0] in remove_set:
            if tracklet_mat['track_cluster_t_idx'][n][0]!=-1:
                err_flag = 1
                import pdb; pdb.set_trace()
                return err_flag
        else:
            t_min_array = np.zeros((len(tracklet_mat['track_cluster'][n]),1))
            t_max_array = np.zeros((len(tracklet_mat['track_cluster'][n]),1))
            for m in range(len(tracklet_mat['track_cluster'][n])):
                track_id = tracklet_mat['track_cluster'][n][m]
                t_min_array[m,0] = tracklet_mat['track_interval'][track_id,0]
                t_max_array[m,0] = tracklet_mat['track_interval'][track_id,1]
            min_time_cluster_idx = int(np.floor(max(np.min(t_min_array)
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
            max_time_cluster_idx = int(np.floor(min(np.max(t_max_array)
                                            +track_struct['track_params']['t_dist_thresh']+5,tracklet_mat['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
            temp_t_idx = list(range(min_time_cluster_idx,max_time_cluster_idx+1))
            for m in range(len(temp_t_idx)):
                if n not in tracklet_mat['time_cluster'][temp_t_idx[m]]:
                    err_flag = 1
                    import pdb; pdb.set_trace()
                    return err_flag
            
    for n in range(len(tracklet_mat['time_cluster'])):
        for m in range(len(tracklet_mat['time_cluster'][n])):
            cluster_id = tracklet_mat['time_cluster'][n][m]
            
            if len(tracklet_mat['track_cluster'][cluster_id])==0:
                err_flag = 1
                import pdb; pdb.set_trace()
                return err_flag
            elif tracklet_mat['track_cluster'][cluster_id][0] in remove_set:
                err_flag = 1
                import pdb; pdb.set_trace()
                return err_flag
            else:
                t_min_array = np.zeros((len(tracklet_mat['track_cluster'][cluster_id]),1))
                t_max_array = np.zeros((len(tracklet_mat['track_cluster'][cluster_id]),1))
                for k in range(len(tracklet_mat['track_cluster'][cluster_id])):
                    track_id = tracklet_mat['track_cluster'][cluster_id][k]
                    t_min_array[k,0] = tracklet_mat['track_interval'][track_id,0]
                    t_max_array[k,0] = tracklet_mat['track_interval'][track_id,1]
                min_time_cluster_idx = int(np.floor(max(np.min(t_min_array)
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
                max_time_cluster_idx = int(np.floor(min(np.max(t_max_array)
                                            +track_struct['track_params']['t_dist_thresh']+5,tracklet_mat['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
                temp_t_idx = list(range(min_time_cluster_idx,max_time_cluster_idx+1))
                if n not in temp_t_idx:
                    err_flag = 1
                    import pdb; pdb.set_trace()
                    return err_flag
    return err_flag        
    
def TC_tracker(): 
    M = load_detection(det_path, 'KITTI_3d_2') 
    global track_struct
    global all_fea_mat
    global all_fea_label
    track_struct = {'track_params':{}} 
    track_struct['track_params']['num_fr'] = int(M[-1,0]-M[0,0]+1) 
    track_struct['track_params']['IOU_thresh'] = 0.3 
    track_struct['track_params']['color_thresh'] = 0.05
    track_struct['track_params']['det_thresh'] = -2 
    track_struct['track_params']['linear_pred_thresh'] = 5 
    track_struct['track_params']['t_dist_thresh'] = 10 
    track_struct['track_params']['track_overlap_thresh'] = 0.1 
    track_struct['track_params']['search_radius'] = 1.5
    track_struct['track_params']['const_fr_thresh'] = 5 
    track_struct['track_params']['crop_size'] = 182 
    track_struct['track_params']['loc_scales'] = [100,30,5,5]
    track_struct['track_params']['time_cluster_dist'] = 100
    track_struct['track_params']['num_time_cluster'] = int(np.ceil(track_struct['track_params']['num_fr']
                                                               /track_struct['track_params']['time_cluster_dist']))
    track_struct['track_obj'] = {'track_id':[], 'bbox':[], 'bbox_3d':[] ,'det_score':[], 'mean_color':[]} 
    track_struct['tracklet_mat'] = {'xmin_mat':[], 'ymin_mat':[], 'xmax_mat':[], 'ymax_mat':[], 'x_3d_mat':[], 'y_3d_mat':[],
                                    'w_3d_mat':[], 'h_3d_mat':[], 'det_score_mat':[]}

    img_list = os.listdir(img_folder)
    #track_struct['track_params']['num_fr'] = len(img_list)
    for n in range(track_struct['track_params']['num_fr']):
        
        
        # fr idx starts from 1
        fr_idx = n+1
        idx = np.where(np.logical_and(M[:,0]==fr_idx,M[:,5]>track_struct['track_params']['det_thresh']))[0]
        if len(idx)>1:
            choose_idx, _ = merge_bbox(M[idx,1:5], 0.3, M[idx,5])
            #import pdb; pdb.set_trace()
            temp_M = M[idx[choose_idx],:]
        else:
            temp_M = M[idx,:]
        num_bbox = len(temp_M)
        
        img_name = file_name(fr_idx-1,10)+'.png'
        if img_name in img_list:
            img_path = img_folder+'/'+img_name
            img = misc.imread(img_path) 
        else:
            num_bbox = 0
        
        track_struct['track_obj']['track_id'].append([])
        if num_bbox==0:
            track_struct['track_obj']['bbox'].append([])
            track_struct['track_obj']['bbox_3d'].append([])
            track_struct['track_obj']['det_score'].append([])
            track_struct['track_obj']['mean_color'].append([])
        else:
            track_struct['track_obj']['bbox'].append(temp_M[:,1:5])
            track_struct['track_obj']['bbox_3d'].append(temp_M[:,6:10])
            track_struct['track_obj']['det_score'].append(temp_M[:,5])
            temp_mean_color = np.zeros((num_bbox,3))
            for k in range(num_bbox):
                xmin = int(max(0,temp_M[k,1]))
                ymin = int(max(0,temp_M[k,2]))
                xmax = int(min(img.shape[1]-1,temp_M[k,1]+temp_M[k,3]))
                ymax = int(min(img.shape[0]-1,temp_M[k,2]+temp_M[k,4]))
                temp_mean_color[k,0] = np.mean(img[ymin:ymax+1,xmin:xmax+1,0])
                temp_mean_color[k,1] = np.mean(img[ymin:ymax+1,xmin:xmax+1,1])
                temp_mean_color[k,2] = np.mean(img[ymin:ymax+1,xmin:xmax+1,2])
            temp_mean_color = temp_mean_color/255.0
            #import pdb; pdb.set_trace()
            track_struct['track_obj']['mean_color'].append(temp_mean_color.copy())
        #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()
    # forward tracking
    init_num = 2000
    track_struct['tracklet_mat']['xmin_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['ymin_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['xmax_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['ymax_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['det_score_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['x_3d_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['y_3d_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['w_3d_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['h_3d_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    
    max_id = 0
    for n in range(track_struct['track_params']['num_fr']-1):
        print(n)
        #print(max_id)
        track_struct['tracklet_mat'], track_struct['track_obj']['track_id'][n], track_struct['track_obj']['track_id'][n+1], max_id \
            = forward_tracking(track_struct['track_obj']['track_id'][n], track_struct['track_obj']['track_id'][n+1], 
                     track_struct['track_obj']['bbox'][n], track_struct['track_obj']['bbox'][n+1],
                     track_struct['track_obj']['bbox_3d'][n], track_struct['track_obj']['bbox_3d'][n+1],
                     track_struct['track_obj']['det_score'][n], track_struct['track_obj']['det_score'][n+1], 
                     track_struct['track_obj']['mean_color'][n], track_struct['track_obj']['mean_color'][n+1],
                     n+2, track_struct['track_params'], track_struct['tracklet_mat'], max_id)
    mask = track_struct['tracklet_mat']['xmin_mat']==-1
    mask = np.sum(mask,axis=1)
    neg_idx = np.where(mask==track_struct['track_params']['num_fr'])[0]
    track_struct['tracklet_mat']['xmin_mat'] = np.delete(track_struct['tracklet_mat']['xmin_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['ymin_mat'] = np.delete(track_struct['tracklet_mat']['ymin_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['xmax_mat'] = np.delete(track_struct['tracklet_mat']['xmax_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['ymax_mat'] = np.delete(track_struct['tracklet_mat']['ymax_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['det_score_mat'] = np.delete(track_struct['tracklet_mat']['det_score_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['x_3d_mat'] = np.delete(track_struct['tracklet_mat']['x_3d_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['y_3d_mat'] = np.delete(track_struct['tracklet_mat']['y_3d_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['w_3d_mat'] = np.delete(track_struct['tracklet_mat']['w_3d_mat'], neg_idx, axis=0)
    track_struct['tracklet_mat']['h_3d_mat'] = np.delete(track_struct['tracklet_mat']['h_3d_mat'], neg_idx, axis=0)
    #import pdb; pdb.set_trace()
    
    # tracklet clustering
    iters = 20
    track_struct['tracklet_mat'] = preprocessing(track_struct['tracklet_mat'], 3)
    
    '''
    # remove large bbox
    #import pdb; pdb.set_trace()
    for n in range(len(track_struct['tracklet_mat']['xmin_mat'])):
        cand_t = np.where(track_struct['tracklet_mat']['xmin_mat'][n,:]!=-1)[0]
        temp_h = track_struct['tracklet_mat']['ymax_mat'][n,cand_t]-track_struct['tracklet_mat']['ymin_mat'][n,cand_t]
        max_h = np.max(temp_h)
        if max_h>400:
            remove_set.append(n)
    '''
    
    #import pdb; pdb.set_trace()
    
    num_patch, img_size = crop_det(track_struct['tracklet_mat'], track_struct['track_params']['crop_size'], 
                               img_folder, crop_det_folder, 0)
    track_struct['tracklet_mat']['appearance_fea_mat'] = feature_extract(feature_size, num_patch, max_length, 
                                                                     crop_det_folder, triplet_model)
    #import pdb; pdb.set_trace()
    #*******************
    track_struct['tracklet_mat']['appearance_fea_mat'][:,2:] = 10*track_struct['tracklet_mat']['appearance_fea_mat'][:,2:]
    track_struct['track_params']['img_size'] = img_size
    track_struct['tracklet_mat']['comb_track_cost'] = np.zeros((len(track_struct['tracklet_mat']['xmin_mat']),
                                                                len(track_struct['tracklet_mat']['xmin_mat'])))
    track_struct['tracklet_mat']['comb_track_cost_mask'] = np.zeros((len(track_struct['tracklet_mat']['xmin_mat']),
                                                                len(track_struct['tracklet_mat']['xmin_mat'])))

    #import pdb; pdb.set_trace()

    # load nn
    batch_X_x = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_y = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_w = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_h = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_a = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 1])
    batch_mask_1 = tf.placeholder(tf.float32, [None, 1, max_length, 2])
    batch_mask_2 = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 2])
    batch_Y = tf.placeholder(tf.int32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = seq_nn_3d.seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,
                       batch_mask_2,batch_Y,max_length,feature_size,keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_Y, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(batch_Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()    

    with tf.Session() as sess:

        saver.restore(sess, seq_model)
        print("Model restored.")

        for n in range(iters):
            print("iteration")
            print(n)
            change_flag = tracklet_clustering(sess,
                                                        batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1,
                                                        batch_mask_2, batch_Y, keep_prob, y_conv)
            if change_flag==0:
                #import pdb; pdb.set_trace()
                time_check_flag = time_cluster_check()
                break
                
            
        #pickle.dump(save_fea_mat, open(save_fea_path, 'wb'))
        #pickle.dump(track_set, open(save_label_path,'wb'))
        #pickle.dump(remove_set, open(save_remove_path,'wb'))
        
        '''
        print(np.sum(track_struct['tracklet_mat']['comb_track_cost_mask']))
        global all_fea_mat
        global all_fea_label
        remove_idx = []
        for n in range(len(all_fea_mat)):
            if np.sum(all_fea_mat[n,0,:,1])==0:
                remove_idx.append(n) 
                
        all_fea_mat = np.delete(all_fea_mat, np.array(remove_idx), axis=0)
        all_fea_label = np.delete(all_fea_label, np.array(remove_idx), axis=0)        
        
        print(len(all_fea_mat))
        #import pdb; pdb.set_trace()
        pickle.dump(all_fea_mat, open(save_all_fea_path,'wb'))
        pickle.dump(all_fea_label, open(save_all_label_path,'wb'))
        
        
        
        save_batch_size = 5000
        save_batch_num = int(np.ceil(len(all_fea_mat)/save_batch_size))
        for k in range(save_batch_num):
            if k!=save_batch_num-1:
                temp_fea = all_fea_mat[k*save_batch_size:(k+1)*save_batch_size,:,:,:]
                temp_label = all_fea_label[k*save_batch_size:(k+1)*save_batch_size,:]
            else:
                temp_fea = all_fea_mat[k*save_batch_size:,:,:,:]
                temp_label = all_fea_label[k*save_batch_size:,:]
            temp_fea_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+'_all'+str(k)+'.obj'
            temp_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+'_all_label'+str(k)+'.obj'
            pickle.dump(temp_fea, open(temp_fea_path,'wb'))
            pickle.dump(temp_label, open(temp_label_path,'wb'))
        '''

    post_processing()
    
    pickle.dump(track_struct, open(track_struct_path,'wb'))
    
    wrt_txt(track_struct['final_tracklet_mat'])

    draw_result(img_folder, tracking_img_folder)

    convert_frames_to_video(tracking_img_folder+'/', tracking_video_path, 20)
    
    return track_struct

def refine_track_set():

    track_struct = pickle.load(open(track_struct_path,'rb'))
    track_set = pickle.load(open(save_label_path,'rb'))
    
    track_interval = track_struct['tracklet_mat']['track_interval']
    new_track_cluster = []
    for n in range(len(track_struct['final_tracklet_mat']['track_cluster'])):
        new_track_cluster.append(track_struct['final_tracklet_mat']['track_cluster'][n].copy())
    new_track_class = track_struct['final_tracklet_mat']['track_class'].copy()
    #import pdb; pdb.set_trace()
    
    # split track
    for n in range(len(track_set)):
            
        if track_set[n,2]==1:
            continue
        track_class1 = new_track_class[track_set[n,0]][0]
        track_class2 = new_track_class[track_set[n,1]][0]
        if track_class1!=track_class2:
            continue
        temp_track_cluster = new_track_cluster[track_class1].copy()
        sort_idx = np.argsort(track_interval[np.array(temp_track_cluster),1])
        before_track_ids = []
        for k in range(len(sort_idx)):
            before_track_ids.append(temp_track_cluster[sort_idx[k]])
            new_track_class[temp_track_cluster[sort_idx[k]]][0] = len(new_track_cluster)

            new_track_cluster[track_class1].remove(temp_track_cluster[sort_idx[k]])

            if temp_track_cluster[sort_idx[k]]==track_set[n,0]:
                break
        new_track_cluster.append(before_track_ids)
        
    #import pdb; pdb.set_trace()
    
    # merge track
    for n in range(len(track_set)):
        if track_set[n,2]==0:
            continue
        track_class1 = new_track_class[track_set[n,0]][0]
        track_class2 = new_track_class[track_set[n,1]][0]
        
        if track_class1==track_class2:
            continue
        if track_set[n,0] not in track_struct['tracklet_mat']['neighbor_track_idx'][track_set[n,1]]:
            continue
            
        for k in range(len(new_track_cluster[track_class2])):
            new_track_class[new_track_cluster[track_class2][k]] = track_class1
        new_track_cluster[track_class1] = new_track_cluster[track_class1].copy()+new_track_cluster[track_class2].copy()
        new_track_cluster[track_class2] = []
        
        #if track_set[n,0]==271 and track_set[n,1]==290:
        #    import pdb; pdb.set_trace()
            
    remove_idx = []
    for n in range(len(new_track_cluster)):
        if len(new_track_cluster[n])==0:
            remove_idx.append(n)
        
    new_track_cluster = list(np.delete(new_track_cluster, remove_idx))

    #import pdb; pdb.set_trace()
    
    # update track class
    N_tracklet = track_struct['tracklet_mat']['xmin_mat'].shape[0]
    new_track_class = -1*np.ones((N_tracklet,1),dtype=int)
    for n in range(len(new_track_cluster)):
        for k in range(len(new_track_cluster[n])):
            track_id = new_track_cluster[n][k]
            new_track_class[track_id,0] = n
    
    #import pdb; pdb.set_trace()
    track_struct['gt_tracklet_mat'] = {'track_cluster':[], 'track_class':[]}
    track_struct['gt_tracklet_mat']['track_cluster'] = new_track_cluster.copy()
    track_struct['gt_tracklet_mat']['track_class'] = new_track_class.copy()
    
    #import pdb; pdb.set_trace()
    
    # update label
    all_fea_label = pickle.load(open(save_all_label_path,'rb'))
    '''
    all_fea_label2 = pickle.load(open(save_all_label_path2,'rb'))
    all_fea_label3 = pickle.load(open(save_all_label_path3,'rb'))
    all_fea_label4 = pickle.load(open(save_all_label_path4,'rb'))
    all_fea_label = np.concatenate((all_fea_label1, all_fea_label2), axis=0)
    all_fea_label = np.concatenate((all_fea_label, all_fea_label3), axis=0)
    all_fea_label = np.concatenate((all_fea_label, all_fea_label4), axis=0)
    '''
    
    for n in range(len(all_fea_label)):
        track_class1 = track_struct['gt_tracklet_mat']['track_class'][int(all_fea_label[n,0])]
        track_class2 = track_struct['gt_tracklet_mat']['track_class'][int(all_fea_label[n,1])]
        if track_class1==track_class2:
            all_fea_label[n,2] = 1
            all_fea_label[n,3] = 0
        else:
            all_fea_label[n,2] = 0
            all_fea_label[n,3] = 1
    
    pickle.dump(all_fea_label, open(save_all_label_path,'wb'))
    '''
    save_batch_size = 5000
    save_batch_num = int(np.ceil(len(all_fea_label)/save_batch_size))
    #import pdb; pdb.set_trace()
    for k in range(save_batch_num):
        if k!=save_batch_num-1:
            temp_label = all_fea_label[k*save_batch_size:(k+1)*save_batch_size,:]
        else:
            temp_label = all_fea_label[k*save_batch_size:,:]
        temp_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+'_all_label'+str(k)+'.obj'
        pickle.dump(temp_label, open(temp_label_path,'wb'))
    '''
    pickle.dump(track_struct,open(track_struct_path,'wb'))

def refine_track():
    det_thresh = 0
    overlap_thresh = 0.5
    linear_len_thresh = 5
    img_size = track_struct['track_params']['img_size']
    
    global track_struct
    track_struct = pickle.load(open(track_struct_path,'rb'))
    M = load_detection(det_path, 'MOT') 
    num_det = M.shape[0]
    cand_mask = np.ones((num_det,1),dtype=int)
    
    # remove detection with low score
    cand_mask[M[:,5]<det_thresh,0] = 0
    
    # remove detection with overlap
    cand_idx = np.where(cand_mask[:,0]==1)[0]
    for n in range(len(cand_idx)):
        temp_bbox = M[cand_idx[n],1:5]
        fr_idx = M[cand_idx[n],0]
        cand_track_idx = np.where(track_struct['final_tracklet_mat']['xmin_mat'][:,fr_idx-1]!=-1)[0]
        cand_track_bbox = np.zeros((len(cand_track_idx),4))
        cand_track_bbox[:,0] = track_struct['final_tracklet_mat']['xmin_mat'][cand_track_idx,fr_idx-1]
        cand_track_bbox[:,1] = track_struct['final_tracklet_mat']['ymin_mat'][cand_track_idx,fr_idx-1]
        cand_track_bbox[:,2] = track_struct['final_tracklet_mat']['xmax_mat'][cand_track_idx,fr_idx-1]- \
            track_struct['final_tracklet_mat']['xmin_mat'][cand_track_idx,fr_idx-1]+1
        cand_track_bbox[:,3] = track_struct['final_tracklet_mat']['ymax_mat'][cand_track_idx,fr_idx-1]- \
            track_struct['final_tracklet_mat']['ymin_mat'][cand_track_idx,fr_idx-1]+1
        overlap_mat = get_overlap(temp_bbox, cand_track_bbox)
        max_overlap = np.max(overlap_mat)
        if max_overlap>overlap_thresh:
            cand_mask[cand_idx[n]] = 0
    
    # get mean color of detection
    cand_idx = np.where(cand_mask[:,0]==1)[0]
    mean_color_mat = np.zeros((num_det,3))
    for n in range(len(cand_idx)):
        xmin = int(max(0,M[cand_idx[n],1]))
        ymin = int(max(0,M[cand_idx[n],2]))
        xmax = int(min(img_size[1]-1,M[cand_idx[n],1]+M[cand_idx[n],3]))
        ymax = int(min(img_size[0]-1,M[cand_idx[n],2]+M[cand_idx[n],4]))
        mean_color_mat[cand_idx[n],0] = np.mean(img[ymin:ymax+1,xmin:xmax+1,0])
        mean_color_mat[cand_idx[n],1] = np.mean(img[ymin:ymax+1,xmin:xmax+1,1])
        mean_color_mat[cand_idx[n],2] = np.mean(img[ymin:ymax+1,xmin:xmax+1,2])
    
    # assign detection to track
    tracklet_mat = track_struct['final_tracklet_mat'].copy()
    num_track = len(tracklet_mat['xmin_mat'])
    det_to_track_overlap = np.zeros((len(cand_idx),num_track))
    det_to_track_mask = np.zeros((len(cand_idx),num_track))
    det_to_track_dist = np.zeros((len(cand_idx),num_track))
    for n in range(len(cand_idx)):
        fr_idx = M[cand_idx[n],0]
        for m in range(len(tracklet_mat['xmin_mat'])):
            non_neg_idx = np.where(tracklet_mat['xmin_mat'][m,:]!=-1)[0]
            t_min = np.min(non_neg_idx)
            t_max = np.max(non_neg_idx)
            if fr_idx-1>=t_min and fr_idx-1<=t_max:
                continue
            det_to_track_dist[n,m] = min(abs(fr_idx-1-t_min),abs(fr_idx-1-t_max))
            det_to_track_mask[n,m] = 1
            track_bbox = np.zeros((1,4))
            if abs(fr_idx-1-t_min)<abs(fr_idx-1-t_max):
                temp_len = min(linear_len_thresh,t_max-t_min+1)
                temp_x = (tracklet_mat['xmin_mat'][m,t_min:t_min+temp_len]+tracklet_mat['xmax_mat'][m,t_min:t_min+temp_len])/2
                temp_y = (tracklet_mat['ymin_mat'][m,t_min:t_min+temp_len]+tracklet_mat['ymax_mat'][m,t_min:t_min+temp_len])/2
                temp_w = tracklet_mat['xmax_mat'][m,t_min:t_min+temp_len]-tracklet_mat['xmin_mat'][m,t_min:t_min+temp_len]+1
                temp_h = tracklet_mat['ymax_mat'][m,t_min:t_min+temp_len]-tracklet_mat['ymin_mat'][m,t_min:t_min+temp_len]+1
                pred_xx = linear_pred(temp_x,t=fr_idx-1-t_min)
                pred_yy = linear_pred(temp_y,t=fr_idx-1-t_min)
                pred_ww = linear_pred(temp_w,t=fr_idx-1-t_min)
                pred_hh = linear_pred(temp_h,t=fr_idx-1-t_min)
            else:
                temp_len = min(linear_len_thresh,t_max-t_min+1)
                temp_x = (tracklet_mat['xmin_mat'][m,t_max+1-temp_len:t_max+1]+tracklet_mat['xmax_mat'][m,t_max+1-temp_len:t_max+1])/2
                temp_y = (tracklet_mat['ymin_mat'][m,t_max+1-temp_len:t_max+1]+tracklet_mat['ymax_mat'][m,t_max+1-temp_len:t_max+1])/2
                temp_w = tracklet_mat['xmax_mat'][m,t_max+1-temp_len:t_max+1]-tracklet_mat['xmin_mat'][m,t_max+1-temp_len:t_max+1]+1
                temp_h = tracklet_mat['ymax_mat'][m,t_max+1-temp_len:t_max+1]-tracklet_mat['ymin_mat'][m,t_max+1-temp_len:t_max+1]+1
                pred_xx = linear_pred(temp_x,t=fr_idx-1-t_min)
                pred_yy = linear_pred(temp_y,t=fr_idx-1-t_min)
                pred_ww = linear_pred(temp_w,t=fr_idx-1-t_min)
                pred_hh = linear_pred(temp_h,t=fr_idx-1-t_min)
                
                track_bbox[0,0] = pred_xx-pred_ww/2
                track_bbox[0,1] = pred_yy-pred_hh/2
                track_bbox[0,2] = pred_ww
                track_bbox[0,3] = pred_hh
        
    while 1:
        for n in range(len(cand_idx)):
            aa=1
            
    
    

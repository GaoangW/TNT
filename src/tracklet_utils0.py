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
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import seq_nn


det_path = 'D:/Data/MOT/MOT16Labels/test/MOT16-07/det/det.txt'
img_folder = 'D:/Data/MOT/MOT17Det/test/MOT17-07/img1'
crop_det_folder = 'D:/Data/MOT/crop_det/MOT16-07'
triplet_model = 'D:/Data/UA-Detrac/UA_Detrac_model/MOT'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
seq_model = 'D:/Data/UA-Detrac/cnn_MOT/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/fine_tune_model/model.ckpt'
tracking_img_folder = 'D:/Data/MOT/tracking_img/MOT16-07'
tracking_video_path = 'D:/Data/MOT/tracking_video/MOT16-07.avi'
save_fea_path = 'D:/Data/MOT/save_fea_mat/MOT16-07.obj'
save_label_path = 'D:/Data/MOT/save_fea_mat/MOT16-07_label.obj'
save_remove_path = 'D:/Data/MOT/save_fea_mat/MOT16-07_remove_set.obj'
txt_result_path = 'D:/Data/MOT/txt_result/MOT16-07.txt'
track_struct_path = 'D:/Data/MOT/track_struct/MOT16-07.obj'
max_length = 64
feature_size = 4+512
batch_size = 64
num_classes = 2


track_set = []
remove_set = []

'''
remove_set = [6,8,10,16,15,14,34,29,92,69,71,38,40,45,48,43,54,75,98,97,100,102,108,109,111,112,114,122,123,126,130,135,129,
             132,139,134,144,146,147,159,165,171,173,183,204,205,209,210,216,222,227,233,242,250]
track_set = np.array([[5,11,1],
                     [3,31,1],
                     [52,55,0],
                     [36,42,0],
                     [36,58,1],
                     [47,55,1],
                     [77,82,0],
                     [46,82,1],
                     [52,82,1],
                     [85,115,1],
                     [137,143,0],
                     [119,143,1],
                     [143,150,0],
                     [150,152,0],
                     [155,163,0],
                     [192,195,0],
                     [206,213,0]])

save_fea_mat = np.zeros((len(track_set),feature_size,max_length,2))
'''

def linear_pred(y):
    if len(y)==1:
        return y
    else:
        x = np.array(range(0,len(y)))
        slope, intercept, _, _, _ = stats.linregress(x,y)
        return slope*len(y)+intercept
    
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
        if len(idx)<len_thresh: 
            remove_idx.append(n) 
            
    new_tracklet_mat['xmin_mat'] = np.delete(new_tracklet_mat['xmin_mat'], remove_idx, 0) 
    new_tracklet_mat['ymin_mat'] = np.delete(new_tracklet_mat['ymin_mat'], remove_idx, 0) 
    new_tracklet_mat['xmax_mat'] = np.delete(new_tracklet_mat['xmax_mat'], remove_idx, 0) 
    new_tracklet_mat['ymax_mat'] = np.delete(new_tracklet_mat['ymax_mat'], remove_idx, 0) 
    new_tracklet_mat['det_score_mat'] = np.delete(new_tracklet_mat['det_score_mat'], remove_idx, 0) 
    return new_tracklet_mat

#M = [fr_idx, x, y, w, h, score]
def forward_tracking(track_id1, track_id2, bbox1, bbox2, det_score1, det_score2, mean_color1, mean_color2, 
                     fr_idx2, track_params, tracklet_mat): 
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
    if fr_idx2==2:
        new_track_id1 = list(range(1,num1+1))
        new_tracklet_mat['xmin_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['ymin_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['xmax_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['ymax_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['det_score_mat'] = -np.ones((num1, num_fr))
        new_tracklet_mat['xmin_mat'][:,0] = bbox1[:,0]
        new_tracklet_mat['ymin_mat'][:,0] = bbox1[:,1]
        new_tracklet_mat['xmax_mat'][:,0] = bbox1[:,0]+bbox1[:,2]-1
        new_tracklet_mat['ymax_mat'][:,0] = bbox1[:,1]+bbox1[:,3]-1
        new_tracklet_mat['det_score_mat'][:,0] = det_score1
        
    max_id = new_tracklet_mat['xmin_mat'].shape[0]
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
        max_id = max_id+num2
        new_tracklet_mat['xmin_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,0]
        new_tracklet_mat['ymin_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,1]
        new_tracklet_mat['xmax_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,0]+bbox2[:,2]-1
        new_tracklet_mat['ymax_mat'][max_id-num2:max_id,fr_idx2-1] = bbox2[:,1]+bbox2[:,3]-1
        new_tracklet_mat['det_score_mat'][max_id-num2:max_id,fr_idx2-1] = det_score2
    elif len(idx1)>0:
        new_track_id2 = []
        for n in range(num2):
            #import pdb; pdb.set_trace()
            temp_idx = np.where(idx2==n)[0]
            if len(temp_idx)==0:
                max_id = max_id+1
                new_track_id2.append(max_id)
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
                #import pdb; pdb.set_trace()
                new_tracklet_mat['xmin_mat'][-1,fr_idx2-1] = bbox2[n,0]
                new_tracklet_mat['ymin_mat'][-1,fr_idx2-1] = bbox2[n,1]
                new_tracklet_mat['xmax_mat'][-1,fr_idx2-1] = bbox2[n,0]+bbox2[n,2]-1
                new_tracklet_mat['ymax_mat'][-1,fr_idx2-1] = bbox2[n,1]+bbox2[n,3]-1
                new_tracklet_mat['det_score_mat'][-1,fr_idx2-1] = det_score2[n]
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
    else:
        new_track_id2 = []

    #import pdb; pdb.set_trace()
    return new_tracklet_mat, new_track_id1, new_track_id2
def init_clustering(track_struct): 
    N_tracklet = track_struct['tracklet_mat']['xmin_mat'].shape[0]

    # track interval
    track_struct['tracklet_mat']['track_interval'] = np.zeros((N_tracklet, 2))

    # track cluster
    track_struct['tracklet_mat']['track_cluster'] = []

    # track class
    track_struct['tracklet_mat']['track_class'] = np.arange(N_tracklet, dtype=int)

    for n in range(N_tracklet):
        idx = np.where(track_struct['tracklet_mat']['xmin_mat'][n,:]!=-1)[0]
        track_struct['tracklet_mat']['track_interval'][n,0] = np.min(idx)
        track_struct['tracklet_mat']['track_interval'][n,1] = np.max(idx)
        track_struct['tracklet_mat']['track_cluster'].append([n])

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
            overlap_len = min(t_max2,t_max1)-max(t_min1,t_min2)
            overlap_r = overlap_len/(t_max1-t_min1+1+t_max2-t_min2+1-overlap_len)
            if overlap_len>0 and overlap_r>track_struct['track_params']['track_overlap_thresh']:
                track_struct['tracklet_mat']['conflict_track_idx'][n].append(m)
                track_struct['tracklet_mat']['conflict_track_idx'][m].append(n)
            if overlap_len>0 and overlap_r<=track_struct['track_params']['track_overlap_thresh']:
                # check the search region
                t1 = int(max(t_min1,t_min2))
                t2 = int(min(t_max2,t_max1))
                if (t_min1<t_min2 and t_max1>t_max2) or (t_min1>t_min2 and t_max1<t_max2):
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
                min_dist_x1 = np.min(dist_x/w1)
                min_dist_y1 = np.min(dist_y/h1)
                min_dist_x2 = np.min(dist_x/w2)
                min_dist_y2 = np.min(dist_y/h2)
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
                #import pdb; pdb.set_trace()
                dist_x = abs(track_struct['tracklet_mat']['center_x'][n,t1] \
                         -track_struct['tracklet_mat']['center_x'][m,t2])
                dist_y = abs(track_struct['tracklet_mat']['center_y'][n,t1] \
                         -track_struct['tracklet_mat']['center_y'][m,t2])
                w1 = track_struct['tracklet_mat']['w'][n,t1]
                h1 = track_struct['tracklet_mat']['h'][n,t1]
                w2 = track_struct['tracklet_mat']['w'][m,t2]
                h2 = track_struct['tracklet_mat']['h'][m,t2]
                min_dist_x1 = dist_x/w1
                min_dist_y1 = dist_y/h1
                min_dist_x2 = dist_x/w2
                min_dist_y2 = dist_y/h2
                if min_dist_x1<track_struct['track_params']['search_radius'] \
                    and min_dist_y1<track_struct['track_params']['search_radius'] \
                    and min_dist_x2<track_struct['track_params']['search_radius'] \
                    and min_dist_y2<track_struct['track_params']['search_radius']:
                    track_struct['tracklet_mat']['neighbor_track_idx'][n].append(m)
                    track_struct['tracklet_mat']['neighbor_track_idx'][m].append(n)

    # cluster cost
    track_struct['tracklet_mat']['cluster_cost'] = []
    for n in range(N_tracklet):
        track_struct['tracklet_mat']['cluster_cost'].append(0)

    # save all comb cost for two tracklets
    # comb_track_cost [track_id1, track_id2, cost]
    track_struct['tracklet_mat']['comb_track_cost'] = []

    # save feature mat for training
    '''
    if len(track_struct['tracklet_mat']['track_set'])>0:
        track_struct['tracklet_mat']['save_fea_mat'] = np.zeros((len(track_struct['tracklet_mat']['track_set']), feature_size, max_length, 2))
    else:
        track_struct['tracklet_mat']['save_fea_mat'] = []
    '''
    return track_struct

def comb_cost(tracklet_set, tracklet_mat, feature_size, max_length, img_size, sess, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv): 
    comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    comb_track_cost = np.array(tracklet_mat['comb_track_cost'].copy()) 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy() 
    #track_set = tracklet_mat['track_set'].copy()

    # cnn classifier
    N_tracklet = len(tracklet_set)
    track_interval = tracklet_mat['track_interval']
    sort_idx = np.argsort(track_interval[np.array(tracklet_set),1])
    cost = 0
    if len(sort_idx)<=1:
        return cost, comb_track_cost_list


    remove_ids = []
    comb_fea_mat = np.zeros((len(sort_idx)-1,feature_size,max_length,2))
    temp_cost_list = []
    for n in range(0, len(sort_idx)-1):
        track_id1 = tracklet_set[sort_idx[n]]
        track_id2 = tracklet_set[sort_idx[n+1]]
        if len(comb_track_cost)>0:
            search_idx = np.where(np.logical_and(comb_track_cost[:,0]==track_id1, comb_track_cost[:,1]==track_id2))
            if len(search_idx[0])>0:
                remove_ids.append(n)
                #import pdb; pdb.set_trace()
                cost = cost+comb_track_cost[search_idx[0][0],2]
                continue

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
            comb_fea_mat[n,0,t1_min-t_min:t1_max-t_min+1,1] = 1
            comb_fea_mat[n,0,t1_min-t_min:t1_max-t_min+1,0] = 0.5*(tracklet_mat['xmin_mat'][track_id1,t1_min:t1_max+1]
                                                     +tracklet_mat['xmax_mat'][track_id1,t1_min:t1_max+1])/img_size[1]
            comb_fea_mat[n,1,t1_min-t_min:t1_max-t_min+1,0] = 0.5*(tracklet_mat['ymin_mat'][track_id1,t1_min:t1_max+1]
                                                     +tracklet_mat['ymax_mat'][track_id1,t1_min:t1_max+1])/img_size[0]
            comb_fea_mat[n,2,t1_min-t_min:t1_max-t_min+1,0] = (tracklet_mat['xmax_mat'][track_id1,t1_min:t1_max+1]
                                                 -tracklet_mat['xmin_mat'][track_id1,t1_min:t1_max+1]+1)/img_size[1]
            comb_fea_mat[n,3,t1_min-t_min:t1_max-t_min+1,0] = (tracklet_mat['ymax_mat'][track_id1,t1_min:t1_max+1]
                                                 -tracklet_mat['ymin_mat'][track_id1,t1_min:t1_max+1]+1)/img_size[0]
            cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id1+1)[0]

            if comb_fea_mat[n,4:,t1_min-t_min:t1_max-t_min+1,0].shape[1]!=np.transpose(tracklet_mat['appearance_fea_mat'] \
                                                                                       [cand_idx,2:]).shape[1]:
                import pdb; pdb.set_trace()
            comb_fea_mat[n,4:,t1_min-t_min:t1_max-t_min+1,0] = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])

            comb_fea_mat[n,0,t2_min-t_min:t2_max-t_min+1,1] = 1
            #print(t_min)
            #print(t2_min)
            #print(t2_max)
            #import pdb; pdb.set_trace()
            comb_fea_mat[n,0,t2_min-t_min:t2_max-t_min+1,0] = 0.5*(tracklet_mat['xmin_mat'][track_id2,t2_min:t2_max+1]
                                                     +tracklet_mat['xmax_mat'][track_id2,t2_min:t2_max+1])/img_size[1]
            comb_fea_mat[n,1,t2_min-t_min:t2_max-t_min+1,0] = 0.5*(tracklet_mat['ymin_mat'][track_id2,t2_min:t2_max+1]
                                                     +tracklet_mat['ymax_mat'][track_id2,t2_min:t2_max+1])/img_size[0]
            comb_fea_mat[n,2,t2_min-t_min:t2_max-t_min+1,0] = (tracklet_mat['xmax_mat'][track_id2,t2_min:t2_max+1]
                                                 -tracklet_mat['xmin_mat'][track_id2,t2_min:t2_max+1]+1)/img_size[1]
            comb_fea_mat[n,3,t2_min-t_min:t2_max-t_min+1,0] = (tracklet_mat['ymax_mat'][track_id2,t2_min:t2_max+1]
                                                 -tracklet_mat['ymin_mat'][track_id2,t2_min:t2_max+1]+1)/img_size[0]
            cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id2+1)[0]
            if comb_fea_mat[n,4:,t2_min-t_min:t2_max-t_min+1,0].shape[1]!=np.transpose(tracklet_mat['appearance_fea_mat'] \
                                                                                       [cand_idx,2:]).shape[1]:
                import pdb; pdb.set_trace()
                
            comb_fea_mat[n,4:,t2_min-t_min:t2_max-t_min+1,0] = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])
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

            comb_fea_mat[n,:,0:t1_max-t1_start+1,1] = comb_fea_mat[n,0,0:t1_max-t1_start+1,1]+1
            if comb_fea_mat[n,0,0:t1_max-t1_start+1,0].shape[0] \
                !=tracklet_mat['xmax_mat'][track_id1,t1_start:t1_max+1].shape[0]:
                import pdb; pdb.set_trace()
            comb_fea_mat[n,0,0:t1_max-t1_start+1,0] = 0.5*(tracklet_mat['xmin_mat'][track_id1,t1_start:t1_max+1]
                                                     +tracklet_mat['xmax_mat'][track_id1,t1_start:t1_max+1])/img_size[1]
            comb_fea_mat[n,1,0:t1_max-t1_start+1,0] = 0.5*(tracklet_mat['ymin_mat'][track_id1,t1_start:t1_max+1]
                                                     +tracklet_mat['ymax_mat'][track_id1,t1_start:t1_max+1])/img_size[0]
            comb_fea_mat[n,2,0:t1_max-t1_start+1,0] = (tracklet_mat['xmax_mat'][track_id1,t1_start:t1_max+1]
                                                 -tracklet_mat['xmin_mat'][track_id1,t1_start:t1_max+1]+1)/img_size[1]
            comb_fea_mat[n,3,0:t1_max-t1_start+1,0] = (tracklet_mat['ymax_mat'][track_id1,t1_start:t1_max+1]
                                                 -tracklet_mat['ymin_mat'][track_id1,t1_start:t1_max+1]+1)/img_size[0]
            cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id1+1)[0]
            cand_idx = cand_idx[t1_start-t1_min:]
            comb_fea_mat[n,4:,0:t1_max-t1_start+1,0] = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])

            comb_fea_mat[n,:,t2_min-t1_start:t2_end-t1_start+1,1] = comb_fea_mat[n,0,t2_min-t1_start:t2_end-t1_start+1,1]+1
            if comb_fea_mat[n,0,t2_min-t1_start:t2_end-t1_start+1,0].shape[0] \
                !=tracklet_mat['xmin_mat'][track_id2,t2_min:t2_end+1].shape[0]:
                import pdb; pdb.set_trace()
            comb_fea_mat[n,0,t2_min-t1_start:t2_end-t1_start+1,0] = 0.5*(tracklet_mat['xmin_mat'][track_id2,t2_min:t2_end+1]
                                                     +tracklet_mat['xmax_mat'][track_id2,t2_min:t2_end+1])/img_size[1]
            comb_fea_mat[n,1,t2_min-t1_start:t2_end-t1_start+1,0] = 0.5*(tracklet_mat['ymin_mat'][track_id2,t2_min:t2_end+1]
                                                     +tracklet_mat['ymax_mat'][track_id2,t2_min:t2_end+1])/img_size[0]
            comb_fea_mat[n,2,t2_min-t1_start:t2_end-t1_start+1,0] = (tracklet_mat['xmax_mat'][track_id2,t2_min:t2_end+1]
                                                 -tracklet_mat['xmin_mat'][track_id2,t2_min:t2_end+1]+1)/img_size[1]
            comb_fea_mat[n,3,t2_min-t1_start:t2_end-t1_start+1,0] = (tracklet_mat['ymax_mat'][track_id2,t2_min:t2_end+1]
                                                 -tracklet_mat['ymin_mat'][track_id2,t2_min:t2_end+1]+1)/img_size[0]
            cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id2+1)[0]
            #import pdb; pdb.set_trace()
            cand_idx = cand_idx[0:t2_end-t2_min+1]
            comb_fea_mat[n,4:,t2_min-t1_start:t2_end-t1_start+1,0] \
                = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])

        #if track_id1==34 and track_id2==39:
        #    import pdb; pdb.set_trace()

        # remove overlap detections
        t_overlap = np.where(comb_fea_mat[n,0,:,1]>1)
        if len(t_overlap)>0:
            t_overlap = t_overlap[0]
            comb_fea_mat[n,:,t_overlap,:] = 0

        if len(track_set)>0:
            search_idx = np.where(np.logical_and(track_set[:,0]==track_id1, track_set[:,1]==track_id2))
            if len(search_idx[0])>0:
                save_fea_mat[search_idx[0][0],:,:,:] = comb_fea_mat[n,:,:,:]

    comb_fea_mat = np.delete(comb_fea_mat, np.array(remove_ids), axis=0)
    if len(comb_fea_mat)>0:
        batch_size = comb_fea_mat.shape[0]
        x = np.zeros((batch_size,1,max_length,1))
        y = np.zeros((batch_size,1,max_length,1))
        w = np.zeros((batch_size,1,max_length,1))
        h = np.zeros((batch_size,1,max_length,1))
        ap = np.zeros((batch_size,feature_size-4,max_length,1))
        mask_1 = np.zeros((batch_size,1,max_length,1))
        mask_2 = np.zeros((batch_size,feature_size-4,max_length,1))
        x[:,0,:,0] = comb_fea_mat[:,0,:,0]
        y[:,0,:,0] = comb_fea_mat[:,1,:,0]
        w[:,0,:,0] = comb_fea_mat[:,2,:,0]
        h[:,0,:,0] = comb_fea_mat[:,3,:,0]
        ap[:,:,:,0] = comb_fea_mat[:,4:,:,0]
        mask_1[:,0,:,0] = 1-comb_fea_mat[:,0,:,1]
        mask_2[:,:,:,0] = 1-comb_fea_mat[:,4:,:,1]
        pred_y = sess.run(y_conv, feed_dict={batch_X_x: x,
                                     batch_X_y: y,
                                     batch_X_w: w,
                                     batch_X_h: h,
                                     batch_X_a: ap,
                                     batch_mask_1: mask_1,
                                     batch_mask_2: mask_2,
                                     batch_Y: np.zeros((batch_size,2)), 
                                     keep_prob: 1.0})
        cost = cost+np.sum(pred_y[:,1]-pred_y[:,0])
        for n in range(pred_y.shape[0]):
            temp_cost_list[n].append(pred_y[n,1]-pred_y[n,0])
        comb_track_cost_list = comb_track_cost_list+temp_cost_list
    return cost, comb_track_cost_list

def get_split_cost(tracklet_mat, track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                   batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    new_cluster_cost = np.zeros((2,1))
    if len(tracklet_mat['track_cluster'][track_id])<2:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    track_interval = tracklet_mat['track_interval']
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
                return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

        #*********************************
        new_cluster_cost[1,0], comb_track_cost_list = comb_cost(remain_tracks, tracklet_mat, feature_size, 
                                                                         max_length, 
                                                            img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                            batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                            batch_Y, keep_prob, y_conv)
        tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
        
    #if len(remain_tracks)==9 and track_id==251:
    #    import pdb; pdb.set_trace()
    cost = np.sum(new_cluster_cost)  
    prev_cost = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]]
    diff_cost = cost-prev_cost

    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx, comb_track_cost_list

def get_assign_cost(tracklet_mat, track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, 
                    batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    #import pdb; pdb.set_trace()
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]]
    new_cluster_cost = np.zeros((2,1))
    new_cluster_set = []
    new_cluster_set.append(cluster1.copy())
    new_cluster_set[0].remove(track_id)
    track_interval = tracklet_mat['track_interval']
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
                return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

        new_cluster_cost[0,0], comb_track_cost_list = comb_cost(new_cluster_set[0], tracklet_mat, feature_size, 
                                                                         max_length, 
                                                            img_size, sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                                                            batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, 
                                                            y_conv)
        tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()

    N_cluster = len(tracklet_mat['track_cluster'])
    temp_new_cluster_cost = float("inf")*np.ones((N_cluster,1))
    prev_cost_vec = np.zeros((N_cluster,1))

    for n in range(N_cluster):
        # the original cluster
        if tracklet_mat['track_class'][track_id]==n:
            continue

        # check neighbor and conflict track
        cluster2 = tracklet_mat['track_cluster'][n]
        neighbor_flag = 0
        conflict_flag = 0
        remove_flag = 0
        for m in range(len(cluster2)):
            if cluster2[m] in remove_set:
                remove_flag = 1
                break
            if track_id in tracklet_mat['neighbor_track_idx'][cluster2[m]]:
                neighbor_flag = 1
            if track_id in tracklet_mat['conflict_track_idx'][cluster2[m]]:
                conflict_flag = 1
        if neighbor_flag==0 or conflict_flag==1 or remove_flag==1:
            continue

        # get cost
        temp_set = cluster2.copy()
        temp_set.append(track_id)
        temp_new_cluster_cost[n,0], comb_track_cost_list = comb_cost(temp_set, tracklet_mat, feature_size, 
                                                                              max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
        tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()

        #import pdb; pdb.set_trace()
        prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
            +tracklet_mat['cluster_cost'][n]

    cost_vec = temp_new_cluster_cost+new_cluster_cost[0,0] 
    diff_cost_vec = cost_vec-prev_cost_vec
    min_idx = np.argmin(diff_cost_vec)
    cost = cost_vec[min_idx]
    if cost==float("inf"):
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    diff_cost = diff_cost_vec[min_idx]
    new_cluster_cost[1,0] = temp_new_cluster_cost[min_idx,0]
    change_cluster_idx = [tracklet_mat['track_class'][track_id],min_idx]
    temp_set = tracklet_mat['track_cluster'][min_idx].copy()
    temp_set.append(track_id)
    new_cluster_set.append(temp_set)

    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

def get_merge_cost(tracklet_mat, track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                   batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]]
    if len(cluster1)==1:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    N_cluster = len(tracklet_mat['track_cluster'])
    new_cluster_cost_vec = float("inf")*np.ones((N_cluster,1))
    prev_cost_vec = np.zeros((N_cluster,1))

    for n in range(N_cluster):
        # the original cluster
        if tracklet_mat['track_class'][track_id]==n:
            continue

        # check neighbor and conflict track
        cluster2 = tracklet_mat['track_cluster'][n]
        if len(cluster2)<=1:
            continue
        
        neighbor_flag = 0
        conflict_flag = 0
        remove_flag = 0
        for m in range(len(cluster1)):
            for k in range(len(cluster2)):
                if cluster2[k] in remove_set:
                    remove_flag = 1
                    break
                if cluster1[m] in tracklet_mat['neighbor_track_idx'][cluster2[k]]:
                    neighbor_flag = 1
                if cluster1[m] in tracklet_mat['conflict_track_idx'][cluster2[k]]:
                    conflict_flag = 1
        if neighbor_flag==0 or conflict_flag==1 or remove_flag==1:
            continue

            
        # get cost
        new_cluster_cost_vec[n,0], comb_track_cost_list = comb_cost(cluster1+cluster2, tracklet_mat, feature_size, 
                                                                max_length, img_size, sess, batch_X_x, batch_X_y, 
                                                                batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
                                                                batch_mask_2, batch_Y, keep_prob, y_conv)
        tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()

        prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
            +tracklet_mat['cluster_cost'][n]

    diff_cost_vec = new_cluster_cost_vec-prev_cost_vec
    min_idx = np.argmin(diff_cost_vec)
    cost = new_cluster_cost_vec[min_idx,0]
    if cost==float("inf"):
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    diff_cost = diff_cost_vec[min_idx,0]
    new_cluster_cost = np.zeros((2,1))
    new_cluster_cost[0,0] = cost
    change_cluster_idx = [tracklet_mat['track_class'][track_id], min_idx]
    new_cluster_set = []
    temp_set = cluster1.copy()
    temp_set = temp_set+tracklet_mat['track_cluster'][min_idx]
    new_cluster_set.append(temp_set)
    new_cluster_set.append([])
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

def get_switch_cost(tracklet_mat, track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                    batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]]
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
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    N_cluster = len(tracklet_mat['track_cluster'])
    cost_vec = float("inf")*np.ones((N_cluster,1))
    prev_cost_vec = np.zeros((N_cluster,1))
    new_cluster_cost_vec1 = float("inf")*np.ones((N_cluster,1))
    new_cluster_cost_vec2 = float("inf")*np.ones((N_cluster,1))
    track_id_set = []
    for n in range(N_cluster):
        track_id_set.append([])

    for n in range(N_cluster):
        # the original cluster
        if tracklet_mat['track_class'][track_id]==n:
            continue

        # switch availability check
        S3 = []
        S4 = []
        remove_flag = 0
        cluster2 = tracklet_mat['track_cluster'][n]
        for k in range(len(cluster2)):
            temp_id = cluster2[k]
            if temp_id in remove_set:
                remove_flag = 1
                break
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
        
        if remove_flag==1:
            continue
            
        neighbor_flag1 = 0
        conflict_flag1 = 0
        if len(S3)==0:
            neighbor_flag1 = 1
            conflict_flag1 = 0
        else:
            for k in range(len(S3)):
                for kk in range(len(S2)):
                    if S3[k] in tracklet_mat['neighbor_track_idx'][S2[kk]]:
                        neighbor_flag1 = 1
                    if S3[k] in tracklet_mat['conflict_track_idx'][S2[kk]]:
                        conflict_flag1 = 1
        neighbor_flag2 = 0
        conflict_flag2 = 0
        if len(S4)==0:
            neighbor_flag2 = 1
            conflict_flag2 = 0
        else:
            for k in range(len(S4)):
                for kk in range(len(S1)):
                    if S4[k] in tracklet_mat['neighbor_track_idx'][S1[kk]]:
                        neighbor_flag2 = 1
                    if S4[k] in tracklet_mat['conflict_track_idx'][S1[kk]]:
                        conflict_flag2 = 1 
        if neighbor_flag1==0 or conflict_flag1==1 or neighbor_flag2==0 or conflict_flag2==1:
            continue

        # get cost
        S_1 = S1+S4
        S_2 = S2+S3
        new_cluster_cost_vec1[n,0], comb_track_cost_list = comb_cost(S_1, tracklet_mat, feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
        tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
        new_cluster_cost_vec2[n,0], comb_track_cost_list = comb_cost(S_2, tracklet_mat, feature_size, max_length, 
                                                                 img_size, sess, batch_X_x, batch_X_y, batch_X_w, 
                                                                 batch_X_h, batch_X_a, batch_mask_1, batch_mask_2, 
                                                                 batch_Y, keep_prob, y_conv)
        tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
        #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
        cost_vec[n,0] = new_cluster_cost_vec1[n,0]+new_cluster_cost_vec2[n,0]

        track_id_set[n].append(S_1.copy())
        track_id_set[n].append(S_2.copy())
        prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
                        +tracklet_mat['cluster_cost'][n]

    diff_cost_vec = cost_vec-prev_cost_vec
    min_idx = np.argmin(diff_cost_vec)
    cost = cost_vec[min_idx,0]
    if cost==float("inf"):
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    diff_cost = diff_cost_vec[min_idx,0]
    new_cluster_cost = np.zeros((2,1))
    new_cluster_cost[0,0] = new_cluster_cost_vec1[min_idx,0]
    new_cluster_cost[1,0] = new_cluster_cost_vec2[min_idx,0]

    change_cluster_idx = [tracklet_mat['track_class'][track_id], min_idx]
    new_cluster_set = []
    new_cluster_set.append(track_id_set[min_idx][0])
    new_cluster_set.append(track_id_set[min_idx][1])
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

def get_break_cost(tracklet_mat, track_id, sess, img_size, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                   batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    comb_track_cost_list = tracklet_mat['comb_track_cost'].copy() 
    #save_fea_mat = tracklet_mat['save_fea_mat'].copy()

    new_cluster_cost = np.zeros((2,1))
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]]
    if len(cluster1)<=2:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

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
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    before_ids = list(set(cluster1)-set(after_ids))
    if len(before_ids)<=1:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

    change_cluster_idx = [len(tracklet_mat['track_cluster']), tracklet_mat['track_class'][track_id]]
    new_cluster_set = []
    new_cluster_set.append(before_ids)
    remain_tracks = after_ids
    new_cluster_set.append(remain_tracks)
    new_cluster_cost[0,0], comb_track_cost_list = comb_cost(new_cluster_set[0], tracklet_mat, feature_size, 
                                                                     max_length, 
                                                        img_size, sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                                                        batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, 
                                                        y_conv)
    tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
    #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
    new_cluster_cost[1,0], comb_track_cost_list = comb_cost(new_cluster_set[1], tracklet_mat, feature_size, 
                                                                     max_length, 
                                                        img_size, sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, 
                                                        batch_X_a, batch_mask_1, batch_mask_2, batch_Y, keep_prob, 
                                                        y_conv)
    tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
    #tracklet_mat['save_fea_mat'] = save_fea_mat.copy()
    cost = np.sum(new_cluster_cost)
    diff_cost = cost-tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]]
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx,comb_track_cost_list

def update_tracklet_mat(tracklet_mat): 
    new_tracklet_mat = tracklet_mat.copy() 
    final_tracklet_mat = tracklet_mat.copy() 
    track_interval = tracklet_mat['track_interval'] 
    num_cluster = len(tracklet_mat['track_cluster'])

    new_xmin_mat = -1*np.ones((num_cluster, new_tracklet_mat['xmin_mat'].shape[1]))
    new_ymin_mat = -1*np.ones((num_cluster, new_tracklet_mat['ymin_mat'].shape[1]))
    new_xmax_mat = -1*np.ones((num_cluster, new_tracklet_mat['xmax_mat'].shape[1]))
    new_ymax_mat = -1*np.ones((num_cluster, new_tracklet_mat['ymax_mat'].shape[1]))
    new_det_score_mat = -1*np.ones((num_cluster, new_tracklet_mat['det_score_mat'].shape[1]))
    
    final_xmin_mat = -1*np.ones((num_cluster, new_tracklet_mat['xmin_mat'].shape[1]))
    final_ymin_mat = -1*np.ones((num_cluster, new_tracklet_mat['ymin_mat'].shape[1]))
    final_xmax_mat = -1*np.ones((num_cluster, new_tracklet_mat['xmax_mat'].shape[1]))
    final_ymax_mat = -1*np.ones((num_cluster, new_tracklet_mat['ymax_mat'].shape[1]))
    final_det_score_mat = -1*np.ones((num_cluster, new_tracklet_mat['det_score_mat'].shape[1]))

    for n in range(num_cluster):
        for k in range(len(new_tracklet_mat['track_cluster'][n])):
            temp_id = new_tracklet_mat['track_cluster'][n][k]
            #import pdb; pdb.set_trace()
            new_xmin_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = new_tracklet_mat['xmin_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            new_ymin_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = new_tracklet_mat['ymin_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            new_xmax_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = new_tracklet_mat['xmax_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            new_ymax_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = new_tracklet_mat['ymax_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]
            new_det_score_mat[n,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1] \
                = new_tracklet_mat['det_score_mat'][temp_id,int(track_interval[temp_id,0]):int(track_interval[temp_id,1])+1]

    for n in range(num_cluster):
        det_idx = np.where(new_xmin_mat[n,:]!=-1)[0]
        t_min = np.min(det_idx)
        t_max = np.max(det_idx)
        miss_idx = np.where(new_xmin_mat[n,t_min:t_max+1]==-1)[0]
        if len(miss_idx)==0:
            new_xmin_mat[n,:] = -1
            new_ymin_mat[n,:] = -1
            new_xmax_mat[n,:] = -1
            new_ymax_mat[n,:] = -1
            new_det_score_mat[n,:] = -1
            continue
        miss_idx = miss_idx+t_min
        new_xmin_mat[n,miss_idx] = np.interp(miss_idx, det_idx, new_xmin_mat[n,det_idx])
        final_xmin_mat[n,miss_idx] = new_xmin_mat[n,miss_idx].copy()
        new_xmin_mat[n,det_idx] = -1
        
        new_ymin_mat[n,miss_idx] = np.interp(miss_idx, det_idx, new_ymin_mat[n,det_idx])
        final_ymin_mat[n,miss_idx] = new_ymin_mat[n,miss_idx].copy()
        new_ymin_mat[n,det_idx] = -1
        
        new_xmax_mat[n,miss_idx] = np.interp(miss_idx, det_idx, new_xmax_mat[n,det_idx])
        final_xmax_mat[n,miss_idx] = new_xmax_mat[n,miss_idx].copy()
        new_xmax_mat[n,det_idx] = -1
        
        new_ymax_mat[n,miss_idx] = np.interp(miss_idx, det_idx, new_ymax_mat[n,det_idx])
        final_ymax_mat[n,miss_idx] = new_ymax_mat[n,miss_idx].copy()
        new_ymax_mat[n,det_idx] = -1
        
        new_det_score_mat[n,miss_idx] = np.interp(miss_idx, det_idx, new_det_score_mat[n,det_idx])
        final_det_score_mat[n,miss_idx] = new_det_score_mat[n,miss_idx].copy()
        new_det_score_mat[n,det_idx] = -1

    new_tracklet_mat['xmin_mat'] = new_xmin_mat
    new_tracklet_mat['ymin_mat'] = new_ymin_mat
    new_tracklet_mat['xmax_mat'] = new_xmax_mat
    new_tracklet_mat['ymax_mat'] = new_ymax_mat
    new_tracklet_mat['det_score_mat'] = new_det_score_mat
    
    final_tracklet_mat['xmin_mat'] = final_xmin_mat
    final_tracklet_mat['ymin_mat'] = final_ymin_mat
    final_tracklet_mat['xmax_mat'] = final_xmax_mat
    final_tracklet_mat['ymax_mat'] = final_ymax_mat
    final_tracklet_mat['det_score_mat'] = final_det_score_mat
    return new_tracklet_mat, final_tracklet_mat

def post_processing(tracklet_mat, track_params): 
    new_tracklet_mat = tracklet_mat.copy()

    # update track cluster
    N_cluster = len(tracklet_mat["track_cluster"])
    remove_idx = []
    for n in range(N_cluster):
        if len(tracklet_mat["track_cluster"][n])==0:
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


    # assign tracklet
    missing_mat, final_tracklet_mat = update_tracklet_mat(new_tracklet_mat.copy())
    return new_tracklet_mat, missing_mat, final_tracklet_mat

def tracklet_clustering(track_struct, sess, batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, 
                        batch_mask_1, batch_mask_2, batch_Y, keep_prob, y_conv): 
    if 'track_interval' not in track_struct['tracklet_mat']: 
        track_struct = init_clustering(track_struct) 
        
    track_interval = track_struct['tracklet_mat']['track_interval'] 
    N_tracklet = track_interval.shape[0] 
    change_flag = 0 
    img_size = track_struct['track_params']['img_size']

    # sort tracklet in ascending order
    sort_idx = np.argsort(track_interval[:,1])
    for n in range(N_tracklet):
        track_id = sort_idx[n]
        if track_struct['tracklet_mat']['track_class'][track_id]<0:
            continue

        if track_id in remove_set:
            continue
            
        diff_cost = np.zeros((5,1))
        new_C = [] # new cost
        new_set = []
        change_idx = []

        cluster_cost = track_struct['tracklet_mat']['cluster_cost']
        track_class = track_struct['tracklet_mat']['track_class']

        # get split cost
        #import pdb; pdb.set_trace()
        diff_cost[0,0],temp_new_C,temp_new_set,temp_change_idx,comb_track_cost_list \
            = get_split_cost(track_struct['tracklet_mat'].copy(), track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get assign cost
        #import pdb; pdb.set_trace()
        diff_cost[1,0],temp_new_C,temp_new_set,temp_change_idx,comb_track_cost_list \
            = get_assign_cost(track_struct['tracklet_mat'].copy(), track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get merge cost
        diff_cost[2,0],temp_new_C,temp_new_set,temp_change_idx,comb_track_cost_list \
            = get_merge_cost(track_struct['tracklet_mat'].copy(), track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get switch cost
        diff_cost[3,0],temp_new_C,temp_new_set,temp_change_idx,comb_track_cost_list \
            = get_switch_cost(track_struct['tracklet_mat'].copy(), track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
        #track_struct['tracklet_mat']['save_fea_mat'] = save_fea_mat.copy()
        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get break cost
        diff_cost[4,0],temp_new_C,temp_new_set,temp_change_idx,comb_track_cost_list \
            = get_break_cost(track_struct['tracklet_mat'].copy(), track_id, sess, img_size, 
              batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
              batch_mask_2, batch_Y, keep_prob, y_conv)
        track_struct['tracklet_mat']['comb_track_cost'] = comb_track_cost_list.copy()
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
        if change_idx[min_idx][0]>=len(track_struct['tracklet_mat']['track_cluster']):
            for m in range(len(track_struct['tracklet_mat']['track_cluster']),change_idx[min_idx][0]):
                track_struct['tracklet_mat']['track_cluster'].append([])
            track_struct['tracklet_mat']['track_cluster'].append(new_set[min_idx][0])
        else:
            track_struct['tracklet_mat']['track_cluster'][change_idx[min_idx][0]] = new_set[min_idx][0]

        if change_idx[min_idx][1]>=len(track_struct['tracklet_mat']['track_cluster']):
            for m in range(len(track_struct['tracklet_mat']['track_cluster']),change_idx[min_idx][1]):
                track_struct['tracklet_mat']['track_cluster'].append([])
            track_struct['tracklet_mat']['track_cluster'].append(new_set[min_idx][1])  
        else:
            track_struct['tracklet_mat']['track_cluster'][change_idx[min_idx][1]] = new_set[min_idx][1]

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
    return track_struct, change_flag

def crop_det(tracklet_mat, crop_size, img_folder, crop_det_folder, flag): 
    N_tracklet = tracklet_mat['xmin_mat'].shape[0] 
    T = tracklet_mat['xmin_mat'].shape[1] 
    img_list = os.listdir(img_folder) 
    cnt = 0 
    for n in range(T): 
        track_ids = np.where(tracklet_mat['xmax_mat'][:,n]!=-1) 
        if len(track_ids)==0: 
            continue 
        track_ids = track_ids[0] 
        img_path = img_folder+'/'+img_list[n] 
        img = misc.imread(img_path) 
        img_size = img.shape 

        for m in range(len(track_ids)): 
            if flag==0: 
                xmin = int(max(0,tracklet_mat['xmin_mat'][track_ids[m],n])) 
                xmax = int(min(img.shape[1]-1,tracklet_mat['xmax_mat'][track_ids[m],n])) 
                ymin = int(max(0,tracklet_mat['ymin_mat'][track_ids[m],n])) 
                ymax = int(min(img.shape[0]-1,tracklet_mat['ymax_mat'][track_ids[m],n])) 
                img_patch = img[ymin:ymax,xmin:xmax,:] 
                img_patch = misc.imresize(img_patch, size=[crop_size,crop_size]) 
                class_name = file_name(track_ids[m]+1,4) 
                patch_name = class_name+'_'+file_name(n+1,4)+'.png' 
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

def draw_result(tracklet_mat, missing_mat, img_folder, save_folder): 
    img_list = os.listdir(img_folder) 
    table = color_table(len(tracklet_mat['track_cluster'])) 
    for n in range(len(img_list)): 
        img_path = img_folder+'/'+img_list[n] 
        img = misc.imread(img_path)

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(img)

        # Create Rectangle patches
        
        for k in range(tracklet_mat['xmin_mat'].shape[0]):
            track_class = tracklet_mat['track_class'][k,0]
            track_id = k
            if track_class==-1:
                continue
            if tracklet_mat['xmin_mat'][k,n]!=-1:
                xmin = tracklet_mat['xmin_mat'][k,n]
                ymin = tracklet_mat['ymin_mat'][k,n]
                xmax = tracklet_mat['xmax_mat'][k,n]
                ymax = tracklet_mat['ymax_mat'][k,n]
                w = xmax-xmin
                h = ymax-ymin
                rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='#'+table[track_class], facecolor='none')
                img_text = plt.text(xmin,ymin,str(track_class)+'_'+str(track_id),fontsize=6,color='#'+table[track_class])
                # Add the patch to the Axes
                ax.add_patch(rect)

        for k in range(missing_mat['xmin_mat'].shape[0]):       
            if missing_mat['xmin_mat'][k,n]!=-1:
                xmin = missing_mat['xmin_mat'][k,n]
                ymin = missing_mat['ymin_mat'][k,n]
                xmax = missing_mat['xmax_mat'][k,n]
                ymax = missing_mat['ymax_mat'][k,n]
                w = xmax-xmin
                h = ymax-ymin
                rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='#'+table[k], facecolor='none')
                img_text = plt.text(xmin,ymin,str(k)+'_',fontsize=6,color='#'+table[k])
                # Add the patch to the Axes
                ax.add_patch(rect)

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder+'/'+img_list[n]
        plt.savefig(save_path,bbox_inches='tight',dpi=400)
        #plt.show()
        #import pdb; pdb.set_trace()
        
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
        size = (width,height)
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
    
def TC_tracker(): 
    M = load_detection(det_path, 'MOT') 
    track_struct = {'track_params':{}} 
    track_struct['track_params']['num_fr'] = int(M[-1,0]-M[0,0]+1) 
    track_struct['track_params']['IOU_thresh'] = 0.5 
    track_struct['track_params']['color_thresh'] = 0.1 
    track_struct['track_params']['det_thresh'] = -1 
    track_struct['track_params']['linear_pred_thresh'] = 5 
    track_struct['track_params']['t_dist_thresh'] = 30 
    track_struct['track_params']['track_overlap_thresh'] = 0.1 
    track_struct['track_params']['search_radius'] = 2 
    track_struct['track_params']['const_fr_thresh'] = 5 
    track_struct['track_params']['crop_size'] = 182 
    track_struct['track_obj'] = {'track_id':[], 'bbox':[], 'det_score':[], 'mean_color':[]} 
    track_struct['tracklet_mat'] = {'xmin_mat':[], 'ymin_mat':[], 'xmax_mat':[], 'ymax_mat':[], 
                                    'det_score_mat':[]}

    img_list = os.listdir(img_folder)
    for n in range(track_struct['track_params']['num_fr']):
        img_path = img_folder+'/'+img_list[n] 
        img = misc.imread(img_path) 
        
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
        
        track_struct['track_obj']['track_id'].append([])
        if num_bbox==0:
            track_struct['track_obj']['bbox'].append([])
            track_struct['track_obj']['det_score'].append([])
            track_struct['track_obj']['mean_color'].append([])
        else:
            track_struct['track_obj']['bbox'].append(temp_M[:,1:5])
            track_struct['track_obj']['det_score'].append(temp_M[:,5])
            temp_mean_color = np.zeros((num_bbox,3))
            for k in range(num_bbox):
                xmin = int(max(0,temp_M[k,1]))
                ymin = int(max(0,temp_M[k,2]))
                xmax = int(min(img.shape[1]-1,temp_M[k,1]+M[k,3]))
                ymax = int(min(img.shape[0]-1,temp_M[k,2]+M[k,4]))
                temp_mean_color[k,0] = np.mean(img[ymin:ymax+1,xmin:xmax+1,0])
                temp_mean_color[k,1] = np.mean(img[ymin:ymax+1,xmin:xmax+1,1])
                temp_mean_color[k,2] = np.mean(img[ymin:ymax+1,xmin:xmax+1,2])
            temp_mean_color = temp_mean_color/255.0
            #import pdb; pdb.set_trace()
            track_struct['track_obj']['mean_color'].append(temp_mean_color.copy())
        #import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # forward tracking
    for n in range(track_struct['track_params']['num_fr']-1):
        #print(n)
        track_struct['tracklet_mat'], track_struct['track_obj']['track_id'][n], track_struct['track_obj']['track_id'][n+1] \
            = forward_tracking(track_struct['track_obj']['track_id'][n], track_struct['track_obj']['track_id'][n+1], 
                     track_struct['track_obj']['bbox'][n], track_struct['track_obj']['bbox'][n+1], 
                     track_struct['track_obj']['det_score'][n], track_struct['track_obj']['det_score'][n+1], 
                     track_struct['track_obj']['mean_color'][n], track_struct['track_obj']['mean_color'][n+1],
                     n+2, track_struct['track_params'], track_struct['tracklet_mat'])

    # tracklet clustering
    iters = 10
    track_struct['tracklet_mat'] = preprocessing(track_struct['tracklet_mat'], 3)
    #import pdb; pdb.set_trace()
    
    num_patch, img_size = crop_det(track_struct['tracklet_mat'], track_struct['track_params']['crop_size'], 
                               img_folder, crop_det_folder, 1)
    track_struct['tracklet_mat']['appearance_fea_mat'] = feature_extract(feature_size, num_patch, max_length, 
                                                                     crop_det_folder, triplet_model)
    #import pdb; pdb.set_trace()
    #*******************
    track_struct['tracklet_mat']['appearance_fea_mat'][:,2:] = 10*track_struct['tracklet_mat']['appearance_fea_mat'][:,2:]
    track_struct['track_params']['img_size'] = img_size

    #import pdb; pdb.set_trace()

    # load nn
    batch_X_x = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_y = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_w = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_h = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_X_a = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 1])
    batch_mask_1 = tf.placeholder(tf.float32, [None, 1, max_length, 1])
    batch_mask_2 = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 1])
    batch_Y = tf.placeholder(tf.int32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    y_conv = seq_nn.seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,
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
            track_struct, change_flag = tracklet_clustering(track_struct, sess,
                                                        batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1,
                                                        batch_mask_2, batch_Y, keep_prob, y_conv)
            if change_flag==0:
                #import pdb; pdb.set_trace()
                break
                
        #pickle.dump(save_fea_mat, open(save_fea_path, 'wb'))
        #pickle.dump(track_set, open(save_label_path,'wb'))
        #pickle.dump(remove_set, open(save_remove_path,'wb'))
        

    track_struct['tracklet_mat'], missing_mat, track_struct['final_tracklet_mat'] = post_processing(track_struct['tracklet_mat'].copy(), 
                                                      track_struct['track_params'].copy())

    draw_result(track_struct['tracklet_mat'], missing_mat, img_folder, tracking_img_folder)

    convert_frames_to_video(tracking_img_folder+'/', tracking_video_path, 20)
    
    #wrt_txt(track_struct['final_tracklet_mat'])

    #pickle.dump(track_struct, open(track_struct_path,'wb'))
    
    return track_struct
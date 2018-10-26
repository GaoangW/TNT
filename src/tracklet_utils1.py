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
import seq_nn

seq_name = 'MOT16-06'
img_name = 'MOT17-06'
sub_seq_name = '-02'
det_path = 'D:/Data/MOT/MOT16Labels/test/'+seq_name+'/det/det.txt'
img_folder = 'D:/Data/MOT/MOT17Det/test/'+img_name+sub_seq_name+'/img1'
crop_det_folder = 'D:/Data/MOT/crop_det/'+seq_name+sub_seq_name
triplet_model = 'D:/Data/UA-Detrac/UA_Detrac_model/MOT'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
seq_model = 'D:/Data/UA-Detrac/cnn_MOT/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/cnn_MOT_fine_tune_model_MOT_06/model.ckpt'
tracking_img_folder = 'D:/Data/MOT/tracking_img/'+seq_name+sub_seq_name
tracking_video_path = 'D:/Data/MOT/tracking_video/'+seq_name+sub_seq_name+'.avi'
'''
save_fea_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'.obj'
save_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_label.obj'
save_remove_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_remove_set.obj'
save_all_fea_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all.obj'
save_all_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label.obj'
'''
txt_result_path = 'D:/Data/MOT/txt_result/'+seq_name+sub_seq_name+'.txt'
track_struct_path = 'D:/Data/MOT/track_struct/'+seq_name+sub_seq_name+'.obj'


max_length = 64
feature_size = 4+512
batch_size = 64
num_classes = 2


track_set = []
remove_set = []

'''
track_set = pickle.load(open(save_label_path,'rb'))
remove_set = pickle.load(open(save_remove_path,'rb'))
'''

'''
remove_set = [28,32,24,33,42,46,65,66,62,76,88,91,87,85,102,53,35,37,48,70,43,78,82,98,86,93,107,114,117,132,126,105,131,137,
             156,151,148,161,145,134,129,170,167,176,182,184,179,181,193,105,153,140,231,232,234,215,212,257,274,285,219,327,
             326,266,264,233,302,340,432,408,417,430,431,425,394,412,396,358,329,259,202,289,299,273,229,240,327,314,301,340,
             350,344,385,375,394,466,468,480,477,500,516,502,460,470,480,478,444,433,425,412,449,498,552,553,537,549,629,665,
             682,561,570,592,577,568,619,627,683,598,774,831,837,953,965,1081,1226,1519,1569,1559,1645,1544,1485,1444,1518,
             1634,1681,1709,1700,1754,1957,2067,1995,2038,2094,2161,2192,2201,2225,2235,2272,2288,2323,2366,2425,2458,2495,
             2518,2610,2624,2626,2658,2685,2729,2764,2793,2824,2551,2618,2463,2569,2289,2210,2177,2180,2191,2199,2551,1879,
             1949,1994,2037,2031,2008,2123,2443,2620,3075,3084,3080,3073,3208,3215,3238,3239,3252,3290,3300,3310,3322,3326,
             3333,3337,3351,3369,3124,3141,3173,3260,3283,3360,3394,3420,3435,3480,3584,3629,3660,3730,3738,3727,3678,3657,
             3617,3582,3561,3549,3492,3445,3419,3369,3351,3322,3205,3562,3706,2618,2868,3417,3532,3562,3596,3718,3645]

track_set = np.array([[99,141,1],
                      [6,150,1],
                     [143,192,0],
                     [243,260,0],
                     [7,192,1],
                     [112,224,1],
                     [300,392,1],
                     [400,442,1],
                     [384,461,0],
                     [186,267,1],
                     [226,267,0],
                     [254,271,1],
                     [271,282,0],
                     [271,288,1],
                     [360,368,1],
                     [368,387,1],
                     [368,390,0],
                     [341,363,1],
                     [287,306,1],
                     [72,398,1],
                     [386,395,0],
                     [300,389,0],
                     [383,413,0],
                     [382,409,0],
                     [346,414,0],
                     [418,429,1],
                     [488,493,0],
                     [437,446,0],
                     [476,483,0],
                     [494,510,0],
                     [489,519,0],
                     [522,539,0],
                     [538,567,0],
                     [567,605,0],
                     [557,575,0],
                     [529,587,1],
                     [575,587,0],
                     [605,617,0],
                     [608,635,0],
                     [636,640,0],
                     [605,624,1],
                     [300,392,1],
                     [527,564,0],
                     [488,529,1],
                     [654,733,0],
                     [705,733,1],
                     [838,894,0],
                     [652,883,1],
                     [883,893,1],
                     [748,883,0],
                     [959,976,1],
                     [1013,1060,0],
                     [896,922,0],
                     [922,946,1],
                     [888,922,1],
                     [991,1049,0],
                     [1107,1151,0],
                     [1166,1208,0],
                     [1166,1198,1]
                     [980,1033,1],
                     [1213,1233,1],
                     [1233,1269,0],
                     [742,1176,1],
                     [1201,1221,1],
                     [1479,1507,1],
                     [1513,1529,1],
                     [1560,1575,0],
                     [556,1575,1],
                     [1429,1532,0],
                     [1380,1441,0],
                     [1399,1423,1],
                     [1540,1554,0],
                     [1423,1628,1],
                     [1624,1692,1],
                     [1722,1753,1],
                     [1778,1798,0],
                     [1387,1830,1],
                     [1876,1927,0],
                     [1850,1887,1],
                     [999,2024,1],
                     [2034,2060,0]
                     [185,2060,1],
                     [2060,2077,1],
                     [2100,2129,0],
                     [2507,2585,1],
                     [2585,2601,0],
                     [2583,2611,1],
                     [2611,2650,0],
                     [2169,2342,1],
                     [2249,2315,1],
                     [2128,2229,1],
                     [2237,2252,1],
                     [2145,2166,1],
                     [2144,2165,1],
                     [1876,1927,0],
                     [2051,2059,0],
                     [2096,2121,1],
                     [2091,2140,1],
                     [2091,2121,0],
                     [2140,2165,0],
                     [2145,2206,0],
                     [2258,2270,1],
                     [1941,2073,0],
                     [2016,2045,0],
                     [1904,2015,1],
                     [1921,1974,0],
                     [1855,1974,1],
                     [1958,2100,0],
                     [2100,2129,0],
                     [2126,2183,1],
                     [2040,2092,0],
                     [2260,2302,1],
                     [2183,2393,1],
                     [2407,2449,1],
                     [2507,2589,0],
                     [2702,2734,1],
                     [2816,2892,1],
                     [2892,2907,1],
                     [2907,2919,1],
                     [2929,2961,1],
                     [2961,3013,1],
                     [2961,3067,0],
                     [2734,3067,1],
                     [3241,3318,1],
                     [3303,3341,1],
                     [3341,3385,1],
                     [3385,3413,1],
                     [3413,3427,1],
                     [3413,3447,0],
                     [3366,3385,0],
                     [3251,3359,1],
                     [3427,3501,1],
                     [3640,3693,0],
                     [2583,2585,0],
                     [2717,2779,0],
                     [2758,2799,1],
                     [2799,2845,0],
                     [2828,2904,1],
                     [2977,2985,1],
                     [3021,3039,1],
                     [2828,2985,0],
                     [2860,3024,0],
                     [2884,2897,0],
                     [2874,2917,1],
                     [2971,3009,0],
                     [3048,3056,0],
                     [3157,3169,1],
                     [3157,3163,0],
                     [3186,3213,0],
                     [3198,3213,1],
                     [3213,3225,1],
                     [3220,3250,1],
                     [3250,3271,1],
                     [3105,3133,0],
                     [2971,3077,1],
                     [3105,3163,1],
                     [3180,3220,1],
                     [3235,3250,1],
                     [3250,3261,1],
                     [3250,3280,0],
                     [3217,3296,1],
                     [3217,3298,0],
                     [3357,3479,0],
                     [3505,3547,0],
                     [3479,3505,0],
                     [3521,3541,1],
                     [3541,3543,0],
                     [3669,3692,0],
                     [3639,3662,0],
                     [3662,3680,0],
                     [3675,3721,0],
                     [3355,3641,1],
                     [3573,3586,1]])

'''
save_fea_mat = np.zeros((len(track_set),feature_size,max_length,2))

global all_fea_mat
global all_fea_label
all_fea_mat = np.zeros((100,feature_size,max_length,2))
all_fea_label = np.zeros((100,2))


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
    tracklet_mat = track_struct['tracklet_mat']
    
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
    comb_fea_mat = np.zeros((len(sort_idx)-1,feature_size,max_length,2))
    comb_fea_label = np.zeros((len(sort_idx)-1,2))
    
    temp_cost_list = []
    #print(len(comb_track_cost))
    for n in range(0, len(sort_idx)-1):
        track_id1 = tracklet_set[sort_idx[n]]
        track_id2 = tracklet_set[sort_idx[n+1]]
        if tracklet_mat['comb_track_cost_mask'][track_id1,track_id2]==1:
            cost = cost+tracklet_mat['comb_track_cost'][track_id1,track_id2]
            remove_ids.append(n)
            continue
            
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
            comb_fea_mat[n,:,t1_min-t_min:t1_max-t_min+1,1] = 1
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

            comb_fea_mat[n,:,t2_min-t_min:t2_max-t_min+1,1] = 1
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

            comb_fea_mat[n,:,0:t1_max-t1_start+1,1] = comb_fea_mat[n,:,0:t1_max-t1_start+1,1]+1
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

            comb_fea_mat[n,:,t2_min-t1_start:t2_end-t1_start+1,1] = comb_fea_mat[n,:,t2_min-t1_start:t2_end-t1_start+1,1]+1
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
                if track_set[search_idx[0][0],2]==1:
                    comb_fea_label[n,0] = 1
                else:
                    comb_fea_label[n,1] = 1
    
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
            mask_1 = np.zeros((batch_size,1,max_length,1))
            mask_2 = np.zeros((batch_size,feature_size-4,max_length,1))
            x[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,0,:,0]
            y[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,1,:,0]
            w[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,2,:,0]
            h[:,0,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,3,:,0]
            ap[:,:,:,0] = comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,4:,:,0]
            mask_1[:,0,:,0] = 1-comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,0,:,1]
            mask_2[:,:,:,0] = 1-comb_fea_mat[n*max_batch_size:n*max_batch_size+batch_size,4:,:,1]
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
            if np.sum(comb_fea_label[n,:])==1:
                continue
            if pred_y[n,0]>pred_y[n,1]:
                comb_fea_label[n,0] = 1
            else:
                comb_fea_label[n,1] = 1
        '''
        all_fea_mat[fea_id:fea_id+len(pred_y),:,:,:] = comb_fea_mat
        all_fea_label[fea_id:fea_id+len(pred_y),:] = comb_fea_label
        '''
        
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
        
    #if len(remain_tracks)==9 and track_id==251:
    #    import pdb; pdb.set_trace()
    cost = np.sum(new_cluster_cost)  
    prev_cost = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]]
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

    cost_vec = temp_new_cluster_cost+new_cluster_cost[0,0] 
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
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    if len(cluster1)==1:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    N_cluster = len(tracklet_mat['track_cluster'])
    new_cluster_cost_vec = float("inf")*np.ones((N_cluster,1))
    prev_cost_vec = np.zeros((N_cluster,1))

    for n in range(N_cluster):
        # the original cluster
        if tracklet_mat['track_class'][track_id]==n:
            continue

        # check neighbor and conflict track
        cluster2 = tracklet_mat['track_cluster'][n].copy()
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
        new_cluster_cost_vec[n,0] = comb_cost(cluster1+cluster2, feature_size, 
                                                                max_length, img_size, sess, batch_X_x, batch_X_y, 
                                                                batch_X_w, batch_X_h, batch_X_a, batch_mask_1, 
                                                                batch_mask_2, batch_Y, keep_prob, y_conv)
        #tracklet_mat['comb_track_cost'] = comb_track_cost_list.copy()
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
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    diff_cost = diff_cost_vec[min_idx,0]
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
        cluster2 = tracklet_mat['track_cluster'][n].copy()
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

    diff_cost_vec = cost_vec-prev_cost_vec
    min_idx = np.argmin(diff_cost_vec)
    cost = cost_vec[min_idx,0]
    if cost==float("inf"):
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    diff_cost = diff_cost_vec[min_idx,0]
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
        
    # moving average
    for n in range(num_cluster):  
        cand_t = np.where(final_xmin_mat[n,:]!=-1)[0]
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


    # assign tracklet
    track_struct['final_tracklet_mat'] = update_tracklet_mat(new_tracklet_mat.copy())
    
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
    return change_flag

def crop_det(tracklet_mat, crop_size, img_folder, crop_det_folder, flag): 
    
    if not os.path.isdir(crop_det_folder): 
        os.makedirs(crop_det_folder)
    import pdb; pdb.set_trace()    
    N_tracklet = tracklet_mat['xmin_mat'].shape[0] 
    T = tracklet_mat['xmin_mat'].shape[1] 
    img_list = os.listdir(img_folder) 
    cnt = 0 
    for n in range(T): 
        track_ids = np.where(tracklet_mat['xmax_mat'][:,n]!=-1) 
        if len(track_ids)==0: 
            continue 
        track_ids = track_ids[0] 
        img_name = file_name(n+1,6)+'.jpg'
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

def draw_result(img_folder, save_folder): 
    
    global track_struct
    tracklet_mat = track_struct['final_tracklet_mat']
    img_list = os.listdir(img_folder) 
    table = color_table(len(tracklet_mat['track_cluster'])) 
    #import pdb; pdb.set_trace()
    for n in range(len(img_list)): 
        img_path = img_folder+'/'+img_list[n] 
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
        save_path = save_folder+'/'+img_list[n]
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
    
def TC_tracker(): 
    M = load_detection(det_path, 'MOT') 
    global track_struct
    track_struct = {'track_params':{}} 
    track_struct['track_params']['num_fr'] = int(M[-1,0]-M[0,0]+1) 
    track_struct['track_params']['IOU_thresh'] = 0.5 
    track_struct['track_params']['color_thresh'] = 0.12 
    track_struct['track_params']['det_thresh'] = -2 
    track_struct['track_params']['linear_pred_thresh'] = 5 
    track_struct['track_params']['t_dist_thresh'] = 30 
    track_struct['track_params']['track_overlap_thresh'] = 0.1 
    track_struct['track_params']['search_radius'] = 2
    track_struct['track_params']['const_fr_thresh'] = 1 
    track_struct['track_params']['crop_size'] = 182 
    track_struct['track_obj'] = {'track_id':[], 'bbox':[], 'det_score':[], 'mean_color':[]} 
    track_struct['tracklet_mat'] = {'xmin_mat':[], 'ymin_mat':[], 'xmax_mat':[], 'ymax_mat':[], 
                                    'det_score_mat':[]}

    img_list = os.listdir(img_folder)
    #track_struct['track_params']['num_fr'] = len(img_list)
    for n in range(track_struct['track_params']['num_fr']):
        
        
        # fr idx starts from 1
        fr_idx = n+1
        idx = np.where(np.logical_and(M[:,0]==fr_idx,M[:,5]>track_struct['track_params']['det_thresh']))[0]
        if len(idx)>1:
            choose_idx, _ = merge_bbox(M[idx,1:5], 0.7, M[idx,5])
            #import pdb; pdb.set_trace()
            temp_M = M[idx[choose_idx],:]
        else:
            temp_M = M[idx,:]
        num_bbox = len(temp_M)
        
        img_name = file_name(fr_idx,6)+'.jpg'
        if img_name in img_list:
            img_path = img_folder+'/'+img_name
            img = misc.imread(img_path) 
        else:
            num_bbox = 0
        
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
    init_num = 20000
    track_struct['tracklet_mat']['xmin_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['ymin_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['xmax_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['ymax_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['det_score_mat'] = -1*np.ones((init_num,track_struct['track_params']['num_fr']))
    
    max_id = 0
    for n in range(track_struct['track_params']['num_fr']-1):
        print(n)
        #print(max_id)
        track_struct['tracklet_mat'], track_struct['track_obj']['track_id'][n], track_struct['track_obj']['track_id'][n+1], max_id \
            = forward_tracking(track_struct['track_obj']['track_id'][n], track_struct['track_obj']['track_id'][n+1], 
                     track_struct['track_obj']['bbox'][n], track_struct['track_obj']['bbox'][n+1], 
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
    #import pdb; pdb.set_trace()
    
    # tracklet clustering
    iters = 10
    #track_struct['tracklet_mat'] = preprocessing(track_struct['tracklet_mat'], 3)
    
    '''
    # remove large bbox
    #import pdb; pdb.set_trace()
    for n in range(len(track_struct['tracklet_mat']['xmin_mat'])):
        cand_t = np.where(track_struct['tracklet_mat']['xmin_mat'][n,:]!=-1)[0]
        temp_h = track_struct['tracklet_mat']['ymax_mat'][n,cand_t]-track_struct['tracklet_mat']['ymin_mat'][n,cand_t]
        max_h = np.max(temp_h)
        if max_h>300:
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
            change_flag = tracklet_clustering(sess,
                                                        batch_X_x, batch_X_y, batch_X_w, batch_X_h, batch_X_a, batch_mask_1,
                                                        batch_mask_2, batch_Y, keep_prob, y_conv)
            if change_flag==0:
                #import pdb; pdb.set_trace()
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
            save_batch_size = 5000
            save_batch_num = int(np.ceil(len(all_fea_mat)/save_batch_size))
            for k in range(save_batch_num):
                if k!=save_batch_num-1:
                    temp_fea = all_fea_mat[k*save_batch_size:(k+1)*save_batch_size,:,:,:]
                    temp_label = all_fea_label[k*save_batch_size:(k+1)*save_batch_size,:]
                else:
                    temp_fea = all_fea_mat[k*save_batch_size:,:,:,:]
                    temp_label = all_fea_label[k*save_batch_size:,:]
                temp_fea_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/save_fea_mat/'+seq_name+'_all'+str(k)+'.obj'
                temp_label_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/save_fea_mat/'+seq_name+'_all_label'+str(k)+'.obj'
                pickle.dump(temp_fea, open(temp_fea_path,'wb'))
                pickle.dump(temp_label, open(temp_label_path,'wb'))
            '''

    post_processing()

    draw_result(img_folder, tracking_img_folder)
    
    wrt_txt(track_struct['final_tracklet_mat'])

    convert_frames_to_video(tracking_img_folder+'/', tracking_video_path, 20)

    pickle.dump(track_struct, open(track_struct_path,'wb'))
    
    return track_struct
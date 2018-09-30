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

def check_bbox_near_img_bnd(bbox, img_size, margin):
    # img_size = [x,y]
    xmin = bbox[0,0]
    ymin = bbox[0,1]
    xmax = bbox[0,2]+bbox[0,0]
    ymax = bbox[0,3]+bbox[0,1]
    
    check_flag = 0
    if xmin<margin or ymin<margin:
        check_flag = 1
        return check_flag
    if img_size[0]-xmax<margin or img_size[1]-ymax<margin:
        check_flag = 1
        return check_flag
    
    return check_flag

def pred_bbox_by_F(bbox, F, show_flag, img1, img2):
    #model, _, _, _, _ = estimateF(img1, img2)
    #F = model.params
    
    # Create figure and axes
    if show_flag==1:
        fig1,ax1 = plt.subplots(1)

    # Display the image
    if show_flag==1:
        ax1.imshow(img1)
    
    pred_bbox = np.zeros((len(bbox),4))
    for n in range(len(bbox)):
        xmin = bbox[n,0]
        ymin = bbox[n,1]
        xmax = bbox[n,2]+bbox[n,0]
        ymax = bbox[n,3]+bbox[n,1]
        w = bbox[n,2]
        h = bbox[n,3]
        if show_flag==1:
            rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='#FF0000', facecolor='none')
            ax1.add_patch(rect)
    
    if show_flag==1:
        plt.show()
        
    # Create figure and axes
    if show_flag==1:
        fig2,ax2 = plt.subplots(1)

    # Display the image
    if show_flag==1:
        ax2.imshow(img2)
    
    for n in range(len(bbox)):
        xmin = bbox[n,0]
        ymin = bbox[n,1]
        xmax = bbox[n,2]+bbox[n,0]
        ymax = bbox[n,3]+bbox[n,1]
        w = bbox[n,2]
        h = bbox[n,3]
        
        temp_A = np.zeros((4,2))
        temp_b = np.zeros((4,1));
        temp_pt = np.zeros((1,3))
        temp_pt[0,:] = np.array([xmin,ymin,1])
        A1 = np.matmul(temp_pt, np.transpose(F))
        #
        temp_A[0,0] = A1[0,0]
        temp_A[0,1] = A1[0,1]
        temp_b[0,0] = -A1[0,2]
        
        temp_pt[0,:] = np.array([xmax,ymin,1])
        A2 = np.matmul(temp_pt, np.transpose(F))
        temp_A[1,0] = A2[0,0]
        temp_A[1,1] = A2[0,1]
        temp_b[1,0] = -w*A2[0,0]-A2[0,2]
        
        temp_pt[0,:] = np.array([xmin,ymax,1])
        A3 = np.matmul(temp_pt, np.transpose(F))
        temp_A[2,0] = A3[0,0]
        temp_A[2,1] = A3[0,1]
        temp_b[2,0] = -h*A3[0,1]-A3[0,2]
        
        temp_pt[0,:] = np.array([xmax,ymax,1])
        A4 = np.matmul(temp_pt, np.transpose(F))
        temp_A[3,0] = A4[0,0]
        temp_A[3,1] = A4[0,1]
        temp_b[3,0] = -w*A4[0,0]-h*A4[0,1]-A4[0,2]
        
        new_loc = np.matmul(np.linalg.pinv(temp_A),temp_b)
        xmin = new_loc[0,0]
        ymin = new_loc[1,0]
        xmax = new_loc[0,0]+w
        ymax = new_loc[1,0]+h
        
        pred_bbox[n,0] = xmin
        pred_bbox[n,1] = ymin
        pred_bbox[n,2] = w
        pred_bbox[n,3] = h
        #import pdb; pdb.set_trace()
        if show_flag==1:
            rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='#FF0000', facecolor='none')
            ax2.add_patch(rect)
    
    if show_flag==1:
        plt.show()
    return pred_bbox    
    
def crop_bbox_in_image(bbox, img_size):
    new_bbox = bbox.copy()
    new_bbox[bbox[:,0]<0,0] = 0
    new_bbox[bbox[:,1]<0,1] = 0
    xmax = bbox[:,0]+bbox[:,2]-1
    ymax = bbox[:,1]+bbox[:,3]-1
    xmax[xmax>img_size[1]] = img_size[1]
    ymax[ymax>img_size[0]] = img_size[0]
    new_bbox[:,2] = xmax-new_bbox[:,0]+1
    new_bbox[:,3] = ymax-new_bbox[:,1]+1
    return new_bbox
    
def estimateF(img1, img2):
    
    np.random.seed(0)

    #img1, img2, groundtruth_disp = data.stereo_motorcycle()

    img1_gray, img2_gray = map(rgb2gray, (img1, img2))
    
    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(img1_gray)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2_gray)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                            cross_check=True)

    # Estimate the epipolar geometry between the left and right image.

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                         keypoints_right[matches[:, 1]]),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=5000)

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

    print("Number of matches:", matches.shape[0])
    print("Number of inliers:", inliers.sum())

    # Visualize the results.
    '''
    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], img1, img2, keypoints_left, keypoints_right,
                 matches[inliers], only_matches=True)
    ax[0].axis("off")
    ax[0].set_title("Inlier correspondences")
    
    plt.show()
    '''
    #import pdb; pdb.set_trace()
    
    return model, matches.shape[0], inliers.sum(), inlier_keypoints_left, inlier_keypoints_right

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
    
    if dataset=='KITTI_3d':
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        mask = np.zeros((len(f),1))
        for n in range(len(f)):
            # only for pedestrian
            #*******************
            if f[n][7]==4 or f[n][7]==5 or f[n][7]==6:
                mask[n,0] = 1
        num = int(np.sum(mask))
        
        M = np.zeros((num, 10))
        cnt = 0
        for n in range(len(f)):
            if mask[n,0]==1:
                M[cnt,0] = int(float(f[n][0]))
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
    
    if dataset=='MOT_tr':
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        M = np.zeros((f.shape[0], 6))
        M[:,0] = f[:,0]
        M[:,1:6] = f[:,2:7]
        #import pdb; pdb.set_trace()
        return M
    if dataset=='MOT_gt':
        # fr_id, x, y, w, h, obj_id, class_id
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        M = np.zeros((f.shape[0], 7))
        M[:,0] = f[:,0]
        M[:,1:5] = f[:,2:6]
        M[:,5] = f[:,1]
        M[:,6] = f[:,7]
        #import pdb; pdb.set_trace()
        return M
    if dataset=='MOT_1':
        # fr_id, x, y, w, h, det_score, svm_score, h_score, y_score, IOU_gt
        f = np.loadtxt(file_name, delimiter=',')
        f = np.array(f)
        M = np.zeros((f.shape[0], 10))
        M[:,0] = f[:,0]
        M[:,1:6] = f[:,2:7]
        M[:,6:10] = f[:,10:14]
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
                M[cnt,0] = int(float(f[n][0]))
                M[cnt,1] = int(float(f[n][1]))
                M[cnt,2] = int(float(f[n][2]))
                M[cnt,3] = int(float(f[n][3]))
                M[cnt,4] = int(float(f[n][4]))
                M[cnt,5] = float(f[n][10])/100.0
                M[cnt,6] = float(f[n][5])
                M[cnt,7] = float(f[n][7])
                M[cnt,8] = float(f[n][8])
                M[cnt,9] = float(f[n][9])
                cnt = cnt+1
            #import pdb; pdb.set_trace()
        return M
    
def bbox_associate(overlap_mat, IOU_thresh): 
    idx1 = [] 
    idx2 = [] 
    new_overlap_mat = overlap_mat.copy()
    while 1: 
        idx = np.unravel_index(np.argmax(new_overlap_mat, axis=None), new_overlap_mat.shape) 
        if new_overlap_mat[idx]<IOU_thresh: 
            break 
        else: 
            idx1.append(idx[0]) 
            idx2.append(idx[1]) 
            new_overlap_mat[idx[0],:] = 0 
            new_overlap_mat[:,idx[1]] = 0

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

def estimate_h_y(hloc, yloc):
    # h
    A = np.ones((hloc.shape[0],2))
    A[:,0] = yloc
    iters = 10
    W = np.identity(hloc.shape[0])
    for k in range(iters):
        A_w = np.matmul(W,A)
        b_w = np.matmul(W,hloc)
        ph = np.matmul(np.linalg.pinv(A_w),b_w)
        y_err = np.matmul(A,ph)-hloc
        err_std = np.std(y_err)
        w = np.exp(-np.power(y_err,2)/err_std*err_std)
        W = np.diag(w)
        
    # y
    A = np.ones((hloc.shape[0],2))
    A[:,0] = hloc
    iters = 10
    W = np.identity(hloc.shape[0])
    for k in range(iters):
        A_w = np.matmul(W,A)
        b_w = np.matmul(W,yloc)
        py = np.matmul(np.linalg.pinv(A_w),b_w)
        y_err = np.matmul(A,py)-yloc
        err_std = np.std(y_err)
        w = np.exp(-np.power(y_err,2)/err_std*err_std)
        W = np.diag(w)
    return ph, py

def extract_tracklet_feature(tracklet_mat, k, idx):
    tracklet_fea = np.zeros(17)
    tracklet_fea[0] = len(idx)
    tracklet_fea[1] = np.min(tracklet_mat['det_score_mat'][k,idx])
    tracklet_fea[2] = np.max(tracklet_mat['det_score_mat'][k,idx])
    tracklet_fea[3] = np.mean(tracklet_mat['det_score_mat'][k,idx])
    tracklet_fea[4] = np.std(tracklet_mat['det_score_mat'][k,idx])
    tracklet_fea[5] = np.min(tracklet_mat['svm_score_mat'][k,idx])
    tracklet_fea[6] = np.max(tracklet_mat['svm_score_mat'][k,idx])
    tracklet_fea[7] = np.mean(tracklet_mat['svm_score_mat'][k,idx])
    tracklet_fea[8] = np.std(tracklet_mat['svm_score_mat'][k,idx])
    tracklet_fea[9] = np.min(tracklet_mat['h_score_mat'][k,idx])
    tracklet_fea[10] = np.max(tracklet_mat['h_score_mat'][k,idx])
    tracklet_fea[11] = np.mean(tracklet_mat['h_score_mat'][k,idx])
    tracklet_fea[12] = np.std(tracklet_mat['h_score_mat'][k,idx])
    tracklet_fea[13] = np.min(tracklet_mat['y_score_mat'][k,idx])
    tracklet_fea[14] = np.max(tracklet_mat['y_score_mat'][k,idx])
    tracklet_fea[15] = np.mean(tracklet_mat['y_score_mat'][k,idx])
    tracklet_fea[16] = np.std(tracklet_mat['y_score_mat'][k,idx])
    return tracklet_fea
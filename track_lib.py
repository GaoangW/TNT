    
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
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, PairwiseKernel, DotProduct, RationalQuadratic
from sklearn.decomposition import SparseCoder

def tracklet_classify(A, pca, D, knn, clf_coding):
    encode_fea = np.zeros((len(A),len(D)))
    for n in range(len(A)):
        pca_fea = pca.transform(A[n])
        dist = distance.cdist(pca_fea, D, 'euclidean')
        x = np.zeros((len(pca_fea),len(D)))
        for k in range(len(dist)):
            sort_idx = np.argsort(dist[k,:])
            temp_D = D[sort_idx[0:knn],:]
            temp_coder = SparseCoder(dictionary=temp_D, transform_n_nonzero_coefs=10, 
                                 transform_alpha=0.05, transform_algorithm='lasso_lars')
            #import pdb; pdb.set_trace()
            xx = np.zeros((1,D.shape[1]))
            xx[:,:] = pca_fea[k,:]
            temp_x = temp_coder.transform(xx)
            x[k,sort_idx[0:knn]] = temp_x

        encode_fea[n,:] = np.max(x, axis=0)
    pred_set_label = clf_coding.predict(encode_fea)
    return pred_set_label

def interp_batch(total_batch_x):
    interp_batch_x = total_batch_x.copy()
    N_batch = total_batch_x.shape[0]
    for n in range(N_batch):
        temp_idx = np.where(total_batch_x[n,0,:,1]==1)[0]
        t1 = int(temp_idx[-1])
        temp_idx = np.where(total_batch_x[n,0,:,2]==1)[0]
        t2 = int(temp_idx[0])
        if t2-t1<=1:
            continue
        interp_t = np.array(range(t1+1,t2))
        for k in range(total_batch_x.shape[1]):
            #temp_std = np.std(total_batch_x[n,k,total_batch_x[n,k,:,0]!=0,0])
            
            temp_std1 = np.std(total_batch_x[n,k,total_batch_x[n,0,:,1]!=0,0])
            temp_std2 = np.std(total_batch_x[n,k,total_batch_x[n,0,:,2]!=0,0])
            
            x_p = [t1,t2]
            f_p = [total_batch_x[n,k,t1,0],total_batch_x[n,k,t2,0]]
            #interp_batch_x[n,k,t1+1:t2,0] = np.interp(interp_t,x_p,f_p)#+np.random.normal(0, temp_std, t2-t1-1)
            interp_batch_x[n,k,t1+1:t2,0] = np.interp(interp_t,x_p,f_p)+np.random.normal(0, (temp_std1+temp_std2)*0.5, t2-t1-1)
    return interp_batch_x

def GP_regression(tr_x,tr_y,test_x):
    A = np.ones((len(tr_x),2))
    A[:,0] = tr_x[:,0]
    p = np.matmul(np.linalg.pinv(A),tr_y)
    mean_tr_y = np.matmul(A,p)
    A = np.ones((len(test_x),2))
    A[:,0] = test_x[:,0]
    mean_test_y = np.matmul(A,p)
    kernel = ConstantKernel(100,(1e-5, 1e5))*RBF(1, (1e-5, 1e5))+RBF(1, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1, n_restarts_optimizer=9)
    gp.fit(tr_x, tr_y-mean_tr_y)
    test_y, sigma = gp.predict(test_x, return_std=True)
    test_y = test_y+mean_test_y
    #import pdb; pdb.set_trace()
    return test_y

def show_trajectory(tracklet_mat, obj_id):
    max_len = 60
    check_fr = np.where(tracklet_mat['xmin_mat'][obj_id,:]!=-1)[0]
    
    test_xmin = tracklet_mat['xmin_mat'][obj_id,:].copy()
    test_ymin = tracklet_mat['ymin_mat'][obj_id,:].copy()
    test_xmax = tracklet_mat['xmax_mat'][obj_id,:].copy()
    test_ymax = tracklet_mat['ymax_mat'][obj_id,:].copy()
    
    t1 = max(0,check_fr[0]-max_len)
    t2 = min(check_fr[-1]+max_len,tracklet_mat['xmin_mat'].shape[1]-1)
    test_t = np.concatenate((np.array(range(t1,check_fr[0])),np.array(range(check_fr[-1],t2))))
    test_t = test_t.astype(int)
    aa = np.zeros((len(check_fr),1))
    bb = np.zeros((len(check_fr),1))
    cc = np.zeros((len(test_t),1))
    aa[:,0] = check_fr
    cc[:,0] = test_t
    bb[:,0] = tracklet_mat['xmin_mat'][obj_id,check_fr]
    dd = GP_regression(aa,bb,cc)
    #import pdb; pdb.set_trace()
    test_xmin[test_t] = dd[:,0]
    bb[:,0] = tracklet_mat['ymin_mat'][obj_id,check_fr]
    dd = GP_regression(aa,bb,cc)
    test_ymin[test_t] = dd[:,0]
    bb[:,0] = tracklet_mat['xmax_mat'][obj_id,check_fr]
    dd = GP_regression(aa,bb,cc)
    test_xmax[test_t] = dd[:,0]
    bb[:,0] = tracklet_mat['ymax_mat'][obj_id,check_fr]
    dd = GP_regression(aa,bb,cc)
    test_ymax[test_t] = dd[:,0]
    
    t_range = np.where(test_xmin!=-1)[0]
    #if obj_id==2:
    #    import pdb; pdb.set_trace()
    plt.plot(t_range,test_xmin[t_range],'k.',t_range,test_ymin[t_range],'k.',
             t_range,test_xmax[t_range],'k.',t_range,test_ymax[t_range],'k.',
             check_fr,tracklet_mat['xmin_mat'][obj_id,check_fr],'b.',check_fr,tracklet_mat['ymin_mat'][obj_id,check_fr],'r.',
            check_fr,tracklet_mat['xmax_mat'][obj_id,check_fr],'g.',check_fr,tracklet_mat['ymax_mat'][obj_id,check_fr],'y.')
    plt.show()
    #import pdb; pdb.set_trace()
    #plt.close('all')
    return
    
def remove_det(det_M, det_thresh, y_thresh, h_thresh, y_thresh2, ratio_1, h_thresh2, y_thresh3, y_thresh4):
    
    remove_idx = []
    
    # remove low det score
    for n in range(len(det_M)):
        if det_M[n,-1]<det_thresh:
            remove_idx.append(n)
            
    # remove det upper the ground plane
    for n in range(len(det_M)):
        if det_M[n,2]<y_thresh:
            remove_idx.append(n)
            
    # remove det below the ground plane
    for n in range(len(det_M)):
        if det_M[n,2]>y_thresh2:
            remove_idx.append(n)
    
    # remove thin objects
    for n in range(len(det_M)):
        if (det_M[n,4]/det_M[n,3])>ratio_1:
            remove_idx.append(n)
            
    # remove small object
    for n in range(len(det_M)):
        if det_M[n,4]<h_thresh:
            remove_idx.append(n)
            
    # remove large object
    for n in range(len(det_M)):
        if det_M[n,4]>h_thresh2:
            remove_idx.append(n)
    
    # remove ymax
    for n in range(len(det_M)):
        if det_M[n,2]+det_M[n,4]>y_thresh3:
            remove_idx.append(n)
            
    # remove ymax
    for n in range(len(det_M)):
        if det_M[n,2]+det_M[n,4]<y_thresh4:
            remove_idx.append(n)
            
    remove_idx = np.array(list(set(remove_idx)),dtype=int)
    new_M = det_M.copy()
    new_M = np.delete(new_M,remove_idx,axis=0)
    return new_M

def track_extend(xmins, ymins, xmaxs, ymaxs, img_size, bnd_margin, min_len, extend_len, reg_thresh, speed_thresh, static_len, fr_id):
    # img_size = [x,y]
    
    extend_xmins = xmins.copy()
    extend_ymins = ymins.copy()
    extend_xmaxs = xmaxs.copy()
    extend_ymaxs = ymaxs.copy()
    N_fr = len(xmins)
    
    check_flag = 0
    fr_idx = np.where(xmins!=-1)[0]
    if len(fr_idx)==0:
        check_flag = 1
        return check_flag,extend_xmins,extend_ymins,extend_xmaxs,extend_ymaxs
    time_interval = [np.min(fr_idx),np.max(fr_idx)]
    if time_interval[1]-time_interval[0]+1<min_len:
        check_flag = 1
        return check_flag,extend_xmins,extend_ymins,extend_xmaxs,extend_ymaxs
    
    x_center = np.zeros((min_len,1))
    y_center = np.zeros((min_len,1))
    w = np.zeros((min_len,1))
    h = np.zeros((min_len,1))
    
    for drt in range(2):
        if drt==0:
            # start direction
            start_fr = time_interval[0]
            end_fr = time_interval[0]+min_len
            ext_fr = max(time_interval[0]-extend_len,0)
            
        else:
            # end direction
            if time_interval[1]>=N_fr-1:
                continue
            start_fr = time_interval[1]+1-min_len
            end_fr = time_interval[1]+1
            ext_fr = min(time_interval[1]+extend_len+1,N_fr-1)
            
        A = np.ones((min_len,2))
        A[:,0] = np.array(range(start_fr,end_fr))
        w[:,0] = xmaxs[start_fr:end_fr]-xmins[start_fr:end_fr]
        h[:,0] = ymaxs[start_fr:end_fr]-ymins[start_fr:end_fr]
        mean_w = 0
        mean_h = 0
        if drt==0:
            mean_w = np.mean(w[int(min_len/2):,0])
            mean_h = np.mean(h[int(min_len/2):,0])
        else:
            mean_w = np.mean(w[0:int(min_len/2),0])
            mean_h = np.mean(h[0:int(min_len/2),0])
            
        dist1 = (ymins[int((start_fr+end_fr)/2)]+ymaxs[int((start_fr+end_fr)/2)])/2
        dist2 = (xmins[int((start_fr+end_fr)/2)]+xmaxs[int((start_fr+end_fr)/2)])/2
        v_flag = 0 #top
        h_flag = 0 #left
        if dist1>img_size[1]/2:
            dist1 = img_size[1]-dist1
            v_flag = 1
        if dist2>img_size[0]/2:
            dist2 = img_size[0]-dist2
            h_flag = 1
        # top bnd
        if dist1<dist2 and v_flag==0:
            x_center[:,0] = (xmins[start_fr:end_fr]+xmaxs[start_fr:end_fr])/2
            y_center[:,0] = (ymaxs[start_fr:end_fr]-mean_h/2)
        # bot bnd
        elif dist1<dist2 and v_flag==1:
            x_center[:,0] = (xmins[start_fr:end_fr]+xmaxs[start_fr:end_fr])/2
            y_center[:,0] = (ymins[start_fr:end_fr]+mean_h/2)
        # left bnd
        elif dist1>=dist2 and h_flag==0:
            x_center[:,0] = (xmaxs[start_fr:end_fr]-mean_w/2)
            y_center[:,0] = (ymins[start_fr:end_fr]+ymaxs[start_fr:end_fr])/2
        # right bnd
        elif dist1>=dist2 and h_flag==1:
            x_center[:,0] = (xmins[start_fr:end_fr]+mean_w/2)
            y_center[:,0] = (ymins[start_fr:end_fr]+ymaxs[start_fr:end_fr])/2
            
        #x_center[:,0] = (xmins[start_fr:end_fr]+xmaxs[start_fr:end_fr])/2
        #y_center[:,0] = (ymins[start_fr:end_fr]+ymaxs[start_fr:end_fr])/2
        #if fr_id==10:
        #    import pdb; pdb.set_trace()
        px = np.matmul(np.linalg.pinv(A),x_center)
        err_x = np.sum(np.absolute(np.matmul(A,px)-x_center)/mean_w)/min_len
        if err_x>reg_thresh: # trajectory cannot be predicted
            continue
        py = np.matmul(np.linalg.pinv(A),y_center)
        err_y = np.sum(np.absolute(np.matmul(A,py)-y_center)/mean_h)/min_len
        if err_y>reg_thresh: # trajectory cannot be predicted
            continue
    
        # slow motion check
        static_flag = 0
        diff_x = abs((xmins[time_interval[1]]+xmaxs[time_interval[1]])/2-(xmins[time_interval[0]]+xmaxs[time_interval[0]])/2)
        diff_y = abs((ymins[time_interval[1]]+ymaxs[time_interval[1]])/2-(ymins[time_interval[0]]+ymaxs[time_interval[0]])/2)
        speed = np.sqrt(np.power(diff_x,2)+np.power(diff_y,2))/(time_interval[1]-time_interval[0]+1)
        if speed<speed_thresh and time_interval[1]-time_interval[0]>static_len: # static person
            static_flag = 1
    
        #ext_fr = max(time_interval[0]-extend_len,0)
        if static_flag==1:
            mean_x = np.mean(x_center[:,0])
            mean_y = np.mean(y_center[:,0])
            #mean_w = np.mean(w[:,0])
            #mean_h = np.mean(h[:,0])
            if drt==0:
                extend_xmins[0:time_interval[0]] = mean_x-mean_w/2
                extend_ymins[0:time_interval[0]] = mean_y-mean_h/2
                extend_xmaxs[0:time_interval[0]] = mean_x+mean_w/2
                extend_ymaxs[0:time_interval[0]] = mean_y+mean_h/2
            else:
                if time_interval[1]<N_fr-1:
                    extend_xmins[time_interval[1]+1:] = mean_x-mean_w/2
                    extend_ymins[time_interval[1]+1:] = mean_y-mean_h/2
                    extend_xmaxs[time_interval[1]+1:] = mean_x+mean_w/2
                    extend_ymaxs[time_interval[1]+1:] = mean_y+mean_h/2
                
        else:
            if drt==0:
                x0 = x_center[0,0]
                y0 = y_center[0,0]
                t1 = ext_fr
                t2 = time_interval[0]
            else:
                x0 = x_center[-1,0]
                y0 = y_center[-1,0]
                t1 = time_interval[1]+1
                t2 = ext_fr
            # check whether the track near img bnd
            if abs(x0)>bnd_margin and abs(img_size[0]-x0)>bnd_margin and abs(y0)>bnd_margin and abs(img_size[1]-y0)>bnd_margin:
                continue
        
            t_test = np.array(range(t1,t2))
            test_t = np.zeros((len(t_test),1))
            test_t[:,0] = t_test
            N_t = t2-t1
            if N_t==0:
                continue
            A = np.ones((N_t,2))
            A[:,0] = t_test
            
            tr_t = np.zeros((end_fr-start_fr,1))
            tr_t[:,0] = np.array(range(start_fr,end_fr))
            tr_x = np.zeros((end_fr-start_fr,1))
            tr_x[:,0] = xmins[start_fr:end_fr]
            tr_y = np.zeros((end_fr-start_fr,1))
            tr_y[:,0] = ymins[start_fr:end_fr]
            tr_w = np.zeros((end_fr-start_fr,1))
            tr_w[:,0] = xmaxs[start_fr:end_fr]-xmins[start_fr:end_fr]
            tr_h = np.zeros((end_fr-start_fr,1))
            tr_h[:,0] = ymaxs[start_fr:end_fr]-ymins[start_fr:end_fr]

            test_x = GP_regression(tr_t,tr_x,test_t)
            test_y = GP_regression(tr_t,tr_y,test_t)
            test_w = GP_regression(tr_t,tr_w,test_t)
            test_h = GP_regression(tr_t,tr_h,test_t)

            if drt==0:

                max_idx = np.where(np.logical_or(test_h<test_w,test_w<test_h/4))[0]
                if len(max_idx)!=0:
                    relative_t = int(np.max(max_idx))
                    max_t = int(np.max(max_idx)+t1)
                else:
                    relative_t = 0
                    max_t = int(t1)
                extend_xmins[max_t:time_interval[0]] = test_x[relative_t:,0]
                extend_ymins[max_t:time_interval[0]] = test_y[relative_t:,0]
                extend_xmaxs[max_t:time_interval[0]] = test_x[relative_t:,0]+test_w[relative_t:,0]
                extend_ymaxs[max_t:time_interval[0]] = test_y[relative_t:,0]+test_h[relative_t:,0]
            else:

                min_idx = np.where(np.logical_or(test_h<test_w,test_w<test_h/4))[0]
                if len(min_idx)!=0:
                    relative_t = int(np.min(min_idx))
                    min_t = int(np.min(min_idx)+t1)
                else:
                    min_t = int(t2)
                    relative_t = min_t-t1
                extend_xmins[t1:min_t] = test_x[0:relative_t,0]
                extend_ymins[t1:min_t] = test_y[0:relative_t,0]
                extend_xmaxs[t1:min_t] = test_x[0:relative_t,0]+test_w[0:relative_t,0]
                extend_ymaxs[t1:min_t] = test_y[0:relative_t,0]+test_h[0:relative_t,0]
                #if fr_id==10:
                #    import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    # constraint output inside image  
    neg_idx = np.where(extend_xmins==-1)[0]
    extend_xmins[extend_xmins<1] = 1
    extend_xmins[extend_xmins>img_size[0]-1] = img_size[0]-1
    extend_ymins[extend_ymins<1] = 1
    extend_ymins[extend_ymins>img_size[1]-1] = img_size[1]-1
    extend_xmaxs[extend_xmaxs<1] = 1
    extend_xmaxs[extend_xmaxs>img_size[0]-1] = img_size[0]-1
    extend_ymaxs[extend_ymaxs<1] = 1
    extend_ymaxs[extend_ymaxs>img_size[1]-1] = img_size[1]-1
    if len(neg_idx)!=0:
        extend_xmins[neg_idx] = -1
        extend_ymins[neg_idx] = -1
        extend_xmaxs[neg_idx] = -1
        extend_ymaxs[neg_idx] = -1
    
    
    neg_idx = np.where(np.logical_or(extend_ymaxs-extend_ymins<extend_xmaxs-extend_xmins,
                                  extend_xmaxs-extend_xmins<(extend_ymaxs-extend_ymins)/5))[0] 
    if len(neg_idx)!=0:
        extend_xmins[neg_idx] = -1
        extend_ymins[neg_idx] = -1
        extend_xmaxs[neg_idx] = -1
        extend_ymaxs[neg_idx] = -1
        
    extend_xmins[extend_ymaxs-extend_ymins<80] = -1
    extend_ymins[extend_ymaxs-extend_ymins<80] = -1
    extend_xmaxs[extend_ymaxs-extend_ymins<80] = -1
    extend_ymaxs[extend_ymaxs-extend_ymins<80] = -1
    #if len(neg_idx)==len(extend_ymaxs):
    #    import pdb; pdb.set_trace()
    return check_flag,extend_xmins,extend_ymins,extend_xmaxs,extend_ymaxs
    
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
    return ratio,overlap_area,area1,area2

def get_overlap(bbox1, bbox2): 
    num1 = bbox1.shape[0] 
    num2 = bbox2.shape[0] 
    overlap_mat = np.zeros((num1, num2)) 
    overlap_area = np.zeros((num1, num2)) 
    area1 = np.zeros(num1)
    area2 = np.zeros(num2)
    for n in range(num1): 
        for m in range(num2):

            #import pdb; pdb.set_trace()
            overlap_mat[n,m],overlap_area[n,m],area1[n],area2[m] = get_IOU(bbox1[n,:], bbox2[m,:])

    return overlap_mat,overlap_area,area1,area2

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
    if dataset=='YOLO':
        f = np.loadtxt(file_name, dtype=str, delimiter=',')
        f = np.array(f)
        M = np.zeros((f.shape[0], 6))
        cnt = 0
        for n in range(len(f)):
            M[cnt,0] = int(float(f[n][0]))+1
            M[cnt,1] = int(float(f[n][2]))
            M[cnt,2] = int(float(f[n][3]))
            M[cnt,3] = int(float(f[n][4]))
            M[cnt,4] = int(float(f[n][5]))
            M[cnt,5] = float(f[n][6])/100.0
            cnt = cnt+1
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
    if dataset=='chongqing':
        f = np.loadtxt(file_name, dtype=str, delimiter=',')
        f = np.array(f)
        num = len(f)
        M = np.zeros((num, 10))
        cnt = 0
        for n in range(len(f)):
            M[cnt,0] = int(float(f[n][0]))
            M[cnt,1] = int(float(f[n][2]))
            M[cnt,2] = int(float(f[n][3]))
            M[cnt,3] = int(float(f[n][4]))
            M[cnt,4] = int(float(f[n][5]))
            M[cnt,5] = float(f[n][6])/100
            M[cnt,6] = float(f[n][2])
            M[cnt,7] = float(f[n][3])
            M[cnt,8] = float(f[n][4])
            M[cnt,9] = float(f[n][5])
            cnt = cnt+1
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

def merge_bbox(bbox, IOU_thresh, det_score, merge_mode): 
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
            r,overlap_area,area1,area2 = get_overlap(a, b)
            r = r[0,0]
            overlap_area = overlap_area[0,0]
            r1 = overlap_area/area1[0]
            r2 = overlap_area/area2[0]
            s1 = det_score[n1]
            s2 = det_score[n2]
            if merge_mode==0:
                if r>IOU_thresh:
                    if s1>s2:
                        cand_idx[n2] = 0
                    else:
                        cand_idx[n1] = 0
            if merge_mode==1:
                if r1>IOU_thresh or r2>IOU_thresh:
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

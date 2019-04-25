    
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
from sklearn import svm
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import track_lib

train_seqs = ['MOT17-02-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN','MOT17-11-FRCNN','MOT17-13-FRCNN']
             #'MOT17-02-DPM','MOT17-04-DPM','MOT17-05-DPM','MOT17-09-DPM','MOT17-10-DPM','MOT17-11-DPM','MOT17-13-DPM',
        #['MOT17-02-SDP','MOT17-04-SDP','MOT17-05-SDP','MOT17-09-SDP','MOT17-10-SDP','MOT17-11-SDP','MOT17-13-SDP']
             
det_path = 'D:/Data/MOT/MOT17Labels/train'
gt_path = 'D:/Data/MOT/MOT17Labels/train'
cnn_fea_path = 'D:/Data/MOT/MOT17_train_det_crop'
save_cnn_svm_path = 'D:/Data/MOT/MOT17_train_det_crop/cnn_svm_MOT17.pkl'
save_det_path = 'D:/Data/MOT/MOT17Labels/train'
track_struct_path = 'D:/Data/MOT/track_struct'
save_classifier_path = 'D:/Data/MOT/MOT17_train_det_crop/rand_forest_MOT17_FRCNN.pkl'
F_set_path = 'D:/Data/MOT/geometry_info'
img_folder = 'D:/Data/MOT/MOT17Det/train'

fea_size = 512
img_size = [[1920,1080],[1920,1080],[1920,1080],[1920,1080],[1920,1080]]
           #[1920,1080],[1920,1080],[640,480],[1920,1080],[1920,1080],[1920,1080],[1920,1080],
           #[1920,1080],[1920,1080],[640,480],[1920,1080],[1920,1080],[1920,1080],[1920,1080]]
F_use_flag = [0,0,1,1,1]

def proj_bbox(M, F_set, max_fr_dist, img_size, img_list):
    max_rows = 1000000
    ext_M = np.zeros((max_rows,5))
    ext_M[0:len(M),:] = M[:,0:5]
    cnt = len(M)
    max_fr = int(np.max(M[:,0]))
    
    for n in range(len(M)):
        bbox = np.zeros((1,4))
        bbox[0,:] = M[n,1:5]
        fr_idx = int(M[n,0])
        
        # forward
        prev_bbox = bbox.copy()
        for k in range(fr_idx, min(max_fr,fr_idx+max_fr_dist)):
            #import pdb; pdb.set_trace()
            pred_bbox = track_lib.pred_bbox_by_F(prev_bbox, F_set[:,:,k-1], 0, [], [])
            check_flag = track_lib.check_bbox_near_img_bnd(pred_bbox, img_size, 10)
            if check_flag==1:
                break
                
            ext_M[cnt,0] = fr_idx+1
            ext_M[cnt,1:] = pred_bbox.copy()
            prev_bbox = pred_bbox.copy()
            cnt = cnt+1
        
        # backward
        prev_bbox = bbox.copy()
        if fr_idx==1:
            continue
        for k in range(fr_idx,max(1,fr_idx-max_fr_dist),-1):
            #img1 = misc.imread(img_list[k])
            #img2 = misc.imread(img_list[k-1])
            pred_bbox = track_lib.pred_bbox_by_F(prev_bbox, np.transpose(F_set[:,:,k-2]), 0, [], [])
            check_flag = track_lib.check_bbox_near_img_bnd(pred_bbox, img_size, 10)
            if check_flag==1:
                break
                
            ext_M[cnt,0] = fr_idx+1
            ext_M[cnt,1:] = pred_bbox.copy()
            prev_bbox = pred_bbox.copy()
            cnt = cnt+1
    
    remove_range = range(cnt,max_rows)
    np.delete(ext_M, np.array(remove_range), axis=0)
    return ext_M

def estimate_GP(bbox,err_sigma):
    # h = ax+by+c
    N_pt = 2*len(bbox)
    A = np.zeros((N_pt,3))
    b = np.zeros((N_pt,1))
    w = np.ones(N_pt)/N_pt
    
    for n in range(0,len(bbox)):
        xmin = bbox[n,0]
        xmax = bbox[n,0]+bbox[n,2]
        ymax = bbox[n,1]+bbox[n,3]
        h = bbox[n,3]
        A[2*n,0] = xmin
        A[2*n,1] = ymax
        A[2*n,2] = 1
        b[2*n,0] = h
        A[2*n+1,0] = xmax
        A[2*n+1,1] = ymax
        A[2*n+1,2] = 1
        b[2*n+1,0] = h
            
    iters = 20
    for k in range(iters):
        W = np.diag(w)
        p = np.matmul(np.linalg.pinv(np.matmul(W,A)),np.matmul(W,b))
        err_ratio = np.absolute(np.matmul(A,p)-b)/np.absolute(np.matmul(A,p))
        w = np.exp(-np.power(err_ratio[:,0],2)/np.power(err_sigma,2))
        ww = (w[::2]+w[1::2])/2
        w[::2] = ww
        w[1::2] = ww
        w = w/np.sum(w)
        #import pdb; pdb.set_trace()
    return p

def h_err_pred(p,bbox,err_sigma):
    x_center = (bbox[:,0]+bbox[:,2])/2
    ymax = bbox[:,1]+bbox[:,3]
    h = bbox[:,3]
    A = np.ones((len(bbox),3))
    A[:,0] = x_center
    A[:,1] = ymax
    h_pred = np.matmul(A,p)
    #import pdb; pdb.set_trace()
    err_ratio = np.absolute(h_pred[:,0]-h)/np.absolute(h_pred[:,0])
    err_ratio[h_pred[:,0]==0] = 0
    import pdb; pdb.set_trace()
    '''
    for n in range(len(h_pred)):
        if h_pred[n,0]==0:
            import pdb; pdb.set_trace()
    '''
    return err_ratio
    
def extract_classifier_features():
    tr_M = []
    gt_M = []
    cnn_fea_mat = []
    cnn_label = []
    num_det = []
    #loc_score_h = []
    #loc_score_y = []
    err_ratio_h = []
    det_IOU = []
    
    bnd_thresh = 10
    max_fr_dist = 10
    err_sigma = 1
    for n in range(len(train_seqs)):
        print(n)
        det_file_path = det_path+'/'+train_seqs[n]+'/det/det.txt'
        temp_det = track_lib.load_detection(det_file_path, 'MOT_tr')
        tr_M.append(temp_det)
        gt_file_path = gt_path+'/'+train_seqs[n]+'/gt/gt.txt'
        temp_gt = track_lib.load_detection(gt_file_path, 'MOT_gt')
        gt_M.append(temp_gt)
        cnn_fea_file_path = cnn_fea_path+'/'+train_seqs[n]+'.csv'
        f = np.loadtxt(cnn_fea_file_path, delimiter=',')
        f = np.array(f)
        cnn_fea_mat.append(f)
        num_det.append(f.shape[0])
        
        img_list = []
        for kk in range(int(np.max(temp_gt[:,0]))):
            img_path = img_folder+'/'+train_seqs[n][0:8]+'/img1/'+track_lib.file_name(kk+1,6)+'.jpg'
            img_list.append(img_path)
            
        if F_use_flag[n]==1:
            F_set_file_path = F_set_path+'/'+train_seqs[n][0:8]+'_F_set.mat'
            F_set = loadmat(F_set_file_path)
            F_set = F_set['F_set']
            
        loc_train_idx = np.zeros((f.shape[0],1),dtype=int)
        temp_label = np.zeros((f.shape[0],1))
        temp_IOU = np.zeros((f.shape[0],1))
        for k in range(temp_det.shape[0]):
            temp_det_bbox = np.zeros((1,4))
            temp_det_bbox[0,:] = temp_det[k,1:5]
            #import pdb; pdb.set_trace()
            if abs(temp_det_bbox[0,0])>bnd_thresh and abs(temp_det_bbox[0,1])>bnd_thresh and \
                abs(temp_det_bbox[0,0]+temp_det_bbox[0,2]-img_size[n][0])>bnd_thresh and \
                abs(temp_det_bbox[0,1]+temp_det_bbox[0,3]-img_size[n][1])>bnd_thresh:
                loc_train_idx[k,0] = 1
                
            choose_idx1 = list(np.where(np.logical_and(temp_gt[:,0]==temp_det[k,0],temp_gt[:,6]==1))[0])
            choose_idx2 = list(np.where(np.logical_and(temp_gt[:,0]==temp_det[k,0],temp_gt[:,6]==2))[0])
            choose_idx3 = list(np.where(np.logical_and(temp_gt[:,0]==temp_det[k,0],temp_gt[:,6]==7))[0])
            choose_idx = []
            choose_idx.extend(choose_idx1)
            choose_idx.extend(choose_idx2)
            choose_idx.extend(choose_idx3)
            choose_idx = np.array(choose_idx,dtype=int)
            choose_idx = np.unique(choose_idx)
            
            if len(choose_idx)==0:
                temp_label[k] = 0
                continue
            temp_gt_bbox = temp_gt[choose_idx,1:5]
            xmax = temp_gt_bbox[:,0].copy()+temp_gt_bbox[:,2].copy()
            ymax = temp_gt_bbox[:,1].copy()+temp_gt_bbox[:,3].copy()
            xmin = temp_gt_bbox[:,0].copy()
            xmin[xmin<0] = 0
            ymin = temp_gt_bbox[:,1].copy()
            ymin[ymin<0] = 0
            w = xmax-xmin
            h = ymax-ymin
            temp_gt_bbox[:,0] = xmin
            temp_gt_bbox[:,1] = ymin
            temp_gt_bbox[:,2] = w
            temp_gt_bbox[:,3] = h
            #import pdb; pdb.set_trace()
            overlap_mat = track_lib.get_overlap(temp_det_bbox, temp_gt_bbox)
            temp_IOU[k,0] = np.max(overlap_mat)
            if np.max(overlap_mat)>0.5:
                temp_label[k] = 1
            else:
                temp_label[k] = 0
        cnn_label.append(temp_label)
        det_IOU.append(temp_IOU)
        
        # train location
        M = temp_det[loc_train_idx[:,0]==1,:]
        err_ratio = np.zeros((len(temp_det),1))
        if F_use_flag[n]==1:
            ext_M = proj_bbox(M, F_set, max_fr_dist, img_size[n], img_list)
            for t in range(int(np.min(M[:,0])), int(np.max(M[:,0]))+1):
                temp_bbox = ext_M[ext_M[:,0]==t,1:5]
                GP_p = estimate_GP(temp_bbox,err_sigma)
                temp_det_bbox = temp_det[temp_det[:,0]==t,1:5]
                #import pdb; pdb.set_trace()
                err_ratio[temp_det[:,0]==t,0] = h_err_pred(GP_p,temp_det_bbox,err_sigma)
        else:
            GP_p = estimate_GP(M[:,1:5],err_sigma)
            #import pdb; pdb.set_trace()
            err_ratio[:,0] = h_err_pred(GP_p,temp_det[:,1:5],err_sigma)
        err_ratio_h.append(err_ratio)    
            
        '''
        # train location
        # h
        yloc = temp_det[loc_train_idx[:,0]==1,2]+temp_det[loc_train_idx[:,0]==1,4]
        hloc = temp_det[loc_train_idx[:,0]==1,4]
        ph, py = track_lib.estimate_h_y(hloc, yloc)
        
        A = np.ones((temp_label.shape[0],2))
        A[:,0] = temp_det[:,2]+temp_det[:,4]
        y_err = (np.matmul(A,ph)-temp_det[:,4])/temp_det[:,4]
        err_std = np.std(y_err)
        w = np.zeros((y_err.shape[0],1))
        w[:,0] = np.exp(-np.power(y_err,2)/(err_std*err_std))
        #import pdb; pdb.set_trace()
        loc_score_h.append(w)


        # y
        A = np.ones((temp_label.shape[0],2))
        A[:,0] = temp_det[:,4]
        y_err = np.matmul(A,py)-(temp_det[:,2]+temp_det[:,4])
        err_std = np.std(y_err)
        w = np.zeros((y_err.shape[0],1))
        w[:,0] = np.exp(-np.power(y_err,2)/(err_std*err_std))
        loc_score_y.append(w)
        '''
        
    total_num_det = int(np.sum(np.array(num_det)))
    tr_fea_mat = np.zeros((total_num_det,fea_size))
    tr_label = np.zeros((total_num_det,1))
    cnt = 0
    for n in range(len(train_seqs)):    
        tr_fea_mat[cnt:cnt+num_det[n],:] = cnn_fea_mat[n]
        tr_label[cnt:cnt+num_det[n],0] = cnn_label[n][:,0]
        cnt = cnt+num_det[n]
    
    # train svm for cnn features
    #import pdb; pdb.set_trace()
    clf = svm.LinearSVC()
    clf.fit(tr_fea_mat, tr_label[:,0])
    pred_label = clf.predict(tr_fea_mat)
    err = np.sum(np.absolute(pred_label-tr_label[:,0]))/tr_label.shape[0]
    #print(err)
    #import pdb; pdb.set_trace()
    pred_s = np.zeros((pred_label.shape[0],1))
    pred_s[:,0] = clf.decision_function(tr_fea_mat)
    joblib.dump(clf, save_cnn_svm_path) 
    
    # save det to file
    cnt = 0
    for n in range(len(train_seqs)):
        det_file_path = det_path+'/'+train_seqs[n]+'/det/det.txt'
        f = np.loadtxt(det_file_path, delimiter=',')
        f = np.array(f)
        #import pdb; pdb.set_trace()
        f = np.concatenate([f[:,0:10],pred_s[cnt:cnt+f.shape[0],:]],axis=1)
        #f = np.concatenate([f,loc_score_h[n]],axis=1)
        #f = np.concatenate([f,loc_score_y[n]],axis=1)
        f = np.concatenate([f,err_ratio_h[n]],axis=1)
        f = np.concatenate([f,det_IOU[n]],axis=1)
        #import pdb; pdb.set_trace()
        np.savetxt(det_file_path, f, delimiter=',')
        cnt = cnt+f.shape[0]
    return

def train_classifier():
    track_struct = []
    num_tracklet = []
    for n in range(len(train_seqs)):
        track_struct_file_path = track_struct_path+'/'+train_seqs[n]+'.obj'
        temp_track_struct = pickle.load(open(track_struct_file_path,'rb'))
        num_tracklet.append(temp_track_struct['tracklet_mat']['xmin_mat'].shape[0])
        track_struct.append(temp_track_struct)
    total_num_tracklet = np.sum(np.array(num_tracklet))
    tracklet_fea = np.zeros((total_num_tracklet,17))
    tracklet_label = np.zeros((total_num_tracklet,1))
    cnt = 0
    
    # t_duration, det_score, svm_score, h_score, y_score
    for n in range(len(train_seqs)):
        print(n)
        for k in range(num_tracklet[n]):
            #if n==2:
            #    import pdb; pdb.set_trace()
            idx = np.where(track_struct[n]['tracklet_mat']['xmin_mat'][k,:]!=-1)[0]
            tracklet_fea[cnt,:] = track_lib.extract_tracklet_feature(track_struct[n]['tracklet_mat'], k, idx)
            mean_IOU = np.mean(track_struct[n]['tracklet_mat']['IOU_gt_mat'][k,idx])
            if mean_IOU>0.5:
                tracklet_label[cnt,0] = 1
            cnt = cnt+1
    
    # train random forest
    iters = 1
    train_num = int(total_num_tracklet)
    avg_err = np.zeros(iters)
    for n in range(iters):
        clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0)
        #clf = svm.SVC()
    
        shuffle_idx = np.random.permutation(total_num_tracklet)
        train_fea = tracklet_fea[shuffle_idx[0:train_num],:]
        train_label = tracklet_label[shuffle_idx[0:train_num],0]
        clf.fit(train_fea, train_label)
        
        test_fea = tracklet_fea[shuffle_idx,:]
        test_label = tracklet_label[shuffle_idx,0]
        pred_label = clf.predict(test_fea)
        avg_err[n] = np.sum(np.absolute(pred_label-test_label))/test_label.shape[0]
    print(np.mean(avg_err))
    
    
    joblib.dump(clf, save_classifier_path)
    
    '''
    # train svm
    clf = svm.SVC()
    clf.fit(tracklet_fea, tracklet_label[:,0])
    pred_label = clf.predict(tracklet_fea)
    err = np.sum(np.absolute(pred_label-tracklet_label[:,0]))/tracklet_label.shape[0]
    print(err)
    '''

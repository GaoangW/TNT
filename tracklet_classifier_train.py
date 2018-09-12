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

train_seqs = ['MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13']
det_path = 'D:/Data/MOT/MOT16Labels/train'
gt_path = 'D:/Data/MOT/MOT16Labels/train'
cnn_fea_path = 'D:/Data/MOT/MOT16_train_det_crop'
save_cnn_svm_path = 'D:/Data/MOT/MOT16_train_det_crop/cnn_svm.pkl'
save_det_path = 'D:/Data/MOT/MOT16Labels/train'
track_struct_path = 'D:/Data/MOT/track_struct'
save_classifier_path = 'D:/Data/MOT/MOT16_train_det_crop/rand_forest.pkl'

fea_size = 512
img_size = [[1920,1080],[1920,1080],[640,480],[1920,1080],[1920,1080],[1920,1080],[1920,1080]]

def extract_classifier_features():
    tr_M = []
    gt_M = []
    cnn_fea_mat = []
    cnn_label = []
    num_det = []
    loc_score_h = []
    loc_score_y = []
    det_IOU = []
    bnd_thresh = 10
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
                
            choose_idx = np.where(np.logical_and(temp_gt[:,0]==temp_det[k,0],temp_gt[:,6]==1))[0]
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
        # h
        yloc = temp_det[loc_train_idx[:,0]==1,2]+temp_det[loc_train_idx[:,0]==1,4]
        hloc = temp_det[loc_train_idx[:,0]==1,4]
        ph, py = track_lib.estimate_h_y(hloc, yloc)
        
        A = np.ones((temp_label.shape[0],2))
        A[:,0] = temp_det[:,2]+temp_det[:,4]
        y_err = np.matmul(A,ph)-temp_det[:,4]
        err_std = np.std(y_err)
        w = np.zeros((y_err.shape[0],1))
        w[:,0] = np.exp(-np.power(y_err,2)/(err_std*err_std))
        #import pdb; pdb.set_trace()
        loc_score_h.append(w)
        '''
        for k in range(temp_label.shape[0]):   
            if temp_label[k]==1:
                plt.plot(A[k,0], temp_det[k,4], 'ro')
            else:
                plt.plot(A[k,0], temp_det[k,4], 'ko')
                
        line_x = np.array(range(1400))
        line_y = line_x*p[0]+p[1]
        loc_score.append(w)
        plt.plot(line_x, line_y, 'b.')
        plt.show()
        '''

        # y
        A = np.ones((temp_label.shape[0],2))
        A[:,0] = temp_det[:,4]
        y_err = np.matmul(A,py)-(temp_det[:,2]+temp_det[:,4])
        err_std = np.std(y_err)
        w = np.zeros((y_err.shape[0],1))
        w[:,0] = np.exp(-np.power(y_err,2)/(err_std*err_std))
        loc_score_y.append(w)
        '''
        for k in range(temp_label.shape[0]):   
            if temp_label[k]==1:
                plt.plot(A[k,0], temp_det[k,2]+temp_det[k,4], 'ro')
            else:
                plt.plot(A[k,0], temp_det[k,2]+temp_det[k,4], 'ko')
                
        line_x = np.array(range(1000))
        line_y = line_x*p[0]+p[1]
        loc_score.append(w)
        plt.plot(line_x, line_y, 'b.')
        plt.show()
        import pdb; pdb.set_trace()
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
        f = np.concatenate([f,loc_score_h[n]],axis=1)
        f = np.concatenate([f,loc_score_y[n]],axis=1)
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
        for k in range(num_tracklet[n]):
            idx = np.where(track_struct[n]['tracklet_mat']['xmin_mat'][k,:]!=-1)[0]
            tracklet_fea[cnt,:] = track_lib.extract_tracklet_feature(track_struct[n]['tracklet_mat'], k, idx)
            mean_IOU = np.mean(track_struct[n]['tracklet_mat']['IOU_gt_mat'][k,idx])
            if mean_IOU>0.5:
                tracklet_label[cnt,0] = 1
            cnt = cnt+1
    
    # train random forest
    clf = RandomForestClassifier(n_estimators=50, max_depth=30, random_state=0)
    clf.fit(tracklet_fea, tracklet_label[:,0])
    pred_label = clf.predict(tracklet_fea)
    err = np.sum(np.absolute(pred_label-tracklet_label[:,0]))/tracklet_label.shape[0]
    print(err)
    joblib.dump(clf, save_classifier_path)
    
    '''
    # train svm
    clf = svm.SVC()
    clf.fit(tracklet_fea, tracklet_label[:,0])
    pred_label = clf.predict(tracklet_fea)
    err = np.sum(np.absolute(pred_label-tracklet_label[:,0]))/tracklet_label.shape[0]
    print(err)
    '''
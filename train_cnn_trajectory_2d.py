    
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
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy import spatial
import matplotlib.pyplot as plt
import seq_nn_3d_v2
import random
import math
import scipy
import shutil


MAT_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/gt_mat'
img_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/MOT17Det/train'
temp_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/temp'
triplet_model = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/MOT_appearance'
save_dir = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/MOT_2d/model.ckpt'

bbox_size = 182
max_length = 64
feature_size = 4+512
batch_size = 32
num_classes = 2
margin = 0.15
sample_prob = [0.0852,0.1996,0.2550,0.0313,0.0854,0.1546,0.1890]
#sample_prob = np.ones(25)
#remove_file_idx = [7,23]
#sample_prob[remove_file_idx] = 0
lr = 1e-3


# In[3]:
def draw_traj(x,mask_1):
    fig, ax = plt.subplots()
    ax.plot(x,color=[0.5,0.5,0.5],marker='o',linestyle='None')
    t1 = np.where(mask_1[:,0]==1)[0]
    t2 = np.where(mask_1[:,1]==1)[0]
    ax.plot(t1,x[mask_1[:,0]==1],color=[0.2,0.6,0.86],marker='o',linestyle='None')
    ax.plot(t2,x[mask_1[:,1]==1],color=[0.18,0.8,0.44],marker='o',linestyle='None')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim(0,64)
    y_range = np.max(x)-np.min(x)
    plt.ylim(np.min(x)-y_range/50, np.max(x)+y_range/20)
    plt.show()
    
def draw_fea_map(x):
    fig, ax = plt.subplots()
    ax.imshow(np.power(np.transpose(x),0.2),cmap='gray')
    ax.set_aspect(0.1)
    plt.axis('off')
    #plt.xticks(range(8)) 
    plt.show()
                    
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
            #*************************************
            #interp_batch_x[n,k,t1+1:t2,0] = np.interp(interp_t,x_p,f_p)+np.random.normal(0, temp_std, t2-t1-1)
            #*************************************
            interp_batch_x[n,k,t1+1:t2,0] = np.interp(interp_t,x_p,f_p)+np.random.normal(0, (temp_std1+temp_std2)*0.5, t2-t1-1)
    return interp_batch_x
    
def num_str(num, length):
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

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, batch_size, distance_metric):
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
        
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
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
    
def split_track(X_2d,Y_2d,W_2d,H_2d,V_2d,img_size,obj_id,noise_scale,connect_thresh):
    
    err_flag = 0
    part_W_mat = W_2d[:,obj_id]
    #import pdb; pdb.set_trace()        
    non_zero_idx = np.where(part_W_mat>0)[0]
    if len(non_zero_idx)<=1 or np.max(non_zero_idx)-np.min(non_zero_idx)+1!=len(non_zero_idx):
        err_flag = 1
        return [], [], err_flag
                
    st_fr = np.min(non_zero_idx)
    end_fr = np.max(non_zero_idx)
            
    bbox_tracklet = []
    bbox_num = []
            
    v_flag = 1
    rand_num = random.uniform(0.0,1.0)
    if rand_num<0.5 or len(V_2d)==0:   # 0.5
        v_flag = 0
    
    
    #v_flag = 0
    
    
    for k in range(st_fr, end_fr+1):
        rand_num = np.zeros((1,4))
        for kk in range(4):
            while 1:
                rand_num[0,kk] = np.random.normal(0,0.05,size=1)[0]
                if abs(rand_num[0,kk])<noise_scale:
                    break
        x0 = max(0,X_2d[k,obj_id])
        x1 = min(img_size[0]-1,X_2d[k,obj_id]+W_2d[k,obj_id])
        y0 = max(0,Y_2d[k,obj_id])
        y1 = min(img_size[1]-1,Y_2d[k,obj_id]+H_2d[k,obj_id])
        w0 = max(1,x1-x0)
        h0 = max(1,y1-y0)
        
        xmin = max(0,x0+rand_num[0,0]*w0)
        xmin = min(img_size[0]-2,xmin)
        ymin = max(0,y0+rand_num[0,1]*h0)
        ymin = min(img_size[1]-2,ymin)
        ww = max(1,w0+rand_num[0,2]*w0)
        hh = max(1,h0+rand_num[0,3]*h0)
        xmax = min(xmin+ww,img_size[0]-1)
        ymax = min(ymin+hh,img_size[1]-1)
        ww = max(1,xmax-xmin+1)
        hh = max(1,ymax-ymin+1)
        #print(rand_num)
        temp_bbox = [int(k), int(xmin), int(ymin), int(xmax), int(ymax), float(V_2d[k,obj_id])]
        
        if k==st_fr:
            bbox = []
            
        if temp_bbox[-1]<0.5:
            if len(bbox)==0:
                continue
            else:
                bbox_tracklet.append(np.array(bbox))
                bbox = []
                continue
                
        rand_num = random.uniform(0.0,1.0)
        if v_flag==0:
            temp_connect = connect_thresh
        else:
            temp_connect = connect_thresh*math.sqrt(V_2d[k,obj_id])
                       
        if rand_num<temp_connect:
            bbox.append(temp_bbox)
            if k==end_fr and len(bbox)!=0:
                bbox_tracklet.append(np.array(bbox))
                #bbox_num.append(len(bbox))
        else:
            if len(bbox)!=0:
                bbox_tracklet.append(np.array(bbox))
                #bbox_num.append(len(bbox))
            bbox = []
            bbox.append(temp_bbox)
            
        '''    
        if k==st_fr:
            bbox = []
            bbox.append(temp_bbox)
        else:
            rand_num = random.uniform(0.0,1.0)
            if v_flag==0:
                temp_connect = connect_thresh
            else:
                temp_connect = connect_thresh*math.sqrt(V_2d[k,obj_id])
                       
            if rand_num<temp_connect:
                bbox.append(temp_bbox)
                if k==end_fr:
                    bbox_tracklet.append(np.array(bbox))
                    bbox_num.append(len(bbox))
            else:
                bbox_tracklet.append(np.array(bbox))
                bbox_num.append(len(bbox))
                bbox = []
                bbox.append(temp_bbox)
         '''
    #import pdb; pdb.set_trace()    
    t_interval = np.zeros((len(bbox_tracklet),2))
    for k in range(len(bbox_tracklet)):
        #print(bbox_tracklet[k])
        t_interval[k,0] = bbox_tracklet[k][0,0]
        t_interval[k,1] = bbox_tracklet[k][-1,0]
    return bbox_tracklet, t_interval, err_flag
    
def generate_data(feature_size, max_length, batch_size, MAT_folder, img_folder):

    noise_scale = 0.15    # 0.15
    #connect_thresh = 0.95
    connect_thresh = np.random.uniform(0.6,1)
    #sample_p = np.array([0.0852,0.1996,0.2550,0.0313,0.0854,0.1546,0.1890])
    sample_p = np.array(sample_prob)
    sample_p = list(sample_p/sum(sample_p))
    
    # load mat files
    Mat_paths = os.listdir(MAT_folder)
    choose_idx = np.random.choice(len(Mat_paths), size=batch_size, p=sample_p)
    Mat_files = []
    for n in range(batch_size):
        temp_path = MAT_folder+'/'+Mat_paths[choose_idx[n]]
        temp_mat_file = loadmat(temp_path)
        Mat_files.append(temp_mat_file)
    
    
    #X = np.zeros((batch_size,1,max_length,1+4+512))
    X = np.zeros((batch_size,feature_size,max_length,3))
    Y = np.zeros((batch_size,2))
    crop_bbox = []
    
    # positive 
    for n in range(int(batch_size/2)):
        #print(n)
        fr_num = Mat_files[n]['gtInfo'][0][0][0].shape[0]
        id_num = Mat_files[n]['gtInfo'][0][0][0].shape[1]
        Y[n,0] = 1
        
        X_2d = Mat_files[n]['gtInfo'][0][0][0]
        Y_2d = Mat_files[n]['gtInfo'][0][0][1]
        W_2d = Mat_files[n]['gtInfo'][0][0][2]
        H_2d = Mat_files[n]['gtInfo'][0][0][3]
        
        #########################################
        X_2d = X_2d-margin*W_2d
        Y_2d = Y_2d-margin*H_2d
        W_2d = (1+2*margin)*W_2d
        H_2d = (1+2*margin)*H_2d
        ##########################################
        
        
        if len(Mat_files[n]['gtInfo'][0][0])<=4:
            V_2d = []
        else:
            V_2d = Mat_files[n]['gtInfo'][0][0][4]
        if len(Mat_files[n]['gtInfo'][0][0])==6:
            img_size = Mat_files[n]['gtInfo'][0][0][5][0]
        else:
            img_size = [1920,1080]
        #V_2d = []
        
        #import pdb; pdb.set_trace()
        
        while 1:
            obj_id = np.random.randint(id_num, size=1)[0]
            
            bbox_tracklet, t_interval, err_flag = split_track(X_2d,Y_2d,W_2d,H_2d,V_2d,img_size,obj_id,noise_scale,connect_thresh)
            if err_flag==1:
                continue
                
            cand_pairs = []
            if len(bbox_tracklet)<=1:
                continue
            for k1 in range(len(bbox_tracklet)-1):
                for k2 in range(k1+1,len(bbox_tracklet)):
                    if t_interval[k1,0]+max_length>t_interval[k2,0]:
                        t_dist = t_interval[k2,0]-t_interval[k1,1]
                        cand_pairs.append([k1,k2,t_dist])
            if len(cand_pairs)==0:
                continue
            
            cand_pairs = np.array(cand_pairs)
            rand_num = np.random.rand(1)[0]
            #print(rand_num)
            if rand_num<0.7:
                select_p = np.exp(-np.power(cand_pairs[:,2],2)/100)
                select_p = select_p/sum(select_p)
                #print(select_p)
                pair_idx = np.random.choice(len(cand_pairs), size=1, p=select_p)[0]
            else:
                pair_idx = np.random.randint(len(cand_pairs), size=1)[0]
            select_pair = cand_pairs[pair_idx]
            select_pair = select_pair.astype(int)
            
            abs_fr_t1 = int(t_interval[select_pair[0],0])
            abs_fr_t2 = int(t_interval[select_pair[0],1])
            abs_fr_t3 = int(t_interval[select_pair[1],0])
            abs_fr_t4 = int(min(abs_fr_t1+max_length-1,t_interval[select_pair[1],1]))
            
            t1 = 0
            t2 = abs_fr_t2-abs_fr_t1
            t3 = abs_fr_t3-abs_fr_t1
            t4 = abs_fr_t4-abs_fr_t1
            
            # mask
            X[n,:,t1:t2+1,1] = 1
            X[n,:,t3:t4+1,2] = 1
            #X[n,4:,t1:t2+1,1] = bbox_tracklet[select_pair[0]][:,5]
            #X[n,4:,t3:t4+1,2] = bbox_tracklet[select_pair[1]][0:t4-t3+1,5]
            #import pdb; pdb.set_trace() 
            
            # X    
            X[n,0,t1:t2+1,0] = 0.5*(bbox_tracklet[select_pair[0]][:,1]+bbox_tracklet[select_pair[0]][:,3])/img_size[0]
            X[n,0,t3:t4+1,0] = 0.5*(bbox_tracklet[select_pair[1]][0:t4-t3+1,1]+bbox_tracklet[select_pair[1]][0:t4-t3+1,3])/img_size[0]
            
            # Y
            X[n,1,t1:t2+1,0] = 0.5*(bbox_tracklet[select_pair[0]][:,2]+bbox_tracklet[select_pair[0]][:,4])/img_size[1]
            X[n,1,t3:t4+1,0] = 0.5*(bbox_tracklet[select_pair[1]][0:t4-t3+1,2]+bbox_tracklet[select_pair[1]][0:t4-t3+1,4])/img_size[1]
            
            # W
            X[n,2,t1:t2+1,0] = (bbox_tracklet[select_pair[0]][:,3]-bbox_tracklet[select_pair[0]][:,1])/img_size[0]
            X[n,2,t3:t4+1,0] = (bbox_tracklet[select_pair[1]][0:t4-t3+1,3]-bbox_tracklet[select_pair[1]][0:t4-t3+1,1])/img_size[0]
            
            # H
            X[n,3,t1:t2+1,0] = (bbox_tracklet[select_pair[0]][:,4]-bbox_tracklet[select_pair[0]][:,2])/img_size[1]
            X[n,3,t3:t4+1,0] = (bbox_tracklet[select_pair[1]][0:t4-t3+1,4]-bbox_tracklet[select_pair[1]][0:t4-t3+1,2])/img_size[1]
            '''       
            plt.plot(X[n,0,:,0], 'ro')
            plt.show()
            plt.plot(X[n,1,:,0], 'ro')
            plt.show()
            plt.plot(X[n,2,:,0], 'ro')
            plt.show()
            plt.plot(X[n,3,:,0], 'ro')
            plt.show()
            plt.plot(X[n,0,:,1], 'ro')
            plt.show()
            plt.plot(X[n,0,:,2], 'ro')
            plt.show()    
            import pdb; pdb.set_trace() 
            '''
            
            # save all bbox
            temp_crop_bbox = np.concatenate((bbox_tracklet[select_pair[0]],bbox_tracklet[select_pair[1]][0:t4-t3+1,:]), axis=0)
            temp_crop_bbox = temp_crop_bbox.astype(int)
            crop_bbox.append(temp_crop_bbox)
            break
  
    #import pdb; pdb.set_trace() 
    
    # negative
    for n in range(int(batch_size/2),batch_size):
        fr_num = Mat_files[n]['gtInfo'][0][0][0].shape[0]
        id_num = Mat_files[n]['gtInfo'][0][0][0].shape[1]
        Y[n,1] = 1
        
        X_2d = Mat_files[n]['gtInfo'][0][0][0]
        Y_2d = Mat_files[n]['gtInfo'][0][0][1]
        W_2d = Mat_files[n]['gtInfo'][0][0][2]
        H_2d = Mat_files[n]['gtInfo'][0][0][3]
        
        #########################################
        X_2d = X_2d-margin*W_2d
        Y_2d = Y_2d-margin*H_2d
        W_2d = (1+2*margin)*W_2d
        H_2d = (1+2*margin)*H_2d
        ##########################################
        
        if len(Mat_files[n]['gtInfo'][0][0])<=4:
            V_2d = []
        else:
            V_2d = Mat_files[n]['gtInfo'][0][0][4]
        if len(Mat_files[n]['gtInfo'][0][0])==6:
            img_size = Mat_files[n]['gtInfo'][0][0][5][0]
        else:
            img_size = [1920,1080]
        #V_2d = []
        #img_size = [1920,1080]
    
        # check candidate obj pairs
        #pair_mat = np.zeros((id_num,id_num))
        cand_idx_pairs = []
        for n1 in range(id_num-1):
            for n2 in range(n1+1,id_num):
                cand_fr1 = np.where(W_2d[:,n1]>0)[0]
                cand_fr2 = np.where(W_2d[:,n2]>0)[0]
                if max(cand_fr1[0],cand_fr2[0])<min(cand_fr1[-1],cand_fr2[-1]):
                    cand_idx_pairs.append([n1,n2])
                    #pair_mat[n1,n2] = 1
        
        #cand_pairs = np.nonzero(pair_mat)        
        while 1:
            #
            if len(cand_idx_pairs)==0:
                import pdb; pdb.set_trace() 
            pair_idx = np.random.randint(len(cand_idx_pairs), size=1)[0]
            obj_id1 = cand_idx_pairs[pair_idx][0]
            obj_id2 = cand_idx_pairs[pair_idx][1]
            #import pdb; pdb.set_trace() 
            
            part_W_mat1 = W_2d[:,obj_id1]        
            non_zero_idx1 = np.where(part_W_mat1>0)[0]
            part_W_mat2 = W_2d[:,obj_id2]        
            non_zero_idx2 = np.where(part_W_mat2>0)[0]
            if len(non_zero_idx1)==0 or len(non_zero_idx2)==0 or \
            max(non_zero_idx1)+max_length<min(non_zero_idx2) or min(non_zero_idx1)>max(non_zero_idx2):
                continue
                
            bbox_tracklet1, t_interval1, err_flag = split_track(X_2d,Y_2d,W_2d,H_2d,V_2d,img_size,obj_id1,noise_scale,connect_thresh)
            if err_flag==1:
                continue
            bbox_tracklet2, t_interval2, err_flag = split_track(X_2d,Y_2d,W_2d,H_2d,V_2d,img_size,obj_id2,noise_scale,connect_thresh)
            if err_flag==1:
                continue
            
            cand_pairs = []
            if len(bbox_tracklet1)<=1 or len(bbox_tracklet2)<=1:
                continue
            for k1 in range(len(bbox_tracklet1)):
                for k2 in range(len(bbox_tracklet2)):
                    if t_interval1[k1,0]+max_length>t_interval2[k2,0] and t_interval1[k1,1]<t_interval2[k2,0]:
                        t_dist = t_interval2[k2,0]-t_interval1[k1,1]
                        cand_pairs.append([k1,k2,t_dist])
            if len(cand_pairs)==0:
                continue
    
            cand_pairs = np.array(cand_pairs)
            rand_num = np.random.rand(1)[0]
            #print(rand_num)
            if rand_num<0.7:
                select_p = np.exp(-np.power(cand_pairs[:,2],2)/100)
                select_p = select_p/sum(select_p)
                #print(select_p)
                pair_idx = np.random.choice(len(cand_pairs), size=1, p=select_p)[0]
            else:
                pair_idx = np.random.randint(len(cand_pairs), size=1)[0]
            select_pair = cand_pairs[pair_idx]
            select_pair = select_pair.astype(int)
            
            abs_fr_t1 = int(t_interval1[select_pair[0],0])
            abs_fr_t2 = int(t_interval1[select_pair[0],1])
            abs_fr_t3 = int(t_interval2[select_pair[1],0])
            abs_fr_t4 = int(min(abs_fr_t1+max_length-1,t_interval2[select_pair[1],1]))
    
            t1 = 0
            t2 = abs_fr_t2-abs_fr_t1
            t3 = abs_fr_t3-abs_fr_t1
            t4 = abs_fr_t4-abs_fr_t1
            
            # mask
            X[n,:,t1:t2+1,1] = 1
            X[n,:,t3:t4+1,2] = 1
            #X[n,4:,t1:t2+1,1] = bbox_tracklet1[select_pair[0]][:,5]
            #X[n,4:,t3:t4+1,2] = bbox_tracklet2[select_pair[1]][0:t4-t3+1,5]
            
            # X    
            X[n,0,t1:t2+1,0] = 0.5*(bbox_tracklet1[select_pair[0]][:,1]+bbox_tracklet1[select_pair[0]][:,3])/img_size[0]
            X[n,0,t3:t4+1,0] = 0.5*(bbox_tracklet2[select_pair[1]][0:t4-t3+1,1]+bbox_tracklet2[select_pair[1]][0:t4-t3+1,3])/img_size[0]
            
            # Y
            X[n,1,t1:t2+1,0] = 0.5*(bbox_tracklet1[select_pair[0]][:,2]+bbox_tracklet1[select_pair[0]][:,4])/img_size[1]
            X[n,1,t3:t4+1,0] = 0.5*(bbox_tracklet2[select_pair[1]][0:t4-t3+1,2]+bbox_tracklet2[select_pair[1]][0:t4-t3+1,4])/img_size[1]
            
            # W
            X[n,2,t1:t2+1,0] = (bbox_tracklet1[select_pair[0]][:,3]-bbox_tracklet1[select_pair[0]][:,1])/img_size[0]
            X[n,2,t3:t4+1,0] = (bbox_tracklet2[select_pair[1]][0:t4-t3+1,3]-bbox_tracklet2[select_pair[1]][0:t4-t3+1,1])/img_size[0]
            
            # H
            X[n,3,t1:t2+1,0] = (bbox_tracklet1[select_pair[0]][:,4]-bbox_tracklet1[select_pair[0]][:,2])/img_size[1]
            X[n,3,t3:t4+1,0] = (bbox_tracklet2[select_pair[1]][0:t4-t3+1,4]-bbox_tracklet2[select_pair[1]][0:t4-t3+1,2])/img_size[1]
            '''
            plt.plot(X[n,0,:,0], 'ro')
            plt.show()
            plt.plot(X[n,1,:,0], 'ro')
            plt.show()
            plt.plot(X[n,2,:,0], 'ro')
            plt.show()
            plt.plot(X[n,3,:,0], 'ro')
            plt.show()
            plt.plot(X[n,0,:,1], 'ro')
            plt.show()
            plt.plot(X[n,0,:,2], 'ro')
            plt.show()    
            import pdb; pdb.set_trace() 
            '''
            # save all bbox
            temp_crop_bbox = np.concatenate((bbox_tracklet1[select_pair[0]],bbox_tracklet2[select_pair[1]][0:t4-t3+1,:]), axis=0)
            temp_crop_bbox = temp_crop_bbox.astype(int)
            crop_bbox.append(temp_crop_bbox)
            break
            
    # crop data to a temp folder
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    all_paths = []
    for n in range(batch_size):
        temp_all_path = []
        seq_name = Mat_paths[choose_idx[n]][:-4]
        img_path = img_folder+'/'+seq_name+'/img1/'
        #img_path = img_folder+'/'+seq_name+'/'
        track_name = file_name(n+1,4)
        save_path = temp_folder+'/'+track_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for k in range(len(crop_bbox[n])):
            fr_id = crop_bbox[n][k,0]+1
            temp_img_path = img_path+file_name(fr_id,6)+'.jpg'
            img = scipy.ndimage.imread(temp_img_path)
            bbox_img = img[crop_bbox[n][k,2]:crop_bbox[n][k,4],crop_bbox[n][k,1]:crop_bbox[n][k,3],:]
            #import pdb; pdb.set_trace() 
            bbox_img = scipy.misc.imresize(bbox_img, size=(bbox_size,bbox_size))
            bbox_img_path = save_path+'/'+file_name(k,4)+'.png'
            temp_all_path.append(bbox_img_path)
            scipy.misc.imsave(bbox_img_path,bbox_img)
        all_paths.append(temp_all_path)
    #import pdb; pdb.set_trace() 
    
    
    f_image_size = 160
    distance_metric = 0
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            

            #import pdb; pdb.set_trace()
            
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
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(triplet_model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#              
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            
            for n in range(len(all_paths)):
                #print(n)
                lfw_batch_size = len(all_paths[n])
                
                emb_array = evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                         batch_size_placeholder, control_placeholder, embeddings, label_batch, all_paths[n], lfw_batch_size, distance_metric)
                
                if X[n,4:,X[n,0,:,1]+X[n,0,:,2]>0.5,0].shape[0]!=emb_array.shape[0]:
                    aa = 0
                    import pdb; pdb.set_trace()  
                    
                X[n,4:,X[n,0,:,1]+X[n,0,:,2]>0.5,0] = emb_array
              
    #import pdb; pdb.set_trace()
    return X, Y

# In[4]:
batch_X_x = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_y = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_w = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_h = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_a = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 1])
batch_mask_1 = tf.placeholder(tf.float32, [None, 1, max_length, 2])
batch_mask_2 = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 2])
batch_Y = tf.placeholder(tf.int32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

y_conv = seq_nn_3d_v2.seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,batch_mask_2,batch_Y,max_length,feature_size,keep_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_Y, logits=y_conv))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(batch_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
 
    
    if os.path.isfile(save_dir+'.meta')==True:
        saver.restore(sess, save_dir)
        print("Model restored.")

    cnt = 0
    for i in range(2000000):
        total_batch_x, total_batch_y = generate_data(feature_size, max_length, batch_size*10, MAT_folder, img_folder)
        total_batch_x = interp_batch(total_batch_x)

        # delete temp folder
        shutil.rmtree(temp_folder)
        
        remove_idx = []
        for k in range(len(total_batch_x)):
            if np.sum(total_batch_x[k,0,:,1])==0:
                remove_idx.append(k)     
        total_batch_x = np.delete(total_batch_x, np.array(remove_idx), axis=0)
        total_batch_y = np.delete(total_batch_y, np.array(remove_idx), axis=0) 
        print(len(total_batch_y))
        
        total_batch_x[:,4:,:,0] = 10*total_batch_x[:,4:,:,0]
        temp_X = np.copy(total_batch_x)
        temp_Y = np.copy(total_batch_y)
        idx = np.arange(total_batch_x.shape[0])
        np.random.shuffle(idx)
        for k in range(len(idx)):
            total_batch_x[idx[k],:,:,:] = temp_X[k,:,:,:]
            total_batch_y[idx[k],:] = temp_Y[k,:]
        num_batch = int(np.ceil(len(total_batch_y)/batch_size))
        
        # shuffle 4 times
        acc = []
        acc2 = []
        for kk in range(num_batch):
            temp_batch_size = batch_size
            if kk==num_batch-1:
                temp_batch_size = len(total_batch_y)-batch_size*(num_batch-1)
            
            cnt = cnt+1
            batch_x = total_batch_x[kk*batch_size:kk*batch_size+temp_batch_size,:,:,:]
            batch_y = total_batch_y[kk*batch_size:kk*batch_size+temp_batch_size,:]
            
            x = np.zeros((temp_batch_size,1,max_length,1))
            y = np.zeros((temp_batch_size,1,max_length,1))
            w = np.zeros((temp_batch_size,1,max_length,1))
            h = np.zeros((temp_batch_size,1,max_length,1))
            ap = np.zeros((temp_batch_size,feature_size-4,max_length,1))
            mask_1 = np.zeros((temp_batch_size,1,max_length,2))
            mask_2 = np.zeros((temp_batch_size,feature_size-4,max_length,2))

            x[:,0,:,0] = batch_x[:,0,:,0]
            y[:,0,:,0] = batch_x[:,1,:,0]
            w[:,0,:,0] = batch_x[:,2,:,0]
            h[:,0,:,0] = batch_x[:,3,:,0]

            ap[:,:,:,0] = batch_x[:,4:,:,0]

            mask_1[:,0,:,:] = batch_x[:,0,:,1:]
            mask_2[:,:,:,:] = batch_x[:,4:,:,1:]
            
            if cnt % 1 == 0:

                temp_c = 0
                while 1:
                    y_pred = sess.run(y_conv,feed_dict={batch_X_x: x,
                                                          batch_X_y: y,
                                                          batch_X_w: w,
                                                          batch_X_h: h,
                                                          batch_X_a: ap,
                                                          batch_mask_1: mask_1,
                                                          batch_mask_2: mask_2,
                                                          batch_Y: batch_y, 
                                                          keep_prob: 1.0})
                
                    wrong_idx = []
                    for mm in range(len(y_pred)):
                        if (y_pred[mm,0]>y_pred[mm,1] and batch_y[mm,0]==0) or (y_pred[mm,0]<=y_pred[mm,1] and batch_y[mm,0]==1):
                            wrong_idx.append(mm)
                
                    train_accuracy = (len(y_pred)-len(wrong_idx))/len(y_pred)
                    if temp_c==0:
                        acc.append(train_accuracy)
                    temp_c = temp_c+1
                    
                    print(train_accuracy)
                    if train_accuracy>0.9:
                        break

                    train_step.run(feed_dict={batch_X_x: x, 
                                              batch_X_y: y, 
                                              batch_X_w: w, 
                                              batch_X_h: h, 
                                              batch_X_a: ap, 
                                              batch_mask_1: mask_1, 
                                              batch_mask_2: mask_2, 
                                              batch_Y: batch_y, 
                                              keep_prob: 0.75})
            
                
                print('step %d, training accuracy %g' % (cnt, train_accuracy))
        
        acc = np.array(acc)
        print(np.mean(acc))
   
        
        if cnt % 100 == 0:
            save_path = saver.save(sess, save_dir)
            print("Model saved in path: %s" % save_path)

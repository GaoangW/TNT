    
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
import shutil
import seq_nn_3d_v2
import track_lib

global remove_set
global track_set
remove_set = []
track_set = []

def convert_frames_to_video(pathIn,pathOut,fps): 
    frame_array = [] 
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))

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
    
    
def wrt_missing_det(save_mat):
    # fr_id, track_id, xmin, ymin, xmax, ymax, x, y, w, h, det_score
    global track_struct
    for n in range(len(save_mat)):
        fr_id = int(save_mat[n,0])
        obj_id = int(save_mat[n,1])
        if track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,0]==-1:
            continue
            
        num_miss_fr = int(fr_id-track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,0]-1)
        if num_miss_fr<=0:
            continue
        
        temp_save_mat = np.zeros((num_miss_fr,12))
        fr_range = np.array(range(int(track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,0])+1,fr_id))
        interp_xmin = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,2],save_mat[n,3]])
        interp_ymin = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,3],save_mat[n,4]])
        interp_xmax = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,4],save_mat[n,5]])
        interp_ymax = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,5],save_mat[n,6]])
        interp_x_3d = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,6],save_mat[n,7]])
        interp_y_3d = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,7],save_mat[n,8]])
        interp_w_3d = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,8],save_mat[n,9]])
        interp_h_3d = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                                [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,9],save_mat[n,10]])
        #interp_class_name = np.interp(fr_range, [fr_range[0]-1, fr_id], 
             #                   [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,11],save_mat[n,12]])
        #interp_dist2cam = np.interp(fr_range, [fr_range[0]-1, fr_id], 
                 #               [track_struct['tracklet_mat']['obj_end_fr_info'][obj_id,12],save_mat[n,13]])
        temp_save_mat[:,0] = fr_range
        temp_save_mat[:,1] = obj_id
        temp_save_mat[:,2] = -1
        temp_save_mat[:,3] = interp_xmin
        temp_save_mat[:,4] = interp_ymin
        temp_save_mat[:,5] = interp_xmax
        temp_save_mat[:,6] = interp_ymax
        temp_save_mat[:,7] = interp_x_3d
        temp_save_mat[:,8] = interp_y_3d
        temp_save_mat[:,9] = interp_w_3d
        temp_save_mat[:,10] = interp_h_3d
        temp_save_mat[:,11] = -1
        #temp_save_mat[:,12] = interp_class_name
        #temp_save_mat[:,13] = interp_dist2cam
    
        f = open(track_struct['file_path']['txt_result_path'], 'a')  
        np.savetxt(f, temp_save_mat, delimiter=',')
        f.close()
         
    # update track_struct['tracklet_mat']['obj_end_fr_info']
    track_struct['tracklet_mat']['obj_end_fr_info'][save_mat[:,1].astype(int),0] = save_mat[:,0]
    track_struct['tracklet_mat']['obj_end_fr_info'][save_mat[:,1].astype(int),1:] = save_mat[:,2:]
    return
        
def draw_result(img, save_mat, fr_id): 
    
    global track_struct
    save_folder = track_struct['file_path']['tracking_img_folder']
    table = track_struct['tracklet_mat']['color_table']
    save_path = save_folder+'/'+track_lib.file_name(fr_id,10)+'.jpg'
    
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)
    
    # Create Rectangle patches
    # save_mat = [fr_id, obj_id, track_id, x, y, w, h, x_3d, y_3d, w_3d, h_3d, det_score]
    for k in range(len(save_mat)):
        obj_id = int(save_mat[k,1])
        tracklet_id = int(save_mat[k,2])
        xmin = int(save_mat[k,3])
        ymin = int(save_mat[k,4])
        w = int(save_mat[k,5])
        h = int(save_mat[k,6])
        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor='#'+table[obj_id], facecolor='none')
        img_text = plt.text(xmin,ymin,str(obj_id)+'_'+str(tracklet_id),fontsize=6,color='#'+table[obj_id])
        
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        
    plt.savefig(save_path,bbox_inches='tight',dpi=400)
    plt.clf()
    plt.close('all')
    return

def post_processing(debug_mode): 
    
    global track_struct
    #import pdb; pdb.set_trace()
    
    # update comb_cost
    cand_track_idx = np.where(track_struct['tracklet_mat']['track_id_mat']!=-1)[0]
    for n in range(len(cand_track_idx)):
        track_struct['tracklet_mat']['comb_track_cost'][cand_track_idx[n],cand_track_idx] \
            = track_struct['sub_tracklet_mat']['comb_track_cost'][n,:].copy()
        track_struct['tracklet_mat']['comb_track_cost_mask'][cand_track_idx[n],cand_track_idx] \
            = track_struct['sub_tracklet_mat']['comb_track_cost_mask'][n,:].copy()
        
    # 
    tracklet_mat = track_struct['sub_tracklet_mat']
    track_params = track_struct['track_params']
    new_tracklet_mat = tracklet_mat.copy()
    #import pdb; pdb.set_trace()
    
    # update track cluster
    N_cluster = len(tracklet_mat["track_cluster"])
    new_assigned_id_mask = track_struct['tracklet_mat']['save_obj_id_mask'].copy()
    avai_ids = np.where(track_struct['tracklet_mat']['assigned_obj_id_mask']==0)[0]
    
    check_save_idx = list(np.where(new_assigned_id_mask==1)[0])
    check_assigned_idx = list(np.where(avai_ids==1)[0])
    
    new_cnt = -1
    for n in range(N_cluster):
        if len(tracklet_mat["track_cluster"][n])==0:
            continue
        
        #if debug_mode==1:
        #    import pdb; pdb.set_trace()
        
        finish_check_idx = 0
        
        # check save_obj_id_mask
        obj_ids = tracklet_mat['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)]
        obj_mask = track_struct['tracklet_mat']['save_obj_id_mask'][obj_ids]
        save_idx = np.where(obj_mask==1)[0]
        
        for k in range(len(tracklet_mat["track_cluster"][n])):
            temp_id = tracklet_mat['obj_id_mat'][tracklet_mat["track_cluster"][n][k]]
            if temp_id in check_save_idx:
                track_struct['sub_tracklet_mat']['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)] = temp_id
                finish_check_idx = 1
                check_save_idx.remove(temp_id)
                if temp_id in check_assigned_idx:
                    check_assigned_idx.remove(temp_id)
                break
        
        if finish_check_idx==1:
            continue
            
        # check assigned_obj_id_mask
        for k in range(len(tracklet_mat["track_cluster"][n])):
            temp_id = tracklet_mat['obj_id_mat'][tracklet_mat["track_cluster"][n][k]]
            if temp_id in check_assigned_idx:
                track_struct['sub_tracklet_mat']['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)] = temp_id
                finish_check_idx = 1
                check_assigned_idx.remove(temp_id)
                break
        
        if finish_check_idx==1:
            continue
            
        new_cnt = new_cnt+1
        track_struct['sub_tracklet_mat']['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)] = avai_ids[new_cnt]
        
        '''
        # check save_obj_id_mask
        obj_ids = tracklet_mat['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)]
        obj_mask = track_struct['tracklet_mat']['save_obj_id_mask'][obj_ids]
        save_idx = np.where(obj_mask==1)[0]
        if len(save_idx)>0:
            track_struct['sub_tracklet_mat']['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)] = obj_ids[save_idx[0]]
            continue
        
        # check assigned_obj_id_mask
        obj_mask = track_struct['tracklet_mat']['assigned_obj_id_mask'][obj_ids]
        assigned_idx = np.where(obj_mask==1)[0]
        if len(assigned_idx)==0:
            new_cnt = new_cnt+1
            track_struct['sub_tracklet_mat']['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)] = avai_ids[new_cnt]
        else:
            check_flag = 0
            for k in range(len(assigned_idx)):
                temp_obj_id = obj_ids[assigned_idx[k]]
                if new_assigned_id_mask[temp_obj_id]==1:
                    continue
                else:
                    track_struct['sub_tracklet_mat']['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)] \
                        = temp_obj_id
                    check_flag = 1
                    new_assigned_id_mask[temp_obj_id] = 1
                    break
            if check_flag==0:
                new_cnt = new_cnt+1
                track_struct['sub_tracklet_mat']['obj_id_mat'][np.array(tracklet_mat["track_cluster"][n],dtype=int)] = avai_ids[new_cnt]
        '''        
    # copy to tracklet_mat
    #import pdb; pdb.set_trace()
    cand_track_idx = np.where(track_struct['tracklet_mat']['track_id_mat']!=-1)[0]
    track_struct['tracklet_mat']['obj_id_mat'][cand_track_idx] = track_struct['sub_tracklet_mat']['obj_id_mat'].copy()
    
    return

def comb_cost(tracklet_set, sess): 

    global track_struct
    #global all_fea_mat
    #global all_fea_label
    
    img_size = track_struct['track_params']['img_size']
    feature_size = track_struct['track_params']['feature_size']
    max_length = track_struct['track_params']['max_length']
    
    tracklet_mat = track_struct['sub_tracklet_mat']
    loc_scales = track_struct['track_params']['loc_scales']
    
    '''
    temp_sum = np.sum(all_fea_mat[:,4,:,1], axis=1)
    if len(np.where(temp_sum!=0)[0])==0:
        fea_id = 0
    else:
        fea_id = int(np.max(np.where(temp_sum!=0)[0]))+1
    '''
    
    # cnn classifier
    N_tracklet = len(tracklet_set)
    track_interval = tracklet_mat['track_interval']
    sort_idx = np.argsort(track_interval[np.array(tracklet_set),1])
    cost = 0
    if len(sort_idx)<=1:
        return cost

    remove_ids = []
    
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
                
            #import pdb; pdb.set_trace()
            if tracklet_mat['comb_track_cost_mask'][track_id1,track_id2]==1:
                cost = cost+tracklet_mat['comb_track_cost'][track_id1,track_id2]
                remove_ids.append(cnt)
                continue
            
            comb_fea_label[cnt,0] = track_id1 
            comb_fea_label[cnt,1] = track_id2 

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
                
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id1)[0]
                if len(cand_idx)>0:
                    temp_frs = tracklet_mat['appearance_fea_mat'][cand_idx,1]
                    temp_sort_idx = np.argsort(temp_frs)
                    cand_idx = cand_idx[temp_sort_idx]
                
                if comb_fea_mat[cnt,4:,t1_min-t_min:t1_max-t_min+1,0].shape[1]!=np.transpose(tracklet_mat['appearance_fea_mat'] \
                                                                                       [cand_idx,2:]).shape[1]:
                    import pdb; pdb.set_trace()
                comb_fea_mat[cnt,4:,t1_min-t_min:t1_max-t_min+1,0] = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])

                comb_fea_mat[cnt,:,t2_min-t_min:t2_max-t_min+1,2] = 1

                comb_fea_mat[cnt,0,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['x_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[0]
                
                comb_fea_mat[cnt,1,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['y_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[1]
                
                comb_fea_mat[cnt,2,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['w_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[2]
                
                comb_fea_mat[cnt,3,t2_min-t_min:t2_max-t_min+1,0] = tracklet_mat['h_3d_mat'][track_id2,t2_min:t2_max+1]/loc_scales[3]
                
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id2)[0]
                if len(cand_idx)>0:
                    temp_frs = tracklet_mat['appearance_fea_mat'][cand_idx,1]
                    temp_sort_idx = np.argsort(temp_frs)
                    cand_idx = cand_idx[temp_sort_idx]
                    
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
                
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id1)[0]
                if len(cand_idx)>0:
                    temp_frs = tracklet_mat['appearance_fea_mat'][cand_idx,1]
                    temp_sort_idx = np.argsort(temp_frs)
                    cand_idx = cand_idx[temp_sort_idx]
                    
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
                    
                cand_idx = np.where(tracklet_mat['appearance_fea_mat'][:,0]==track_id2)[0]
                if len(cand_idx)>0:
                    temp_frs = tracklet_mat['appearance_fea_mat'][cand_idx,1]
                    temp_sort_idx = np.argsort(temp_frs)
                    cand_idx = cand_idx[temp_sort_idx]
                #import pdb; pdb.set_trace()
                cand_idx = cand_idx[0:t2_end-t2_min+1]
                comb_fea_mat[cnt,4:,t2_min-t1_start:t2_end-t1_start+1,0] \
                    = np.transpose(tracklet_mat['appearance_fea_mat'][cand_idx,2:])
                
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
        
        comb_fea_mat = track_lib.interp_batch(comb_fea_mat)
        
        max_batch_size = 16
        num_batch = int(np.ceil(comb_fea_mat.shape[0]/max_batch_size))
        pred_y = np.zeros((comb_fea_mat.shape[0],2))
        for n in range(num_batch):
            if n!=num_batch-1:
                batch_size = max_batch_size
            else:
                batch_size = int(comb_fea_mat.shape[0]-(num_batch-1)*max_batch_size)
                
            # batch_size = comb_fea_mat.shape[0]
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
            
        '''
        all_fea_mat[fea_id:fea_id+len(pred_y),:,:,:] = comb_fea_mat
        all_fea_label[fea_id:fea_id+len(pred_y),:] = comb_fea_label
        '''

        
        cost = cost+np.sum(pred_y[:,1]-pred_y[:,0])
        #import pdb; pdb.set_trace()
        
        if pred_y.shape[0]!=len(temp_cost_list):
            import pdb; pdb.set_trace()
        for n in range(pred_y.shape[0]):

            tracklet_mat['comb_track_cost_mask'][temp_cost_list[n][0],temp_cost_list[n][1]] = 1
            tracklet_mat['comb_track_cost'][temp_cost_list[n][0],temp_cost_list[n][1]] = pred_y[n,1]-pred_y[n,0]
            
    return cost

def get_split_cost(track_id, sess): 

    global track_struct     
    tracklet_mat = track_struct['sub_tracklet_mat']
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
        new_cluster_cost[1,0] = comb_cost(remain_tracks, sess)
        
    # cross cost
    comb_cluster = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    sort_idx = np.argsort(track_interval[np.array(comb_cluster),1])
    cross_cost = np.zeros((2,1))

    cost = np.sum(new_cluster_cost)-cross_cost[1,0]
    prev_cost = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]]-cross_cost[0,0]
    diff_cost = cost-prev_cost

    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

def get_assign_cost(track_id, sess): 
    
    global track_struct
    tracklet_mat = track_struct['sub_tracklet_mat']
    
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

        new_cluster_cost[0,0] = comb_cost(new_cluster_set[0], sess)

    track_class = track_struct['sub_tracklet_mat']['track_class'][track_id]
    t_cluster_idx = track_struct['sub_tracklet_mat']['track_cluster_t_idx'][track_class]
    
    NN_cluster = len(tracklet_mat['track_cluster'])
    temp_new_cluster_cost = float("inf")*np.ones((NN_cluster,1))
    prev_cost_vec = np.zeros((NN_cluster,1))
    cross_cost_vec = np.zeros((NN_cluster,2))

    for nn in range(len(t_cluster_idx)):
        N_cluster = len(track_struct['sub_tracklet_mat']['time_cluster'][t_cluster_idx[nn]])
        for mm in range(N_cluster):
            n = track_struct['sub_tracklet_mat']['time_cluster'][t_cluster_idx[nn]][mm]
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
            temp_new_cluster_cost[n,0] = comb_cost(temp_set, sess)

            prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
                +tracklet_mat['cluster_cost'][n]      
                

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

def get_merge_cost(track_id, sess): 

    global track_struct
    tracklet_mat = track_struct['sub_tracklet_mat']
    track_interval = tracklet_mat['track_interval'].copy()
    cluster1 = tracklet_mat['track_cluster'][tracklet_mat['track_class'][track_id]].copy()
    if len(cluster1)==1:
        cost = float("inf")
        diff_cost = float("inf")
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

    track_class = track_struct['sub_tracklet_mat']['track_class'][track_id]
    t_cluster_idx = track_struct['sub_tracklet_mat']['track_cluster_t_idx'][track_class]
    
    NN_cluster = len(tracklet_mat['track_cluster'])
    new_cluster_cost_vec = float("inf")*np.ones((NN_cluster,1))
    prev_cost_vec = np.zeros((NN_cluster,1))
    cross_cost_vec = np.zeros((NN_cluster,2))

    for nn in range(len(t_cluster_idx)):
        N_cluster = len(track_struct['sub_tracklet_mat']['time_cluster'][t_cluster_idx[nn]])
        
        for mm in range(N_cluster):
            n = track_struct['sub_tracklet_mat']['time_cluster'][t_cluster_idx[nn]][mm]
        
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
            new_cluster_cost_vec[n,0] = comb_cost(cluster1+cluster2, sess)

            prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
                +tracklet_mat['cluster_cost'][n]

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

def get_switch_cost(track_id, sess): 

    global track_struct
    tracklet_mat = track_struct['sub_tracklet_mat']
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

    track_class = track_struct['sub_tracklet_mat']['track_class'][track_id]
    t_cluster_idx = track_struct['sub_tracklet_mat']['track_cluster_t_idx'][track_class]
    
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
        N_cluster = len(track_struct['sub_tracklet_mat']['time_cluster'][t_cluster_idx[nn]])  
        
        for mm in range(N_cluster):
            n = track_struct['sub_tracklet_mat']['time_cluster'][t_cluster_idx[nn]][mm]
            
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

            new_cluster_cost_vec1[n,0] = comb_cost(S_1, sess)

            new_cluster_cost_vec2[n,0] = comb_cost(S_2, sess)

            cost_vec[n,0] = new_cluster_cost_vec1[n,0]+new_cluster_cost_vec2[n,0]

            track_id_set[n].append(S_1.copy())
            track_id_set[n].append(S_2.copy())
            prev_cost_vec[n,0] = tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]] \
                        +tracklet_mat['cluster_cost'][n]  

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

def get_break_cost(track_id, sess): 

    global track_struct
    tracklet_mat = track_struct['sub_tracklet_mat']
    
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
    new_cluster_cost[0,0] = comb_cost(new_cluster_set[0], sess)

    new_cluster_cost[1,0] = comb_cost(new_cluster_set[1], sess)

    cost = np.sum(new_cluster_cost)
    diff_cost = cost-tracklet_mat['cluster_cost'][tracklet_mat['track_class'][track_id]]
    return diff_cost,new_cluster_cost,new_cluster_set,change_cluster_idx

def copy_sub_mat():
    global track_struct
    track_struct['sub_tracklet_mat'] = {}
    cand_track_idx = np.where(track_struct['tracklet_mat']['track_id_mat']!=-1)[0]
    track_struct['sub_tracklet_mat']['xmin_mat'] = track_struct['tracklet_mat']['xmin_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['ymin_mat'] = track_struct['tracklet_mat']['ymin_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['xmax_mat'] = track_struct['tracklet_mat']['xmax_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['ymax_mat'] = track_struct['tracklet_mat']['ymax_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['x_3d_mat'] = track_struct['tracklet_mat']['x_3d_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['y_3d_mat'] = track_struct['tracklet_mat']['y_3d_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['w_3d_mat'] = track_struct['tracklet_mat']['w_3d_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['h_3d_mat'] = track_struct['tracklet_mat']['h_3d_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['det_score_mat'] = track_struct['tracklet_mat']['det_score_mat'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['track_interval'] = track_struct['tracklet_mat']['track_interval'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['obj_id_mat'] = track_struct['tracklet_mat']['obj_id_mat'][cand_track_idx].copy()
    track_struct['sub_tracklet_mat']['track_id_mat'] = track_struct['tracklet_mat']['track_id_mat'][cand_track_idx].copy()
    #track_struct['sub_tracklet_mat']['save_obj_id_mask'] = track_struct['tracklet_mat']['save_obj_id_mask'].copy()
    #track_struct['sub_tracklet_mat']['assigned_obj_id_mask'] = track_struct['tracklet_mat']['assigned_obj_id_mask'].copy()
    
    # update comb_track_cost
    change_idx = np.zeros(track_struct['track_params']['num_track'], dtype=int)
    for n in range(track_struct['track_params']['num_track']):
        if track_struct['tracklet_mat']['track_interval'][n,1]-track_struct['tracklet_mat']['track_interval'][n,0] \
            !=track_struct['tracklet_mat']['prev_track_interval'][n,1]-track_struct['tracklet_mat']['prev_track_interval'][n,0] \
            or (track_struct['tracklet_mat']['track_interval'][n,0]==0 
                and track_struct['tracklet_mat']['prev_track_interval'][n,0]==0
                and track_struct['tracklet_mat']['track_interval'][n,1]==track_struct['track_params']['num_fr']-1 
                and track_struct['tracklet_mat']['prev_track_interval'][n,1]==track_struct['track_params']['num_fr']-1):
            change_idx[n] = 1
            
    track_struct['tracklet_mat']['comb_track_cost'][change_idx==1,:] = 0
    track_struct['tracklet_mat']['comb_track_cost'][:,change_idx==1] = 0
    track_struct['tracklet_mat']['comb_track_cost_mask'][change_idx==1,:] = 0
    track_struct['tracklet_mat']['comb_track_cost_mask'][:,change_idx==1] = 0
    
    temp_mat = track_struct['tracklet_mat']['comb_track_cost'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['comb_track_cost'] = temp_mat[:,cand_track_idx].copy()
    
    temp_mat = track_struct['tracklet_mat']['comb_track_cost_mask'][cand_track_idx,:].copy()
    track_struct['sub_tracklet_mat']['comb_track_cost_mask'] = temp_mat[:,cand_track_idx].copy()
    
    fea_cand_idx = np.where(track_struct['tracklet_mat']['appearance_fea_mat'][:,0]!=-1)[0]
    track_struct['sub_tracklet_mat']['appearance_fea_mat'] = track_struct['tracklet_mat']['appearance_fea_mat'][fea_cand_idx,:].copy()
    
    # update track_id for sub_tracklet_mat
    for n in range(len(cand_track_idx)):
        temp_idx = np.where(track_struct['sub_tracklet_mat']['appearance_fea_mat'][:,0]==cand_track_idx[n])[0]
        track_struct['sub_tracklet_mat']['appearance_fea_mat'][temp_idx,0] = n
    
    return
    
def init_clustering(): 
    
    global track_struct
    
    # copy the sub tracklet_mat
    copy_sub_mat()
        
    N_tracklet = track_struct['sub_tracklet_mat']['xmin_mat'].shape[0]

    # track cluster
    track_struct['sub_tracklet_mat']['track_cluster'] = []

    # track class
    track_struct['sub_tracklet_mat']['track_class'] = np.arange(N_tracklet, dtype=int)

    # time cluster
    track_struct['sub_tracklet_mat']['time_cluster'] = []
    for n in range(track_struct['track_params']['num_time_cluster']):
        track_struct['sub_tracklet_mat']['time_cluster'].append([])
    
    track_struct['sub_tracklet_mat']['track_cluster_t_idx'] = []
    for n in range(N_tracklet):
        idx = np.where(track_struct['sub_tracklet_mat']['xmax_mat'][n,:]!=-1)[0]
        if len(idx)==0:
            import pdb; pdb.set_trace()
        track_struct['sub_tracklet_mat']['track_interval'][n,0] = np.min(idx)
        track_struct['sub_tracklet_mat']['track_interval'][n,1] = np.max(idx)
        track_struct['sub_tracklet_mat']['track_cluster'].append([n])
        
        if n in remove_set:
            track_struct['sub_tracklet_mat']['track_cluster_t_idx'].append([-1])
        else:
            min_time_cluster_idx = int(np.floor(max(track_struct['sub_tracklet_mat']['track_interval'][n,0]
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
            max_time_cluster_idx = int(np.floor(min(track_struct['sub_tracklet_mat']['track_interval'][n,1]
                                            +track_struct['track_params']['t_dist_thresh']+5,
                                                    track_struct['sub_tracklet_mat']['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
            track_struct['sub_tracklet_mat']['track_cluster_t_idx'].append(list(range(min_time_cluster_idx,max_time_cluster_idx+1)))
            for k in range(min_time_cluster_idx,max_time_cluster_idx+1):
                track_struct['sub_tracklet_mat']['time_cluster'][k].append(n)

    # get center position of each detection location
    mask = track_struct['sub_tracklet_mat']['xmin_mat']==-1
    track_struct['sub_tracklet_mat']['center_x'] = \
        (track_struct['sub_tracklet_mat']['xmin_mat']+track_struct['sub_tracklet_mat']['xmax_mat'])/2
    track_struct['sub_tracklet_mat']['center_y'] = \
        (track_struct['sub_tracklet_mat']['ymin_mat']+track_struct['sub_tracklet_mat']['ymax_mat'])/2
    track_struct['sub_tracklet_mat']['w'] = \
        track_struct['sub_tracklet_mat']['xmax_mat']-track_struct['sub_tracklet_mat']['xmin_mat']+1
    track_struct['sub_tracklet_mat']['h'] = \
        track_struct['sub_tracklet_mat']['ymax_mat']-track_struct['sub_tracklet_mat']['ymin_mat']+1
    track_struct['sub_tracklet_mat']['center_x'][mask] = -1
    track_struct['sub_tracklet_mat']['center_y'][mask] = -1
    track_struct['sub_tracklet_mat']['w'][mask] = -1
    track_struct['sub_tracklet_mat']['h'][mask] = -1

    # neighbor_track_idx and conflict_track_idx
    track_struct['sub_tracklet_mat']['neighbor_track_idx'] = []
    track_struct['sub_tracklet_mat']['conflict_track_idx'] = []
    for n in range(N_tracklet):
        track_struct['sub_tracklet_mat']['neighbor_track_idx'].append([])
        track_struct['sub_tracklet_mat']['conflict_track_idx'].append([])
    for n in range(N_tracklet-1):
        for m in range(n+1, N_tracklet):
            t_min1 = track_struct['sub_tracklet_mat']['track_interval'][n,0]
            t_max1 = track_struct['sub_tracklet_mat']['track_interval'][n,1]
            t_min2 = track_struct['sub_tracklet_mat']['track_interval'][m,0]
            t_max2 = track_struct['sub_tracklet_mat']['track_interval'][m,1]
            overlap_len = min(t_max2,t_max1)-max(t_min1,t_min2)+1
            overlap_r = overlap_len/(t_max1-t_min1+1+t_max2-t_min2+1-overlap_len)
            if overlap_len>0 and overlap_r>track_struct['track_params']['track_overlap_thresh']:
                track_struct['sub_tracklet_mat']['conflict_track_idx'][n].append(m)
                track_struct['sub_tracklet_mat']['conflict_track_idx'][m].append(n)
                continue
            if overlap_len>0 and overlap_r<=track_struct['track_params']['track_overlap_thresh']:
                # check the search region
                t1 = int(max(t_min1,t_min2))
                t2 = int(min(t_max2,t_max1))
                if (t_min1<=t_min2 and t_max1>=t_max2) or (t_min1>=t_min2 and t_max1<=t_max2) or overlap_len>4:
                    track_struct['sub_tracklet_mat']['conflict_track_idx'][n].append(m)
                    track_struct['sub_tracklet_mat']['conflict_track_idx'][m].append(n)
                    continue

                cand_t = np.array(range(t1,t2+1))
                dist_x = abs(track_struct['sub_tracklet_mat']['center_x'][n,cand_t] \
                         -track_struct['sub_tracklet_mat']['center_x'][m,cand_t])
                dist_y = abs(track_struct['sub_tracklet_mat']['center_y'][n,cand_t] \
                         -track_struct['sub_tracklet_mat']['center_y'][m,cand_t])
                w1 = track_struct['sub_tracklet_mat']['w'][n,cand_t]
                h1 = track_struct['sub_tracklet_mat']['h'][n,cand_t]
                w2 = track_struct['sub_tracklet_mat']['w'][m,cand_t]
                h2 = track_struct['sub_tracklet_mat']['h'][m,cand_t]
                
                min_len = np.min([np.min(w1),np.min(h1),np.min(w2),np.min(h2)])
                min_dist_x1 = np.min(dist_x/min_len)
                min_dist_y1 = np.min(dist_y/min_len)
                min_dist_x2 = np.min(dist_x/min_len)
                min_dist_y2 = np.min(dist_y/min_len)
                if min_dist_x1<track_struct['track_params']['search_radius'] \
                    and min_dist_y1<track_struct['track_params']['search_radius'] \
                    and min_dist_x2<track_struct['track_params']['search_radius'] \
                    and min_dist_y2<track_struct['track_params']['search_radius']:
                    track_struct['sub_tracklet_mat']['neighbor_track_idx'][n].append(m)
                    track_struct['sub_tracklet_mat']['neighbor_track_idx'][m].append(n)

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
                tr_x1 = track_struct['sub_tracklet_mat']['center_x'][n,int(t_min1):int(t_max1+1)]
                tr_y1 = track_struct['sub_tracklet_mat']['center_y'][n,int(t_min1):int(t_max1+1)]
                if len(tr_t1)>10:
                    if t_min1>=t_max2:
                        tr_t1 = tr_t1[0:10]
                        tr_x1 = tr_x1[0:10]
                        tr_y1 = tr_y1[0:10]
                    else:
                        tr_t1 = tr_t1[-10:]
                        tr_x1 = tr_x1[-10:]
                        tr_y1 = tr_y1[-10:]         
                ts_x1 = track_lib.linear_pred_v2(tr_t1, tr_x1, np.array([t2]))
                ts_y1 = track_lib.linear_pred_v2(tr_t1, tr_y1, np.array([t2]))
                dist_x1 = abs(ts_x1[0]-track_struct['sub_tracklet_mat']['center_x'][m,t2])
                dist_y1 = abs(ts_y1[0]-track_struct['sub_tracklet_mat']['center_y'][m,t2])
                
                tr_t2 = np.array(range(int(t_min2),int(t_max2+1)))
                tr_x2 = track_struct['sub_tracklet_mat']['center_x'][m,int(t_min2):int(t_max2+1)]
                tr_y2 = track_struct['sub_tracklet_mat']['center_y'][m,int(t_min2):int(t_max2+1)]
                if len(tr_t2)>10:
                    if t_min2>t_max1:
                        tr_t2 = tr_t2[0:10]
                        tr_x2 = tr_x2[0:10]
                        tr_y2 = tr_y2[0:10]
                    else:
                        tr_t2 = tr_t2[-10:]
                        tr_x2 = tr_x2[-10:]
                        tr_y2 = tr_y2[-10:]   
                        
                ts_x2 = track_lib.linear_pred_v2(tr_t2, tr_x2, np.array([t1]))
                ts_y2 = track_lib.linear_pred_v2(tr_t2, tr_y2, np.array([t1]))
                dist_x2 = abs(ts_x2[0]-track_struct['sub_tracklet_mat']['center_x'][n,t1])
                dist_y2 = abs(ts_y2[0]-track_struct['sub_tracklet_mat']['center_y'][n,t1])
                
                dist_x = min(dist_x1, dist_x2)
                dist_y = min(dist_y1, dist_y2)
                #***********************************
                
                
                w1 = track_struct['sub_tracklet_mat']['w'][n,t1]
                h1 = track_struct['sub_tracklet_mat']['h'][n,t1]
                w2 = track_struct['sub_tracklet_mat']['w'][m,t2]
                h2 = track_struct['sub_tracklet_mat']['h'][m,t2]
                
                min_len = np.min([np.min(w1),np.min(h1),np.min(w2),np.min(h2)])
                min_dist_x1 = dist_x/min_len
                min_dist_y1 = dist_y/min_len
                min_dist_x2 = dist_x/min_len
                min_dist_y2 = dist_y/min_len
                
                if min_dist_x1<track_struct['track_params']['search_radius'] \
                    and min_dist_y1<track_struct['track_params']['search_radius'] \
                    and min_dist_x2<track_struct['track_params']['search_radius'] \
                    and min_dist_y2<track_struct['track_params']['search_radius']:
                    track_struct['sub_tracklet_mat']['neighbor_track_idx'][n].append(m)
                    track_struct['sub_tracklet_mat']['neighbor_track_idx'][m].append(n)

    # cluster cost
    track_struct['sub_tracklet_mat']['cluster_cost'] = []
    for n in range(N_tracklet):
        track_struct['sub_tracklet_mat']['cluster_cost'].append(0)

    return 

def tracklet_clustering(sess, t_iter): 
    
    global track_struct
    if t_iter==0:
        init_clustering()

    track_interval = track_struct['sub_tracklet_mat']['track_interval'] 
    N_tracklet = track_interval.shape[0] 
    change_flag = 0 
    img_size = track_struct['track_params']['img_size']
    
    # sort tracklet in ascending order
    sort_idx = np.argsort(track_interval[:,1])
    for n in range(N_tracklet):
        # print(n)
        track_id = sort_idx[n]
        track_class = track_struct['sub_tracklet_mat']['track_class'][track_id]
        t_cluster_idx = track_struct['sub_tracklet_mat']['track_cluster_t_idx'][track_class]

        # remove_set
        if t_cluster_idx[0]==-1:
            continue
                     
        diff_cost = np.zeros((5,1))
        new_C = [] # new cost
        new_set = []
        change_idx = []

        # get split cost
        #import pdb; pdb.set_trace()
        diff_cost[0,0],temp_new_C,temp_new_set,temp_change_idx = get_split_cost(track_id, sess)

        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get assign cost
        #import pdb; pdb.set_trace()
        diff_cost[1,0],temp_new_C,temp_new_set,temp_change_idx = get_assign_cost(track_id, sess)

        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get merge cost
        diff_cost[2,0],temp_new_C,temp_new_set,temp_change_idx = get_merge_cost(track_id, sess)

        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get switch cost
        diff_cost[3,0],temp_new_C,temp_new_set,temp_change_idx = get_switch_cost(track_id, sess)

        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # get break cost
        diff_cost[4,0],temp_new_C,temp_new_set,temp_change_idx = get_break_cost(track_id, sess)

        new_C.append(temp_new_C)
        new_set.append(temp_new_set)
        change_idx.append(temp_change_idx)

        # update cluster
        min_idx = np.argmin(diff_cost[:,0])
        min_cost = diff_cost[min_idx,0]
        if min_cost>=0:
            continue

        change_flag = 1
            
        #****************
        #import pdb; pdb.set_trace()
        # print(min_idx)
        # print(new_set)
        new_t_idx = []
        if len(new_set[min_idx][0])==0:
            new_t_idx.append([-1])
        else:
            t_min_array = np.zeros((len(new_set[min_idx][0]),1))
            t_max_array = np.zeros((len(new_set[min_idx][0]),1))
            for m in range(len(new_set[min_idx][0])):
                t_min_array[m,0] = track_struct['sub_tracklet_mat']['track_interval'][new_set[min_idx][0][m],0]
                t_max_array[m,0] = track_struct['sub_tracklet_mat']['track_interval'][new_set[min_idx][0][m],1]
                                   
            min_time_cluster_idx = int(np.floor(max(np.min(t_min_array)
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
            max_time_cluster_idx = int(np.floor(min(np.max(t_max_array)
                                            +track_struct['track_params']['t_dist_thresh']+5,
                                                    track_struct['sub_tracklet_mat']['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
            new_t_idx.append(list(range(min_time_cluster_idx,max_time_cluster_idx+1)))
                           
        if len(new_set[min_idx][1])==0:
            new_t_idx.append([-1])
        else:
            t_min_array = np.zeros((len(new_set[min_idx][1]),1))
            t_max_array = np.zeros((len(new_set[min_idx][1]),1))
            for m in range(len(new_set[min_idx][1])):
                t_min_array[m,0] = track_struct['sub_tracklet_mat']['track_interval'][new_set[min_idx][1][m],0]
                t_max_array[m,0] = track_struct['sub_tracklet_mat']['track_interval'][new_set[min_idx][1][m],1]
                                   
            min_time_cluster_idx = int(np.floor(max(np.min(t_min_array)
                                            -track_struct['track_params']['t_dist_thresh']-5,0)
                                        /track_struct['track_params']['time_cluster_dist']))
            max_time_cluster_idx = int(np.floor(min(np.max(t_max_array)
                                            +track_struct['track_params']['t_dist_thresh']+5,
                                                    track_struct['sub_tracklet_mat']['xmin_mat'].shape[1]-1)
                                        /track_struct['track_params']['time_cluster_dist']))
            new_t_idx.append(list(range(min_time_cluster_idx,max_time_cluster_idx+1)))
                                   
        if change_idx[min_idx][0]>=len(track_struct['sub_tracklet_mat']['track_cluster']):
            for m in range(len(track_struct['sub_tracklet_mat']['track_cluster']),change_idx[min_idx][0]):
                track_struct['sub_tracklet_mat']['track_cluster'].append([])
                track_struct['sub_tracklet_mat']['track_cluster_t_idx'].append([-1])
            track_struct['sub_tracklet_mat']['track_cluster'].append(new_set[min_idx][0])   
            track_struct['sub_tracklet_mat']['track_cluster_t_idx'].append(new_t_idx[0])
        else:
            track_struct['sub_tracklet_mat']['track_cluster'][change_idx[min_idx][0]] = new_set[min_idx][0]
            track_struct['sub_tracklet_mat']['track_cluster_t_idx'][change_idx[min_idx][0]] = new_t_idx[0]

        if change_idx[min_idx][1]>=len(track_struct['sub_tracklet_mat']['track_cluster']):
            for m in range(len(track_struct['sub_tracklet_mat']['track_cluster']),change_idx[min_idx][1]):
                track_struct['sub_tracklet_mat']['track_cluster'].append([])
                track_struct['sub_tracklet_mat']['track_cluster_t_idx'].append([-1])
            track_struct['sub_tracklet_mat']['track_cluster'].append(new_set[min_idx][1])  
            track_struct['sub_tracklet_mat']['track_cluster_t_idx'].append(new_t_idx[1])
        else:
            track_struct['sub_tracklet_mat']['track_cluster'][change_idx[min_idx][1]] = new_set[min_idx][1]
            track_struct['sub_tracklet_mat']['track_cluster_t_idx'][change_idx[min_idx][1]] = new_t_idx[1]
        
        #import pdb; pdb.set_trace()
        for m in range(track_struct['track_params']['num_time_cluster']):
            #import pdb; pdb.set_trace()
            if change_idx[min_idx][0] in track_struct['sub_tracklet_mat']['time_cluster'][m]:
                track_struct['sub_tracklet_mat']['time_cluster'][m].remove(change_idx[min_idx][0])                   
            if change_idx[min_idx][1] in track_struct['sub_tracklet_mat']['time_cluster'][m]:
                track_struct['sub_tracklet_mat']['time_cluster'][m].remove(change_idx[min_idx][1])
                                   
        for m in range(track_struct['track_params']['num_time_cluster']):
            if m in new_t_idx[0]:
                track_struct['sub_tracklet_mat']['time_cluster'][m].append(change_idx[min_idx][0])                   
            if m in new_t_idx[1]:
                track_struct['sub_tracklet_mat']['time_cluster'][m].append(change_idx[min_idx][1])
                                   
        if change_idx[min_idx][0]>=len(track_struct['sub_tracklet_mat']['cluster_cost']):
            for m in range(len(track_struct['sub_tracklet_mat']['cluster_cost']),change_idx[min_idx][0]):
                track_struct['sub_tracklet_mat']['cluster_cost'].append(0)
            track_struct['sub_tracklet_mat']['cluster_cost'].append(new_C[min_idx][0])
        else:
            track_struct['sub_tracklet_mat']['cluster_cost'][change_idx[min_idx][0]] = new_C[min_idx][0]

        if change_idx[min_idx][1]>=len(track_struct['sub_tracklet_mat']['cluster_cost']):
            for m in range(len(track_struct['sub_tracklet_mat']['cluster_cost']),change_idx[min_idx][1]):
                track_struct['sub_tracklet_mat']['cluster_cost'].append([])
            track_struct['sub_tracklet_mat']['cluster_cost'].append(new_C[min_idx][1])  
        else:
            track_struct['sub_tracklet_mat']['cluster_cost'][change_idx[min_idx][1]] = new_C[min_idx][1]

        for k in range(len(track_struct['sub_tracklet_mat']['track_cluster'][change_idx[min_idx][0]])):
            track_struct['sub_tracklet_mat']['track_class'][track_struct['sub_tracklet_mat'] \
                                                    ['track_cluster'][change_idx[min_idx][0]][k]] = change_idx[min_idx][0]

        for k in range(len(track_struct['sub_tracklet_mat']['track_cluster'][change_idx[min_idx][1]])):
            track_struct['sub_tracklet_mat']['track_class'][track_struct['sub_tracklet_mat'] \
                                                    ['track_cluster'][change_idx[min_idx][1]][k]] = change_idx[min_idx][1]
        #import pdb; pdb.set_trace()
    return change_flag

def feature_encode(sess, image_paths, batch_size):

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

    sess.run(eval_enqueue_op, {image_paths_placeholder: image_paths_array, 
                      labels_placeholder: labels_array, control_placeholder: control_array})

    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            # print('.', end='')
            sys.stdout.flush()
    #import pdb; pdb.set_trace()
    #np.savetxt("emb_array.csv", emb_array, delimiter=",")
    return emb_array

def crop_det(det_M, img): 
    global track_struct
    crop_det_folder = track_struct['file_path']['crop_det_folder']
    crop_size = track_struct['track_params']['crop_size']
    if not os.path.isdir(crop_det_folder): 
        os.makedirs(crop_det_folder) 
    
    save_patch_list = []
    for n in range(len(det_M)):
        xmin = int(max(0,det_M[n,1])) 
        xmax = int(min(img.shape[1]-1,det_M[n,1]+det_M[n,3])) 
        ymin = int(max(0,det_M[n,2])) 
        ymax = int(min(img.shape[0]-1,det_M[n,2]+det_M[n,4])) 
        img_patch = img[ymin:ymax,xmin:xmax,:] 
        img_patch = misc.imresize(img_patch, size=[crop_size,crop_size]) 
        patch_name = track_lib.file_name(n,4)+'.png'
        save_path = crop_det_folder+'/'+patch_name 
        misc.imsave(save_path, img_patch)
        save_patch_list.append(save_path)
  
    return save_patch_list

def init_tracklet_model():
    global track_struct
    global tracklet_graph
    global tracklet_sess
    
    global batch_X_x
    global batch_X_y
    global batch_X_w
    global batch_X_h
    global batch_X_a
    global batch_mask_1
    global batch_mask_2
    global batch_Y
    global keep_prob
    global y_conv
    
    max_length = track_struct['track_params']['max_length']
    batch_size = track_struct['track_params']['batch_size']
    feature_size = track_struct['track_params']['feature_size']
    num_classes = track_struct['track_params']['num_classes']
    
    # build tracklet graph
    tracklet_graph = tf.Graph()
    with tracklet_graph.as_default():
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

        y_conv = seq_nn_3d_v2.seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,
                       batch_mask_2,batch_Y,max_length,feature_size,keep_prob)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_Y, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(batch_Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tracklet_init = tf.global_variables_initializer()
        tracklet_saver = tf.train.Saver()    
    
        tracklet_sess = tf.Session(graph=tracklet_graph)   
        with tracklet_sess.as_default():
            tracklet_saver.restore(tracklet_sess, track_struct['file_path']['seq_model'])
            print("Tracklet model restored.")       
    return
            
def init_triplet_model():
    global track_struct
    global triplet_graph
    global triplet_sess
    
    global eval_enqueue_op
    global image_paths_placeholder
    global labels_placeholder
    global phase_train_placeholder
    global batch_size_placeholder
    global control_placeholder
    global embeddings
    global label_batch
    global distance_metric
    f_image_size = 160 
    distance_metric = 0 

    triplet_graph = tf.Graph()
    with triplet_graph.as_default():
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
    triplet_sess = tf.Session(graph=triplet_graph)   
    with triplet_sess.as_default():
        with triplet_graph.as_default():
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(track_struct['file_path']['triplet_model'], input_map=input_map)
            
            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=triplet_sess)
    return

def TC_online(det_M, img, t_pointer, fr_idx, end_flag):
    global track_struct
    global triplet_graph
    global triplet_sess
    global tracklet_graph
    global tracklet_sess
    
    prev_t_pointer = t_pointer
    num_bbox = len(det_M)
    #print(num_bbox)
    track_struct['track_params']['img_size'] = img.shape
    track_struct['tracklet_mat']['imgs'].append(img)   
    
    # last frame in the time window
    max_track_id = np.max(track_struct['tracklet_mat']['track_id_mat'])
    if t_pointer==track_struct['track_params']['num_fr']:
        #import pdb; pdb.set_trace()
        # save tracking to file
        # fr_id, obj_id, track_id, x, y, w, h, x_3d, y_3d, w_3d, h_3d, det_score
        track_idx = np.where(track_struct['tracklet_mat']['xmin_mat'][:,0]!=-1)[0]
        num_save_id = len(track_idx)
        if num_save_id!=0:
            save_mat = np.zeros((num_save_id, 12))
            save_mat[:,0] = int(fr_idx-track_struct['track_params']['num_fr'])
            save_mat[:,1] = track_struct['tracklet_mat']['obj_id_mat'][track_idx]
            track_struct['tracklet_mat']['save_obj_id_mask'][save_mat[:,1].astype(int)] = 1
            save_mat[:,2] = track_struct['tracklet_mat']['track_id_mat'][track_idx]
            save_mat[:,3] = track_struct['tracklet_mat']['xmin_mat'][track_idx,0]
            save_mat[:,4] = track_struct['tracklet_mat']['ymin_mat'][track_idx,0]
            save_mat[:,5] = track_struct['tracklet_mat']['xmax_mat'][track_idx,0] \
                -track_struct['tracklet_mat']['xmin_mat'][track_idx,0]
            save_mat[:,6] = track_struct['tracklet_mat']['ymax_mat'][track_idx,0] \
                -track_struct['tracklet_mat']['ymin_mat'][track_idx,0]
            save_mat[:,7] = track_struct['tracklet_mat']['x_3d_mat'][track_idx,0]
            save_mat[:,8] = track_struct['tracklet_mat']['y_3d_mat'][track_idx,0]
            save_mat[:,9] = track_struct['tracklet_mat']['w_3d_mat'][track_idx,0]
            save_mat[:,10] = track_struct['tracklet_mat']['h_3d_mat'][track_idx,0]
            save_mat[:,11] = track_struct['tracklet_mat']['det_score_mat'][track_idx,0]
            #save_mat[:,12] = track_struct['tracklet_mat']['class_name'][track_idx,0]
            #save_mat[:,13] = track_struct['tracklet_mat']['dist2cam'][track_idx,0]
            
            #import pdb; pdb.set_trace()
            f = open(track_struct['file_path']['txt_result_path'], 'a')  
            np.savetxt(f, save_mat, delimiter=',')
            f.close()
            wrt_missing_det(save_mat)
            
        else:
            save_mat = []
            
        #draw_result(track_struct['tracklet_mat']['imgs'][0], save_mat, fr_idx-track_struct['track_params']['num_fr'])
        #del track_struct['tracklet_mat']['imgs'][0]
        
        # Slide the time window
        track_struct['tracklet_mat']['xmin_mat'][:,:-1] = track_struct['tracklet_mat']['xmin_mat'][:,1:]
        track_struct['tracklet_mat']['xmin_mat'][:,-1] = -1
        track_struct['tracklet_mat']['ymin_mat'][:,:-1] = track_struct['tracklet_mat']['ymin_mat'][:,1:]
        track_struct['tracklet_mat']['ymin_mat'][:,-1] = -1
        track_struct['tracklet_mat']['xmax_mat'][:,:-1] = track_struct['tracklet_mat']['xmax_mat'][:,1:]
        track_struct['tracklet_mat']['xmax_mat'][:,-1] = -1
        track_struct['tracklet_mat']['ymax_mat'][:,:-1] = track_struct['tracklet_mat']['ymax_mat'][:,1:]
        track_struct['tracklet_mat']['ymax_mat'][:,-1] = -1
        track_struct['tracklet_mat']['x_3d_mat'][:,:-1] = track_struct['tracklet_mat']['x_3d_mat'][:,1:]
        track_struct['tracklet_mat']['x_3d_mat'][:,-1] = -1
        track_struct['tracklet_mat']['y_3d_mat'][:,:-1] = track_struct['tracklet_mat']['y_3d_mat'][:,1:]
        track_struct['tracklet_mat']['y_3d_mat'][:,-1] = -1
        track_struct['tracklet_mat']['w_3d_mat'][:,:-1] = track_struct['tracklet_mat']['w_3d_mat'][:,1:]
        track_struct['tracklet_mat']['w_3d_mat'][:,-1] = -1
        track_struct['tracklet_mat']['h_3d_mat'][:,:-1] = track_struct['tracklet_mat']['h_3d_mat'][:,1:]
        track_struct['tracklet_mat']['h_3d_mat'][:,-1] = -1
        track_struct['tracklet_mat']['det_score_mat'][:,:-1] = track_struct['tracklet_mat']['det_score_mat'][:,1:]
        track_struct['tracklet_mat']['det_score_mat'][:,-1] = -1
        track_struct['tracklet_mat']['track_interval'] = track_struct['tracklet_mat']['track_interval']-1
        track_struct['tracklet_mat']['track_interval'][track_struct['tracklet_mat']['track_interval'][:,0]<0,0] = 0
        track_struct['tracklet_mat']['track_interval'][track_struct['tracklet_mat']['track_interval'][:,1]<0,0] = -1
        track_struct['tracklet_mat']['track_interval'][track_struct['tracklet_mat']['track_interval'][:,1]<0,1] = -1
        
        track_struct['tracklet_mat']['obj_id_mat'][track_struct['tracklet_mat']['track_interval'][:,1]==-1] = -1
        track_struct['tracklet_mat']['track_id_mat'][track_struct['tracklet_mat']['track_interval'][:,1]==-1] = -1

        #track_struct['tracklet_mat']['class_name'][:,:-1] = track_struct['tracklet_mat']['class_name'][:,1:]
        #track_struct['tracklet_mat']['class_name'][:,-1] = -1
        #track_struct['tracklet_mat']['dist2cam'][:,:-1] = track_struct['tracklet_mat']['dist2cam'][:,1:]
        #track_struct['tracklet_mat']['dist2cam'][:,-1] = -1
        
        t_pointer = t_pointer-1
        
        remove_fr_idx = fr_idx-track_struct['track_params']['num_fr']
        remove_fea_idx = np.where(track_struct['tracklet_mat']['appearance_fea_mat'][:,1]==remove_fr_idx)[0]
        track_struct['tracklet_mat']['appearance_fea_mat'][remove_fea_idx,:] = -1
    
    track_struct['tracklet_mat']['assigned_obj_id_mask'] = track_struct['tracklet_mat']['save_obj_id_mask'].copy()
    assigned_ids = track_struct['tracklet_mat']['obj_id_mat'][track_struct['tracklet_mat']['obj_id_mat']!=-1]
    track_struct['tracklet_mat']['assigned_obj_id_mask'][assigned_ids] = 1
    avai_ids = np.where(track_struct['tracklet_mat']['assigned_obj_id_mask']==0)[0]

    #if fr_idx-track_struct['track_params']['num_fr']==214:
    #    import pdb; pdb.set_trace()
    empty_idx = np.where(track_struct['tracklet_mat']['track_id_mat']==-1)[0]
    empty_fea_idx = np.where(track_struct['tracklet_mat']['appearance_fea_mat'][:,0]==-1)[0]
    
    # crop detection results and extract cnn features
    if num_bbox!=0:
        patch_list = crop_det(det_M, img)
        track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[0:len(patch_list)],2:] \
            = 10*feature_encode(triplet_sess, patch_list, len(patch_list)) 
            
        # remove folder
        shutil.rmtree(track_struct['file_path']['crop_det_folder'])
    
    # Forward tracking    
    if t_pointer==0 and num_bbox!=0:
        track_struct['tracklet_mat']['obj_id_mat'][empty_idx[0:num_bbox]] = avai_ids[0:num_bbox]
        track_struct['tracklet_mat']['track_id_mat'][empty_idx[0:num_bbox]] = np.array(range(num_bbox),dtype=int)+max_track_id+1
        track_struct['tracklet_mat']['xmin_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,1]
        track_struct['tracklet_mat']['ymin_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,2]
        track_struct['tracklet_mat']['xmax_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,1]+det_M[:,3]
        track_struct['tracklet_mat']['ymax_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,2]+det_M[:,4]
        track_struct['tracklet_mat']['x_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,6]
        track_struct['tracklet_mat']['y_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,7]
        track_struct['tracklet_mat']['w_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,8]
        track_struct['tracklet_mat']['h_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,9]
        track_struct['tracklet_mat']['det_score_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,5]
        track_struct['tracklet_mat']['track_interval'][empty_idx[0:num_bbox],0] = t_pointer
        track_struct['tracklet_mat']['track_interval'][empty_idx[0:num_bbox],1] = t_pointer
        track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[0:len(patch_list)],0] = empty_idx[0:num_bbox]
        track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[0:len(patch_list)],1] = fr_idx
        #track_struct['tracklet_mat']['class_name'][empty_idx[0:num_bbox],t_pointer] = det_M[:,10]
        #track_struct['tracklet_mat']['dist2cam'][empty_idx[0:num_bbox],t_pointer] = det_M[:,11]
        
    elif t_pointer!=0 and num_bbox!=0:
        #import pdb; pdb.set_trace()
        prev_bbox_idx = np.where(track_struct['tracklet_mat']['xmin_mat'][:,t_pointer-1]!=-1)[0]
        prev_num_bbox = len(prev_bbox_idx)
        if prev_num_bbox==0:
            track_struct['tracklet_mat']['obj_id_mat'][empty_idx[0:num_bbox]] = avai_ids[0:num_bbox]
            track_struct['tracklet_mat']['track_id_mat'][empty_idx[0:num_bbox]] = np.array(range(num_bbox),dtype=int)+max_track_id+1
            track_struct['tracklet_mat']['xmin_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,1]
            track_struct['tracklet_mat']['ymin_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,2]
            track_struct['tracklet_mat']['xmax_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,1]+det_M[:,3]
            track_struct['tracklet_mat']['ymax_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,2]+det_M[:,4]
            track_struct['tracklet_mat']['x_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,6]
            track_struct['tracklet_mat']['y_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,7]
            track_struct['tracklet_mat']['w_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,8]
            track_struct['tracklet_mat']['h_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,9]
            track_struct['tracklet_mat']['det_score_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,5]
            track_struct['tracklet_mat']['track_interval'][empty_idx[0:num_bbox],0] = t_pointer
            track_struct['tracklet_mat']['track_interval'][empty_idx[0:num_bbox],1] = t_pointer
            track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[0:len(patch_list)],0] = empty_idx[0:num_bbox]
            track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[0:len(patch_list)],1] = fr_idx
            #track_struct['tracklet_mat']['class_name'][empty_idx[0:num_bbox],t_pointer] = det_M[:,10]
            #track_struct['tracklet_mat']['dist2cam'][empty_idx[0:num_bbox],t_pointer] = det_M[:,11]
        else:
            # predict bbox location
            bbox1 = np.zeros((prev_num_bbox,4))
            bbox1[:,0] = track_struct['tracklet_mat']['xmin_mat'][prev_bbox_idx,t_pointer-1]
            bbox1[:,1] = track_struct['tracklet_mat']['ymin_mat'][prev_bbox_idx,t_pointer-1]
            bbox1[:,2] = track_struct['tracklet_mat']['xmax_mat'][prev_bbox_idx,t_pointer-1] \
                -track_struct['tracklet_mat']['xmin_mat'][prev_bbox_idx,t_pointer-1]+1
            bbox1[:,3] = track_struct['tracklet_mat']['ymax_mat'][prev_bbox_idx,t_pointer-1] \
                -track_struct['tracklet_mat']['ymin_mat'][prev_bbox_idx,t_pointer-1]+1
            pred_bbox1 = np.zeros((prev_num_bbox,4))
            
            bbox2 = np.zeros((num_bbox,4))
            bbox2[:,:] = det_M[:,1:5]
            
            # bbox association
            for k in range(prev_num_bbox):
                temp_track_id = prev_bbox_idx[k]
                t_idx = np.where(track_struct['tracklet_mat']['xmin_mat'][temp_track_id,:]!=-1)[0]
                t_min = np.min(t_idx)
                if t_min<t_pointer-1-track_struct['track_params']['linear_pred_thresh']:
                    t_min = t_pointer-1-track_struct['track_params']['linear_pred_thresh']
                xx = (track_struct['tracklet_mat']['xmin_mat'][temp_track_id,int(t_min):t_pointer]
                      +track_struct['tracklet_mat']['xmax_mat'][temp_track_id,int(t_min):t_pointer])/2
                yy = (track_struct['tracklet_mat']['ymin_mat'][temp_track_id,int(t_min):t_pointer]
                      +track_struct['tracklet_mat']['ymax_mat'][temp_track_id,int(t_min):t_pointer])/2
                ww = (track_struct['tracklet_mat']['xmax_mat'][temp_track_id,int(t_min):t_pointer]
                      -track_struct['tracklet_mat']['xmin_mat'][temp_track_id,int(t_min):t_pointer])+1
                hh = (track_struct['tracklet_mat']['ymax_mat'][temp_track_id,int(t_min):t_pointer]
                      -track_struct['tracklet_mat']['ymin_mat'][temp_track_id,int(t_min):t_pointer])+1
                if len(xx)==0:
                    import pdb; pdb.set_trace()
                pred_x = track_lib.linear_pred(xx)
                pred_y = track_lib.linear_pred(yy)
                pred_w = track_lib.linear_pred(ww)
                pred_h = track_lib.linear_pred(hh)
                pred_bbox1[k,2] = max(pred_w,1)
                pred_bbox1[k,3] = max(pred_h,1)
                pred_bbox1[k,0] = pred_x-pred_w/2
                pred_bbox1[k,1] = pred_y-pred_h/2
                
            overlap_mat,_,_,_ = track_lib.get_overlap(pred_bbox1, bbox2)
            # color dist
            color_dist = np.zeros((prev_num_bbox,num_bbox))
            for n1 in range(prev_num_bbox):
                for n2 in range(num_bbox):
                    cnn_fea_idx1 = np.where(np.logical_and(track_struct['tracklet_mat']['appearance_fea_mat'][:,0]==prev_bbox_idx[n1],
                                                          track_struct['tracklet_mat']['appearance_fea_mat'][:,1]==fr_idx-1))[0]
                    cnn_fea_idx2 = empty_fea_idx[n2]
                    cnn_fea1 = track_struct['tracklet_mat']['appearance_fea_mat'][cnn_fea_idx1,2:]
                    cnn_fea2 = track_struct['tracklet_mat']['appearance_fea_mat'][cnn_fea_idx2,2:]
                    color_dist[n1,n2] = np.linalg.norm(cnn_fea1-cnn_fea2, 2)
                               
            overlap_mat[color_dist>track_struct['track_params']['color_thresh']] = 0    
            idx1, idx2 = track_lib.bbox_associate(overlap_mat, track_struct['track_params']['IOU_thresh'])
            #if fr_idx==14:
            #import pdb; pdb.set_trace()
            
            # assign the tracklet_mat
            if len(idx1)==0:
                track_struct['tracklet_mat']['obj_id_mat'][empty_idx[0:num_bbox]] = avai_ids[0:num_bbox]
                track_struct['tracklet_mat']['track_id_mat'][empty_idx[0:num_bbox]] = np.array(range(num_bbox),dtype=int)+max_track_id+1
                track_struct['tracklet_mat']['xmin_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,1]
                track_struct['tracklet_mat']['ymin_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,2]
                track_struct['tracklet_mat']['xmax_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,1]+det_M[:,3]
                track_struct['tracklet_mat']['ymax_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,2]+det_M[:,4]
                track_struct['tracklet_mat']['x_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,6]
                track_struct['tracklet_mat']['y_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,7]
                track_struct['tracklet_mat']['w_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,8]
                track_struct['tracklet_mat']['h_3d_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,9]
                track_struct['tracklet_mat']['det_score_mat'][empty_idx[0:num_bbox],t_pointer] = det_M[:,5]
                track_struct['tracklet_mat']['track_interval'][empty_idx[0:num_bbox],0] = t_pointer
                track_struct['tracklet_mat']['track_interval'][empty_idx[0:num_bbox],1] = t_pointer
                track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[0:len(patch_list)],0] = empty_idx[0:num_bbox]
                track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[0:len(patch_list)],1] = fr_idx
                #track_struct['tracklet_mat']['class_name'][empty_idx[0:num_bbox],t_pointer] = det_M[:,10]
                #track_struct['tracklet_mat']['dist2cam'][empty_idx[0:num_bbox],t_pointer] = det_M[:,11]
            else:
                cnt1 = -1
                cnt2 = -1
                for n in range(num_bbox):
                    if n not in list(idx2):
                        cnt1 = cnt1+1
                        track_struct['tracklet_mat']['obj_id_mat'][empty_idx[cnt1]] \
                            = avai_ids[cnt1]
                        track_struct['tracklet_mat']['track_id_mat'][empty_idx[cnt1]] \
                            = cnt1+max_track_id+1
                        track_struct['tracklet_mat']['xmin_mat'][empty_idx[cnt1],t_pointer] = det_M[n,1]
                        track_struct['tracklet_mat']['ymin_mat'][empty_idx[cnt1],t_pointer] = det_M[n,2]
                        track_struct['tracklet_mat']['xmax_mat'][empty_idx[cnt1],t_pointer] = det_M[n,1]+det_M[n,3]
                        track_struct['tracklet_mat']['ymax_mat'][empty_idx[cnt1],t_pointer] = det_M[n,2]+det_M[n,4]
                        track_struct['tracklet_mat']['x_3d_mat'][empty_idx[cnt1],t_pointer] = det_M[n,6]
                        track_struct['tracklet_mat']['y_3d_mat'][empty_idx[cnt1],t_pointer] = det_M[n,7]
                        track_struct['tracklet_mat']['w_3d_mat'][empty_idx[cnt1],t_pointer] = det_M[n,8]
                        track_struct['tracklet_mat']['h_3d_mat'][empty_idx[cnt1],t_pointer] = det_M[n,9]
                        track_struct['tracklet_mat']['det_score_mat'][empty_idx[cnt1],t_pointer] = det_M[n,5]
                        track_struct['tracklet_mat']['track_interval'][empty_idx[cnt1],0] = t_pointer
                        track_struct['tracklet_mat']['track_interval'][empty_idx[cnt1],1] = t_pointer
                        track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[n],0] = empty_idx[cnt1]
                        track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[n],1] = fr_idx
                        #track_struct['tracklet_mat']['class_name'][empty_idx[cnt1],t_pointer] = det_M[n,10]
                        #track_struct['tracklet_mat']['dist2cam'][empty_idx[cnt1],t_pointer] = det_M[n,11]
                    else:
                        temp_idx = np.where(idx2==n)[0]                       
                        track_struct['tracklet_mat']['xmin_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,1]
                        track_struct['tracklet_mat']['ymin_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,2]
                        track_struct['tracklet_mat']['xmax_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,1]+det_M[n,3]
                        track_struct['tracklet_mat']['ymax_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,2]+det_M[n,4]
                        track_struct['tracklet_mat']['x_3d_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,6]
                        track_struct['tracklet_mat']['y_3d_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,7]
                        track_struct['tracklet_mat']['w_3d_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,8]
                        track_struct['tracklet_mat']['h_3d_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,9]
                        track_struct['tracklet_mat']['det_score_mat'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,5]
                        track_struct['tracklet_mat']['track_interval'][prev_bbox_idx[idx1[temp_idx[0]]],1] = t_pointer
                        track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[n],0] = prev_bbox_idx[idx1[temp_idx[0]]]
                        track_struct['tracklet_mat']['appearance_fea_mat'][empty_fea_idx[n],1] = fr_idx
                        #track_struct['tracklet_mat']['class_name'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,10]
                        #track_struct['tracklet_mat']['dist2cam'][prev_bbox_idx[idx1[temp_idx[0]]],t_pointer] = det_M[n,11]
    
    track_struct['tracklet_mat']['assigned_obj_id_mask'] = track_struct['tracklet_mat']['save_obj_id_mask'].copy()
    assigned_ids = track_struct['tracklet_mat']['obj_id_mat'][track_struct['tracklet_mat']['obj_id_mat']!=-1]
    track_struct['tracklet_mat']['assigned_obj_id_mask'][assigned_ids] = 1
    avai_ids = np.where(track_struct['tracklet_mat']['assigned_obj_id_mask']==0)[0]
    
    # Tracklet clustering
    
    iters = 20
    if fr_idx%track_struct['track_params']['clustering_period']==track_struct['track_params']['clustering_period']-1 or end_flag==1:
        for n in range(iters):
            # print("iteration")
            # print(n)
            change_flag = tracklet_clustering(tracklet_sess, n)
            if change_flag==0:
                #import pdb; pdb.set_trace()
                #time_check_flag = time_cluster_check()
                break
    
        # Update tracklet
        debug_mode = 0
        if fr_idx-track_struct['track_params']['num_fr']>190:
            debug_mode = 1
        
        print('-------')
        print(debug_mode)
        print(fr_idx-track_struct['track_params']['num_fr'])
        print('-------')
        post_processing(debug_mode)
    
    # for the last frame, save all the info to file
    if end_flag==1:
        for n in range(track_struct['tracklet_mat']['xmin_mat'].shape[1]):
            
            track_idx = np.where(track_struct['tracklet_mat']['xmin_mat'][:,n]!=-1)[0]
            num_save_id = len(track_idx)
            if num_save_id!=0:
                save_mat = np.zeros((num_save_id, 12))
                save_mat[:,0] = fr_idx-track_struct['track_params']['num_fr']+n+1
                save_mat[:,1] = track_struct['tracklet_mat']['obj_id_mat'][track_idx]
                track_struct['tracklet_mat']['save_obj_id_mask'][save_mat[:,1].astype(int)] = 1
                save_mat[:,2] = track_struct['tracklet_mat']['track_id_mat'][track_idx]
                save_mat[:,3] = track_struct['tracklet_mat']['xmin_mat'][track_idx,n]
                save_mat[:,4] = track_struct['tracklet_mat']['ymin_mat'][track_idx,n]
                save_mat[:,5] = track_struct['tracklet_mat']['xmax_mat'][track_idx,n] \
                    -track_struct['tracklet_mat']['xmin_mat'][track_idx,n]
                save_mat[:,6] = track_struct['tracklet_mat']['ymax_mat'][track_idx,n] \
                    -track_struct['tracklet_mat']['ymin_mat'][track_idx,n]
                save_mat[:,7] = track_struct['tracklet_mat']['x_3d_mat'][track_idx,n]
                save_mat[:,8] = track_struct['tracklet_mat']['y_3d_mat'][track_idx,n]
                save_mat[:,9] = track_struct['tracklet_mat']['w_3d_mat'][track_idx,n]
                save_mat[:,10] = track_struct['tracklet_mat']['h_3d_mat'][track_idx,n]
                save_mat[:,11] = track_struct['tracklet_mat']['det_score_mat'][track_idx,n]
                #save_mat[:,12] = track_struct['tracklet_mat']['class_name'][track_idx,n]
                #save_mat[:,13] = track_struct['tracklet_mat']['dist2cam'][track_idx,n]
                f = open(track_struct['file_path']['txt_result_path'], 'a')  
                np.savetxt(f, save_mat, delimiter=',')
                f.close()
                wrt_missing_det(save_mat)
        
    #import pdb; pdb.set_trace()
    t_pointer = prev_t_pointer
    return

def init_TC_tracker():
    global track_struct
    
    track_struct = {'track_params':{}, 'file_path':{}} 
    track_struct['file_path']['seq_name'] = '1'
    track_struct['file_path']['img_name'] = '1'
    track_struct['file_path']['sub_seq_name'] = ''
    # track_struct['file_path']['det_path'] = 'D:/Data/KITTI/'+track_struct['file_path']['seq_name']+'/dets2.txt'
    # track_struct['file_path']['img_folder'] = 'D:/Data/KITTI/'+track_struct['file_path']['img_name'] \
    #     +track_struct['file_path']['sub_seq_name']+'/image_02/data'
    # track_struct['file_path']['crop_det_folder'] = 'D:/Data/KITTI/temp_crop'
    # track_struct['file_path']['triplet_model'] = 'D:/Data/UA-Detrac/UA_Detrac_model/KITTI_model'
    # track_struct['file_path']['seq_model'] = 'D:/Data/UA-Detrac/KITTI_model/model.ckpt'
    # track_struct['file_path']['tracking_img_folder'] = 'D:/Data/KITTI/tracking_img/'+track_struct['file_path']['seq_name'] \
    #     +track_struct['file_path']['sub_seq_name']
    # track_struct['file_path']['tracking_video_path'] = 'D:/Data/KITTI/tracking_video/'+track_struct['file_path']['seq_name'] \
    #     +track_struct['file_path']['sub_seq_name']+'.avi'
    # track_struct['file_path']['txt_result_path'] = 'D:/Data/KITTI/txt_result/'+track_struct['file_path']['seq_name'] \
    #     +track_struct['file_path']['sub_seq_name']+'.txt'
    # if os.path.isfile(track_struct['file_path']['txt_result_path']):
    #     os.remove(track_struct['file_path']['txt_result_path'])
    # track_struct['file_path']['track_struct_path'] = 'D:/Data/KITTI/track_struct/'+track_struct['file_path']['seq_name'] \
    #     +track_struct['file_path']['sub_seq_name']+'.obj'



    track_struct['file_path']['det_path'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/chongqing/det/1_new.txt'
    track_struct['file_path']['img_folder'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/chongqing/img_folder/1'
    track_struct['file_path']['crop_det_folder'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/chongqing/temp'
    track_struct['file_path']['triplet_model'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/update_facenet/UA_Detrac_model/MOT'
    track_struct['file_path']['seq_model'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/MOT_2d_v2/model.ckpt'
    track_struct['file_path']['tracking_img_folder'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/chongqing/tracking_img/'+track_struct['file_path']['seq_name'] \
        +track_struct['file_path']['sub_seq_name']
    track_struct['file_path']['tracking_video_path'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/chongqing/tracking_video/'+track_struct['file_path']['seq_name'] \
        +track_struct['file_path']['sub_seq_name']+'.avi'
    track_struct['file_path']['txt_result_path'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/chongqing/txt_result/'+track_struct['file_path']['seq_name'] \
        +track_struct['file_path']['sub_seq_name']+'.txt'
        
    if os.path.isfile(track_struct['file_path']['txt_result_path']):
        os.remove(track_struct['file_path']['txt_result_path'])
    
    track_struct['file_path']['track_struct_path'] = 'C:/Users/tangz/OneDrive/Documents/Gaoang/chongqing/appear_mat/'+track_struct['file_path']['seq_name'] \
        +track_struct['file_path']['sub_seq_name']+'.obj'


    track_struct['track_params']['num_fr'] = 64 
    track_struct['track_params']['num_track'] = 1000
    track_struct['track_params']['num_max_det'] = 10000
    track_struct['track_params']['max_num_obj'] = 10000
    track_struct['track_params']['IOU_thresh'] = 0.5#0.3 
    track_struct['track_params']['color_thresh'] = 5
    track_struct['track_params']['det_thresh'] = -2 
    track_struct['track_params']['linear_pred_thresh'] = 5 
    track_struct['track_params']['t_dist_thresh'] = 60
    track_struct['track_params']['track_overlap_thresh'] = 0 
    track_struct['track_params']['search_radius'] = 1
    track_struct['track_params']['const_fr_thresh'] = 1 
    track_struct['track_params']['crop_size'] = 182 
    track_struct['track_params']['loc_scales'] = [1352,700,1352,700]#[100,30,5,5]
    track_struct['track_params']['clustering_period'] = 20
    track_struct['track_params']['time_cluster_dist'] = 100
    track_struct['track_params']['file_name_len'] = 6
    track_struct['track_params']['num_time_cluster'] \
        = int(np.ceil(track_struct['track_params']['num_fr']/track_struct['track_params']['time_cluster_dist']))
        
    track_struct['track_params']['max_length'] = 64
    track_struct['track_params']['feature_size'] = 4+512
    track_struct['track_params']['batch_size'] = 64
    track_struct['track_params']['num_classes'] = 2


    track_struct['tracklet_mat'] = {'track_id_mat':[], 'xmin_mat':[], 'ymin_mat':[], 'xmax_mat':[], 'ymax_mat':[], 'x_3d_mat':[], 
                                    'y_3d_mat':[], 'w_3d_mat':[], 'h_3d_mat':[], 'det_score_mat':[], 'track_interval':[], 
                                    'obj_id_mat':[], 'appearance_fea_mat':[]}
    
    track_struct['tracklet_mat']['track_id_mat'] = -np.ones(track_struct['track_params']['num_track'], dtype=int)
    track_struct['tracklet_mat']['obj_id_mat'] = -np.ones(track_struct['track_params']['num_track'], dtype=int)
    track_struct['tracklet_mat']['xmin_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['ymin_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['xmax_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['ymax_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['x_3d_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['y_3d_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['w_3d_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['h_3d_mat'] = -np.ones((track_struct['track_params']['num_track'],
                                                         track_struct['track_params']['num_fr']))
    track_struct['tracklet_mat']['det_score_mat'] = \
        -np.ones((track_struct['track_params']['num_track'],track_struct['track_params']['num_fr']))

    #track_struct['tracklet_mat']['class_name'] = -np.ones((track_struct['track_params']['num_track'],
     #                                                    track_struct['track_params']['num_fr']))
    #track_struct['tracklet_mat']['dist2cam'] = -np.ones((track_struct['track_params']['num_track'],
       #                                                  track_struct['track_params']['num_fr']))

    track_struct['tracklet_mat']['track_interval'] = -np.ones((track_struct['track_params']['num_track'],2), dtype=int)
    track_struct['tracklet_mat']['prev_track_interval'] = -np.ones((track_struct['track_params']['num_track'],2), dtype=int)
    track_struct['tracklet_mat']['appearance_fea_mat'] = -np.ones((track_struct['track_params']['num_max_det'],
                                                                   track_struct['track_params']['feature_size']-4+2))
    
    track_struct['tracklet_mat']['comb_track_cost'] = np.zeros((track_struct['track_params']['num_track'],
                                                                track_struct['track_params']['num_track']))
    track_struct['tracklet_mat']['comb_track_cost_mask'] = np.zeros((track_struct['track_params']['num_track'],
                                                                track_struct['track_params']['num_track']),dtype=int) 
    track_struct['tracklet_mat']['save_obj_id_mask'] = np.zeros(track_struct['track_params']['max_num_obj'],dtype=int)
    track_struct['tracklet_mat']['assigned_obj_id_mask'] = np.zeros(track_struct['track_params']['max_num_obj'],dtype=int)
    track_struct['tracklet_mat']['imgs'] = []
    track_struct['tracklet_mat']['color_table'] = track_lib.color_table(track_struct['track_params']['max_num_obj'])
    
    # fr_id, track_id, xmin, ymin, xmax, ymax, x, y, w, h, det_score
    track_struct['tracklet_mat']['obj_end_fr_info'] = -np.ones((track_struct['track_params']['max_num_obj'],11))
    
    # remove folder
    if os.path.isdir(track_struct['file_path']['crop_det_folder']):
        shutil.rmtree(track_struct['file_path']['crop_det_folder'])
        
    return
    
def TC_tracker():
    global track_struct
    init_TC_tracker()
    
    # initialize triplet model
    global triplet_graph
    global triplet_sess
    init_triplet_model()

    # initialize tracklet model
    global tracklet_graph
    global tracklet_sess
    init_tracklet_model()
    
    M = track_lib.load_detection(track_struct['file_path']['det_path'], 'chongqing') 
    total_num_fr = int(M[-1,0]+1) 
    
    t_pointer = 0
    for n in range(total_num_fr):
        print("Frame %d" % n)
        # print("t_pointer %d" % t_pointer)
        fr_idx = n
        idx = np.where(np.logical_and(M[:,0]==fr_idx,M[:,5]>track_struct['track_params']['det_thresh']))[0]
        if len(idx)>1:
            choose_idx, _ = track_lib.merge_bbox(M[idx,1:5], 0.7, M[idx,5],1)
            #import pdb; pdb.set_trace()
            temp_M = M[idx[choose_idx],:]
        else:
            temp_M = M[idx,:]
        
        img_name = track_lib.file_name(fr_idx+1,track_struct['track_params']['file_name_len'])+'.jpg'
        img_path = track_struct['file_path']['img_folder']+'/'+img_name
        img = misc.imread(img_path) 

        if fr_idx==total_num_fr-1:
            end_flag = 1
        else:
            end_flag = 0
            
        TC_online(temp_M, img, t_pointer, fr_idx, end_flag)  
        t_pointer = t_pointer+1
        if t_pointer>track_struct['track_params']['num_fr']:
            t_pointer = track_struct['track_params']['num_fr']
    
    
    # draw all results
    M = np.loadtxt(track_struct['file_path']['txt_result_path'], delimiter=',')
    M = np.array(M)
    for n in range(total_num_fr):
        fr_idx = n
        img_name = track_lib.file_name(fr_idx+1,track_struct['track_params']['file_name_len'])+'.jpg'
        img_path = track_struct['file_path']['img_folder']+'/'+img_name
        img = misc.imread(img_path) 
        
        temp_M = M[M[:,0]==fr_idx,:]
        draw_result(img, temp_M, fr_idx)
    
    convert_frames_to_video(track_struct['file_path']['tracking_img_folder']+'/', track_struct['file_path']['tracking_video_path'], 30)
    
    return track_struct

if __name__ == '__main__':

    TC_tracker()


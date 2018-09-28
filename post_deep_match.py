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
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn import svm
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import seq_nn_3d
import track_lib
import tracklet_utils_3c

seq_name = 'MOT17-03-FRCNN'
img_name = 'MOT17-03'
sub_seq_name = ''
track_struct_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/track_struct/'+seq_name+sub_seq_name+'.obj'
appear_mat_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/appear_mat/'+seq_name+'.obj'
tracking_img_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/tracking_img/'+seq_name+sub_seq_name
img_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/MOT17Det/test/'+img_name+sub_seq_name+'/img1'
tracking_video_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/tracking_video/'+seq_name+sub_seq_name+'.avi'
txt_result_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/txt_result/'+seq_name+sub_seq_name+'.txt'

max_time_dist = 130
cost_thresh = 9.5 #8.7
min_cost_thresh = 6 #6.1
x_dist_thresh = 200
y_dist_thresh = 200

global track_struct
global remove_set
remove_set = []
#remove_set = [4,25,142,183,275,346]

track_struct = pickle.load(open(track_struct_path,'rb'))
appear_mat = pickle.load(open(appear_mat_path,'rb'))

#import pdb; pdb.set_trace()

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
    
def draw_result(img_folder, save_folder): 
    #track_struct = pickle.load(open(track_struct_path,'rb'))
    
    global track_struct
    tracklet_mat = track_struct['final_tracklet_mat']
    img_list = os.listdir(img_folder) 
    table = track_lib.color_table(len(tracklet_mat['track_cluster'])) 
    #import pdb; pdb.set_trace()
    for n in range(track_struct['final_tracklet_mat']['xmin_mat'].shape[1]): 
        img_name = track_lib.file_name(n+1,6)+'.jpg'
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


N_cluster = len(track_struct['final_tracklet_mat']['track_cluster'])
track_interval = np.zeros((N_cluster,2), dtype=int)
for n in range(N_cluster):
    if n in remove_set:
        track_struct['final_tracklet_mat']['track_cluster'][n] = []
        continue
    idx = np.where(track_struct['final_tracklet_mat']['xmin_mat'][n,:]!=-1)[0]
    track_interval[n,0] = np.min(idx)
    track_interval[n,1] = np.max(idx)

mean_cost_D = np.Inf*np.ones((N_cluster,N_cluster))
min_cost_D = np.Inf*np.ones((N_cluster,N_cluster))
max_cost_D = np.Inf*np.ones((N_cluster,N_cluster))
motion_cost = np.Inf*np.ones((N_cluster,N_cluster))
angle_cost = np.Inf*np.ones((N_cluster,N_cluster))


cost_mask = np.zeros((N_cluster,N_cluster), dtype=int)
conflict_mask = np.zeros((N_cluster,N_cluster), dtype=int)
for n1 in range(N_cluster-1):
    print(n1)
    for n2 in range(n1+1,N_cluster):
        if ((n1 in remove_set) or (n2 in remove_set)):
            continue
            
        if ((track_interval[n1,0]>=track_interval[n2,0] and track_interval[n1,0]<=track_interval[n2,1]) or 
            (track_interval[n1,1]>=track_interval[n2,0] and track_interval[n1,1]<=track_interval[n2,1])):
            conflict_mask[n1,n2] = 1
            conflict_mask[n2,n1] = 1
            continue
        
        if ((track_interval[n1,0]-track_interval[n2,1]>=0 and track_interval[n1,0]-track_interval[n2,1]<=max_time_dist) or
            (track_interval[n2,0]-track_interval[n1,1]>=0 and track_interval[n2,0]-track_interval[n1,1]<=max_time_dist)):
            track_set1 = track_struct['final_tracklet_mat']['track_cluster'][n1]
            track_set2 = track_struct['final_tracklet_mat']['track_cluster'][n2]
            if np.mean(np.array(track_set1))<np.mean(np.array(track_set2)):
                track_id1 = int(np.max(np.array(track_set1)))
                track_id2 = int(np.min(np.array(track_set2)))
            else:
                track_id1 = int(np.max(np.array(track_set2)))
                track_id2 = int(np.min(np.array(track_set1)))
            t_idx1 = np.where(track_struct['tracklet_mat']['xmin_mat'][track_id1,:]!=-1)[0]
            t_max = np.max(t_idx1)
            x_center1 = (track_struct['tracklet_mat']['xmin_mat'][track_id1,t_max]
                        +track_struct['tracklet_mat']['xmax_mat'][track_id1,t_max])/2
            y_center1 = (track_struct['tracklet_mat']['ymin_mat'][track_id1,t_max]
                        +track_struct['tracklet_mat']['ymax_mat'][track_id1,t_max])/2
            
            t_idx2 = np.where(track_struct['tracklet_mat']['xmin_mat'][track_id2,:]!=-1)[0]
            t_min = np.min(t_idx2)
            x_center2 = (track_struct['tracklet_mat']['xmin_mat'][track_id2,t_min]
                        +track_struct['tracklet_mat']['xmax_mat'][track_id2,t_min])/2
            y_center2 = (track_struct['tracklet_mat']['ymin_mat'][track_id2,t_min]
                        +track_struct['tracklet_mat']['ymax_mat'][track_id2,t_min])/2
            if abs(x_center1-x_center2)>x_dist_thresh or abs(y_center1-y_center2)>y_dist_thresh:
                continue
                
            idx1 = []
            idx2 = []
            for k in range(len(track_set1)):
                idx1.extend(np.where(appear_mat[:,0]==track_set1[k]+1)[0])
            for k in range(len(track_set2)):
                idx2.extend(np.where(appear_mat[:,0]==track_set2[k]+1)[0])
            idx1 = np.array(idx1,dtype=int)
            idx2 = np.array(idx2,dtype=int)
            X1 = appear_mat[idx1,2:]
            X2 = appear_mat[idx2,2:]
            #print(X2.shape)
            temp_dist = spatial.distance.cdist(X1, X2, 'euclidean')
            cost_mask[n1,n2] = 1
            cost_mask[n2,n1] = 1
            mean_cost_D[n1,n2] = np.mean(temp_dist)
            mean_cost_D[n2,n1] = mean_cost_D[n1,n2]
            min_cost_D[n1,n2] = np.min(temp_dist)
            min_cost_D[n2,n1] = min_cost_D[n1,n2]
            max_cost_D[n1,n2] = np.max(temp_dist)
            max_cost_D[n2,n1] = max_cost_D[n1,n2]

new_cost_D = mean_cost_D.copy()
merge_idx = []
merge_cost = []
while 1:
    min_idx = np.unravel_index(np.argmin(new_cost_D, axis=None), new_cost_D.shape)
    mean_v = new_cost_D[min_idx]
    if mean_v>cost_thresh:
        break
    min_v = min_cost_D[min_idx]
    max_v = max_cost_D[min_idx]   
    idx1 = np.where(conflict_mask[min_idx[0],:]==1)[0]
    idx2 = np.where(conflict_mask[min_idx[1],:]==1)[0]
    new_cost_D[min_idx[1],idx1] = np.Inf
    new_cost_D[idx1,min_idx[1]] = np.Inf
    new_cost_D[min_idx[0],idx2] = np.Inf
    new_cost_D[idx2,min_idx[0]] = np.Inf
    new_cost_D[min_idx[0],min_idx[1]] = np.Inf
    new_cost_D[min_idx[1],min_idx[0]] = np.Inf
    conflict_mask[min_idx[1],idx1] = 1
    conflict_mask[idx1,min_idx[1]] = 1
    conflict_mask[min_idx[0],idx2] = 1
    conflict_mask[idx2,min_idx[0]] = 1
    if min_v<min_cost_thresh:
        merge_idx.append(min_idx)
        merge_cost.append([mean_v,min_v,max_v])
merge_idx = np.array(merge_idx,dtype=int)
import pdb; pdb.set_trace()

# update track_struct
track_struct['tracklet_mat']['track_cluster'] = track_struct['final_tracklet_mat']['track_cluster'].copy()
for n in range(len(merge_idx)):
    track_struct['tracklet_mat']['track_cluster'][merge_idx[n,0]].extend(track_struct['tracklet_mat']['track_cluster'][merge_idx[n,1]])
    track_struct['tracklet_mat']['track_cluster'][merge_idx[n,1]] = []
    merge_idx[merge_idx[:,0]==merge_idx[n,1],0] = merge_idx[n,0]
    merge_idx[merge_idx[:,1]==merge_idx[n,1],0] = merge_idx[n,0]

import pdb; pdb.set_trace()
post_processing()
    
wrt_txt(track_struct['final_tracklet_mat'])

draw_result(img_folder, tracking_img_folder)

convert_frames_to_video(tracking_img_folder+'/', tracking_video_path, 20)
import pdb; pdb.set_trace()

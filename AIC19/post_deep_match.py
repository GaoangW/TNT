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


seq_name = 'c029'
img_name = 'c029'
sub_seq_name = ''
file_len = 6
root_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/AI_City_2019'
track_struct_path = root_folder+'/track_struct/'+seq_name+sub_seq_name+'.obj'
#appear_mat_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/appear_mat/'+seq_name+'.obj'
tracking_img_folder = root_folder+'/tracking_img/'+seq_name+sub_seq_name
img_folder = root_folder+'/test_img/'+img_name
tracking_video_path = root_folder+'/tracking_video/'+seq_name+sub_seq_name+'.avi'
txt_result_path = root_folder+'/txt_result/'+seq_name+sub_seq_name+'.txt'
params_path = root_folder+'/params/'+seq_name+'.obj'

'''
seq_name = 'MOT17-14-FRCNN'
img_name = 'MOT17-14'
sub_seq_name = ''
file_len = 6
root_folder = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17'
track_struct_path = root_folder+'/track_struct/'+seq_name+sub_seq_name+'.obj'
#appear_mat_path = 'C:/Users/tangz/OneDrive/Documents/Gaoang/MOT17/appear_mat/'+seq_name+'.obj'
tracking_img_folder = root_folder+'/tracking_img_deep_match/'+seq_name+sub_seq_name
img_folder = root_folder+'/MOT17Det/test/'+img_name+'/img1'
tracking_video_path = root_folder+'/tracking_video_deep_match/'+seq_name+sub_seq_name+'.avi'
txt_result_path = root_folder+'/txt_result_deep_match/'+seq_name+sub_seq_name+'.txt'
params_path = root_folder+'/params/'+seq_name+'.obj'
'''
#params = pickle.load(open(params_path,'rb'))
#import pdb; pdb.set_trace()

global params
params = {}
params['max_time_dist'] = 30  # 60
params['cost_thresh'] = 12
params['min_cost_thresh'] = 6 # car 6.6      # person 8
params['x_dist_thresh'] = 150 #300
params['y_dist_thresh'] = 150 #300
params['angle_cost_thresh'] = 40 #60
params['motion_cost_thresh'] = 50 #1.5 #5 # 50 for car    # 5 for person
params['bnd_margin'] = 130
params['min_len'] = 60 #600
params['extend_len'] = 30
params['reg_thresh'] = 0.15
params['speed_thresh'] = 0.05
params['static_len'] = 1400
params['remove_set'] = []
# mode = [0: appearance_cost, 1: angle_cost]
params['mode'] = 0 #0
max_time_dist = params['max_time_dist']
cost_thresh = params['cost_thresh']
min_cost_thresh =params['min_cost_thresh']
x_dist_thresh = params['x_dist_thresh']
y_dist_thresh = params['y_dist_thresh']
angle_cost_thresh = params['angle_cost_thresh']
motion_cost_thresh = params['motion_cost_thresh']

pickle.dump(params, open(params_path,'wb'))
img_size = [1920,1080]

direction_max_len = 60
direction_min_len = 5 #5

global track_struct
global remove_set
remove_set = params['remove_set']


track_struct = pickle.load(open(track_struct_path,'rb'))
appear_mat = track_struct['tracklet_mat']['appearance_fea_mat'].copy()
#appear_mat = pickle.load(open(appear_mat_path,'rb'))

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
        img_name = track_lib.file_name(n+1,file_len)+'.jpg'
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
    '''
    for n in range(num_cluster):  
        cand_t = np.where(final_xmin_mat[n,:]!=-1)[0]
        if len(cand_t)==0:
            continue
        t1 = int(np.min(cand_t))
        t2 = int(np.max(cand_t))
        for k in range(t1,t2+1):
            t_start = max(k-window_size,t1)
            t_end = min(k+window_size,t2)
            
            final_tracklet_mat['xmin_mat'][n,k] = (np.sum(final_xmin_mat[n,t_start:t_end+1])+abs(t_start-k+window_size)*final_xmin_mat[n,t_start]+abs(t_end-k-window_size)*final_xmin_mat[n,t_end])/(2*window_size+1)
            final_tracklet_mat['ymin_mat'][n,k] = (np.sum(final_ymin_mat[n,t_start:t_end+1])+abs(t_start-k+window_size)*final_ymin_mat[n,t_start]+abs(t_end-k-window_size)*final_ymin_mat[n,t_end])/(2*window_size+1)
            final_tracklet_mat['xmax_mat'][n,k] = (np.sum(final_xmax_mat[n,t_start:t_end+1])+abs(t_start-k+window_size)*final_xmax_mat[n,t_start]+abs(t_end-k-window_size)*final_xmax_mat[n,t_end])/(2*window_size+1)
            final_tracklet_mat['ymax_mat'][n,k] = (np.sum(final_ymax_mat[n,t_start:t_end+1])+abs(t_start-k+window_size)*final_ymax_mat[n,t_start]+abs(t_end-k-window_size)*final_ymax_mat[n,t_end])/(2*window_size+1)
            final_tracklet_mat['det_score_mat'][n,k] = (np.sum(final_det_score_mat[n,t_start:t_end+1])+abs(t_start-k+window_size)*final_det_score_mat[n,t_start]+abs(t_end-k-window_size)*final_det_score_mat[n,t_end])/(2*window_size+1)
    
    '''        
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
    temp_remove_set = []
    for n in range(N_cluster):
        if len(tracklet_mat["track_cluster"][n])==0:
            remove_idx.append(n)
            continue
        if tracklet_mat["track_cluster"][n][0] in temp_remove_set:
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







#import pdb; pdb.set_trace()
new_tracklet_mat = {}

cc = 0
while 1:
    cc = cc+1
    #******************************************************************************************************
    #max_time_dist = max_time_dist*cc
    #*****************************************************************************************************
    new_tracklet_mat['track_cluster'] = track_struct['final_tracklet_mat']['track_cluster'].copy()
    new_tracklet_mat['xmin_mat'] = track_struct['final_tracklet_mat']['xmin_mat'].copy()
    new_tracklet_mat['ymin_mat'] = track_struct['final_tracklet_mat']['ymin_mat'].copy()
    new_tracklet_mat['xmax_mat'] = track_struct['final_tracklet_mat']['xmax_mat'].copy()
    new_tracklet_mat['ymax_mat'] = track_struct['final_tracklet_mat']['ymax_mat'].copy()


    N_cluster = len(new_tracklet_mat['track_cluster'])
    track_interval = np.zeros((N_cluster,2), dtype=int)
    for n in range(N_cluster):
        #track_lib.show_trajectory(new_tracklet_mat, n)
        if n in remove_set and cc==1:
            new_tracklet_mat['track_cluster'][n] = []
            continue
        idx = np.where(new_tracklet_mat['xmin_mat'][n,:]!=-1)[0]
        track_interval[n,0] = np.min(idx)
        track_interval[n,1] = np.max(idx)

    mean_cost_D = np.Inf*np.ones((N_cluster,N_cluster))
    min_cost_D = np.Inf*np.ones((N_cluster,N_cluster))
    max_cost_D = np.Inf*np.ones((N_cluster,N_cluster))
    motion_cost = np.Inf*np.ones((N_cluster,N_cluster))
    angle_cost = np.Inf*np.ones((N_cluster,N_cluster))

    # [start_flag, angle, end_flag, angle], angle from 0 to 360, -1 for unestimated.
    track_angle = np.zeros((N_cluster,4))
    for n in range(N_cluster):
        check_idx = np.where(new_tracklet_mat['xmin_mat'][n,:]!=-1)[0]
        if len(check_idx)<direction_min_len:
            track_angle[n,1] = -1
            track_angle[n,3] = -1
            continue
    
        check_length = min(direction_max_len, len(check_idx))
    
        # start direction
        xmin = new_tracklet_mat['xmin_mat'][n,check_idx[0]:check_idx[0]+check_length]
        ymin = new_tracklet_mat['ymin_mat'][n,check_idx[0]:check_idx[0]+check_length]
        xmax = new_tracklet_mat['xmax_mat'][n,check_idx[0]:check_idx[0]+check_length]
        ymax = new_tracklet_mat['ymax_mat'][n,check_idx[0]:check_idx[0]+check_length]
        x_center = (xmin+xmax)/2
        y_center = (ymin+ymax)/2
        A = np.ones((check_length,2))
        b = np.zeros((check_length,1))
        A[:,0] = x_center
        b[:,0] = y_center
        #import pdb; pdb.set_trace()
        p = np.matmul(np.linalg.pinv(A),b)
        temp_angle = np.arctan(p[0])*180/np.pi
        if x_center[-1]-x_center[0]<0:
            temp_angle = temp_angle+180
        if temp_angle<0:
            temp_angle = temp_angle+360
        track_angle[n,1] = temp_angle
        x_pred = track_lib.linear_pred_v2(np.array(range(len(x_center))), x_center, np.array([-1]))
        y_pred = track_lib.linear_pred_v2(np.array(range(len(y_center))), y_center, np.array([-1]))
        if x_pred<0 or x_pred>img_size[0] or y_pred<0 or y_pred>img_size[1]:
            track_angle[n,0] = 1
            
        # end direction
        xmin = new_tracklet_mat['xmin_mat'][n,check_idx[-1]-check_length+1:check_idx[-1]+1]
        ymin = new_tracklet_mat['ymin_mat'][n,check_idx[-1]-check_length+1:check_idx[-1]+1]
        xmax = new_tracklet_mat['xmax_mat'][n,check_idx[-1]-check_length+1:check_idx[-1]+1]
        ymax = new_tracklet_mat['ymax_mat'][n,check_idx[-1]-check_length+1:check_idx[-1]+1]
        x_center = (xmin+xmax)/2
        y_center = (ymin+ymax)/2
        A = np.ones((check_length,2))
        b = np.zeros((check_length,1))  
        A[:,0] = x_center
        b[:,0] = y_center        
        p = np.matmul(np.linalg.pinv(A),b)
        temp_angle = np.arctan(p[0])*180/np.pi
        if x_center[-1]-x_center[0]<0:
            temp_angle = temp_angle+180
        if temp_angle<0:
            temp_angle = temp_angle+360
        track_angle[n,3] = temp_angle
        x_pred = track_lib.linear_pred(x_center)
        y_pred = track_lib.linear_pred(y_center)
        if x_pred<0 or x_pred>img_size[0] or y_pred<0 or y_pred>img_size[1]:
            track_angle[n,2] = 1    
    

    # compute appearance cost
    cost_mask = np.zeros((N_cluster,N_cluster), dtype=int)
    conflict_mask = np.zeros((N_cluster,N_cluster), dtype=int)
    for n1 in range(N_cluster-1):
        print(n1)
        for n2 in range(n1+1,N_cluster):
            if (((n1 in remove_set) or (n2 in remove_set))) and cc==1:
                continue
            
            #if n1==58 and n2==73:
            #    import pdb; pdb.set_trace()
            if (((track_interval[n1,0]>=track_interval[n2,0] and track_interval[n1,0]<=track_interval[n2,1]) or 
                (track_interval[n1,1]>=track_interval[n2,0] and track_interval[n1,1]<=track_interval[n2,1])) or 
                ((track_interval[n2,0]>=track_interval[n1,0] and track_interval[n2,0]<=track_interval[n1,1]) or 
                (track_interval[n2,1]>=track_interval[n1,0] and track_interval[n2,1]<=track_interval[n1,1]))):
                conflict_mask[n1,n2] = 1
                conflict_mask[n2,n1] = 1
                continue
        
            if ((track_interval[n1,0]-track_interval[n2,1]>=0 and track_interval[n1,0]-track_interval[n2,1]<=max_time_dist) or
                (track_interval[n2,0]-track_interval[n1,1]>=0 and track_interval[n2,0]-track_interval[n1,1]<=max_time_dist)):
                track_set1 = new_tracklet_mat['track_cluster'][n1]
                track_set2 = new_tracklet_mat['track_cluster'][n2]
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
                
                #if n1==10 and n2==38:
                #    import pdb; pdb.set_trace()
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

                
    # compute angle cost and motion cost
    for n1 in range(N_cluster):
        #print(n1)
        for n2 in range(N_cluster):
            if n1==n2 or conflict_mask[n1,n2]==1:
                continue
            
            check_idx1 = np.where(new_tracklet_mat['xmin_mat'][n1,:]!=-1)[0]
            check_idx2 = np.where(new_tracklet_mat['xmin_mat'][n2,:]!=-1)[0]

            # n1 at first
            if track_interval[n1,1]<=track_interval[n2,0]:
                if track_angle[n1,2]==1 or track_angle[n2,0]==1:
                    continue
                
                xmin1 = new_tracklet_mat['xmin_mat'][n1,check_idx1[-1]]
                ymin1 = new_tracklet_mat['ymin_mat'][n1,check_idx1[-1]]
                xmax1 = new_tracklet_mat['xmax_mat'][n1,check_idx1[-1]]
                ymax1 = new_tracklet_mat['ymax_mat'][n1,check_idx1[-1]]
                xmin2 = new_tracklet_mat['xmin_mat'][n2,check_idx2[0]]
                ymin2 = new_tracklet_mat['ymin_mat'][n2,check_idx2[0]]
                xmax2 = new_tracklet_mat['xmax_mat'][n2,check_idx2[0]]
                ymax2 = new_tracklet_mat['ymax_mat'][n2,check_idx2[0]]
                diff_x = (xmax2+xmin2)/2-(xmax1+xmin1)/2
                diff_y = (ymax2+ymin2)/2-(ymax1+ymin1)/2
                if diff_x==0:
                    dist_angle = 90
                else:
                    dist_angle = np.arctan(diff_y/diff_x)*180/np.pi
                
                if diff_x<0:
                    dist_angle = dist_angle+180
                if dist_angle<0:
                    dist_angle = dist_angle+360
                #if n1==106 and n2==122:
                #    import pdb; pdb.set_trace()
                angle_cost[n1,n2] = abs(dist_angle-track_angle[n1,3])
                angle_cost[n1,n2] = min(angle_cost[n1,n2],abs(angle_cost[n1,n2]-360))
                angle_cost[n2,n1] = abs(dist_angle-track_angle[n2,1])
                angle_cost[n2,n1] = min(angle_cost[n2,n1],abs(angle_cost[n2,n1]-360))
                if track_angle[n1,3]==-1:
                    angle_cost[n1,n2] = 0
                if track_angle[n2,1]==-1:
                    angle_cost[n2,n1] = 0
                motion_cost[n1,n2] = np.sqrt(np.power(diff_x,2)+np.power(diff_y,2))/abs(track_interval[n2,0]-track_interval[n1,1])
                motion_cost[n2,n1] = motion_cost[n1,n2]
            else:
                continue
    
    #import pdb; pdb.set_trace()
    # merge tracks        
    if params['mode']==0:
        new_cost_D = min_cost_D.copy()
        mode_cost_thresh = min_cost_thresh
    elif params['mode']==1:
        new_cost_D = angle_cost.copy()
        mode_cost_thresh = angle_cost_thresh
    
    merge_idx = []
    merge_cost = []
    merge_set = []
    for n in range(N_cluster):
        merge_set.append([])
        
    cnt = 0
    while 1:
        min_idx = np.unravel_index(np.argmin(new_cost_D, axis=None), new_cost_D.shape)
        temp_cost = new_cost_D[min_idx]
        #import pdb; pdb.set_trace()
        if temp_cost>mode_cost_thresh:
            break
        min_v = min_cost_D[min_idx]
        max_v = max_cost_D[min_idx]  
        mean_v = mean_cost_D[min_idx]  
        idx1 = np.where(conflict_mask[min_idx[0],:]==1)[0]
        idx2 = np.where(conflict_mask[min_idx[1],:]==1)[0]
        #temp_angle_cost = (angle_cost[min_idx]+angle_cost[min_idx[1],min_idx[0]])/2
        temp_angle_cost = max(angle_cost[min_idx],angle_cost[min_idx[1],min_idx[0]])
        temp_motion_cost = motion_cost[min_idx]
        #import pdb; pdb.set_trace()
        if ((min_v<min_cost_thresh and temp_angle_cost<angle_cost_thresh and temp_motion_cost<motion_cost_thresh and 
        conflict_mask[min_idx[0],min_idx[1]]==0) or (min_v<5 and temp_angle_cost<30 and 
                                                     temp_motion_cost<5 and conflict_mask[min_idx[0],min_idx[1]]==0)):
                           
            merge_idx.append(min_idx)
            merge_cost.append([temp_cost,min_v,max_v])
            merge_set[min_idx[0]].append(min_idx[1])
            merge_set[min_idx[1]].append(min_idx[0])
            cnt = cnt+1
            
            
            for k in range(len(merge_set[min_idx[0]])):
                if k==0:
                    idx = np.where(conflict_mask[merge_set[min_idx[0]][k],:]==1)[0]
                else:
                    idx = np.concatenate((idx,np.where(conflict_mask[merge_set[min_idx[0]][k],:]==1)[0]))
            for k in range(len(merge_set[min_idx[1]])):    
                idx = np.concatenate((idx,np.where(conflict_mask[merge_set[min_idx[1]][k],:]==1)[0]))
            idx = np.array(list(set(list(idx))),dtype=int)
            #import pdb; pdb.set_trace()
            for k in range(len(merge_set[min_idx[0]])):
                new_cost_D[merge_set[min_idx[0]][k],idx] = 1
                new_cost_D[idx,merge_set[min_idx[0]][k]] = 1
                conflict_mask[merge_set[min_idx[0]][k],idx] = 1
                conflict_mask[idx,merge_set[min_idx[0]][k]] = 1
            for k in range(len(merge_set[min_idx[1]])):
                new_cost_D[merge_set[min_idx[1]][k],idx] = 1
                new_cost_D[idx,merge_set[min_idx[1]][k]] = 1
                conflict_mask[merge_set[min_idx[1]][k],idx] = 1
                conflict_mask[idx,merge_set[min_idx[1]][k]] = 1
            new_cost_D[min_idx[0],min_idx[1]] = np.Inf
            new_cost_D[min_idx[1],min_idx[0]] = np.Inf
            '''
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
            '''
        else:
            new_cost_D[min_idx[0],min_idx[1]] = np.Inf
            new_cost_D[min_idx[1],min_idx[0]] = np.Inf
    
    if cnt==0:
        break
        
    merge_idx = np.array(merge_idx,dtype=int)
    print('num_merge ', len(merge_idx))
    #import pdb; pdb.set_trace()

    
    # update track_struct
    track_struct['tracklet_mat']['track_cluster'] = new_tracklet_mat['track_cluster'].copy()
    for n in range(len(merge_idx)):
        track_struct['tracklet_mat']['track_cluster'][merge_idx[n,0]].extend(track_struct['tracklet_mat']['track_cluster'][merge_idx[n,1]])
        track_struct['tracklet_mat']['track_cluster'][merge_idx[n,1]] = []
        merge_idx[merge_idx[:,0]==merge_idx[n,1],0] = merge_idx[n,0]
        merge_idx[merge_idx[:,1]==merge_idx[n,1],0] = merge_idx[n,0]
    
    #import pdb; pdb.set_trace()  
    post_processing()
    #import pdb; pdb.set_trace()  
#import pdb; pdb.set_trace() 

'''
for n in range(len(track_struct['final_tracklet_mat']['xmin_mat'])):
    print(n)
    check_flag,ext_xmins,ext_ymins,ext_xmaxs,ext_ymaxs = track_lib.track_extend(track_struct['final_tracklet_mat']['xmin_mat'][n,:],
                                                                                track_struct['final_tracklet_mat']['ymin_mat'][n,:],
                                                                                track_struct['final_tracklet_mat']['xmax_mat'][n,:],
                                                                                track_struct['final_tracklet_mat']['ymax_mat'][n,:],
                                                                                img_size, params['bnd_margin'], params['min_len'], 
                                                                                params['extend_len'], params['reg_thresh'], 
                                                                                params['speed_thresh'], params['static_len'], n)
    if check_flag==0:
        track_struct['final_tracklet_mat']['xmin_mat'][n,:] = ext_xmins.copy()
        track_struct['final_tracklet_mat']['ymin_mat'][n,:] = ext_ymins.copy()
        track_struct['final_tracklet_mat']['xmax_mat'][n,:] = ext_xmaxs.copy()
        track_struct['final_tracklet_mat']['ymax_mat'][n,:] = ext_ymaxs.copy()
'''        


#import pdb; pdb.set_trace()    
wrt_txt(track_struct['final_tracklet_mat'])

draw_result(img_folder, tracking_img_folder)

#convert_frames_to_video(tracking_img_folder+'/', tracking_video_path, 20)
#import pdb; pdb.set_trace()

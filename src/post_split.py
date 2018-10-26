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

import track_lib


seq_name = 'MOT17-14-FRCNN'
img_name = 'MOT17-14'
sub_seq_name = ''
det_path = 'D:/Data/MOT/MOT17Labels/test/'+seq_name+'/det/det.txt'
img_folder = 'D:/Data/MOT/MOT17Det/test/'+img_name+sub_seq_name+'/img1'
crop_det_folder = 'D:/Data/MOT/crop_det/'+seq_name+sub_seq_name
triplet_model = 'D:/Data/UA-Detrac/UA_Detrac_model/MOT'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/cnn_MOT/model.ckpt'
seq_model = 'D:/Data/UA-Detrac/MOT_2d/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/semi_train_model/model.ckpt'
tracking_img_folder = 'D:/Data/MOT/tracking_img/'+seq_name+sub_seq_name
tracking_video_path = 'D:/Data/MOT/tracking_video/'+seq_name+sub_seq_name+'.avi'
svm_model_path = 'D:/Data/MOT/MOT17_train_det_crop/cnn_svm_MOT17.pkl'
rand_forest_model_path = 'D:/Data/MOT/MOT17_train_det_crop/rand_forest_MOT17_FRCNN.pkl'

save_fea_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'.obj'
save_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_label.obj'
save_remove_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_remove_set.obj'
save_all_fea_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all.obj'
save_all_label_path = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label.obj'

save_all_label_path1 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label0.obj'
save_all_label_path2 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label1.obj'
save_all_label_path3 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label2.obj'
save_all_label_path4 = 'D:/Data/MOT/save_fea_mat/'+seq_name+sub_seq_name+'_all_label3.obj'

txt_result_path = 'D:/Data/MOT/txt_result/'+seq_name+sub_seq_name+'.txt'
track_struct_path = 'D:/Data/MOT/track_struct/'+seq_name+sub_seq_name+'.obj'

models = []
model_info = []
inliers1 = []
inliers2 = []

geometry_folder = 'D:/Data/MOT/geometry_info'
geo_model_path = geometry_folder+'/MOT17-14-FRCNN_models.obj'
geo_model_info_path = geometry_folder+'/MOT17-14-FRCNN_model_info.obj'
geo_inlier1_path = geometry_folder+'/MOT17-14-FRCNN_inliers1.obj'
geo_inlier2_path = geometry_folder+'/MOT17-14-FRCNN_inliers2.obj'

global track_struct
track_struct = pickle.load(open(track_struct_path,'rb'))

global remove_set
remove_set = []

'''
models = pickle.load(open(geo_model_path,'rb'))
model_info = pickle.load(open(geo_model_info_path,'rb'))
inliers1 = pickle.load(open(geo_inlier1_path,'rb'))
inliers2 = pickle.load(open(geo_inlier2_path,'rb'))
'''


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



cost_range = [-2,2]
t_dist_thresh = 30
geometry_cost_thresh = 25

N_cluster = len(track_struct['final_tracklet_mat']['track_cluster'])

num_track = track_struct['tracklet_mat']['comb_track_cost_mask'].shape[0]
geometry_cost = np.zeros((num_track,num_track))

for n in range(N_cluster):
    print(n)
    track_set = track_struct['final_tracklet_mat']['track_cluster'][n]
    track_set = list(np.sort(np.array(track_set,dtype=int)))
    if len(track_set)<=1:
        continue
    for n1 in range(len(track_set)-1):
        track_id1 = track_set[n1]
        track_id2 = track_set[n1+1]
        track_interval1 = track_struct['tracklet_mat']['track_interval'][track_id1,:]
        track_interval2 = track_struct['tracklet_mat']['track_interval'][track_id2,:]
            
        if track_struct['tracklet_mat']['comb_track_cost_mask'][track_id1,track_id2]==0:
            continue
        if (track_struct['tracklet_mat']['comb_track_cost'][track_id1,track_id2]<cost_range[0] 
            or track_struct['tracklet_mat']['comb_track_cost'][track_id1,track_id2]>cost_range[1]):
            continue
            
        if track_interval1[1]>track_interval2[0]:
            continue     
        if track_interval2[0]-track_interval1[1]>t_dist_thresh:
            continue
                
        if len(model_info)>0:
            temp_info = np.array(model_info,dtype=int)
            temp_idx = np.where(np.logical_and(temp_info[:,0]==track_interval1[1], temp_info[:,1]==track_interval2[0]))[0]
        else:
            temp_idx = []
                
        if len(temp_idx)==0:
           
            img_name1 = track_lib.file_name(int(track_interval1[1]+1),6)+'.jpg'
            img_path1 = img_folder+'/'+img_name1
            #import pdb; pdb.set_trace()
            img1 = misc.imread(img_path1) 
            img_name2 = track_lib.file_name(int(track_interval2[0]+1),6)+'.jpg'
            img_path2 = img_folder+'/'+img_name2
            img2 = misc.imread(img_path2) 
            
            model, num_match, num_inlier, inlier_keypoints1, inlier_keypoints2 = track_lib.estimateF(img1, img2)
            models.append(model)
            model_info.append([track_interval1[1],track_interval2[0],num_match,num_inlier])
            inliers1.append(inlier_keypoints1)
            inliers2.append(inlier_keypoints2)
            
        else:
            model = models[temp_idx[0]]
            inlier_keypoints1 = inliers1[temp_idx[0]]
            inlier_keypoints2 = inliers2[temp_idx[0]]
                
        xmin1 = track_struct['tracklet_mat']['xmin_mat'][track_id1,int(track_interval1[1])]
        xmax1 = track_struct['tracklet_mat']['xmax_mat'][track_id1,int(track_interval1[1])]
        ymin1 = track_struct['tracklet_mat']['ymin_mat'][track_id1,int(track_interval1[1])]
        ymax1 = track_struct['tracklet_mat']['ymax_mat'][track_id1,int(track_interval1[1])]
           
        xmin2 = track_struct['tracklet_mat']['xmin_mat'][track_id2,int(track_interval2[0])]
        xmax2 = track_struct['tracklet_mat']['xmax_mat'][track_id2,int(track_interval2[0])]
        ymin2 = track_struct['tracklet_mat']['ymin_mat'][track_id2,int(track_interval2[0])]
        ymax2 = track_struct['tracklet_mat']['ymax_mat'][track_id2,int(track_interval2[0])]
            
        bbox_inlier_idx = []
        for k in range(len(inlier_keypoints1)):
            if (inlier_keypoints1[k,0]>=xmin1 and inlier_keypoints1[k,0]<=xmax1 
                and inlier_keypoints1[k,1]>=ymin1 and inlier_keypoints1[k,1]<=ymax1
                and inlier_keypoints2[k,0]>=xmin2 and inlier_keypoints2[k,0]<=xmax2 
                and inlier_keypoints2[k,1]>=ymin2 and inlier_keypoints2[k,1]<=ymax2):
                bbox_inlier_idx.append(k)
            
        if len(bbox_inlier_idx)!=0:
            bbox_inlier_idx = np.array(bbox_inlier_idx,dtype=int)
            cost = model.residuals(inlier_keypoints1[bbox_inlier_idx,:],inlier_keypoints2[bbox_inlier_idx,:])
            geometry_cost[track_id1,track_id2] = np.mean(cost)
            geometry_cost[track_id2,track_id1] = geometry_cost[track_id1,track_id2]
        else:
            bbox_center1 = np.zeros((1,2))
            bbox_center1[0,0] = (xmin1+xmax1)/2
            bbox_center1[0,1] = (ymin1+ymax1)/2
            bbox_center2 = np.zeros((1,2))
            bbox_center2[0,0] = (xmin2+xmax2)/2
            bbox_center2[0,1] = (ymin2+ymax2)/2
            geometry_cost[track_id1,track_id2] = model.residuals(bbox_center1,bbox_center2)[0]
            geometry_cost[track_id2,track_id1] = geometry_cost[track_id1,track_id2]
                
        print(geometry_cost[track_id1,track_id2])
        
import pdb; pdb.set_trace()
# split track cluster
track_struct['tracklet_mat']['track_cluster'] = track_struct['final_tracklet_mat']['track_cluster'].copy()
for n in range(N_cluster):
    print(n)
    track_set = track_struct['tracklet_mat']['track_cluster'][n]
    track_set = list(np.sort(np.array(track_set,dtype=int)))
    if len(track_set)<=1:
        continue
        
    temp_set = []
    for n1 in range(len(track_set)-1):
        track_id1 = track_set[n1]
        track_id2 = track_set[n1+1]
        temp_set.append(track_id1)
        if geometry_cost[track_id1,track_id2]<geometry_cost_thresh:
            continue
        
        track_struct['tracklet_mat']['track_cluster'].append(temp_set)
        for k in range(len(temp_set)):
            track_struct['tracklet_mat']['track_cluster'][n].remove(temp_set[k])
            
        temp_set = []     

post_processing()
    
wrt_txt(track_struct['final_tracklet_mat'])

pickle.dump(track_struct, open(track_struct_path,'wb'))

draw_result(img_folder, tracking_img_folder)

convert_frames_to_video(tracking_img_folder+'/', tracking_video_path, 20)

import pdb; pdb.set_trace()
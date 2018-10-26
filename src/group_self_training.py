
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import seq_nn_3d
import pickle
import glob
import os
import matplotlib.pyplot as plt

# In[2]:



fea_mat_path = 'D:/Data/MOT/save_fea_mat'
fea_label_path = 'D:/Data/MOT/label_save_fea_mat'
seq_set = ['MOT16-01','MOT16-03','MOT16-06','MOT16-07','MOT16-08','MOT16-12','MOT16-14']
seq_name = 'MOT16-01'
seq_model = 'D:/Data/UA-Detrac/MOT_2d/model.ckpt'
fine_tune_model_path = 'D:/Data/UA-Detrac/semi_train_model/model.ckpt'
temp_fig_path = 'D:/Data/MOT/temp_fig'
max_length = 64
feature_size = 4+512
batch_size = 8
num_classes = 2
train_batch_size = 5000

lr_rate = 1e-5
iters = 10

batch_X_x = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_y = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_w = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_h = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_a = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 1])
batch_mask_1 = tf.placeholder(tf.float32, [None, 1, max_length, 2])
batch_mask_2 = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 2])
batch_Y = tf.placeholder(tf.int32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

y_conv = seq_nn_3d.seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,
                       batch_mask_2,batch_Y,max_length,feature_size,keep_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_Y, logits=y_conv))
train_step = tf.train.AdamOptimizer(lr_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(batch_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()    
    
with tf.Session() as sess:
    #sess.run(init)
    saver.restore(sess, seq_model)
    print("Model restored.")

    # get all feature path
    temp_fea_file_list = os.listdir(fea_mat_path)
    temp_label_file_list = os.listdir(fea_label_path)
    num_file = len(temp_fea_file_list)
    
    cnt = 0
    fea_file_list = []
    label_file_list = []
    for n in range(num_file):
        if temp_fea_file_list[n][0:8]==seq_name:
            fea_file_list.append(fea_mat_path+'/'+temp_fea_file_list[n])
            label_file_list.append(fea_label_path+'/'+temp_label_file_list[n])
    del temp_fea_file_list
    del temp_label_file_list
    num_file = len(fea_file_list)    

    # load all data 
    for n in range(num_file):
        print(n)
        if n==0:
            batch_x = pickle.load(open(fea_file_list[n],'rb'))
            batch_label = pickle.load(open(label_file_list[n],'rb'))
        else:
            batch_x = np.concatenate((batch_x,pickle.load(open(fea_file_list[n],'rb'))),axis=0)
            batch_label = np.concatenate((batch_label,pickle.load(open(label_file_list[n],'rb'))),axis=0)
                
    for t in range(iters):
        print("t ")
        print(t)
        
        if t==0:
            prev_label = batch_label[:,2]
        else:
            prev_label = pred_label[:,0]
            
        # predict loss
        num_samples = batch_label.shape[0]
        pred_loss = np.zeros((num_samples,2))
        num_of_batches = int(np.ceil(num_samples/batch_size))
        confidence = np.zeros(num_samples)
        for k in range(num_of_batches):
            if k!=num_of_batches-1:
                temp_batch_size = batch_size
            else:
                temp_batch_size = int(num_samples-batch_size*(num_of_batches-1))
                    
            x = np.zeros((temp_batch_size,1,max_length,1))
            y = np.zeros((temp_batch_size,1,max_length,1))
            w = np.zeros((temp_batch_size,1,max_length,1))
            h = np.zeros((temp_batch_size,1,max_length,1))
            ap = np.zeros((temp_batch_size,feature_size-4,max_length,1))
            mask_1 = np.zeros((temp_batch_size,1,max_length,2))
            mask_2 = np.zeros((temp_batch_size,feature_size-4,max_length,2))
            x[:,0,:,0] = batch_x[k*batch_size:k*batch_size+temp_batch_size,0,:,0]
            y[:,0,:,0] = batch_x[k*batch_size:k*batch_size+temp_batch_size,1,:,0]
            w[:,0,:,0] = batch_x[k*batch_size:k*batch_size+temp_batch_size,2,:,0]
            h[:,0,:,0] = batch_x[k*batch_size:k*batch_size+temp_batch_size,3,:,0]
            ap[:,:,:,0] = batch_x[k*batch_size:k*batch_size+temp_batch_size,4:,:,0]
            mask_1[:,0,:,:] = batch_x[k*batch_size:k*batch_size+temp_batch_size,0,:,1:]
            mask_2[:,:,:,:] = batch_x[k*batch_size:k*batch_size+temp_batch_size,4:,:,1:]
            pred_loss[k*batch_size:k*batch_size+temp_batch_size,:] = sess.run(y_conv, feed_dict={batch_X_x: x,
                                     batch_X_y: y,
                                     batch_X_w: w,
                                     batch_X_h: h,
                                     batch_X_a: ap,
                                     batch_mask_1: mask_1,
                                     batch_mask_2: mask_2,
                                     batch_Y: np.zeros((temp_batch_size,2)), 
                                     keep_prob: 1.0})
        
        confidence = pred_loss[:,0]-pred_loss[:,1]
        plt.hist(confidence, bins='auto')
        save_fig_name = temp_fig_path+'/'+str(t)+'.png'
        plt.savefig(save_fig_name)  
        
        pred_label = np.zeros((num_samples,2))
        pred_label[pred_loss[:,0]>pred_loss[:,1],0] = 1
        pred_label[pred_label[:,0]==0,1] = 1
        num_change = np.sum(pred_label[:,0]!=prev_label)
        print(num_change)
        
        uniq_tracklet = np.unique(batch_label[:,0:2])
        N_tracklet = len(uniq_tracklet)
        permutation_tracklet = np.random.permutation(uniq_tracklet)
        train_idx = []
        train_label = []
        for n in range(N_tracklet):
            # get neighbors of each tracklet
            idx = np.where(np.logical_or(batch_label[:,0]==permutation_tracklet[n],batch_label[:,1]==permutation_tracklet[n]))[0]
            if len(idx)==0:
                continue
            if len(idx)==1:
                train_idx.append(idx[0])
                train_label.append(pred_label[idx[0],:])
                
            target_to_neighbor_loss = np.zeros((len(idx),2))
            neighbor_track_id = np.zeros(len(idx), dtype=int)
            for k in range(len(idx)):
                if batch_label[idx[k],0]==permutation_tracklet[n]:
                    neighbor_track_id[k] = batch_label[idx[k],1]
                else:
                    neighbor_track_id[k] = batch_label[idx[k],0]
                target_to_neighbor_loss[k,:] = pred_loss[idx[k],:] 
            
            # check whether two neighbors are neighbors
            for k1 in range(len(neighbor_track_id)-1):
                for k2 in range(k1+1,len(neighbor_track_id)):
                    if neighbor_track_id[k1]<neighbor_track_id[k2]:
                        track_id1 = neighbor_track_id[k1]
                        track_id2 = neighbor_track_id[k2]
                    else:
                        track_id1 = neighbor_track_id[k2]
                        track_id2 = neighbor_track_id[k1]
                    temp_idx = np.where(np.logical_and(batch_label[:,0]==track_id1,batch_label[:,1]==track_id2))[0]
                    if len(temp_idx)==0:
                        continue
                    
                    triangle_loss = np.zeros((3,2))
                    triangle_loss[0,:] = pred_loss[temp_idx[0]]
                    triangle_loss[1,:] = target_to_neighbor_loss[k1,:]
                    triangle_loss[2,:] = target_to_neighbor_loss[k2,:]
                    temp_total_loss = np.sum(triangle_loss,axis=0)
                    #import pdb; pdb.set_trace()
                    if temp_total_loss[0]>temp_total_loss[1]:
                        temp_pred_label = [1,0]
                    else:
                        temp_pred_label = [0,1]
                    train_idx.extend([temp_idx[0],idx[k1],idx[k2]])
                    train_label.append(temp_pred_label)
                    train_label.append(temp_pred_label)
                    train_label.append(temp_pred_label)
            
        # train
        train_idx = np.array(train_idx)
        train_label = np.array(train_label)
        num_train_idx = len(train_idx)
        print(num_train_idx)
        perm_idx = np.random.permutation(np.array(range(num_train_idx),dtype=int))
        #import pdb; pdb.set_trace()
        perm_train_idx = train_idx[perm_idx]
        perm_train_label = train_label[perm_idx,:]
        
        num_of_batches = int(np.ceil(num_train_idx/batch_size))
        for k in range(num_of_batches):
            print(k)
            if k!=num_of_batches-1:
                temp_batch_size = batch_size
            else:
                temp_batch_size = int(num_train_idx-batch_size*(num_of_batches-1))
            
            x = np.zeros((temp_batch_size,1,max_length,1))
            y = np.zeros((temp_batch_size,1,max_length,1))
            w = np.zeros((temp_batch_size,1,max_length,1))
            h = np.zeros((temp_batch_size,1,max_length,1))
            ap = np.zeros((temp_batch_size,feature_size-4,max_length,1))
            mask_1 = np.zeros((temp_batch_size,1,max_length,2))
            mask_2 = np.zeros((temp_batch_size,feature_size-4,max_length,2))
            temp_y = np.zeros((temp_batch_size,2))
            x[:,0,:,0] = batch_x[perm_train_idx[k*batch_size:k*batch_size+temp_batch_size],0,:,0]
            y[:,0,:,0] = batch_x[perm_train_idx[k*batch_size:k*batch_size+temp_batch_size],1,:,0]
            w[:,0,:,0] = batch_x[perm_train_idx[k*batch_size:k*batch_size+temp_batch_size],2,:,0]
            h[:,0,:,0] = batch_x[perm_train_idx[k*batch_size:k*batch_size+temp_batch_size],3,:,0]
            ap[:,:,:,0] = batch_x[perm_train_idx[k*batch_size:k*batch_size+temp_batch_size],4:,:,0]
            mask_1[:,0,:,:] = batch_x[perm_train_idx[k*batch_size:k*batch_size+temp_batch_size],0,:,1:]
            mask_2[:,:,:,:] = batch_x[perm_train_idx[k*batch_size:k*batch_size+temp_batch_size],4:,:,1:]
            temp_y = perm_train_label[k*batch_size:k*batch_size+temp_batch_size,:]
            train_step.run(feed_dict={batch_X_x: x,
                                     batch_X_y: y,
                                     batch_X_w: w,
                                     batch_X_h: h,
                                     batch_X_a: ap,
                                     batch_mask_1: mask_1,
                                     batch_mask_2: mask_2,
                                     batch_Y: temp_y, 
                                     keep_prob: 0.75})
        
        
        save_path = saver.save(sess, fine_tune_model_path)
        print("Model saved in path: %s" % save_path)
    
    

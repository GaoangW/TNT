
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
fea_label_path = 'D:/Data/MOT/save_fea_mat'
seq_set = ['MOT16-01','MOT16-03','MOT16-06','MOT16-07','MOT16-08','MOT16-12','MOT16-14']
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
    fea_file_list = os.listdir(fea_mat_path)
    num_file = len(fea_file_list)
    num_file = 2
    num_samples_per_file = np.zeros(num_file, dtype=int)
    
    for n in range(num_file):
        fea_file_list[n] = fea_mat_path+'/'+fea_file_list[n]
    
    for t in range(iters):
        print("t ")
        print(t)
        
        # load all data and predict certainty
        certainty_score = []
        pred_label = []
        sample_idx = [] #[file_idx, sample_idx_in_file]
        for n in range(num_file):
            print(n)
            batch_x = pickle.load(open(fea_file_list[n],'rb'))
            num_samples_per_file[n] = len(batch_x)
            pred_y = np.zeros((num_samples_per_file[n],2))
            
            num_of_batches = int(np.ceil(num_samples_per_file[n]/batch_size))
            for k in range(num_of_batches):
                if k!=num_of_batches-1:
                    temp_batch_size = batch_size
                else:
                    temp_batch_size = int(num_samples_per_file[n]-batch_size*(num_of_batches-1))
                    
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
                pred_y[k*batch_size:k*batch_size+temp_batch_size,:] = sess.run(y_conv, feed_dict={batch_X_x: x,
                                     batch_X_y: y,
                                     batch_X_w: w,
                                     batch_X_h: h,
                                     batch_X_a: ap,
                                     batch_mask_1: mask_1,
                                     batch_mask_2: mask_2,
                                     batch_Y: np.zeros((batch_size,2)), 
                                     keep_prob: 1.0})
            temp_certainty = pred_y[:,1]-pred_y[:,0]
            temp_pred_label = np.zeros((num_samples_per_file[n],2))
            temp_pred_label[pred_y[:,0]>pred_y[:,1],0] = 1
            temp_pred_label[temp_pred_label[:,0]==0,1] = 1
            temp_sample_idx = np.zeros((num_samples_per_file[n],2))
            temp_sample_idx[:,0] = n
            temp_sample_idx[:,1] = np.array(range(num_samples_per_file[n]))
            certainty_score.append(temp_certainty)
            pred_label.append(temp_pred_label)
            sample_idx.append(temp_sample_idx)
        print(num_samples_per_file)
        
        total_num_samples = np.sum(num_samples_per_file)   
        total_certainty_score = np.zeros(total_num_samples)
        total_pred_label = np.zeros((total_num_samples,2))
        total_sample_idx = np.zeros((total_num_samples,2), dtype=int)
        cnt = 0
        for n in range(num_file):
            total_certainty_score[cnt:cnt+num_samples_per_file[n]] = certainty_score[n]
            total_pred_label[cnt:cnt+num_samples_per_file[n],:] = pred_label[n]
            total_sample_idx[cnt:cnt+num_samples_per_file[n],:] = sample_idx[n]
            cnt = cnt+num_samples_per_file[n]
        sort_idx = np.argsort(-np.abolute(total_certainty_score))    
        select_ratio = (t+1)/iters
        select_num = int(select_ratio*total_num_samples)
        select_idx = sort_idx[0:select_num]
        print(select_num)
        plt.hist(total_certainty_score, bins='auto')
        #plt.show()
        save_fig_name = temp_fig_path+'/'+str(t)+'.png'
        plt.savefig(save_fig_name)
        #import pdb; pdb.set_trace()
        
        # split select_idx to batches
        num_train_batch = int(np.ceil(select_num/train_batch_size))
        permutation_select_idx = np.random.permutation(select_idx)
        for n in range(num_train_batch):
            if n!=num_train_batch-1:
                temp_train_batch_size = train_batch_size
            else:
                temp_train_batch_size = select_num-(num_train_batch-1)*train_batch_size
            
            #import pdb; pdb.set_trace()
            temp_select_idx = permutation_select_idx[n*train_batch_size:n*train_batch_size+temp_train_batch_size]
            temp_select_sample_idx = total_sample_idx[temp_select_idx,:]
            
            train_batch_x = np.zeros((temp_train_batch_size,batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]))
            train_batch_label = np.zeros((temp_train_batch_size,2))
            cnt = 0
            for k in range(num_file):
                batch_x = pickle.load(open(fea_file_list[k],'rb'))
                temp_idx = temp_select_sample_idx[temp_select_sample_idx[:,0]==k,1]
                #import pdb; pdb.set_trace()
                train_batch_x[cnt:cnt+len(temp_idx),:,:,:] = batch_x[temp_idx,:,:,:]
                train_batch_label[cnt:cnt+len(temp_idx),:] = pred_label[k][temp_idx,:]
                cnt = cnt+len(temp_idx)
            temp_permutation_idx = np.random.permutation(np.array(range(temp_train_batch_size)))
            permutation_batch_x = train_batch_x[temp_permutation_idx,:,:,:]
            permutation_batch_label = train_batch_label[temp_permutation_idx,:]
            import pdb; pdb.set_trace()
            
            # train
            num_mini_batch = int(np.ceil(temp_train_batch_size/batch_size))
            for k in range(num_mini_batch):
                if k!=num_mini_batch-1:
                    temp_batch_size = batch_size
                else:
                    temp_batch_size = temp_train_batch_size-(num_mini_batch-1)*batch_size
                
                x = np.zeros((temp_batch_size,1,max_length,1))
                y = np.zeros((temp_batch_size,1,max_length,1))
                w = np.zeros((temp_batch_size,1,max_length,1))
                h = np.zeros((temp_batch_size,1,max_length,1))
                ap = np.zeros((temp_batch_size,feature_size-4,max_length,1))
                mask_1 = np.zeros((temp_batch_size,1,max_length,2))
                mask_2 = np.zeros((temp_batch_size,feature_size-4,max_length,2))
                temp_y = np.zeros((temp_batch_size,2))
                x[:,0,:,0] = permutation_batch_x[k*batch_size:k*batch_size+temp_batch_size,0,:,0]
                y[:,0,:,0] = permutation_batch_x[k*batch_size:k*batch_size+temp_batch_size,1,:,0]
                w[:,0,:,0] = permutation_batch_x[k*batch_size:k*batch_size+temp_batch_size,2,:,0]
                h[:,0,:,0] = permutation_batch_x[k*batch_size:k*batch_size+temp_batch_size,3,:,0]
                ap[:,:,:,0] = permutation_batch_x[k*batch_size:k*batch_size+temp_batch_size,4:,:,0]
                mask_1[:,0,:,:] = permutation_batch_x[k*batch_size:k*batch_size+temp_batch_size,0,:,1:]
                mask_2[:,:,:,:] = permutation_batch_x[k*batch_size:k*batch_size+temp_batch_size,4:,:,1:]
                temp_y = permutation_batch_label[k*batch_size:k*batch_size+temp_batch_size,:]
                
                #import pdb; pdb.set_trace()
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
    
    





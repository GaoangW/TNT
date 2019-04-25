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
import matplotlib.pyplot as plt
import seq_nn_3d

# In[2]:

MAT_folder = 'D:/KITTI/raw_data/tracking_annotation'
data_folder = 'D:/KITTI/raw_data/2011_09_26'
img_folder = 'D:/KITTI/raw_data/KITTI_crop_all'
triplet_model = 'D:/KITTI/raw_data/pre_model2'
max_length = 64
feature_size = 4+512
batch_size = 32
num_classes = 2
loc_scales = [100,30,5,5]
img_size = [1242,375]
noise_scales = [0.005,0.005,0.005,0.005]

# In[3]:
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



def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, batch_size, distance_metric):
    # Run forward pass to calculate embeddings
    #print('Runnning forward pass on LFW images')
    
    use_flipped_images = False
    use_fixed_image_standardization = False
    use_random_rotate = True
    use_radnom_crop = True
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
    
def generate_data(feature_size, max_length, batch_size, MAT_folder, img_folder):
    
    # load mat files
    Mat_paths = os.listdir(MAT_folder)
    choose_idx = np.random.randint(len(Mat_paths), size=batch_size)
    Mat_files = []
    seq_names = []
    for n in range(batch_size):
        seq_name = Mat_paths[choose_idx[n]][0:21]+'_sync'
        temp_path = data_folder+'/'+seq_name+'/gt_2.mat'
        temp_mat_file = loadmat(temp_path)
        Mat_files.append(temp_mat_file)
        seq_names.append(seq_name)
    
    X = np.zeros((batch_size,feature_size,max_length,3))
    Y = np.zeros((batch_size,2))
    all_paths = []
    
    
    # positive 
    for n in range(int(batch_size/2)):
        seq_name = seq_names[n]
        fr_num = Mat_files[n]['gtInfo'][0][0][0].shape[0]
        id_num = Mat_files[n]['gtInfo'][0][0][0].shape[1]
        Y[n,0] = 1
        
        X_3d = Mat_files[n]['gtInfo'][0][0][4]
        Y_3d = Mat_files[n]['gtInfo'][0][0][5]
        W_3d = Mat_files[n]['gtInfo'][0][0][6]
        H_3d = Mat_files[n]['gtInfo'][0][0][7]
        try_time = 0
        if try_time>=10:
            continue
        while 1:
            if try_time>=10:
                all_paths.append([])
                #print('err')
                break
            obj_id = np.random.randint(id_num, size=1)[0]
            part_W_mat = Mat_files[n]['gtInfo'][0][0][3][:,obj_id]
            
            non_zero_idx = np.where(part_W_mat>0)[0]
            if np.max(non_zero_idx)-np.min(non_zero_idx)+1!=len(non_zero_idx) or len(non_zero_idx)<=1:
                try_time = try_time+1
                continue
            st_fr = np.min(non_zero_idx)#+np.random.randint(len(non_zero_idx)-1, size=1)[0]
            end_fr = np.max(non_zero_idx)
            abs_fr_t1 = int(st_fr+np.random.randint(len(non_zero_idx)-1, size=1)[0])
            abs_end_fr = min(abs_fr_t1+max_length-1,end_fr)
            abs_fr_t4 = int(abs_end_fr-np.random.randint(abs_end_fr-abs_fr_t1, size=1)[0])
            abs_fr_t2 = int(abs_fr_t1+np.random.randint(abs_fr_t4-abs_fr_t1, size=1)[0])
            abs_fr_t3 = int(abs_fr_t4-np.random.randint(abs_fr_t4-abs_fr_t2, size=1)[0])
            
            t1 = 0
            t2 = abs_fr_t2-abs_fr_t1
            t3 = abs_fr_t3-abs_fr_t1
            t4 = abs_fr_t4-abs_fr_t1
            
            # mask
            X[n,:,t1:t2+1,1] = 1
            X[n,:,t3:t4+1,2] = 1
            
            # X
            X[n,0,t1:t2+1,0] = X_3d[abs_fr_t1:abs_fr_t2+1,obj_id]/loc_scales[0]+noise_scales[0]*np.random.normal(0,1,t2-t1+1)
            X[n,0,t3:t4+1,0] = X_3d[abs_fr_t3:abs_fr_t4+1,obj_id]/loc_scales[0]+noise_scales[0]*np.random.normal(0,1,t4-t3+1)
            
            # Y
            X[n,1,t1:t2+1,0] = Y_3d[abs_fr_t1:abs_fr_t2+1,obj_id]/loc_scales[1]+noise_scales[1]*np.random.normal(0,1,t2-t1+1)
            X[n,1,t3:t4+1,0] = Y_3d[abs_fr_t3:abs_fr_t4+1,obj_id]/loc_scales[1]+noise_scales[1]*np.random.normal(0,1,t4-t3+1)
            
            # W
            X[n,2,t1:t2+1,0] = W_3d[abs_fr_t1:abs_fr_t2+1,obj_id]/loc_scales[2]+noise_scales[2]*np.random.normal(0,1,t2-t1+1)
            X[n,2,t3:t4+1,0] = W_3d[abs_fr_t3:abs_fr_t4+1,obj_id]/loc_scales[2]+noise_scales[2]*np.random.normal(0,1,t4-t3+1)
            
            # H
            X[n,3,t1:t2+1,0] = H_3d[abs_fr_t1:abs_fr_t2+1,obj_id]/loc_scales[3]+noise_scales[3]*np.random.normal(0,1,t2-t1+1)
            X[n,3,t3:t4+1,0] = H_3d[abs_fr_t3:abs_fr_t4+1,obj_id]/loc_scales[3]+noise_scales[3]*np.random.normal(0,1,t4-t3+1)
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
            temp_paths = []
            for k in range(abs_fr_t1,abs_fr_t2+1):
                class_name = seq_name+'_image_02_'+num_str(obj_id+1,4)
                file_name = class_name+'_'+num_str(k+1,4)+'.png'
                temp_path = img_folder+'/'+class_name+'/'+file_name
                temp_paths.append(temp_path)
            for k in range(abs_fr_t3,abs_fr_t4+1):
                class_name = seq_name+'_image_02_'+num_str(obj_id+1,4)
                file_name = class_name+'_'+num_str(k+1,4)+'.png'
                temp_path = img_folder+'/'+class_name+'/'+file_name
                temp_paths.append(temp_path)
            all_paths.append(temp_paths.copy())
            break
            
        
    # negative
    for n in range(int(batch_size/2),batch_size):
        Y[n,1] = 1
        seq_name = seq_names[n]
        fr_num = Mat_files[n]['gtInfo'][0][0][0].shape[0]
        id_num = Mat_files[n]['gtInfo'][0][0][0].shape[1]
        
        X_3d = Mat_files[n]['gtInfo'][0][0][4]
        Y_3d = Mat_files[n]['gtInfo'][0][0][5]
        W_3d = Mat_files[n]['gtInfo'][0][0][6]
        H_3d = Mat_files[n]['gtInfo'][0][0][7]
        try_time = 0
        
        time_interval = np.zeros((id_num,2))
        for obj_id in range(id_num):
            part_W_mat = Mat_files[n]['gtInfo'][0][0][3][:,obj_id]
            non_zero_idx = np.where(part_W_mat>0)[0]
            t_min = np.min(non_zero_idx)
            t_max = np.max(non_zero_idx)
            if len(non_zero_idx)!=t_max-t_min+1:
                time_interval[obj_id,0] = -1
                time_interval[obj_id,1] = -1
            else:
                time_interval[obj_id,0] = t_min
                time_interval[obj_id,1] = t_max
                
        
        if try_time>=10:
            continue
        while 1:
            if try_time>=10:
                all_paths.append([])
                break
            split_fr = 1+np.random.randint(fr_num-2, size=1)[0]
            
            cand_pairs = []
            for id1 in range(id_num):
                for id2 in range(id_num):
                    if id1==id2:
                        continue
                    if time_interval[id1,0]==-1 or time_interval[id2,0]==-1:
                        continue
                    if time_interval[id1,0]<=split_fr and time_interval[id2,1]>split_fr:
                        t_above = min(split_fr,time_interval[id1,1])
                        t_below = max(split_fr+1,time_interval[id2,0])
                        t_dist = t_below-t_above
                        if t_dist<max_length/4:
                            cand_pairs.append([id1,id2,t_dist])
            if len(cand_pairs)==0:
                try_time = try_time+1
                continue
            choose_pair_idx = np.random.randint(len(cand_pairs), size=1)[0]
            obj_id1 = cand_pairs[choose_pair_idx][0]
            obj_id2 = cand_pairs[choose_pair_idx][1]
            t_below = max(split_fr+1,time_interval[obj_id2,0])
            t_above = min(split_fr,time_interval[obj_id1,1])
            t_min = max(t_below-max_length+1,time_interval[obj_id1,0])
            abs_fr_t1 = int(t_min+np.random.randint(t_above-t_min+1, size=1)[0])
            #abs_fr_t2 = int(abs_fr_t1+np.random.randint(t_above-abs_fr_t1+1, size=1)[0])
            abs_fr_t2 = int(t_above)
            abs_fr_t4 = min(abs_fr_t1+max_length-1,time_interval[obj_id2,1])
            abs_fr_t4 = int(abs_fr_t4-np.random.randint(abs_fr_t4-t_below+1, size=1)[0])
            #abs_fr_t3 = int(abs_fr_t4-np.random.randint(abs_fr_t4-t_below+1, size=1)[0])
            abs_fr_t3 = int(t_below)
            '''
            print(abs_fr_t1)
            print(abs_fr_t2)
            print(abs_fr_t3)
            print(abs_fr_t4)
            #import pdb; pdb.set_trace()
            '''
            t1 = 0
            t2 = abs_fr_t2-abs_fr_t1
            t3 = abs_fr_t3-abs_fr_t1
            t4 = abs_fr_t4-abs_fr_t1
            
            # mask
            X[n,:,t1:t2+1,1] = 1
            X[n,:,t3:t4+1,2] = 1
            
            # X
            X[n,0,t1:t2+1,0] = X_3d[abs_fr_t1:abs_fr_t2+1,obj_id1]/loc_scales[0]+noise_scales[0]*np.random.normal(0,1,t2-t1+1)
            X[n,0,t3:t4+1,0] = X_3d[abs_fr_t3:abs_fr_t4+1,obj_id2]/loc_scales[0]+noise_scales[0]*np.random.normal(0,1,t4-t3+1)
            
            # Y
            X[n,1,t1:t2+1,0] = Y_3d[abs_fr_t1:abs_fr_t2+1,obj_id1]/loc_scales[1]+noise_scales[1]*np.random.normal(0,1,t2-t1+1)
            X[n,1,t3:t4+1,0] = Y_3d[abs_fr_t3:abs_fr_t4+1,obj_id2]/loc_scales[1]+noise_scales[1]*np.random.normal(0,1,t4-t3+1)
            
            # W
            X[n,2,t1:t2+1,0] = W_3d[abs_fr_t1:abs_fr_t2+1,obj_id1]/loc_scales[2]+noise_scales[2]*np.random.normal(0,1,t2-t1+1)
            X[n,2,t3:t4+1,0] = W_3d[abs_fr_t3:abs_fr_t4+1,obj_id2]/loc_scales[2]+noise_scales[2]*np.random.normal(0,1,t4-t3+1)
            
            # H
            X[n,3,t1:t2+1,0] = H_3d[abs_fr_t1:abs_fr_t2+1,obj_id1]/loc_scales[3]+noise_scales[3]*np.random.normal(0,1,t2-t1+1)
            X[n,3,t3:t4+1,0] = H_3d[abs_fr_t3:abs_fr_t4+1,obj_id2]/loc_scales[3]+noise_scales[3]*np.random.normal(0,1,t4-t3+1)
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
            temp_paths = []
            for k in range(abs_fr_t1,abs_fr_t2+1):
                class_name = seq_name+'_image_02_'+num_str(obj_id1+1,4)
                file_name = class_name+'_'+num_str(k+1,4)+'.png'
                temp_path = img_folder+'/'+class_name+'/'+file_name
                temp_paths.append(temp_path)
            for k in range(abs_fr_t3,abs_fr_t4+1):
                class_name = seq_name+'_image_02_'+num_str(obj_id2+1,4)
                file_name = class_name+'_'+num_str(k+1,4)+'.png'
                temp_path = img_folder+'/'+class_name+'/'+file_name
                temp_paths.append(temp_path)
            all_paths.append(temp_paths.copy())
            break
            

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
                if lfw_batch_size==0:
                    continue
                emb_array = evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                         batch_size_placeholder, control_placeholder, embeddings, label_batch, all_paths[n], lfw_batch_size, distance_metric)
                
                if X[n,4:,X[n,0,:,1]+X[n,0,:,2]>0.5,0].shape[0]!=emb_array.shape[0]:
                    aa = 0
                    import pdb; pdb.set_trace()
                    
                #import pdb; pdb.set_trace()
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

y_conv = seq_nn_3d.seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,batch_mask_2,batch_Y,max_length,feature_size,keep_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_Y, logits=y_conv))
train_step = tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(batch_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, "C:/Users/tangz/OneDrive/Documents/Gaoang/RNN/KITTI_model/model.ckpt")
    print("Model restored.")
    
    cnt = 0
    for i in range(2000000):
        total_batch_x, total_batch_y = generate_data(feature_size, max_length, batch_size*10, MAT_folder, img_folder)
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
                '''
                y_pred = sess.run(y_conv,feed_dict={batch_X_x: x,
                                                          batch_X_y: y,
                                                          batch_X_w: w,
                                                          batch_X_h: h,
                                                          batch_X_a: ap,
                                                          batch_mask_1: mask_1,
                                                          batch_mask_2: mask_2,
                                                          batch_Y: batch_y, 
                                                          keep_prob: 1.0})
                #import pdb; pdb.set_trace()
                '''
                train_accuracy = accuracy.eval(feed_dict={batch_X_x: x,
                                                          batch_X_y: y,
                                                          batch_X_w: w,
                                                          batch_X_h: h,
                                                          batch_X_a: ap,
                                                          batch_mask_1: mask_1,
                                                          batch_mask_2: mask_2,
                                                          batch_Y: batch_y, 
                                                          keep_prob: 1.0})
                print('step %d, training accuracy %g' % (cnt, train_accuracy))
            '''
            for n in range(10):
                shuffle_x = np.copy(batch_x)
                shuffle_y = np.copy(batch_y)

                if n!=0:
                    shuffle_x2 = np.copy(shuffle_x)
                    shuffle_y2 = np.copy(shuffle_y)
                    idx = np.array(range(4,feature_size))
                    np.random.shuffle(idx)
                    for k in range(len(idx)):
                        shuffle_x[:,idx[k],:,:] = shuffle_x2[:,k+4,:,:]
            '''    

            #import pdb; pdb.set_trace()
            train_step.run(feed_dict={batch_X_x: x, 
                                      batch_X_y: y, 
                                      batch_X_w: w, 
                                      batch_X_h: h, 
                                      batch_X_a: ap, 
                                      batch_mask_1: mask_1, 
                                      batch_mask_2: mask_2, 
                                      batch_Y: batch_y, 
                                      keep_prob: 0.75})
            
        
        if cnt % 100 == 0:
            save_path = saver.save(sess, 'C:/Users/tangz/OneDrive/Documents/Gaoang/RNN/KITTI_model/model.ckpt')
            print("Model saved in path: %s" % save_path)
        

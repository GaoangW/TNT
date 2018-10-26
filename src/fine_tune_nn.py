
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import seq_nn
import pickle


# In[2]:


#fea_mat_path = 'D:/Data/UA-Detrac/save_fea_mat/MVI_39031.obj'
#fea_label_path = 'D:/Data/UA-Detrac/save_fea_mat/MVI_39031_label.obj'
#seq_model = 'D:/Data/UA-Detrac/cnn_appear_model_517_128_16600steps/model.ckpt'
fea_mat_path = 'D:/Data/MOT/save_fea_mat'
fea_label_path = 'D:/Data/MOT/save_fea_mat'
seq_set = ['MOT16-14']#['MOT16-01','MOT16-03','MOT16-06','MOT16-07','MOT16-08','MOT16-12','MOT16-14']
seq_model = 'D:/Data/UA-Detrac/cnn_MOT/model.ckpt'
#seq_model = 'D:/Data/UA-Detrac/cnn_MOT_fine_tune_model_MOT_14/model.ckpt'
fine_tune_model_path = 'D:/Data/UA-Detrac/cnn_MOT_fine_tune_model_MOT_14/model.ckpt'
max_length = 64
feature_size = 4+512
batch_size = 64
num_classes = 2

lr_rate = 1e-5

# In[ ]:


batch_X_x = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_y = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_w = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_h = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_X_a = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 1])
batch_mask_1 = tf.placeholder(tf.float32, [None, 1, max_length, 1])
batch_mask_2 = tf.placeholder(tf.float32, [None, feature_size-4, max_length, 1])
batch_Y = tf.placeholder(tf.int32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
    
y_conv = seq_nn.seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,
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

    for n in range(len(seq_set)):
        temp_fea_mat_path = fea_mat_path+'/'+seq_set[n]+'_all1.obj'
        temp_fea_label_path = fea_label_path+'/'+seq_set[n]+'_all_label1.obj'
        if n==0:
            batch_x = pickle.load(open(temp_fea_mat_path,'rb'))
            batch_label = pickle.load(open(temp_fea_label_path,'rb'))
        else:
            batch_x = np.concatenate((batch_x, pickle.load(open(temp_fea_mat_path,'rb'))), axis=0)
            batch_label = np.concatenate((batch_label, pickle.load(open(temp_fea_label_path,'rb'))), axis=0)
            
        temp_fea_mat_path = fea_mat_path+'/'+seq_set[n]+'_all2.obj'
        temp_fea_label_path = fea_label_path+'/'+seq_set[n]+'_all_label2.obj'
        batch_x = np.concatenate((batch_x, pickle.load(open(temp_fea_mat_path,'rb'))), axis=0)
        batch_label = np.concatenate((batch_label, pickle.load(open(temp_fea_label_path,'rb'))), axis=0)
   
    
    batch_size = batch_x.shape[0]
    import pdb; pdb.set_trace()
    
    '''
    # remove empty rows
    remove_idx = []
    for n in range(batch_size):
        if np.sum(batch_x[n,0,:,1])==0:
            remove_idx.append(n)
    batch_x = np.delete(batch_x, np.array(remove_idx), axis=0)
    batch_label = np.delete(batch_label, np.array(remove_idx), axis=0)
    '''
    
    total_num = batch_x.shape[0]
    total_batch_label = batch_label[:,2:4]
    #total_batch_label = np.zeros((total_num,2))
    #import pdb; pdb.set_trace()
    
    '''
    for n in range(total_num):
        if batch_label[n,2]==1:
            total_batch_label[n,0] = 1
        else:
            total_batch_label[n,1] = 1
    '''
    
    batch_size = 16
    num_batch = int(np.ceil(total_num/batch_size))
    
    total_batch_x = np.zeros((total_num,1,max_length,1))
    total_batch_y = np.zeros((total_num,1,max_length,1))
    total_batch_w = np.zeros((total_num,1,max_length,1))
    total_batch_h = np.zeros((total_num,1,max_length,1))
    total_batch_ap = np.zeros((total_num,feature_size-4,max_length,1))
    total_mask_1 = np.zeros((total_num,1,max_length,1))
    total_mask_2 = np.zeros((total_num,feature_size-4,max_length,1))
    total_batch_x[:,0,:,0] = batch_x[:,0,:,0]
    total_batch_y[:,0,:,0] = batch_x[:,1,:,0]
    total_batch_w[:,0,:,0] = batch_x[:,2,:,0]
    total_batch_h[:,0,:,0] = batch_x[:,3,:,0]
    total_batch_ap[:,:,:,0] = batch_x[:,4:,:,0]
    total_mask_1[:,0,:,0] = 1-batch_x[:,0,:,1]
    total_mask_2[:,:,:,0] = 1-batch_x[:,4:,:,1]
    
    
    for n in range(15):
        print('epoch')
        print(n)
        shuffle_idx = np.random.permutation(total_num)
        acc = []
        for m in range(num_batch):
            
            if m<num_batch-1:
                temp_batch = batch_size
                x = np.zeros((temp_batch,1,max_length,1))
                y = np.zeros((temp_batch,1,max_length,1))
                w = np.zeros((temp_batch,1,max_length,1))
                h = np.zeros((temp_batch,1,max_length,1))
                ap = np.zeros((temp_batch,feature_size-4,max_length,1))
                mask_1 = np.zeros((temp_batch,1,max_length,1))
                mask_2 = np.zeros((temp_batch,feature_size-4,max_length,1))
                temp_y = np.zeros((temp_batch,2))
                temp_idx = shuffle_idx[m*temp_batch:(m+1)*temp_batch]
                x[:,0,:,0] = batch_x[temp_idx,0,:,0]
                y[:,0,:,0] = batch_x[temp_idx,1,:,0]
                w[:,0,:,0] = batch_x[temp_idx,2,:,0]
                h[:,0,:,0] = batch_x[temp_idx,3,:,0]
                ap[:,:,:,0] = batch_x[temp_idx,4:,:,0]
                mask_1[:,0,:,0] = 1-batch_x[temp_idx,0,:,1]
                mask_2[:,:,:,0] = 1-batch_x[temp_idx,4:,:,1]
                temp_y[:,:] = total_batch_label[temp_idx,:]
            else:
                temp_batch = total_num-(num_batch-1)*batch_size
                #import pdb; pdb.set_trace()
                x = np.zeros((temp_batch,1,max_length,1))
                y = np.zeros((temp_batch,1,max_length,1))
                w = np.zeros((temp_batch,1,max_length,1))
                h = np.zeros((temp_batch,1,max_length,1))
                ap = np.zeros((temp_batch,feature_size-4,max_length,1))
                mask_1 = np.zeros((temp_batch,1,max_length,1))
                mask_2 = np.zeros((temp_batch,feature_size-4,max_length,1))
                temp_y = np.zeros((temp_batch,2))
                temp_idx = shuffle_idx[m*batch_size:]
                #import pdb; pdb.set_trace()
                x[:,0,:,0] = batch_x[temp_idx,0,:,0]
                y[:,0,:,0] = batch_x[temp_idx,1,:,0]
                w[:,0,:,0] = batch_x[temp_idx,2,:,0]
                h[:,0,:,0] = batch_x[temp_idx,3,:,0]
                ap[:,:,:,0] = batch_x[temp_idx,4:,:,0]
                mask_1[:,0,:,0] = 1-batch_x[temp_idx,0,:,1]
                mask_2[:,:,:,0] = 1-batch_x[temp_idx,4:,:,1]
                temp_y[:,:] = total_batch_label[temp_idx,:]
            
            train_accuracy = accuracy.eval(feed_dict={batch_X_x: x,
                                                batch_X_y: y,
                                                batch_X_w: w,
                                                batch_X_h: h,
                                                batch_X_a: ap,
                                                batch_mask_1: mask_1,
                                                batch_mask_2: mask_2,
                                                batch_Y: temp_y, 
                                                keep_prob: 1.0})       
            print('training accuracy %g' % (train_accuracy))
            acc.append(train_accuracy)
            
            train_step.run(feed_dict={batch_X_x: x, 
                                      batch_X_y: y, 
                                      batch_X_w: w, 
                                      batch_X_h: h, 
                                      batch_X_a: ap, 
                                      batch_mask_1: mask_1, 
                                      batch_mask_2: mask_2, 
                                      batch_Y: temp_y, 
                                      keep_prob: 0.75})

        
        print(np.mean(np.array(acc)))
        
        save_path = saver.save(sess, fine_tune_model_path)
        print("Model saved in path: %s" % save_path)
    
    


# In[ ]:





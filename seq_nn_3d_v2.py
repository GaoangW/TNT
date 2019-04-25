    
/*
 * Copyright ©2019 Gaoang Wang.  All rights reserved.  Permission is
 * hereby granted for academic use.  No other use, copying, distribution, or modification
 * is permitted without prior written consent. Copyrights for
 * third-party components of this work must be honored.  Instructors
 * interested in reusing these course materials should contact the
 * author.
 */
    
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, s1, s2):
    return tf.nn.max_pool(x, ksize=[1, s1, s2, 1], strides=[1, s1, s2, 1], padding='SAME')

def seq_nn(batch_X_x,batch_X_y,batch_X_w,batch_X_h,batch_X_a,batch_mask_1,batch_mask_2,batch_Y,max_length,feature_size,keep_prob):
    # conv1_x
    W_conv4_1_x = weight_variable([1, 3, 3, 16])
    b_conv4_1_x = bias_variable([16])

    h_concat4_1_x = tf.concat([batch_X_x, batch_mask_1], 3)
    h_conv4_1_x = tf.nn.relu(conv2d(h_concat4_1_x, W_conv4_1_x) + b_conv4_1_x)
    h_pool4_1_x = max_pool(h_conv4_1_x, 1, 2)
    
    W_conv1_1_x = weight_variable([1, 5, 3, 16])
    b_conv1_1_x = bias_variable([16])

    h_concat1_1_x = tf.concat([batch_X_x, batch_mask_1], 3)
    h_conv1_1_x = tf.nn.relu(conv2d(h_concat1_1_x, W_conv1_1_x) + b_conv1_1_x)
    h_pool1_1_x = max_pool(h_conv1_1_x, 1, 2)

    W_conv2_1_x = weight_variable([1, 9, 3, 16])
    b_conv2_1_x = bias_variable([16])

    h_concat2_1_x = tf.concat([batch_X_x, batch_mask_1], 3)
    h_conv2_1_x = tf.nn.relu(conv2d(h_concat2_1_x, W_conv2_1_x) + b_conv2_1_x)
    h_pool2_1_x = max_pool(h_conv2_1_x, 1, 2)
    
    W_conv3_1_x = weight_variable([1, 13, 3, 16])
    b_conv3_1_x = bias_variable([16])

    h_concat3_1_x = tf.concat([batch_X_x, batch_mask_1], 3)
    h_conv3_1_x = tf.nn.relu(conv2d(h_concat3_1_x, W_conv3_1_x) + b_conv3_1_x)
    h_pool3_1_x = max_pool(h_conv3_1_x, 1, 2)
    
    mask_pool4_1 = max_pool(batch_mask_1, 1, 2)
    
    conv1_x = tf.concat([h_pool4_1_x, h_pool1_1_x], 3)
    conv1_x = tf.concat([conv1_x, h_pool2_1_x], 3)
    conv1_x = tf.concat([conv1_x, h_pool3_1_x], 3)
    conv1_x = tf.concat([conv1_x, mask_pool4_1], 3)
    
    # conv2_x
    W_conv4_2_x = weight_variable([1, 3, 66, 64])
    b_conv4_2_x = bias_variable([64])

    h_conv4_2_x = tf.nn.relu(conv2d(conv1_x, W_conv4_2_x) + b_conv4_2_x)
    h_pool4_2_x = max_pool(h_conv4_2_x, 1, 2)
    
    W_conv1_2_x = weight_variable([1, 5, 66, 64])
    b_conv1_2_x = bias_variable([64])

    h_conv1_2_x = tf.nn.relu(conv2d(conv1_x, W_conv1_2_x) + b_conv1_2_x)
    h_pool1_2_x = max_pool(h_conv1_2_x, 1, 2)

    W_conv2_2_x = weight_variable([1, 9, 66, 64])
    b_conv2_2_x = bias_variable([64])

    h_conv2_2_x = tf.nn.relu(conv2d(conv1_x, W_conv2_2_x) + b_conv2_2_x)
    h_pool2_2_x = max_pool(h_conv2_2_x, 1, 2)

    W_conv3_2_x = weight_variable([1, 13, 66, 64])
    b_conv3_2_x = bias_variable([64])

    h_conv3_2_x = tf.nn.relu(conv2d(conv1_x, W_conv3_2_x) + b_conv3_2_x)
    h_pool3_2_x = max_pool(h_conv3_2_x, 1, 2)
      
    mask_pool4_2 = max_pool(mask_pool4_1, 1, 2)

    conv2_x = tf.concat([h_pool4_2_x, h_pool1_2_x], 3)
    conv2_x = tf.concat([conv2_x, h_pool2_2_x], 3)
    conv2_x = tf.concat([conv2_x, h_pool3_2_x], 3)
    conv2_x = tf.concat([conv2_x, mask_pool4_2], 3)
    
    # conv3_x
    W_conv4_3_x = weight_variable([1, 3, 258, 128])
    b_conv4_3_x = bias_variable([128])

    h_conv4_3_x = tf.nn.relu(conv2d(conv2_x, W_conv4_3_x) + b_conv4_3_x)
    h_pool4_3_x = max_pool(h_conv4_3_x, 1, 2)

    h_pool_flat4_x = tf.reshape(h_pool4_3_x, [-1, 1*8*128])
    
    W_conv1_3_x = weight_variable([1, 5, 258, 128])
    b_conv1_3_x = bias_variable([128])

    h_conv1_3_x = tf.nn.relu(conv2d(conv2_x, W_conv1_3_x) + b_conv1_3_x)
    h_pool1_3_x = max_pool(h_conv1_3_x, 1, 2)

    h_pool_flat1_x = tf.reshape(h_pool1_3_x, [-1, 1*8*128])
    
    W_conv2_3_x = weight_variable([1, 9, 258, 128])
    b_conv2_3_x = bias_variable([128])

    h_conv2_3_x = tf.nn.relu(conv2d(conv2_x, W_conv2_3_x) + b_conv2_3_x)
    h_pool2_3_x = max_pool(h_conv2_3_x, 1, 2)

    h_pool_flat2_x = tf.reshape(h_pool2_3_x, [-1, 1*8*128])
    
    W_conv3_3_x = weight_variable([1, 13, 258, 128])
    b_conv3_3_x = bias_variable([128])

    h_conv3_3_x = tf.nn.relu(conv2d(conv2_x, W_conv3_3_x) + b_conv3_3_x)
    h_pool3_3_x = max_pool(h_conv3_3_x, 1, 2)

    h_pool_flat3_x = tf.reshape(h_pool3_3_x, [-1, 1*8*128])
    
    
    
    # conv1_y
    W_conv4_1_y = weight_variable([1, 3, 3, 16])
    b_conv4_1_y = bias_variable([16])

    h_concat4_1_y = tf.concat([batch_X_y, batch_mask_1], 3)
    h_conv4_1_y = tf.nn.relu(conv2d(h_concat4_1_y, W_conv4_1_y) + b_conv4_1_y)
    h_pool4_1_y = max_pool(h_conv4_1_y, 1, 2)
    
    W_conv1_1_y = weight_variable([1, 5, 3, 16])
    b_conv1_1_y = bias_variable([16])

    h_concat1_1_y = tf.concat([batch_X_y, batch_mask_1], 3)
    h_conv1_1_y = tf.nn.relu(conv2d(h_concat1_1_y, W_conv1_1_y) + b_conv1_1_y)
    h_pool1_1_y = max_pool(h_conv1_1_y, 1, 2)
    
    W_conv2_1_y = weight_variable([1, 9, 3, 16])
    b_conv2_1_y = bias_variable([16])

    h_concat2_1_y = tf.concat([batch_X_y, batch_mask_1], 3)
    h_conv2_1_y = tf.nn.relu(conv2d(h_concat2_1_y, W_conv2_1_y) + b_conv2_1_y)
    h_pool2_1_y = max_pool(h_conv2_1_y, 1, 2)
    
    W_conv3_1_y = weight_variable([1, 13, 3, 16])
    b_conv3_1_y = bias_variable([16])

    h_concat3_1_y = tf.concat([batch_X_y, batch_mask_1], 3)
    h_conv3_1_y = tf.nn.relu(conv2d(h_concat3_1_y, W_conv3_1_y) + b_conv3_1_y)
    h_pool3_1_y = max_pool(h_conv3_1_y, 1, 2)
    
    conv1_y = tf.concat([h_pool4_1_y, h_pool1_1_y], 3)
    conv1_y = tf.concat([conv1_y, h_pool2_1_y], 3)
    conv1_y = tf.concat([conv1_y, h_pool3_1_y], 3)
    conv1_y = tf.concat([conv1_y, mask_pool4_1], 3)
    
    # conv2_y
    W_conv4_2_y = weight_variable([1, 3, 66, 64])
    b_conv4_2_y = bias_variable([64])

    h_conv4_2_y = tf.nn.relu(conv2d(conv1_y, W_conv4_2_y) + b_conv4_2_y)
    h_pool4_2_y = max_pool(h_conv4_2_y, 1, 2)
    
    W_conv1_2_y = weight_variable([1, 5, 66, 64])
    b_conv1_2_y = bias_variable([64])

    h_conv1_2_y = tf.nn.relu(conv2d(conv1_y, W_conv1_2_y) + b_conv1_2_y)
    h_pool1_2_y = max_pool(h_conv1_2_y, 1, 2)
    
    W_conv2_2_y = weight_variable([1, 9, 66, 64])
    b_conv2_2_y = bias_variable([64])

    h_conv2_2_y = tf.nn.relu(conv2d(conv1_y, W_conv2_2_y) + b_conv2_2_y)
    h_pool2_2_y = max_pool(h_conv2_2_y, 1, 2)
    
    W_conv3_2_y = weight_variable([1, 13, 66, 64])
    b_conv3_2_y = bias_variable([64])

    h_conv3_2_y = tf.nn.relu(conv2d(conv1_y, W_conv3_2_y) + b_conv3_2_y)
    h_pool3_2_y = max_pool(h_conv3_2_y, 1, 2)
    
    conv2_y = tf.concat([h_pool4_2_y, h_pool1_2_y], 3)
    conv2_y = tf.concat([conv2_y, h_pool2_2_y], 3)
    conv2_y = tf.concat([conv2_y, h_pool3_2_y], 3)
    conv2_y = tf.concat([conv2_y, mask_pool4_2], 3)
    
    # conv3_y
    W_conv4_3_y = weight_variable([1, 3, 258, 128])
    b_conv4_3_y = bias_variable([128])

    h_conv4_3_y = tf.nn.relu(conv2d(conv2_y, W_conv4_3_y) + b_conv4_3_y)
    h_pool4_3_y = max_pool(h_conv4_3_y, 1, 2)

    h_pool_flat4_y = tf.reshape(h_pool4_3_y, [-1, 1*8*128])
    
    W_conv1_3_y = weight_variable([1, 5, 258, 128])
    b_conv1_3_y = bias_variable([128])

    h_conv1_3_y = tf.nn.relu(conv2d(conv2_y, W_conv1_3_y) + b_conv1_3_y)
    h_pool1_3_y = max_pool(h_conv1_3_y, 1, 2)

    h_pool_flat1_y = tf.reshape(h_pool1_3_y, [-1, 1*8*128])
    
    W_conv2_3_y = weight_variable([1, 9, 258, 128])
    b_conv2_3_y = bias_variable([128])

    h_conv2_3_y = tf.nn.relu(conv2d(conv2_y, W_conv2_3_y) + b_conv2_3_y)
    h_pool2_3_y = max_pool(h_conv2_3_y, 1, 2)

    h_pool_flat2_y = tf.reshape(h_pool2_3_y, [-1, 1*8*128])
    
    W_conv3_3_y = weight_variable([1, 13, 258, 128])
    b_conv3_3_y = bias_variable([128])

    h_conv3_3_y = tf.nn.relu(conv2d(conv2_y, W_conv3_3_y) + b_conv3_3_y)
    h_pool3_3_y = max_pool(h_conv3_3_y, 1, 2)

    h_pool_flat3_y = tf.reshape(h_pool3_3_y, [-1, 1*8*128])
    
    
    # conv1_w
    W_conv4_1_w = weight_variable([1, 3, 3, 16])
    b_conv4_1_w = bias_variable([16])

    h_concat4_1_w = tf.concat([batch_X_w, batch_mask_1], 3)
    h_conv4_1_w = tf.nn.relu(conv2d(h_concat4_1_w, W_conv4_1_w) + b_conv4_1_w)
    h_pool4_1_w = max_pool(h_conv4_1_w, 1, 2)
    
    W_conv1_1_w = weight_variable([1, 5, 3, 16])
    b_conv1_1_w = bias_variable([16])

    h_concat1_1_w = tf.concat([batch_X_w, batch_mask_1], 3)
    h_conv1_1_w = tf.nn.relu(conv2d(h_concat1_1_w, W_conv1_1_w) + b_conv1_1_w)
    h_pool1_1_w = max_pool(h_conv1_1_w, 1, 2)
    
    W_conv2_1_w = weight_variable([1, 9, 3, 16])
    b_conv2_1_w = bias_variable([16])

    h_concat2_1_w = tf.concat([batch_X_w, batch_mask_1], 3)
    h_conv2_1_w = tf.nn.relu(conv2d(h_concat2_1_w, W_conv2_1_w) + b_conv2_1_w)
    h_pool2_1_w = max_pool(h_conv2_1_w, 1, 2)
    
    W_conv3_1_w = weight_variable([1, 13, 3, 16])
    b_conv3_1_w = bias_variable([16])

    h_concat3_1_w = tf.concat([batch_X_w, batch_mask_1], 3)
    h_conv3_1_w = tf.nn.relu(conv2d(h_concat3_1_w, W_conv3_1_w) + b_conv3_1_w)
    h_pool3_1_w = max_pool(h_conv3_1_w, 1, 2)
    
    conv1_w = tf.concat([h_pool4_1_w, h_pool1_1_w], 3)
    conv1_w = tf.concat([conv1_w, h_pool2_1_w], 3)
    conv1_w = tf.concat([conv1_w, h_pool3_1_w], 3)
    conv1_w = tf.concat([conv1_w, mask_pool4_1], 3)
    
    # conv2_w
    W_conv4_2_w = weight_variable([1, 3, 66, 64])
    b_conv4_2_w = bias_variable([64])

    h_conv4_2_w = tf.nn.relu(conv2d(conv1_w, W_conv4_2_w) + b_conv4_2_w)
    h_pool4_2_w = max_pool(h_conv4_2_w, 1, 2)
    
    W_conv1_2_w = weight_variable([1, 5, 66, 64])
    b_conv1_2_w = bias_variable([64])

    h_conv1_2_w = tf.nn.relu(conv2d(conv1_w, W_conv1_2_w) + b_conv1_2_w)
    h_pool1_2_w = max_pool(h_conv1_2_w, 1, 2)
    
    W_conv2_2_w = weight_variable([1, 9, 66, 64])
    b_conv2_2_w = bias_variable([64])

    h_conv2_2_w = tf.nn.relu(conv2d(conv1_w, W_conv2_2_w) + b_conv2_2_w)
    h_pool2_2_w = max_pool(h_conv2_2_w, 1, 2)
    
    W_conv3_2_w = weight_variable([1, 13, 66, 64])
    b_conv3_2_w = bias_variable([64])

    h_conv3_2_w = tf.nn.relu(conv2d(conv1_w, W_conv3_2_w) + b_conv3_2_w)
    h_pool3_2_w = max_pool(h_conv3_2_w, 1, 2)
    
    conv2_w = tf.concat([h_pool4_2_w, h_pool1_2_w], 3)
    conv2_w = tf.concat([conv2_w, h_pool2_2_w], 3)
    conv2_w = tf.concat([conv2_w, h_pool3_2_w], 3)
    conv2_w = tf.concat([conv2_w, mask_pool4_2], 3)
    
    # conv3_w
    W_conv4_3_w = weight_variable([1, 3, 258, 128])
    b_conv4_3_w = bias_variable([128])

    h_conv4_3_w = tf.nn.relu(conv2d(conv2_w, W_conv4_3_w) + b_conv4_3_w)
    h_pool4_3_w = max_pool(h_conv4_3_w, 1, 2)

    h_pool_flat4_w = tf.reshape(h_pool4_3_w, [-1, 1*8*128])
    
    W_conv1_3_w = weight_variable([1, 5, 258, 128])
    b_conv1_3_w = bias_variable([128])

    h_conv1_3_w = tf.nn.relu(conv2d(conv2_w, W_conv1_3_w) + b_conv1_3_w)
    h_pool1_3_w = max_pool(h_conv1_3_w, 1, 2)

    h_pool_flat1_w = tf.reshape(h_pool1_3_w, [-1, 1*8*128])
    
    W_conv2_3_w = weight_variable([1, 9, 258, 128])
    b_conv2_3_w = bias_variable([128])

    h_conv2_3_w = tf.nn.relu(conv2d(conv2_w, W_conv2_3_w) + b_conv2_3_w)
    h_pool2_3_w = max_pool(h_conv2_3_w, 1, 2)

    h_pool_flat2_w = tf.reshape(h_pool2_3_w, [-1, 1*8*128])
    
    W_conv3_3_w = weight_variable([1, 13, 258, 128])
    b_conv3_3_w = bias_variable([128])

    h_conv3_3_w = tf.nn.relu(conv2d(conv2_w, W_conv3_3_w) + b_conv3_3_w)
    h_pool3_3_w = max_pool(h_conv3_3_w, 1, 2)

    h_pool_flat3_w = tf.reshape(h_pool3_3_w, [-1, 1*8*128])
    
    
    # conv1_h
    W_conv4_1_h = weight_variable([1, 3, 3, 16])
    b_conv4_1_h = bias_variable([16])

    h_concat4_1_h = tf.concat([batch_X_h, batch_mask_1], 3)
    h_conv4_1_h = tf.nn.relu(conv2d(h_concat4_1_h, W_conv4_1_h) + b_conv4_1_h)
    h_pool4_1_h = max_pool(h_conv4_1_h, 1, 2)
    
    W_conv1_1_h = weight_variable([1, 5, 3, 16])
    b_conv1_1_h = bias_variable([16])

    h_concat1_1_h = tf.concat([batch_X_h, batch_mask_1], 3)
    h_conv1_1_h = tf.nn.relu(conv2d(h_concat1_1_h, W_conv1_1_h) + b_conv1_1_h)
    h_pool1_1_h = max_pool(h_conv1_1_h, 1, 2)
    
    W_conv2_1_h = weight_variable([1, 9, 3, 16])
    b_conv2_1_h = bias_variable([16])

    h_concat2_1_h = tf.concat([batch_X_h, batch_mask_1], 3)
    h_conv2_1_h = tf.nn.relu(conv2d(h_concat2_1_h, W_conv2_1_h) + b_conv2_1_h)
    h_pool2_1_h = max_pool(h_conv2_1_h, 1, 2)
    
    W_conv3_1_h = weight_variable([1, 13, 3, 16])
    b_conv3_1_h = bias_variable([16])

    h_concat3_1_h = tf.concat([batch_X_h, batch_mask_1], 3)
    h_conv3_1_h = tf.nn.relu(conv2d(h_concat3_1_h, W_conv3_1_h) + b_conv3_1_h)
    h_pool3_1_h = max_pool(h_conv3_1_h, 1, 2)
    
    conv1_h = tf.concat([h_pool4_1_h, h_pool1_1_h], 3)
    conv1_h = tf.concat([conv1_h, h_pool2_1_h], 3)
    conv1_h = tf.concat([conv1_h, h_pool3_1_h], 3)
    conv1_h = tf.concat([conv1_h, mask_pool4_1], 3)
    
    # conv2_h
    W_conv4_2_h = weight_variable([1, 3, 66, 64])
    b_conv4_2_h = bias_variable([64])

    h_conv4_2_h = tf.nn.relu(conv2d(conv1_h, W_conv4_2_h) + b_conv4_2_h)
    h_pool4_2_h = max_pool(h_conv4_2_h, 1, 2)
    
    W_conv1_2_h = weight_variable([1, 5, 66, 64])
    b_conv1_2_h = bias_variable([64])

    h_conv1_2_h = tf.nn.relu(conv2d(conv1_h, W_conv1_2_h) + b_conv1_2_h)
    h_pool1_2_h = max_pool(h_conv1_2_h, 1, 2)
    
    W_conv2_2_h = weight_variable([1, 9, 66, 64])
    b_conv2_2_h = bias_variable([64])

    h_conv2_2_h = tf.nn.relu(conv2d(conv1_h, W_conv2_2_h) + b_conv2_2_h)
    h_pool2_2_h = max_pool(h_conv2_2_h, 1, 2)
    
    W_conv3_2_h = weight_variable([1, 13, 66, 64])
    b_conv3_2_h = bias_variable([64])

    h_conv3_2_h = tf.nn.relu(conv2d(conv1_h, W_conv3_2_h) + b_conv3_2_h)
    h_pool3_2_h = max_pool(h_conv3_2_h, 1, 2)
    
    conv2_h = tf.concat([h_pool4_2_h, h_pool1_2_h], 3)
    conv2_h = tf.concat([conv2_h, h_pool2_2_h], 3)
    conv2_h = tf.concat([conv2_h, h_pool3_2_h], 3)
    conv2_h = tf.concat([conv2_h, mask_pool4_2], 3)
    
    # conv3_h
    W_conv4_3_h = weight_variable([1, 3, 258, 128])
    b_conv4_3_h = bias_variable([128])

    h_conv4_3_h = tf.nn.relu(conv2d(conv2_h, W_conv4_3_h) + b_conv4_3_h)
    h_pool4_3_h = max_pool(h_conv4_3_h, 1, 2)

    h_pool_flat4_h = tf.reshape(h_pool4_3_h, [-1, 1*8*128])
    
    W_conv1_3_h = weight_variable([1, 5, 258, 128])
    b_conv1_3_h = bias_variable([128])

    h_conv1_3_h = tf.nn.relu(conv2d(conv2_h, W_conv1_3_h) + b_conv1_3_h)
    h_pool1_3_h = max_pool(h_conv1_3_h, 1, 2)

    h_pool_flat1_h = tf.reshape(h_pool1_3_h, [-1, 1*8*128])
    
    W_conv2_3_h = weight_variable([1, 9, 258, 128])
    b_conv2_3_h = bias_variable([128])

    h_conv2_3_h = tf.nn.relu(conv2d(conv2_h, W_conv2_3_h) + b_conv2_3_h)
    h_pool2_3_h = max_pool(h_conv2_3_h, 1, 2)

    h_pool_flat2_h = tf.reshape(h_pool2_3_h, [-1, 1*8*128])
    
    W_conv3_3_h = weight_variable([1, 13, 258, 128])
    b_conv3_3_h = bias_variable([128])

    h_conv3_3_h = tf.nn.relu(conv2d(conv2_h, W_conv3_3_h) + b_conv3_3_h)
    h_pool3_3_h = max_pool(h_conv3_3_h, 1, 2)

    h_pool_flat3_h = tf.reshape(h_pool3_3_h, [-1, 1*8*128])
    
    
    # conv1_ap
    W_conv4_1_a = weight_variable([1, 3, 3, 16])
    b_conv4_1_a = bias_variable([16])

    h_concat4_1_a = tf.concat([batch_X_a, batch_mask_2], 3)
    h_conv4_1_a = tf.nn.relu(conv2d(h_concat4_1_a, W_conv4_1_a) + b_conv4_1_a)
    h_pool4_1_a = max_pool(h_conv4_1_a, 1, 2)
    
    W_conv1_1_a = weight_variable([1, 5, 3, 16])
    b_conv1_1_a = bias_variable([16])

    h_concat1_1_a = tf.concat([batch_X_a, batch_mask_2], 3)
    h_conv1_1_a = tf.nn.relu(conv2d(h_concat1_1_a, W_conv1_1_a) + b_conv1_1_a)
    h_pool1_1_a = max_pool(h_conv1_1_a, 1, 2)

    W_conv2_1_a = weight_variable([1, 9, 3, 16])
    b_conv2_1_a = bias_variable([16])

    h_concat2_1_a = tf.concat([batch_X_a, batch_mask_2], 3)
    h_conv2_1_a = tf.nn.relu(conv2d(h_concat2_1_a, W_conv2_1_a) + b_conv2_1_a)
    h_pool2_1_a = max_pool(h_conv2_1_a, 1, 2)

    W_conv3_1_a = weight_variable([1, 13, 3, 16])
    b_conv3_1_a = bias_variable([16])

    h_concat3_1_a = tf.concat([batch_X_a, batch_mask_2], 3)
    h_conv3_1_a = tf.nn.relu(conv2d(h_concat3_1_a, W_conv3_1_a) + b_conv3_1_a)
    h_pool3_1_a = max_pool(h_conv3_1_a, 1, 2)

    mask_pool3_1_a = max_pool(batch_mask_2, 1, 2)
    
    conv1_ap = tf.concat([h_pool4_1_a, h_pool1_1_a], 3)
    conv1_ap = tf.concat([conv1_ap, h_pool2_1_a], 3)
    conv1_ap = tf.concat([conv1_ap, h_pool3_1_a], 3)
    conv1_ap = tf.concat([conv1_ap, mask_pool3_1_a], 3)
    
    # conv2_ap
    W_conv4_2_a = weight_variable([1, 3, 66, 64])
    b_conv4_2_a = bias_variable([64])

    h_conv4_2_a = tf.nn.relu(conv2d(conv1_ap, W_conv4_2_a) + b_conv4_2_a)
    h_pool4_2_a = max_pool(h_conv4_2_a, 1, 2)

    W_conv1_2_a = weight_variable([1, 5, 66, 64])
    b_conv1_2_a = bias_variable([64])

    h_conv1_2_a = tf.nn.relu(conv2d(conv1_ap, W_conv1_2_a) + b_conv1_2_a)
    h_pool1_2_a = max_pool(h_conv1_2_a, 1, 2)

    W_conv2_2_a = weight_variable([1, 9, 66, 64])
    b_conv2_2_a = bias_variable([64])

    h_conv2_2_a = tf.nn.relu(conv2d(conv1_ap, W_conv2_2_a) + b_conv2_2_a)
    h_pool2_2_a = max_pool(h_conv2_2_a, 1, 2)

    W_conv3_2_a = weight_variable([1, 13, 66, 64])
    b_conv3_2_a = bias_variable([64])

    h_conv3_2_a = tf.nn.relu(conv2d(conv1_ap, W_conv3_2_a) + b_conv3_2_a)
    h_pool3_2_a = max_pool(h_conv3_2_a, 1, 2)

    mask_pool3_2_a = max_pool(mask_pool3_1_a, 1, 2)
    
    conv2_ap = tf.concat([h_pool4_2_a, h_pool1_2_a], 3)
    conv2_ap = tf.concat([conv2_ap, h_pool2_2_a], 3)
    conv2_ap = tf.concat([conv2_ap, h_pool3_2_a], 3)
    conv2_ap = tf.concat([conv2_ap, mask_pool3_2_a], 3)
    
    # conv3_ap
    W_conv4_3_a = weight_variable([1, 3, 258, 128])
    b_conv4_3_a = bias_variable([128])

    h_conv4_3_a = tf.nn.relu(conv2d(conv2_ap, W_conv4_3_a) + b_conv4_3_a)
    h_pool4_3_a = max_pool(h_conv4_3_a, 1, 2)
    h_pool4_3_a = tf.reduce_mean(h_pool4_3_a, axis=1)
    
    h_pool_flat4_a = tf.reshape(h_pool4_3_a, [-1, 1*8*128])
    
    W_conv1_3_a = weight_variable([1, 5, 258, 128])
    b_conv1_3_a = bias_variable([128])

    h_conv1_3_a = tf.nn.relu(conv2d(conv2_ap, W_conv1_3_a) + b_conv1_3_a)
    h_pool1_3_a = max_pool(h_conv1_3_a, 1, 2)
    h_pool1_3_a = tf.reduce_mean(h_pool1_3_a, axis=1)
    
    h_pool_flat1_a = tf.reshape(h_pool1_3_a, [-1, 1*8*128])
    
    W_conv2_3_a = weight_variable([1, 9, 258, 128])
    b_conv2_3_a = bias_variable([128])

    h_conv2_3_a = tf.nn.relu(conv2d(conv2_ap, W_conv2_3_a) + b_conv2_3_a)
    h_pool2_3_a = max_pool(h_conv2_3_a, 1, 2)
    h_pool2_3_a = tf.reduce_mean(h_pool2_3_a, axis=1)

    h_pool_flat2_a = tf.reshape(h_pool2_3_a, [-1, 1*8*128])

    W_conv3_3_a = weight_variable([1, 13, 258, 128])
    b_conv3_3_a = bias_variable([128])

    h_conv3_3_a = tf.nn.relu(conv2d(conv2_ap, W_conv3_3_a) + b_conv3_3_a)
    h_pool3_3_a = max_pool(h_conv3_3_a, 1, 2)
    h_pool3_3_a = tf.reduce_mean(h_pool3_3_a, axis=1)
    
    h_pool_flat3_a = tf.reshape(h_pool3_3_a, [-1, 1*8*128])
    
    
    # fc
    h_pool_flat = tf.concat([h_pool_flat1_x, h_pool_flat1_y, h_pool_flat1_w, h_pool_flat1_h, h_pool_flat1_a,
                         h_pool_flat2_x, h_pool_flat2_y, h_pool_flat2_w, h_pool_flat2_h, h_pool_flat2_a, 
                         h_pool_flat3_x, h_pool_flat3_y, h_pool_flat3_w, h_pool_flat3_h, h_pool_flat3_a,
                            h_pool_flat4_x, h_pool_flat4_y, h_pool_flat4_w, h_pool_flat4_h, h_pool_flat4_a], 1)

    W_fc1 = weight_variable([20 * 8 * 128, 2048])
    b_fc1 = bias_variable([2048])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([2048, 1024])
    b_fc2 = bias_variable([1024])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([1024, 2])
    b_fc3 = bias_variable([2])

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    
    return y_conv

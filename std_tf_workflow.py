# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:47:53 2017

@author: Jackie
"""
from scipy.io import loadmat as load
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys



batch_size = 256
epoch_num = 3
init_lr = 0.0003
lr_decay = 0.9
train_data_dir = './Data/train_32x32.mat'
test_data_dir = './Data/test_32x32.mat'
tfb_dir = './FileWriter'
#若路径不存在，则创造路径（文件夹）
if not os.path.exists(tfb_dir):
    os.mkdir(tfb_dir)
ckpt_file ='./Checkpoint'
if not os.path.exists(ckpt_file):
    os.mkdir(ckpt_file) 
ckpt_dir = './Checkpoint/model.ckpt'
finial_ckpt_dir='./Checkpoint/finial_model.ckpt'




train_data = load(train_data_dir)
test_data = load(test_data_dir)



#观察数据结构
'''
print(train_data.keys())  #数据探索查看keys
print(train_data['__header__'])
print(train_data['__version__'])
print(train_data['__globals__'])
print(train_data['X'].shape)
print(train_data['y'].shape)
print(train_data['y'][0])
'''



def data_process(data_X,data_y):
    X = np.transpose(data_X,(3,0,1,2))
    #在某一维度加和数据，keepdims决定是否保留合并维度的存在
    #gray_X = np.add.reduce(X,axis=3,keepdims=True)/3.
    normalized_X = X - 128.
    y = [m[0] for m in data_y]
    onehot_list = []
    for i in range(len(y)):
        zero_list = [0 for n in range(10)]
        zero_list[y[i]-1] = 1
        onehot_list.append(zero_list)
        
    onehot_y = np.asarray(onehot_list)
   
    return normalized_X,onehot_y




def distribution(lables):
    labels_dict = {}
    for i in range(len(labels)):
        key = labels[i]
        if key in labels_dict:
            labels_dict[key] += 1
        else:
            labels_dict[key] = 1
    print(labels_dict)
    ##dict转成list，且key与value的list中的元素呈现一一对应的关系
    key_list = labels_dict.keys()#并不是list，而是dict_keys
    key_list = list(key_list)
    value_list = labels_dict.values()
    value_list = list(value_list)
    #print(type(key_list))
    #print(value_list)
    plt.bar(key_list,value_list)
    plt.show()



def batch_iter(data,batch_size):
    batch = len(data)//batch_size
    i = 0
    while i < batch:
        yield data[batch_size*i:batch_size*(i+1)]
        i += 1
#yeild的用法到最后一次next时会产生StopIteration
   
 
    

def weight_variable(shape,name=None):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name=name)



def bias_variable(shape,name=None):
    return tf.Variable(tf.constant(0.1,shape=shape),name=name)
    
 

def conv2d(x,W,b,name=None):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME',name=name) + b



def relu(x,name=None):
    return tf.nn.relu(x,name=name)



def max_pool(x,name=None):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)



def dropout(x,keep_prob=1):
    return tf.nn.dropout(x,keep_prob=keep_prob)



with tf.name_scope('input_datas'):
    input_X = tf.placeholder(tf.float32,[None,32,32,3],name='input_X')
    input_y = tf.placeholder(tf.float32,[None,10],name='input_y')
    tf.summary.histogram('input_X',input_X)
    tf.summary.histogram('input_y',input_y)
    


with tf.name_scope('keep_prob'):
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    

with tf.name_scope('learning_rate'):
    lr = tf.placeholder(tf.float32, name='learning_rate')
    tf.summary.scalar('learning_rate',lr)
 
      

with tf.name_scope('conv1_layer'):
    conv1_w = weight_variable([5,5,3,128],name='weights')
    conv1_b = bias_variable([128],name='bias')
    conv1 = conv2d(input_X,conv1_w,conv1_b,name='conv')
    relu1 = relu(conv1,name='relu')
    pool1 = max_pool(relu1)
    tf.summary.histogram('conv1_w',conv1_w)
    tf.summary.histogram('conv1_b',conv1_b)
    tf.summary.histogram('conv1',conv1)
    tf.summary.histogram('relu1',relu1)
    tf.summary.histogram('pool1',pool1)



with tf.name_scope('conv2_layer'):
    conv2_w = weight_variable([5,5,128,256],name='weights')
    conv2_b = bias_variable([256],name='bias')
    conv2 = conv2d(pool1,conv2_w,conv2_b,name='conv')
    relu2 = relu(conv2,name='relu')
    pool2 = max_pool(relu2)
    tf.summary.histogram('conv2_w',conv2_w)
    tf.summary.histogram('conv2_b',conv2_b)
    tf.summary.histogram('conv2',conv2)
    tf.summary.histogram('relu2',relu2)
    tf.summary.histogram('pool2',pool2)    



with tf.name_scope('flatten_layer'):
    flatten = tf.reshape(pool2,[-1,8*8*256])
    tf.summary.histogram('flatten',flatten)    




with tf.name_scope('fc_layer'):
    fc_w = weight_variable([8*8*256,10],name='weights')
    fc_b = bias_variable([10],name='bias')
    logits = tf.add(tf.matmul(flatten,fc_w),fc_b,name='logits')
    tf.summary.histogram('fc_w',fc_w)
    tf.summary.histogram('fc_b',fc_b)
    tf.summary.histogram('logits',logits)  




with tf.name_scope('dropout_layer'):
    logits = dropout(logits,keep_prob)
    tf.summary.histogram('logits',logits)  
    

 

   
with tf.name_scope('softmax_loss'):
    #tf.nn.softmax_cross_entropy_with_logits求出来的是batch_size的向量
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y,logits=logits))
    tf.summary.scalar('loss',loss)




with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(
            tf.cast(
                    tf.equal(
                            tf.argmax(logits,1),tf.argmax(input_y,1)),tf.float32))
    tf.summary.scalar('accuracy',accuracy)




with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss,name='optimizer')



with tf.name_scope('saver'):
    #max_to_keep：最多保留的检查点数。
    #keep_checkpoint_every_n_hours：保存检查点的时间间隔。
    #也可以在saver中添加你需要保存的变量tf.train.Saver({"v2": v2}) 。
    #可以有多个不同名称的saver
    saver = tf.train.Saver(name='saver',max_to_keep=100) #max_to_keep=1后一次保存的模型会覆盖前一次的，最终只会保存最后一次



train_X,train_y= data_process(train_data['X'],train_data['y'])
train_X,train_y= train_X[0:200],train_y[0:200]
test_X,test_y= data_process(test_data['X'],test_data['y'])
test_X,test_y= test_X[0:200],test_y[0:200]
train_X_iter = batch_iter(train_X,batch_size)
train_y_iter = batch_iter(train_y,batch_size)
test_X_iter = batch_iter(test_X,batch_size)
test_y_iter = batch_iter(test_y,batch_size)  
tf.summary.image('train_X',train_X)


summary_all = tf.summary.merge_all()


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


tfb_writer = tf.summary.FileWriter(tfb_dir,sess.graph)



'''
for epoch in range(epoch_num):
    train_batch_num = len(train_X)//batch_size
    for batch in range(train_batch_num):
        new_lr = init_lr*lr_decay*(epoch+1)
        #train_X_batch = train_X(batch*batch_size:batch*(batch_size+1))
        #train_y_batch = train_y(batch*batch_size:batch*(batch_size+1))
        print(train_X_batch.shape)
        print(train_y_batch.shape)
        print(train_y_batch)
        feed_dict={input_X:train_X_batch,input_y:train_y_batch,keep_prob:1,lr:new_lr}
        _,loss_,accuracy_,summary_all_=sess.run([optimizer,loss,accuracy,summary_all],feed_dict=feed_dict)
        print('epcho:',epoch+1,'       |batch:',batch+1,'       |loss:',loss_,'        |accuracy:',accuracy_)
        tfb_writer.add_summary(summary_all_,batch)#此处batch表示按batch再tensorboard上展示相关数据            
        saver.save(sess,ckpt_dir,global_step=batch+1)#global_step将训练的次数作为后缀加入到模型名字中
#mini_batch训练，张量切分出现问题
'''


for epoch in range(epoch_num):
    new_lr = init_lr*lr_decay
    #train_X_batch = train_X(batch*batch_size:batch*(batch_size+1))
    #train_y_batch = train_y(batch*batch_size:batch*(batch_size+1))
    feed_dict={input_X:train_X,input_y:train_y,keep_prob:1,lr:new_lr}
    _,loss_,accuracy_,summary_all_=sess.run([optimizer,loss,accuracy,summary_all],feed_dict=feed_dict)
    print('epcho:',epoch+1,'       |loss:',loss_,'        |accuracy:',accuracy_)
    #此处batch表示每次迭代都在tensorboard上展示相关数据
    tfb_writer.add_summary(summary_all_,epoch)            
    #global_step将训练的次数作为后缀加入到模型名字中
    saver.save(sess,ckpt_dir,global_step=epoch+1)
#生成最终的ckpt
saver.save(sess,finial_ckpt_dir)




test_accuracy,test_loss = sess.run([accuracy,loss],feed_dict={input_X:test_X,input_y:test_y,keep_prob:1})
print(test_accuracy,test_loss)

 
if __name__ == '__main__':
    pass
    

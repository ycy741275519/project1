# -*- coding: utf-8 -*-
# @File  : cnn_infer.py
# @Author: ycy
# @Date  : 2019/9/6
# @Desc  :定义cnn网络结构
import tensorflow as tf
import load_data as ld

conv1d_stride =1   #卷积步长
pool_stride = [2]    #池化步长
pool_win_shape=[2] #池化窗口
BATCH_SIZE = 50    #数据批量大小

# 第一层卷积大小和深度
CONV1_SIZE = 5
CONV1_DEEP = 32

# 第二层卷积大小和深度
CONV2_SIZE = 5
CONV2_DEEP = 32

# 全连接层节点数
FULL_SIZE=500
# softmax层数
CLASS_NUMBER =3

learning_rate = 0.1
TRAINING_STEPS =20000

# 定义卷积层
# arg:x 为输入
# arg:w 为过滤器权重
def conv1d(x,w):
    return tf.nn.conv1d(x,w,conv1d_stride,'SAME')

# 池化
# arg:x 为输入
# arg:win_shape 窗口大小
# arg:method 池化方式
def pool(x,win_shape,method='MAX'):
    return tf.nn.pool(x,window_shape=win_shape,pooling_type=method, padding="SAME",strides=pool_stride)

# 创建权重
# arg:shape (过滤器宽度，输入通道数，输出通道数)
# arg:name 变量名
def weight_variable(shape,name):
   return  tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=0.1))

# 创建偏置
# arg:name 变量名
# arg:shape 偏置大小
def bias_variable(shape,name):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.0))


# 前向传播
def inference():

    data, label = ld.get_data(ld.dataset_path)
    global_step = tf.Variable(0,trainable=False)
    # shape(批数据，输入宽度即列长，输入通道数)
    x = tf.placeholder(shape=[None, data.shape[1],1], name="input", dtype=tf.float32)
    y = tf.placeholder(shape=[None, 3],dtype=tf.float32)
    # 第一层卷积

    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
        conv1_weight=weight_variable(name='weight',shape=(CONV1_SIZE,1,CONV1_DEEP))
        conv1_bias=bias_variable(name='bis',shape=[CONV1_DEEP])
        conv1_output=tf.nn.relu(conv1d(x,conv1_weight)+conv1_bias)

    #第一层池化
    with tf.variable_scope('pool1',reuse=tf.AUTO_REUSE):
        pool1_output =pool(conv1_output,pool_win_shape,)

    # 第二层卷积
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
        conv2_weight=weight_variable(name='weight',shape=(CONV2_SIZE,CONV1_DEEP,CONV2_DEEP))
        conv2_bias=bias_variable(name='bis',shape=[CONV2_DEEP])
        conv2_output=tf.nn.relu(conv1d(pool1_output,conv2_weight)+conv2_bias)

    # 第二层池化
    with tf.variable_scope('pool2',reuse=tf.AUTO_REUSE):
        pool2_output =pool(conv2_output,pool_win_shape)

    # 将输出拉成向量
    pool2_shape = pool2_output.get_shape().as_list()
    nodes = pool2_shape[1]*pool2_shape[2]
    reshaped = tf.reshape(pool2_output,(-1,nodes))


    #全连接层1
    with tf.variable_scope("full1",reuse=tf.AUTO_REUSE):
        full1_weight = weight_variable([nodes,FULL_SIZE],"weight")
        full1_bias = bias_variable([FULL_SIZE],name='bias')
        full1_output = tf.nn.relu(tf.matmul(reshaped,full1_weight)+full1_bias)

    # 全连接层2
    with tf.variable_scope("full2", reuse=tf.AUTO_REUSE):
        full2_weight = weight_variable([FULL_SIZE, CLASS_NUMBER], "weight")
        full2_bias = bias_variable([CLASS_NUMBER],name='bias')
        full2_output = tf.nn.relu(tf.matmul(full1_output, full2_weight) + full2_bias)
        y_pred = tf.nn.softmax(logits=full2_output)

    # 损失函数
    with tf.name_scope('Loss'):
        # 求交叉熵损失
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=full2_output, name='cross_entropy')
        # 求平均
        loss = tf.reduce_mean(cross_entropy, name='loss')

    # 训练算法
    with tf.name_scope('Optimization'):
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    # 评估节点
    with tf.name_scope('Evaluate'):
        # 返回验证集/测试集预测正确或错误的布尔值
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        # 将布尔值转换为浮点数后，求平均准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    with tf.Session() as sess:
        # 变量初始化
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            start = (i * BATCH_SIZE) % data.shape[0]
            end = min(start + BATCH_SIZE, data.shape[0])
            _,loss_value,step = sess.run([train,loss,global_step],feed_dict={x:data[start:end],y:label[start:end]})
            if i%1000 ==0:
                print("after %d training step(s),loss on training batch is %g."%(step,loss_value))

if __name__ =='__main__':
    inference()
    print('ss')


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#创建一个input数据，-1到1之间300个数，[:,np.newaxis]把x_data变成300维的
x_data=np.linspace(-1,1,300)[:,np.newaxis]
#添加噪点，把他变得更像真实数据
noise=np.random.normal(0,0.05,x_data.shape)
#创建一个input的数据
y_data=np.square(x_data)-0.5+noise

#这里定义了一个添加神经层的方法
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    #定义layer_name是为了在可视化中可以看到这个模块的名字，这里传入的
    #n_layer代表我们现在正创建第几个神经层
    layer_name='layer%s' % n_layer
    #在这里是我们layer_name模块，可视化的时候我们可以看到结果
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            #这里定义的weights模块中，tf.random_normal方法从正态分布中输出随机值
            #输出形状为[in_size，out_size]的矩阵，令其为初始值，名字为W
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            #在这里将这个模块命名为layer_name+weights
            #并用tf.summary.histogram输入到日志文件中
            tf.summary.histogram(layer_name+'/weights',Weights)
            
        with tf.name_scope('biases'):
            #在这里另一个形状为[1,out_size]的矩阵为初始值
            #矩阵的每一个元素均为初始值
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
             
        with tf.name_scope('Wx_plus_b'):
            #这里定义的模块为Wx_plus_b
            #之后加上biases时是矩阵的每一行都去加biases这个数组
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        #在这里如果没有激活函数则直接输出
        #若有激活函数则用激活函数，然后给模块命名
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs',outputs)
        #sess=tf.Session()
        return outputs
        
with tf.name_scope('inputs'):
#这里用tf.placeholder定义一个参数，方便后续为其传值
     xs=tf.placeholder(tf.float32,[None,1],name='x_input')
     ys=tf.placeholder(tf.float32,[None,1],name='y_input')
     
#这里第一层输入参数inputs=xs，Weights是一个1*10的矩阵
#激活函数为relu
l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
#这里第二层输入参数inputs=l1，Weights是一个1*10的矩阵
#激活函数为空
prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)

#这里定义了一个损失函数，
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
#神经网络优化器，这里使用了梯度下降法
#使用优化器去减少每一步的误差
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
sess=tf.Session()
merged= tf.summary.merge_all()
#这里将神经网络结构输入到一个文件中
writer=tf.summary.FileWriter("logs/",sess.graph)

sess=tf.Session()
merged= tf.summary.merge_all()
#这里将神经网络结构输入到一个文件中
writer=tf.summary.FileWriter("logs/",sess.graph)


sess.run(tf.global_variables_initializer())
for i in range(1000):
    #开始训练，设置迭代次数为1000次
    #这里输入的x_data参数为一个300*1的矩阵
    #先在l1网络层运算,将300*10的矩阵Wx_plus_b输入到激活函数Relu中，然后输出
    #输出结果也为300*10的矩阵
    #然后在输出层prediction
    #输入为300*10的矩阵，Weights为10*1的矩阵
    #相乘后为300*1的矩阵然后加上1*1的biases
    #输出为300*91的矩阵
    #然后与之前的y_data去做loss误差分析
    #计算误差
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50==0:
        #每迭代50次输出带日志文件，将所有日志文件都merged合并起来
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
 
 writer=tf.summary.FileWriter("logs/",sess.graph)
 
 #　tensorboard --logdir=C:\Users\yuanninesuns\Desktop\python\logs

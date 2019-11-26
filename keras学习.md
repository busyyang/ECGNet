
# 模型构建及训练
keras是一个使用比较简单的深度学习框架。

首先导入需要使用的库。导入numpy的库，还有keras.layers的Dense,是连接层(fully-connect layer);以及keras.models的Sequential。
~~~python
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
~~~
除了全连接Dense层以外，还有卷积层Conv，池化Pooling等，可查看layers帮助[^https://keras.io/layers/about-keras-layers/]。
加载数据：
~~~python
# 加载保存在csv文件中的数据
dataset = np.loadtxt('datafix.csv', delimiter=',')
Y = dataset[:, 0]
X = dataset[:, 1:]
~~~
加载数据是通过np.loadtxt函数实现的，文件为`csv`文件，由于数据的第一列是标签Y，后面的是数据X，所以加载了之后就把X和Y都分别提出来。

创建模型：模型就是简单的全连接网络，首先创建一个空的序列模型，由于输入数据的X是25个特征，所以`input_dim=25`,前面的数字表示有多少个神经元，以及需要指定激活函数。
~~~python
model = Sequential()
model.add(Dense(64, input_dim=25, init='uniform', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
~~~
这个模型就是只有一个隐藏层的全连接网络，Dense的帮助[^https://keras.io/layers/core/#dense]。\
可以将模型的具体信息打印出来，`model.summary()`,模型的输入输出，复杂程度一目了然。
~~~
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 64)                1664      
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 33        
=================================================================
Total params: 3,777
Trainable params: 3,777
Non-trainable params: 0
_________________________________________________________________
~~~
模型的训练也比较简单：
~~~
# 编译模型，训练模型的方式
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型
model.fit(X, Y, epochs=20, batch_size=512, verbose=1)
~~~
需要先在`model.compile`中设定好方式，比如设置了损失函数为`binary_crossentropy`,使用`adam`优化器，将数据X,Y送入就可以model.fit就可以训练了。

获得准确率：
~~~python
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
~~~
还可以通过`preY = model.predict(X)`等方式查看时间输出的信息。

整体代码整合为：
~~~python
"""
使用keras进行深度学习模型搭建的方式，keras学习测试文件
"""
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# 加载保存在csv文件中的数据
dataset = np.loadtxt('datafix.csv', delimiter=',')
Y = dataset[:, 0]
X = dataset[:, 1:]

# 创建模型
model = Sequential()
model.add(Dense(64, input_dim=25, init='uniform', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 打印模型
model.summary()

# 编译模型，训练模型的方式
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X, Y, epochs=20, batch_size=512, verbose=1)
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
~~~

# 模型保存和载入

训练好了模型以后，可以通过保存模型的方式，让模型在其他地方或者其他时候使用。使用`model.save(path)`就可以把模型保存为一个HDF5文件。文件包括：模型结构，模型权重，训练配置，优化器的状态等信息。通过`keras.models.load_model(path)`调用这个文件。比如在训练模型的代码最后加入：
~~~python
from keras.models import load_model
model.save('test.h5')
print('saved!')
del model
reload_model = load_model('test.h5')
reload_scores = reload_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (reload_model.metrics_names[1], reload_scores[1] * 100))
~~~
那么可以看到加载的模型和训练的模型得到的准确率是一样的（实际上就是一个模型）。还有其他只保存模型结构，只保存权值等的方式参考[^https://blog.csdn.net/jiandanjinxin/article/details/77152530]。

# 绘制loss曲线
在写论文或者总结时候，通常还需要将历史的loss或者acc通过绘图绘制出来。需要在`keras.callbacks.Callback`中派生一个新的类出来，然后重写其中的3个函数，添加一个`loss_plot`函数。
~~~python
import keras
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        # 创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        plt.show()
~~~
在使用时候，首先实例一个对象，然后在`model.fit()`把实例的对象通过`callback`参数传递进去，就可以将数据获取了，然后调用`loss_plot`函数将曲线绘制出来。\
变化部分的代码：
~~~python
# 历史记录
lossHistory = LossHistory()
# 编译模型，训练模型的方式
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型
model.fit(X, Y, epochs=100, batch_size=512, verbose=1, callbacks=[lossHistory])
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
lossHistory.loss_plot('epoch')
~~~
这样就能看到曲线了，如果需要保存数据，也可以在`LossHistory`中再写一个函数保存数据即可。这个方式只能在训练结束之后才能plot出来，如果训练时间比较长，还可以通过tensorboard来plot出来，参考[^https://www.cnblogs.com/tectal/p/9426994.html]。\
上述模型训练并且plot损失，保存模型的代码整合为：
~~~python
"""
使用keras进行深度学习模型搭建的方式，keras学习测试文件
"""
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        # 创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        plt.show()


# 随机种子
seed = 7
np.random.seed(seed)
# 加载保存在csv文件中的数据
dataset = np.loadtxt('datafix.csv', delimiter=',')
Y = dataset[:, 0]
X = dataset[:, 1:]

# 创建模型
model = Sequential()
model.add(Dense(64, input_dim=25, init='uniform', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 打印模型
model.summary()

# 历史记录
lossHistory = LossHistory()

# 编译模型，训练模型的方式
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X, Y, epochs=100, batch_size=512, verbose=1,callbacks=[lossHistory])
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
lossHistory.loss_plot('epoch')
model.save('test.h5')
~~~

# 多分类

多分类时候，需要将标签转化为`one-hot`编码的形式，并且输出层不是输出为1，而是层数，损坏函数需要替换为`categorical_crossentropy`：
~~~python
from keras.utils import np_utils
......
def to_one_hot(y):
    return np_utils.to_categorical(y)

y_train = np.loadtxt('test.txt')
y_train = to_one_hot(y_train)
......
model.add(Dense(7, init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
......
~~~


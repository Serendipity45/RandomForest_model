import random
import numpy as np
import pandas as pd
import itertools
from keras import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras import layers
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from keras.layers import Input, Dense, LSTM, GRU, GlobalMaxPooling1D

#手动创建一个TransformerBlock类
#包含了多头注意力层、前馈神经网络层和层归一化层
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, *args, **kwargs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

def getTrans(T, F) -> Model:
    # 定义模型输入
    inputs = Input(shape=(T, F))
    # 创建 TransformerBlock
    x = TransformerBlock(embed_dim=F, num_heads=3, ff_dim=F)(inputs)
    # 使用全局最大池化层获取序列特征表示
    x = GlobalMaxPooling1D()(x)
    # 添加一个全连接层用于二分类，激活函数为 sigmoid
    outputs = Dense(1, activation='sigmoid')(x)
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    # 编译模型，指定优化器、损失函数和评估指标
    model.compile(optimizer=Adam(learning_rate=1e-3, decay=1e-4),
                  loss=BinaryCrossentropy(),metrics=['accuracy'])
                  # metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
    # 输出模型结构信息
    model.summary()
    return model

#GNN模型代码
def GRUModel(T, F) -> Model:
    inputs = Input(shape=(T, F))

    x = GRU(units=9, return_sequences=True)(inputs)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=1e-3, decay=1e-4), loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
    model.summary()

    return model


# 获取测试数据
def getData(path):
    # 读取数据
    data = pd.read_csv(path)
    data = np.array(data)
    # 打乱顺序
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]

    # 特征与标签分离
    feature = data[:, 1:]
    label = data[:, :1]
    #feature.shape[0] 表示样本的数量，保持不变，即每个样本仍然是一个单独的数据点
    #1表示时间步（timesteps），因为 LSTM 网络需要输入一个序列，这里将每个样本作为一个序列，所以这里设为 1，表示每个序列只包含一个时间步。
    #feature.shape[1] 表示每个时间步的特征数量，保持不变，即每个时间步的特征维度。

    #特征总数为feature.shape[0]*feature.shape[1]
    feature = np.reshape(feature, (feature.shape[0], 1, feature.shape[1]))#feature.shape[0] 条数 feature.shape[1] 特征向量数
    return feature, label

# 返回LSTM网络模型   ReLU 函数：f(x)=max(0,x)
def getLSTM():
    # 定义LSTM网络
    model = Sequential()  #创建一个空的 Sequential 模型，用于按顺序堆叠各层
    model.add(Input(shape=(1, 11)))  # 添加Input层来指定输入形状
    model.add(LSTM(32, activation='relu',return_sequences=True))
          #向模型中添加第一层 LSTM 层，具有 32 个隐藏单元，input_dim=11 表示输入特征的维度为 11。
          # return_sequences=True 表示该层返回完整的输出序列而不是最后一个输出。
    model.add(Dropout(0.2))  #添加一个 Dropout 层，用于在训练过程中随机丢弃 20% 的输入单元，以防止过拟合。
    model.add(LSTM(64, activation='relu', return_sequences=False))
            #添加第二层 LSTM 层，具有 64 个隐藏单元
            # return_sequences=False 表示该层只返回最后一个输出而不是完整的输出序列。
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
            #添加一个全连接层，包含 64 个神经元
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
           #再添加一个全连接层，包含 32 个神经元
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
          #最后添加一个全连接层，包含 1 个神经元，激活函数为 Sigmoid，用于输出二分类结果。
    # print(model_keras.summary())
    return model


# 定义 LossHistory 类，用于保存训练过程中的损失和准确率，并生成损失-准确率曲线图
class LossHistory(keras.callbacks.Callback):
    # 初始化方法，接受保存曲线图的路径作为参数
    def __init__(self, path):
        self.val_acc = None  #记录验证集的准确率
        self.val_loss = None  #记录验证集的损失
        self.accuracy = None  #记录训练集的准确率
        self.losses = None   #记录训练集的损失
        self.path = path     #指定保存损失-准确率曲线图的文件路径,创建类时会传入这个路径参数

    #Batch（批次）  Epoch（时期）
    # 当训练开始时调用的方法，用于初始化损失和准确率的字典
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 在每个批次训练结束时调用的方法，将当前批次的损失和准确率记录到对应的字典中
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    #Batch（批次）  Epoch（时期）
    # 在每个 epoch 结束时调用的方法，将当前 epoch 的损失和准确率记录到对应的字典中
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    #绘制损失-准确率曲线图的方法
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type])) #loss_type 是 'batch' 或 'epoch'
        plt.figure()
        # 训练准确率
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  #横坐标是iters是range(0,len-1) 纵坐标是accuracy
        # 训练损失
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':##对epoch这一类增加
            # 验证准确率
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc') # val_acc验证集上的准确率
            # 验证损失
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss') # val_loss 验证集上的损失
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(self.path)
        plt.show()


# 画图，模型检测效果可视化
class Plot:
    def __init__(self, name): #类的初始化方法，接受一个参数 name，表示保存图片的路径。
        self.path = name

    # 画混淆矩阵
    def plot_confusion_matrix(self, cm, classes):  #接受混淆矩阵cm和类别列表classes作为参数
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)# interpolation='nearest' 表示使用最近邻插值，cmap=plt.cm.Blues 表示使用蓝色调色板。
        plt.title('confusion matrix')
        plt.colorbar() #添加颜色条
        tick_marks = np.arange(len(classes)) #创建类别标记的范围
        plt.xticks(tick_marks, classes)  #设置 x 轴的刻度标签为类别列表
        plt.yticks(tick_marks, classes)  #设置 y 轴的刻度标签为类别列表
        thresh = cm.max() / 2.        #计算混淆矩阵中的最大值的一半
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #使用 itertools.product(笛卡尔积)联合两个变量 遍历混淆矩阵的每个元素的索引
            #在图中的每个单元格中添加文本，文本内容为混淆矩阵中对应位置的值，根据值的大小设置文本颜色为白色或黑色
            #注意i，j的值不要搞错了
            plt.text(j, i, cm[i, j], horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout() #调整子图参数，以使子图适应图形区域
        plt.savefig(self.path + 'confusion_matrix.png', bbox_inches='tight')
        plt.show()

    # 画准确率、精确率、回归率、f1
    def plot_valuation(self, num_list):
        rects = plt.bar(range(len(num_list)), num_list)
        plt.ylim(ymax=110, ymin=0)
        plt.xticks([0, 1, 2, 3], ['accuracy', 'precision', 'recall', 'f1'])
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height, 2)) + '%', ha='center', va='bottom') #文本水平居中对齐||文本底部对齐
        # ax = plt.gca()
        # ax.spines['top'].set_color('none')
        # ax.spines['right'].set_color('none')
        plt.savefig(self.path + 'valuation.png', bbox_inches='tight') #bbox_inches='tight' 边界框调整为最小尺寸
        plt.show()

    # 画ROC
    def plot_roc(self, fpr, tpr, roc_auc):
        plt.figure()
        lw = 2  #设置曲线的线宽
        # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr, tpr, color='darkorange',  #fpr 和 tpr 分别表示假正率和真正率，
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  #表示标签为 ROC 曲线下的面积  %0.2f 是一个占位符  %roc_auc则表示要插入的AUC值
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC曲线')
        plt.legend(loc="lower right") #位置为右下角
        plt.savefig(self.path + 'ROC.png', bbox_inches='tight')
        plt.show()

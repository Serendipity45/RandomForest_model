from __future__ import print_function  # 使用3.0版本

from keras.saving import save_model
from keras.src.optimizers import Adam

from model.utils import LossHistory, getData, getLSTM

batch_size = 20  #用于模型训练时每次处理的样本数量16
epochs = 5      #即训练模型时遍历整个数据集的次数50
split = 0.9      #用于将数据集划分为训练集90%和验证集10%
dataPath = '../data/merge_labeled_data.csv'   #数据集文件的路径
modelPath = '../model_keras/LSTM.keras'  #保存训练好的模型的路径
pngPath = '../pic/accLoss/accLoss_LSTM.png'     #保存损失-准确性曲线图的路径

# 创建一个实例history
history = LossHistory(pngPath)   #用于在训练过程中记录损失和准确性，并生成损失-准确性曲线图

# 获取训练集
feature, label = getData(dataPath)  #特征和标签分别赋值给 feature 和 label，getData函数已经将feature和label两个分开

# 划分训练集和验证集
train_split = int(len(feature) * split)#feature是一个二维数组，支持切片，不包括train_split这一个数据集，[0,train_split-1]
train_feature = feature[:train_split]  #训练集-feature
train_label = label[:train_split]     #训练集-label
val_feature = feature[train_split:]   #测试集-feature
val_label = label[train_split:]       #测试集-label

# 调整数据形状以匹配模型输入
# train_feature = train_feature.reshape(train_feature.shape[0], 1, train_feature.shape[1]) #时间步长为1，表示前后数据没有关联
# val_feature = val_feature.reshape(val_feature.shape[0], 1, val_feature.shape[1])

# 获取LSTM网络模型
model = getLSTM()

###  编译模型
# 损失函数为二元交叉熵---计算模型损失,适用于二分类问题
# 优化器为 Adam---一种基于梯度下降的优化算法
# 监控准确率指标---通过监测准确率，可以了解模型在训练过程中的性能表现，以便进行调优和评估
# 编译模型并设置学习率为0.001
adam = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# shuffle=True: 在每个 epoch 开始之前对训练数据进行随机打乱。通过打乱数据的顺序，可以增加模型的泛化能力，避免模型过度拟合训练数据。
# callbacks=[history] 用于记录训练过程中的损失和准确率
model.fit(train_feature, train_label, batch_size=batch_size, epochs=epochs, shuffle=True,
          validation_data=(val_feature, val_label), callbacks=[history])

# 训练好的模型的检测效果评估，用测试集进行评估，
#二分类，交叉熵函数为：L(y, p) = - (y * log(p) + (1 - y) * log(1 - p))
#多分类，交叉熵函数为：L(y, p) = - Σ (y_i * log(p_i))

loss, accuracy = model.evaluate(val_feature, val_label)
print("\n测试集Loss(损失): %.2f, 测试集Accuracy(准确度): %.2f%%" % (loss, accuracy*100))

# 绘制acc-loss曲线
history.loss_plot('epoch')
# history.loss_plot('batch')

# 保存训练模型
save_model(model, modelPath)  # 保存模型到一个 .keras 文件
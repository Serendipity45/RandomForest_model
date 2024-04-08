from keras.saving import save_model
from model.utils import getData, getTrans, LossHistory

batch_size = 20
epochs = 60
split = 0.9
dataPath = '../data/merge_labeled_data.csv'     #数据打开目录
modelPath = '../model_keras/Transformer.keras'  #模型保存路径
pngPath = '../pic/accLoss/accLoss_Tran.png'     #测试集准确度损失保存路径

# 创建一个实例history
history = LossHistory(pngPath)   #用于在训练过程中记录损失和准确性，并生成损失-准确性曲线图

# 加载数据
feature, label = getData(dataPath)
# 划分训练集和验证集
train_split = int(len(feature) * split)
train_feature = feature[:train_split]
train_label = label[:train_split]
val_feature = feature[train_split:]
val_label = label[train_split:]

# 调整数据形状以匹配模型输入
train_feature = train_feature.reshape(train_feature.shape[0], 1, -1) #-1会自动计算所需的大小，以确保总的元素数量保持不变
val_feature = val_feature.reshape(val_feature.shape[0], 1, -1)  #-1会自动计算所需的大小，以确保总的元素数量保持不变

# 使用 TransModel 函数创建模型
model = getTrans(T=1, F=train_feature.shape[2])

# 模型训练
#early_stop:是一个早停回调（EarlyStopping），通常用来防止模型过拟合
# history--使用自定义的LossHistory回调可以记录训练过程中的损失和准确性，并生成损失-准确性曲线图
model.fit(train_feature, train_label, batch_size=batch_size, epochs=epochs, shuffle=True,
          validation_data=(val_feature, val_label), callbacks=[history])  #callbacks=[early_stop]

# 模型评估
loss, accuracy = model.evaluate(val_feature, val_label)
print("\n测试集Loss(损失): %.2f, 测试集Accuracy(准确度): %.2f%%" % (loss, accuracy*100))

# 绘制损失-准确率曲线
history.loss_plot('epoch')

# 保存模型
save_model(model, modelPath)

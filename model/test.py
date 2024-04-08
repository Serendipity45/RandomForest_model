from __future__ import print_function

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix
from utils import Plot, getData,getLSTM,getTrans

# 各种路径
dataPath = './data/merge_labeled_data.csv'
pngPath = './pic/Conf-Roc-Valua_LSTM/'
# dataPath = './data/merge_adv_data.csv'
# pngPath = './pic/Conf-Roc-Valua_Trans'

modelPath = './model_keras/LSTM.keras'
# modelPath = './model_save/rf_w.pickle'

# 获取测试集
feature, label = getData(dataPath)


#model = getLSTM() #获取getLSTM()模型


# getTrans()获取特征的形状
# T, F = feature.shape
model = getLSTM() #传入的T 表示时间步数，F表示特征数

# 定义模型输入形状
# 加载训练好的模型
model.load_weights(modelPath)
# 表示使用 Adam 优化算法。
# Adam 是一种自适应学习率优化算法，特别适用于训练深度神经网络。
# 它能够在训练过程中自适应地调整学习率
#metrics=['accuracy'] 把准确率来作为评估模型的指标
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','recall','precision','f1'])

# 获得测试结果
prediction = model.predict(feature)
# 记录是否判别成功
y_pred = [0] * len(feature)
for i in range(len(prediction)):
    if prediction[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# 准确率、精确率、召回率、f1
accuracy = accuracy_score(label, y_pred) * 100
recall = recall_score(label, y_pred, average="macro") * 100
precision = precision_score(label, y_pred, average="macro") * 100
f1 = f1_score(label, y_pred, average="macro") * 100

plt = Plot(pngPath)
# 画ROC曲线
fpr, tpr, threshold = roc_curve(label, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot_roc(fpr, tpr, roc_auc)
# 画混淆矩阵
matrix = confusion_matrix(label, y_pred)
plt.plot_confusion_matrix(matrix, classes=[0, 1])
# 画评价指标
valuation = [accuracy, precision, recall, f1]
plt.plot_valuation(valuation)

### 机器学习模型中测试模型的鲁棒性
import tensorflow as tf
import numpy as np  #NumPy用于数值处理
from pandas import DataFrame  # DataFrame 用于数据操作
from model.utils import getData, getLSTM,getTrans

dataPath = '../data/data1/merge_labeled_data.csv'  #标记的合并数据保存路径
saveAdvPath = '../data/data1/merge_adv_data.csv'   #扰动数据保存路径
modelPath = './model_keras/Trans.keras'               #模型加载路径

# FGSM对抗攻击,产生扰动
def create_adversarial_pattern(input_feature, input_label):
    # 获得网络模型
    model = getTrans()
    model.load_weights(modelPath)

    with tf.GradientTape() as tape:
        # 批处理维度已经添加在getData函数中
        # 将输入特征和标签转换为 TensorFlow 张量
        input_feature = tf.convert_to_tensor(input_feature, tf.float32)
        target = tf.convert_to_tensor(input_label, tf.float32)
        # 确保GradientTape跟踪这些张量
        tape.watch(input_feature)
        # 通过模型获取预测，这里假设模型的输出形状与target的形状相匹配
        prediction = model(input_feature)
        # 计算损失函数
        loss = tf.keras.losses.binary_crossentropy(target, prediction)

    # 计算损失函数关于输入特征的梯度
    gradient = tape.gradient(loss, input_feature)
    # 取梯度的符号，并转换为 NumPy 数组
    signed_grad = tf.sign(gradient).numpy()
    return signed_grad

# 生成对抗样本
def FGSM():
    feature, label = getData(dataPath)
    # 设置扰动系数 ε 的值为 0.1
    eps = 0.1
    # 调用 create_adversarial_pattern 函数创建对抗样本
    adv_pattern = create_adversarial_pattern(feature, label)
    # 将对抗梯度乘以 ε 并加到原始特征上，创建对抗样本
    attack = feature + adv_pattern * eps
    #使用 np.clip 函数确保对抗样本的值在合法范围内（通常是0到1）
    attack = np.clip(attack, 0., 1.)
    #确保对抗样本的形状与原始特征相匹配
    adv_attack = attack.reshape((len(feature),feature.shape[2]))
    # 将标签和对抗样本的特征合并成一个 NumPy 数组（竖直拼接，标签拼接为第一列）
    adv_attack = np.concatenate((label, adv_attack), axis=1)
    # 将NumPy 数组转换为 pandas DataFrame
    data = DataFrame(adv_attack)
    # 定义CSV文件的表头
    header = ['label', 'src_ip_entropy', 'port_entropy', 'all_pkt_num', 'tcp_num', 'tcp_rate', 'icmp_num', 'icmp_rate',
              'udp_num', 'udp_rate', 'all_pkt_bytes', 'ave']

    data.to_csv(saveAdvPath, header=header, index=False)
    print('数据抗干扰化成功保存')

if __name__ == '__main__':
    FGSM()

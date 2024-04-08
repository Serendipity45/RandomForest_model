from sklearn import preprocessing
from sklearn.utils import shuffle

# 导入自定义的功能模块，用于特征统计和特征提取
from feature_statistic.trafic_feature import feature_statistic  #导入特征统计工具
from features.load_pcap import feature_extractor  #导入特征提取工具
import numpy as np
import pandas as pd

# 数据预处理类定义
class data_pre_process(object):
    # 类初始化
    def __init__(self, benign_path, mal_path, time_threshold=1):
        # 初始化数据路径和时间阈值
        self.data = None
        self.benign_pcap_path = benign_path
        self.malicious_pcap_path = mal_path
        self.benign_data_path = '../data/data1/benign_data.csv'
        self.malicious_data_path = '../data/data1/malicious_data.csv'
        self.time_threshold = time_threshold

    # 更新良性pcap文件路径
    def update_benign_pcap_path(self, _benign_pcap_path):
        self.benign_pcap_path = _benign_pcap_path

    # 更新恶意pcap文件路径
    def update_malicious_pcap_path(self, _malicious_pcap_path):
        self.malicious_pcap_path = _malicious_pcap_path

    # 更新良性数据文件路径
    def update_benign_data_path(self, _benign_data_path):
        self.benign_data_path = _benign_data_path

    # 更新恶意数据文件路径
    def update_malicious_data_path(self, _malicious_data_path):
        self.malicious_data_path = _malicious_data_path

    # 更新时间阈值
    def update_time_threshold(self, _time_threshold):
        self.time_threshold = _time_threshold

    # 处理良性数据
    def benign_data(self):
        fs_obj = feature_statistic()
        src_pkt_array = feature_extractor(self.benign_pcap_path, self.time_threshold)
        for data_block in src_pkt_array:
            fs_obj.recv_pkt_array(np.array(data_block))
            fs_obj.main()
        fs_obj.write_to_csv(self.benign_data_path)
        return fs_obj

    # 处理恶意数据
    def malicious_data(self):
        _fs_obj = feature_statistic()
        _src_pkt_array = feature_extractor(self.malicious_pcap_path, self.time_threshold)
        for _data_block in _src_pkt_array:#每一个data_block就是一个时间阈值对应的数据包集合
            _fs_obj.recv_pkt_array(np.array(_data_block))
            _fs_obj.main()
        _fs_obj.write_to_csv(self.malicious_data_path)
        return _fs_obj

    # 合并良性和恶意数据，并进行数据预处理
    def merge_data(self):
        benign_data = pd.read_csv(self.benign_data_path)
        malicious_data = pd.read_csv(self.malicious_data_path)

        # 合并数据并忽略索引
        self.data = benign_data._append(malicious_data, ignore_index=True)

        # 数据归一化
        Scaler_data = preprocessing.MinMaxScaler()
        self.data = Scaler_data.fit_transform(self.data)

        # 提取特征数据列
        src_ip_entropy_list = self.data[:, 0]
        port_information_entropy_list = self.data[:, 1]
        all_pkt_num_list = self.data[:, 2]
        tcp_num_list = self.data[:, 3]
        tcp_rate_list = self.data[:, 4]
        
        icmp_num_list = self.data[:, 5]
        icmp_rate_list = self.data[:, 6]

        udp_num_list = self.data[:, 7]
        udp_rate_list = self.data[:, 8]
        
        all_pkt_bytes_list = self.data[:, 9]
        avg_pkt_bytes_list = self.data[:, 10]

        # 制作标签列表
        label_list = []
        for i in range(0, len(benign_data)):
            label_list.append(0)
        for j in range(0, len(malicious_data)):
            label_list.append(1)

        # 构建包含特征和标签的DataFrame
        _data_frame = pd.DataFrame({
            'label': label_list,
            'src_ip_entropy': src_ip_entropy_list,
            'port_entropy': port_information_entropy_list,
            'all_pkt_num': all_pkt_num_list,
            'tcp_num': tcp_num_list,
            'tcp_rate': tcp_rate_list,
            'icmp_num': icmp_num_list,
            'icmp_rate':icmp_rate_list,
            'ucp_num': udp_num_list,
            'udp_rate': udp_rate_list,
            'all_pkt_bytes': all_pkt_bytes_list,
            'avg_pkt_bytes': avg_pkt_bytes_list})

        # 打乱数据集并输出到CSV文件
        _data_frame = shuffle(_data_frame) #shuffle:打乱数据顺序
        _data_frame.to_csv('../data/data1/merge_labeled_data.csv', index=False, sep=',', encoding='utf-8')

if __name__ == '__main__':
    # 设置良性和恶意pcap文件夹的路径
    # benign_path = 'F:\mirai-botnet(正常流量)'
    # malicious_path = 'F:\mirai-botnet(ddos攻击流量)'
    benign_path = '../pcap/benign_pcap_folder'
    malicious_path = '../pcap/malicious_pcap_folder'
    # 创建数据预处理对象
    dp = data_pre_process(benign_path, malicious_path, 1)

    # 提取特性并合并数据
    dp.benign_data()
    dp.malicious_data()
    dp.merge_data()
    pass

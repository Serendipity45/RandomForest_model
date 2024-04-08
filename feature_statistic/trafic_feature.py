import numpy as np
import pandas as pd
import math
from features.load_pcap import feature_extractor

'''
Descripition:一定时间内数据包的特征统计
'''


class feature_statistic(object):
    """
    @:param:src_ip_entropy--外连主机数
    @:param:port_entropy--端口信息熵
    @:param:all_pkt_num--包总数
    @:param:ack_num--ack为1的包数量
    @:param:ack_rate--ack为1的包比例
    @:param:syn_num--syn为1的包数量
    @:param:syn_rate--syn为1的包比例
    @:param:psh_num--psh为1的包数量
    @:param:psh_rate--psh为1的包比例
    @:param:all_pkt_bytes--所有包总字节数
    @:param:avg_pkt_bytes--平均字节数
    """

    def __init__(self):
        # 某含有时间间隔t内数据包特征的二维数组
        self.pkt_array = None
        # 以下为要统计的特征初始化
        self.src_ip_entropy = -1  # 替换diff_ip_num为源IP信息熵
        self.port_entropy = -1
        self.all_pkt_num = -1
        self.tcp_num = -1
        self.tcp_rate = -1
        self.icmp_num = -1  # 新增ICMP统计
        self.icmp_rate = -1  # 新增ICMP比率
        self.udp_num = -1  # 新增UDP统计
        self.udp_rate = -1  # 新增UDP比率
        self.all_pkt_bytes = -1
        self.avg_pkt_bytes = -1
        # 特征对应的列表初始化也相应更改
        self.src_ip_entropy_list = []
        self.port_information_entropy_list = []
        self.all_pkt_num_list = []
        self.tcp_num_list = []  # 新增tcp统计列表
        self.tcp_rate_list = []  # 新增tcp统计列表
        self.icmp_num_list = []  # 新增ICMP统计列表
        self.icmp_rate_list = []  # 新增ICMP比率列表
        self.udp_num_list = []  # 新增UDP统计列表
        self.udp_rate_list = []  # 新增UDP比率列表
        self.all_pkt_bytes_list = []
        self.avg_pkt_bytes_list = []

  #计算源端ip地址的信息熵
    def __src_ip_entropy_statistic(self):
        src_ip_dict = {}
        probability = []
        src_ip_array = self.pkt_array[:, 0]  # 假设源IP地址位于第一列
    
        src_ip_set = set(src_ip_array)
        all_ip_num = len(src_ip_array)
    
        for element in iter(src_ip_set):
            src_ip_dict.update({element: 0})
    
        for ip in src_ip_array:
            src_ip_dict[ip] += 1
    
        # 计算每个IP的概率
        for value in src_ip_dict.values():
            probability.append(value / all_ip_num)
        # 计算信息熵
        self.__information_entropy__src(probability)

    def __information_entropy__src(self, probability):
        entropy = 0
        for p in probability:
            if p > 0:
                entropy += p * math.log(p, 2)
        self.src_ip_entropy= -entropy

    # 端口信息熵（不区分源端口与目的端口）
    def __port_information_entropy_statistic(self):
        port_dict = {}
        probability = []
        src_port_array = self.pkt_array[:, 2]
        dst_port_array = self.pkt_array[:, 3]

        src_port_set = set(src_port_array)
        dst_port_set = set(dst_port_array)
        all_port_set = src_port_set.union(dst_port_set)
        array_len = len(src_port_array)
        all_port_num = array_len * 2

        for element in iter(all_port_set):
            port_dict.update({element: 0})

        for i in range(array_len):
            key_src = src_port_array[i]
            key_dst = dst_port_array[i]
            port_dict[key_src] += 1
            port_dict[key_dst] += 1

        # 计算每个端口的概率
        for value in port_dict.values():
            probability.append(value / all_port_num)
        # 计算信息熵
        self.__information_entropy(probability)

    # 信息熵计算函数
    def __information_entropy(self, probability):
        entropy = 0
        for p in probability:
            entropy += math.log(p, 2)
        self.port_entropy = -entropy

    # 总包数统计
    def __all_pkt_num_statistic(self):
        self.all_pkt_num = len(self.pkt_array)

    def __tcp_num_statistic(self):
        # 确认第4列是TCP标志
        tcp_array = self.pkt_array[:,4].astype(int)
        tcp_num = 0
        for i in tcp_array:
            if i == 1:
                tcp_num += 1
            else:
                pass
        self.tcp_num = tcp_num
        # 计算比率
        self.tcp_rate = self.tcp_num / self.all_pkt_num if self.all_pkt_num > 0 else 0

    #src_ip, dst_ip, src_port, dst_port, tcp_flag, icmp_flag, udp_flag, pkt_len
    # icmp个数以及置1的包所占比率
    def __icmp_num_statistic(self):
        # 确认第6列是icmp标志
        icmp_array = self.pkt_array[:, 5].astype(int)
        icmp_num = 0
        for i in icmp_array:
            if i ==1:
                icmp_num += 1
            else:
                pass
        self.icmp_num = icmp_num
        # 计算比率
        self.icmp_rate = self.icmp_num / self.all_pkt_num if self.all_pkt_num > 0 else 0

    # src_ip, dst_ip, src_port, dst_port, tcp_count, icmp_flag, udp_flag, pkt_len
    # UDP个数以及置1的包所占比率
    def __udp_num_statistic(self):
        udp_array = self.pkt_array[:, 6].astype(int)  # 假设第7列是udp标志
        udp_num = 0
        for i in udp_array:
            if i == 1:
                udp_num += 1
            else:
                pass
        self.udp_num = udp_num
        self.udp_rate = udp_num / self.all_pkt_num if self.all_pkt_num > 0 else 0


    # 窗口时间内总字节数
    def __all_pkt_bytes_statistic(self):
        all_pkt_bytes = 0
        pkt_bytes_array = self.pkt_array[:, 7]
        for pkt_bytes in pkt_bytes_array:
            all_pkt_bytes += int(pkt_bytes)
        self.all_pkt_bytes = all_pkt_bytes

    # 窗口时间内的平均字节数,总包数/总字节数
    def __avg_pkt_bytes_statistic(self):
        self.avg_pkt_bytes = self.all_pkt_bytes / self.all_pkt_num

    # 接收到的上游数据
    def recv_pkt_array(self, pkt_array_src):
        self.pkt_array = pkt_array_src

    # 每个时间块的数加入列表
    def __write_to_list(self):
        self.src_ip_entropy_list.append(self.src_ip_entropy)  # 新增源IP信息熵列表
        self.port_information_entropy_list.append(self.port_entropy)
        self.all_pkt_num_list.append(self.all_pkt_num)
        self.tcp_num_list.append(self.tcp_num) # 新增tcp统计列表
        self.tcp_rate_list.append(self.tcp_rate)  # 新增tcp比率列表
        self.icmp_num_list.append(self.icmp_num)  # 新增ICMP统计列表
        self.icmp_rate_list.append(self.icmp_rate)  # 新增ICMP比率列表
        self.udp_num_list.append(self.udp_num)  # 新增UDP统计列表
        self.udp_rate_list.append(self.udp_rate)  # 新增UDP比率列表
        self.all_pkt_bytes_list.append(self.all_pkt_bytes)
        self.avg_pkt_bytes_list.append(self.avg_pkt_bytes)


    # 写入数据文件
    def write_to_csv(self, _output_file_path):
        _data_frame = pd.DataFrame({
            'src_ip_entropy': self.src_ip_entropy_list,  # 替换'diff_ip_num'为'src_ip_entropy'
            'port_entropy': self.port_information_entropy_list,
            'all_pkt_num': self.all_pkt_num_list,
            'tcp_num': self.tcp_num_list,# 新增TCP数目列
            'tcp_rate': self.tcp_rate_list,# 新增TCP数目列
            'icmp_num': self.icmp_num_list,  # 新增ICMP数目列
            'icmp_rate': self.icmp_rate_list,  # 新增ICMP比率列
            'udp_num': self.udp_num_list,  # 新增UDP数目列
            'udp_rate': self.udp_rate_list,  # 新增UDP比率列
            'all_pkt_bytes': self.all_pkt_bytes_list,
            'avg_pkt_bytes': self.avg_pkt_bytes_list,})
        _data_frame.to_csv(_output_file_path, index=False, encoding='utf-8')

    # 特征统计处理主函数
    def main(self):
        self.__src_ip_entropy_statistic()  # 替换__diff_ip_statistic方法
        self.__port_information_entropy_statistic()
        self.__all_pkt_num_statistic()
        self.__tcp_num_statistic()
        self.__icmp_num_statistic()  # 新增ICMP统计方法
        self.__udp_num_statistic()  # 新增UDP统计方法
        self.__all_pkt_bytes_statistic()
        self.__avg_pkt_bytes_statistic()
        self.__write_to_list()


if __name__ == '__main__':
    # 统计特征
    fs_obj = feature_statistic()
    # 调用时间窗口函数接口 时间为秒级别
    src_pkt_array = feature_extractor( r'F:\数据集\MACCDC\MACCDC 2012' ,2)
    # src_pkt_array = feature_extractor('../pcap/test',1)
    for data_block in src_pkt_array:
        fs_obj.recv_pkt_array(np.array(data_block))
        fs_obj.main()
    # 输出数据文件
    fs_obj.write_to_csv('../data/data2/MACCDC攻击数据.csv')

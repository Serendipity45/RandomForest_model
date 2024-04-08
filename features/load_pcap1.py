from scapy.all import *
import socket
from itertools import groupby
from scapy.layers.inet import IP, TCP, ICMP, UDP
from features.Utils.fileUtils import get_file_path

# 定义 translate_ip 函数，将二进制格式的IP地址转换成字符串格式
def translate_ip(ip):
    try:
        return socket.inet_ntoa(ip)  # 尝试转换IPv4地址
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, ip)  # 转换IPv6地址

# 定义 parsing_packet 函数，解析每个数据包，并提取特征
def parsing_packet(packet):
    feature = []  # 初始化特征列表
    if IP in packet:
        ip_layer = packet[IP]
        src_ip = ip_layer.src  # Source IP address in string format
        dst_ip = ip_layer.dst  # Destination IP address in string format
        pkt_len = len(packet)
        src_port = None
        dst_port = None
        tcp_flag = 0
        icmp_flag = 0
        udp_flag = 0

        if TCP in packet:
            tcp_layer = packet[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            tcp_flag = 1
        elif ICMP in packet:
            icmp_flag = 1
        elif UDP in packet:
            udp_layer = packet[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
            udp_flag = 1

        # Append the extracted features to the list
        feature.extend([src_ip, dst_ip, src_port, dst_port, tcp_flag, icmp_flag, udp_flag, pkt_len])
    return feature

def feature_extractor(pcap_file_list, time_limit, max_packets=100000):
    feature_list = []  # 初始化特征列表
    # 获取文件路径列表
    pcap_files = get_file_path(pcap_file_list, '.pcap')
    # 遍历每个PCAP文件
    for pcap_file in pcap_files:
        # 使用Scapy读取PCAP文件
        packets = rdpcap(pcap_file)
        # 遍历每个数据包
        for packet in packets:
            # 检查是否超过最大数据包数量
            if len(feature_list) >= max_packets:
                break
            timestamp = int(packet.time / time_limit)
            # 解析数据包
            parsed_feature = parsing_packet(packet)
            if parsed_feature is not None:
                # 如果成功解析数据包，则将其特征添加到列表中
                feature_list.append([timestamp] + parsed_feature)
        # 如果已经达到最大数据包数量，提前结束循环
        if len(feature_list) >= max_packets:
            break
    if not feature_list:
        return []  # 如果 feature_list 为空，则返回空列表
    # 对特征列表按时间戳进行分组
    feature_list.sort(key=lambda x: x[0])
    grouped_features = groupby(feature_list, lambda x: x[0])
    result = []
    for key, group in grouped_features:
        group_list = list(group)
        result.append(group_list)
    return result

# 这是程序的入口
if __name__ == '__main__':
    time_window = 1  # 定义时间窗口，例如1秒
    pcap_file_folder = '../pcap/malicious_pcap_folder'  # 替换为你的PCAP文件路径
    # pcap_file_folder = r'F:\数据集\MACCDC'  # 替换为你的PCAP文件路径
    # 调用 feature_extractor 函数，并打印结果
    features_by_time_window = feature_extractor(pcap_file_folder, time_window)
    for features in features_by_time_window:
        print(features)

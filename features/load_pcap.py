import dpkt  # 用于解析PCAP文件
import socket  # 用于IP地址转换
from itertools import groupby  # 用于对特征列表进行分组

from features.Utils.fileUtils import get_file_path  # 导入自定义模块以获取文件路径


# 定义 translate_ip 函数，将二进制格式的IP地址转换成字符串格式
def translate_ip(ip):
    try:
        return socket.inet_ntop(socket.AF_INET, ip)  # 尝试转换IPv4地址
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, ip)  # 转换IPv6地址

# 定义 parsing_packet 函数，解析每个数据包，并提取特征
def parsing_packet(timestamp, pkt):
    feature = []  # 初始化特征列表
    try:
        eth = dpkt.ethernet.Ethernet(pkt)  # 解析数据包的以太网层
        ip = eth.data  # 获取IP层数据
    except Exception as e:
        print("Error parsing packet:", e)
        return None

    # 检查是否存在源IP和目标IP，如果不存在则跳过该数据包
    if not hasattr(ip, 'src') or not hasattr(ip, 'dst'):
        return None
    # 获取IP地址和数据包长度
    src_ip = translate_ip(ip.src)
    dst_ip = translate_ip(ip.dst)
    pkt_len = len(pkt)
    src_port = None
    dst_port = None
    trans = ip.data
    
    # 初始化为None
    if isinstance(trans, dpkt.tcp.TCP) or isinstance(trans, dpkt.udp.UDP):
        src_port = trans.sport
        dst_port = trans.dport

    # tcp标志位检查
    tcp_flag = 1 if isinstance(trans, dpkt.tcp.TCP) else 0

    # ICMP标志位检查
    icmp_flag = 1 if isinstance(trans, dpkt.icmp.ICMP) else 0

    # UDP标志位检查
    udp_flag = 1 if isinstance(trans, dpkt.udp.UDP) else 0

    # 将提取到的特征加入列表中
    feature.extend( [timestamp, src_ip, dst_ip, src_port, dst_port,tcp_flag, icmp_flag, udp_flag, pkt_len])
    return feature


# 定义 feature_extractor 函数，提取特征
def feature_extractor(pcap_file_list, time_limit):
    feature = []  # 初始化特征列表
    #打开PCAP文件并使用dpkt库创建一个阅读器对象
    opened_pcap_files = [dpkt.pcap.Reader(open(file, "rb")) for file in get_file_path(pcap_file_list, '.pcap')]
    # 遍历每个打开的PCAP文件
    for pcap_file in opened_pcap_files:
        # 遍历PCAP文件中的每个时间戳和缓冲区
        for ts, buf in pcap_file:
            timestamp = int(ts / time_limit)  # 根据时间阈值计算时间戳
            parsed_packet = parsing_packet(timestamp, buf)  # 解析数据包
            if parsed_packet is not None:
                feature.append(parsed_packet)  # 如果能成功解析数据包，则将其特征添加到特征列表中
    # 使用groupby对特征列表进行分组
    lstg = groupby(feature, lambda x: x[0])
    result = []
    # 遍历每个分组
    for key, group in lstg:
        temp_list = list(map(lambda x: x[1:], list(group)))
        result.append(temp_list)
    return result

# 这是程序的入口
if __name__ == '__main__':
    # 调用 feature_extractor 函数，并打印结果
    print(feature_extractor('../pcap/malicious_pcap_folder',1))
    # pcap_file_folder = r'F:\数据集\MACCDC'  # 替换为你的PCAP文件路径
    # print(feature_extractor(pcap_file_folder,1))
    # print(feature_extractor('../pcap/benign_pcap_folder',1))
    # print(feature_extractor(r'F:\mirai-botnet(ddos攻击流量)', 1))

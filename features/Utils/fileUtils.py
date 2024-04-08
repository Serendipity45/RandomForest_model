import os


def get_file_path(file_dir, file_type):
    file_list = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        # 判断为文件
        if os.path.isfile(path):
            # 判断为pcap文件
            if os.path.splitext(path)[1] == file_type:
                file_list.append(path)
        # 判断为文件夹，则递归
        if os.path.isdir(path):
            get_file_path(path, file_type)
    return file_list

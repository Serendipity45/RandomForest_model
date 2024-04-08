import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
# 导入numpy模块，用于进行高效的数组计算
import numpy as np
# 导入pandas模块，用于数据分析和操作
import pandas as pd
# 从sklearn.ensemble导入RandomForestClassifier，用于创建随机森林模型
from sklearn.ensemble import RandomForestClassifier
# 导入train_test_split函数，用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 从sklearn.model_selection导入RandomizedSearchCV，用于进行随机参数搜索
from sklearn.model_selection import RandomizedSearchCV

# 以下是data_pre_process模块的导入，但实际代码中没有提供这部分，所以无法确定其具体功能
from process.data_process import data_pre_process


# 定义RandomForest类，封装随机森林模型的训练和预测过程
class RandomForest(object):
	# 初始化方法，接收一个模型加载路径
	def __init__(self, load_model_path):
		self.train_data = None  # 初始化训练数据属性为None
		self.test_data = None  # 初始化测试数据属性为None
		self.file_path = '../../data/data1/merge_labeled_data.csv'  # 数据集文件路径
		self.model_path = '../../model_save/rf_w.pickle'  # 模型保存路径
		self.model_path_adv = '../../model_save/rf_w_adv.pickle'  #优化后的模型保存路径
		self.load_model_path = load_model_path  # 模型加载路径
	
	# 私有方法：读取已标记的数据集文件
	def __load_data_file(self):
		self.data_src = pd.read_csv(self.file_path)  # 使用pandas读取CSV文件数据
	
	# 私有方法：数据预处理
	def __data_pre_process(self):
		# 将数据与标签分离，并且转为np.array形式
		self.labels = np.array(self.data_src['label'])  # 获取标签列并转换为numpy数组
		self.data = self.data_src.drop('label', axis=1)  # 删除标签列，保留数据
		self.data = np.array(self.data)  # 将剩余数据转换为numpy数组
  
		# 划分数据集为训练集和测试集
        #25%的数据会被划分为测试集
        #random_state=0--确保每次划分都能得到相同的数据划分结果
		self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
			self.data, self.labels, test_size=0.25, random_state=0
		)
	
	# 私有方法：随机森林模型的主要执行函数
	def __random_forest_main_execute_function(self):
		# 训练模型并给出预测评分
        # rfc--随机森林分类器，初始化
        # n_estimators=100--随机森林中决策树的数量
        # max_depth=100--这个参数指定了每棵决策树的最大深度（如果没有足够的数据来支持深度为100的树，则实际深度可能会小于100）
        # random_state=1，意味着每次使用相同数据和参数配置运行代码时，得到的随机森林模型都将是相同的
		self.rfc = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=1)
		self.rfc = self.rfc.fit(self.train_x, self.train_y)  # 使用训练数据拟合模型
		score_train = self.rfc.score(self.train_x, self.train_y)  # 计算训练集精确度
		score_test = self.rfc.score(self.test_x, self.test_y)  # 计算测试集精确度
		print("初始随机森林训练集精确度:", format(score_train))  # 打印训练集精确度
		print("初始随机森林测试集精确度:", format(score_test))  # 打印测试集精确度
		self.__save_model()  # 调用保存模型的私有方法
		# # 模型训练及保存模型
		# print('----------------------------------------------------')
		# # 调参(备用)
		# self.__grid_search_function()  # 这部分代码被注释，如果需要进行网格搜索调参则取消注释
		# score_train = self.rfc.score(self.train_x, self.train_y)  # 计算训练集精确度
		# score_test = self.rfc.score(self.test_x, self.test_y)  # 计算测试集精确度
		# print("优化后的随机森林训练集精确度:", format(score_train))  # 打印训练集精确度
		# print("优化后的随机森林测试集精确度:", format(score_test))  # 打印测试集精确度
		# self.__save_model_adv()  # 调用保存模型的私有方法
	
	# 网格搜索寻找最优超参数
	def __grid_search_function(self):
		# 设置随机搜索网格参数
        #10, 15, 20, ..., 195中选择决策树的数量进行模型训练。
		n_estimators = np.arange(10, 200, step=5)
        # "auto"：自动选择特征数，等同于特征总数--"sqrt"：特征数为总特征数的平方根--"log2"：特征数为总特征数的对数值（以2为底）
		max_features = ["sqrt", "log2"]
        # 从4开始--98结束，步长为2，列表末尾为None，表示无限深度
		max_depth = list(np.arange(4, 100, step=2)) + [None]
        # 表示分裂内部节点所需的最小样本数将在2, 4, 6, 8中选择
		min_samples_split = np.arange(2, 10, step=2)
        # 每个叶子节点必须有至少1, 2, 或4个训练样本
		min_samples_leaf = [1, 2, 4]  # 设置min_samples_leaf（最小样本叶节点数）搜索选项
        # 训练每棵决策树时是否使用bootstrap样本（即抽样替换）
		bootstrap = [True, False]  # 设置bootstrap（是否使用bootstrap样本）搜索选项
		
		# 定义参数网格
		param_grid = {
			"n_estimators": n_estimators,
			"max_features": max_features,
			"max_depth": max_depth,
			"min_samples_split": min_samples_split,
			"min_samples_leaf": min_samples_leaf,
			"bootstrap": bootstrap,
		}
		
		# 随机搜索并进行10折交叉验证
        # self.rfc：需要优化的模型
		# param_grid：每个超参数的搜索空间
		# 定义随机搜索的迭代次数，尝试的不同参数组合数量
		# cv=10：数据集将被划分为10个不同的子集，模型将在这10个子集上进行训练和验证--9个数据集作为训练集，1个数据集作为测试集
		# scoring="r2"：这指定了模型评估的标准，即使用 R² 分数，它是回归模型预测性能的指标。
		# n_jobs=-1：这告诉 RandomizedSearchCV 使用所有可用的CPU核心进行计算
		random_cv = RandomizedSearchCV(
			self.rfc, param_grid, n_iter=1000, cv=10, scoring="r2", n_jobs=-1
		)
		# self.rfc 变量被更新为这个最佳估计器
		#random_cv中：
		# best_estimator_属性存储最佳的估计器
		#best_params_属性存储最佳的参数
		self.rfc = random_cv.fit(self.train_x, self.train_y)  # 使用训练数据和参数网格进行随机搜索拟合
		print("最佳参数:")  # 打印最佳参数
		print(random_cv.best_params_)  # 输出最佳参数
	
	# 使用模型进行数据预测
	def __data_predict(self):
		# 使用随机森林带的预测方法进行预测
		_clf = self.__load_model()  # 加载模型
		score_r = _clf.score(self.test_x, self.test_y)  # 使用测试数据计算模型精确度
		print("模型准确率", format(score_r))  # 打印模型精确率
		
		#计算模型准确率方法二：
		# _clf = self.__load_model()  # 加载模型
        # predictions = _clf.predict(self.test_x)
        # # 计算绝对误差
        # errors = abs(predictions - self.test_y)
        # # 如果error是1，则预测错误
        # error_num = 0
        # for error in errors:
        #     if error == 1:
        #         error_num += 1
        # accuracy = 1 - error_num / len(predictions)
        # print('模型准确率:', round(accuracy, 2))
  
	# 保存初始随机森林模型
	def __save_model(self):
		with open(self.model_path, 'wb') as f:  # 打开指定路径的文件以写入二进制模式
			pickle.dump(self.rfc, f)  # 使用pickle将模型序列化并保存到文件中
		
	# 保存优化后的随机森林模型
	def __save_model_adv(self):
		with open(self.model_path_adv, 'wb') as f:  # 打开指定路径的文件以写入二进制模式
			pickle.dump(self.rfc, f)  # 使用pickle将模型序列化并保存到文件中
	
	# 加载模型
	def __load_model(self):
		with open(self.load_model_path, 'rb') as f:  # 打开指定路径的文件以读取二进制模式
			_clf = pickle.load(f)  # 使用pickle加载并反序列化模型
		return _clf  # 返回加载的模型
	
	# 训练执行函数
	def train_main(self):
		self.__load_data_file()  # 调用加载数据文件的方法
		self.__data_pre_process()  # 调用数据预处理方法
		self.__random_forest_main_execute_function()  # 调用随机森林主执行函数
	
	# 预测执行函数
	def predict_main(self):
		self.__load_data_file()  # 调用加载数据文件的方法
		self.__data_pre_process()  # 调用数据预处理方法
		self.__data_predict()  # 调用数据预测方法


# 以下是主执行区域
if __name__ == '__main__':
	execute_type =0  # 执行类型，1代表预测
	model_path = '../../model_save/rf_w.pickle'  # 模型保存路径
	# 针对划分的数据集，对模型进行--（训练+测试）
	if execute_type == 0:  # 如果执行类型为0，则执行数据训练和预测测试过程
		rf_object = RandomForest(model_path)  # 创建RandomForest对象
		rf_object.train_main()  # 调用训练主函数
	
	# 针对划分的数据集，模型没有训练过的数据，加载模型，来做预测
	elif execute_type == 1:  # 如果执行类型为1，则加载模型，只执行预测过程
		rf_object = RandomForest(model_path)  # 创建RandomForest对象
		rf_object.predict_main()  # 调用预测主函数
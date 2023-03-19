
import numpy
from abc import ABCMeta
from common.logger import logs
from sklearn.neural_network import MLPRegressor


class DataPreprocess(metaclass=ABCMeta):

    @staticmethod
    def data_preprocess(train_feature_data=None, train_target_data=None, online_capacity=None,
                        record_restrict=None):
        """
        数据预处理
        :param train_feature_data: 训练输入数据
        :param train_target_data: 训练输出数据
        :param online_capacity: 开机容量
        :param record_restrict: 限电记录
        :return: train_feature_data, train_target_data：训练输入数据，训练输出数据
        """
        train_feature_data, train_target_data = DataPreprocess.data_preprocess_wrongdata(train_feature_data,
                                                                                         train_target_data,
                                                                                         online_capacity)
        train_feature_data, train_target_data = DataPreprocess.data_preprocess_restrictdata(train_feature_data,
                                                                                            train_target_data,
                                                                                            online_capacity,
                                                                                            record_restrict)
        return train_feature_data, train_target_data

    @staticmethod
    def data_preprocess_normalization(train_target_data=None, online_capacity=None):
        """
        数据归一化
        :param train_target_data: 训练输出数据
        :param online_capacity: 开机容量
        :return: train_feature_data, train_target_data：训练输入数据，训练输出数据
        """
        # 数据归一化
        train_target_data = train_target_data / online_capacity

        return train_target_data

    @staticmethod
    def data_preprocess_wrongdata(train_feature_data, train_target_data, online_capacity=1):
        """
        更新训练集数据：识别并去除错误数据
        :param train_feature_data: 训练输入数据
        :param train_target_data: 训练输出数据
        :param online_capacity: 开机容量
        :return: train_feature_data, train_target_data：训练输入数据，训练输出数据
        """
        # 判断错误数据的阈值，建议值50
        # 当连续threshold时段长度的数据变化率小于limit * capacity，认为该数据为错误
        threshold = 50

        # 判断错误数据的阈值，建议值0.01
        # 当连续threshold时段长度的数据变化率小于limit * capacity，认为该数据为错误
        limit = 0.01

        # 标记错误数据
        # 长时间不变的数据
        flag = numpy.zeros((train_feature_data.shape[0], 1))
        for i in range(train_feature_data.shape[0] - threshold):
            ma = max(train_target_data[i:i + threshold])
            mi = min(train_target_data[i:i + threshold])
            if (ma - mi) < limit * online_capacity:
                flag[i: i + threshold, :] = numpy.ones((threshold, 1))

        # 功率小于（-0.05*装机容量）的数据
        for i in range(len(train_target_data)):
            if train_target_data[i] < -0.05 * online_capacity:
                flag[i, :] = 1

        # 去除错误数据
        train_feature_data1 = train_feature_data.copy()
        train_target_data1 = train_target_data.copy()
        flaga = flag.copy()
        for i in range(flaga.shape[0], 0, -1):
            if flaga[i - 1] == 1:
                flaga = numpy.vstack((flaga[:i - 1, :], flaga[i:, :]))
                train_feature_data1 = numpy.vstack((train_feature_data1[:i - 1, :], train_feature_data1[i:, :]))
                train_target_data1 = numpy.hstack((train_target_data1[:i - 1], train_target_data1[i:]))

        from sklearn.neural_network import MLPRegressor
        train_target_data1 = numpy.maximum(train_target_data1, 0)
        clf_bp = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", learning_rate_init=0.01, max_iter=300,
                              random_state=21)  # 定义模型
        clf_bp.fit(train_feature_data1, train_target_data1)  # 模型训练
        for i in range(flag.shape[0]):
            if flag[i] == 1:
                train_target_data[i] = clf_bp.predict(train_feature_data[i, :].reshape(1, -1))
        train_target_data = numpy.maximum(train_target_data, 0)
        train_target_data = numpy.minimum(train_target_data, online_capacity)

        return train_feature_data, train_target_data

    @staticmethod
    def data_preprocess_restrictdata(train_feature_data, train_target_data, online_capacity=1, record_restrict=None):
        """
        更新训练集数据：识别并去除限电数据;判断功率单位是否统一功能分开
        :param train_feature_data: 训练输入数据
        :param train_target_data: 训练输出数据
        :param online_capacity: 开机容量
        :param record_restrict: 限电记录
        :return: train_feature_data, train_target_data：训练输入数据，训练输出数据
        """
        # 识别限电数据self,判断自动生成分两种情况，参数判断是否执行
        if record_restrict is None:
            record_restrict = online_capacity * numpy.ones((train_target_data.shape[0], 1))
            train_target_data = numpy.maximum(train_target_data, 0)
            clf_bp = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", learning_rate_init=0.01, max_iter=300,
                                  random_state=21)
            clf_bp.fit(train_feature_data, train_target_data)  # 模型训练
            predict_target = clf_bp.predict(train_feature_data)  # 预测
            dpower = abs(predict_target - train_target_data)
            thresh = 0.8 * online_capacity
            for i in range(train_target_data.shape[0]):
                if dpower[i] > thresh:
                    record_restrict[i] = 0

        # 去除限电数据
        train_feature_data1 = train_feature_data.copy()
        train_target_data1 = train_target_data.copy()
        online_capacity_list_1 = record_restrict.copy()
        for i in range(record_restrict.shape[0], 0, -1):
            if record_restrict[i - 1] == 0:
                record_restrict = numpy.vstack((record_restrict[:i-1, :], record_restrict[i:, :]))
                train_feature_data1 = numpy.vstack((train_feature_data1[:i - 1, :], train_feature_data1[i:, :]))
                train_target_data1 = numpy.hstack((train_target_data1[:i - 1], train_target_data1[i:]))
        clf_bp = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", learning_rate_init=0.01, max_iter=300,
                              random_state=21)
        clf_bp.fit(train_feature_data1, train_target_data1)  # 模型训练

        for i in range(online_capacity_list_1.shape[0]):
            if online_capacity_list_1[i] == 0:
                train_target_data[i] = clf_bp.predict(train_feature_data[i, :].reshape(1, -1))
        train_target_data = numpy.maximum(train_target_data, 0)
        # 识别功率的单位是否统一
        if online_capacity / max(train_target_data) > 1.25 or online_capacity / max(train_target_data) < 1:
            logs.debug("提示：功率单位可能出错！历史功率最大值/装机容量=" + str(max(train_target_data) / online_capacity))

        return train_feature_data, train_target_data

    @staticmethod
    def data_preprocess_error_data_label(test_target_data, predict_type, online_capacity, limit=0.02):
        """
        更新训练集数据：识别并去除限电数据;判断功率单位是否统一功能分开
        :param test_target_data: 测试输出数据
        :param predict_type: 预测类型
        :param online_capacity: 开机容量
        :param limit：判断为限电的门槛，当连续threshold时段长度的数据变化率小于limit * capacity，认为该数据为错误
        :return: restrict_label：限电记录
        """
        # 判断错误数据的阈值，建议值
        # 当连续threshold时段长度的数据变化率小于limit * capacity，认为该数据为错误
        threshold = 70
        if predict_type == "wind":
            threshold = 40

        # 识别错误数据
        flag = numpy.zeros((test_target_data.shape[0], 1))
        for i in range(test_target_data.shape[0] - threshold):
            ma = max(test_target_data[i:i + threshold, 1])
            mi = min(test_target_data[i:i + threshold, 1])
            if (ma - mi) < limit * online_capacity:
                flag[i: i + threshold, :] = numpy.ones((threshold, 1))
        restrict_label = numpy.hstack((test_target_data[:, 0].reshape(-1, 1), flag))
        return restrict_label

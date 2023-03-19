
from common.logger import logs
import numpy
from sklearn import ensemble, svm
from tensorflow import keras
from app.model_parameter_optimization.model_parameter_search_basic_method import segmentation_of_data, \
    parameter_common_gridsearch, parameter_gridsearch_for_sklearn, parameter_girdsearch
from model import BNN, BLSTM, GRU, LSTM, RandomForest, XGBoost
import torch
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor


def get_best_parameter(predict_term, model_name, train_feature, train_target):
    """
    获取单个场站模型最优参数，并保存
    :param predict_term:预测周期，分为短期‘short’或超短期‘ultra_short’
    :param model_name: 模型名称
    :param train_feature: 训练nwp数据
    :param train_target: 训练功率数据
    :return:
    """
    best_parameter = {}
    if model_name in ['BPNN', 'MFFS', 'BLSTM', 'BNN', 'GRU', 'GBRT', 'DecisionTreeRegressor', 'LSTM', 'RandomForest',
                      'SVM', 'XGBoost']:
        if model_name == 'BNN':
            best_parameter = bnn_parameter_selfsearch(train_feature, train_target, predict_term)
        # 决策树超参数寻优
        if model_name == 'DecisionTreeRegressor':
            best_parameter = decision_tree_parameter_randomsearch_for_sklearn(train_feature, train_target, predict_term)
        # blstm超参数寻优
        if model_name == 'BLSTM':
            best_parameter = blstm_parameter_gridsearch(train_feature, train_target, predict_term)
        # gru超参数寻优
        if model_name == 'GRU':
            best_parameter = gru_parameter_gridsearch(train_feature, train_target, predict_term)
            best_parameter['epochs'] = 20
        # bpnn超参数寻优
        if model_name == 'BPNN':
            best_parameter = bpnn_parameter_Randomsearch(train_feature, train_target, predict_term)
        # mffs超参数寻优
        if model_name == 'MFFS':
            best_parameter = mffs_parameter_gridsearch(train_feature, train_target, predict_term)
        # gbrt超参数寻优
        if model_name == 'GBRT':
            best_parameter = gbrt_parameter_randomsearch_for_sklearn(train_feature, train_target, predict_term)
        # rf超参数寻优
        if model_name == 'RandomForest':
            best_parameter = rf_parameter_randomsearch(train_feature, train_target, predict_term)
        # lstm超参数寻优
        if model_name == 'LSTM':
            best_parameter = lstm_parameter_randomsearch(train_feature, train_target, predict_term)
        # svm超参数寻优
        if model_name == 'SVM':
            best_parameter = svm_parameter_gridsearch(train_feature, train_target, predict_term)
        # XGBoost超参数寻优
        if model_name == 'XGBoost':
            best_parameter = xgb_parameter_randomsearch(train_feature, train_target, predict_term)
    else:
        logs.warning(model_name + '模型名称不在模型寻优列表范围内！')

    return best_parameter


def svm_parameter_gridsearch(feature, target, predict_term):
    """
    基于sklearn的svm的模型参数网格寻优
    :param feature: 训练集/目标集NWP特征量。参数类型为array。
    :param target: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return: 调用基于sklearn的网格寻优方法输出模型最优参数域
    """
    # 设置参数域
    if predict_term == 'ultra_short':
        param_grid = {'C': [1.0], 'gamma': [0.00001, 0.00005, 0.0001, 0.0005, 0.001]}
    else:
        param_grid = [{'C': [1.0], 'gamma': [0.00001, 0.0005, 0.001]},
                      {'C': [0.5], 'gamma': [0.00005]},
                      {'C': [0.7], 'gamma': [0.0001]}]
    # 重新建立模型
    sklearn_model_SVM = svm.SVR(kernel="rbf")

    return parameter_gridsearch_for_sklearn(optimization_model=sklearn_model_SVM,
                                            feature=feature,
                                            target=target, param_grid=param_grid)


def blstm_parameter_gridsearch(train_feature=None, train_target=None, predict_term=None):
    """
    blstm模型超参数网格寻优
    :param train_feature: 训练集/目标集NWP特征量。参数类型为array。
    :param train_target: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return: 输出模型最优参数域（字典）
    """
    # 设置参数域
    param_grid = {
        "blstm_alpha": [0.3, 0.35, 0.4],
        "blstm_units": [50],
        "epochs": [30]
    }

    # 读取数据
    feature, target = segmentation_of_data(train_feature, train_target, predict_term)
    # 初始化模型
    model = BLSTM(input_size=feature.shape[2])
    # 将keras的神经网络转成sklearn的网络
    sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=model.build_blstmmodel)

    return parameter_girdsearch(model=sklearn_model,
                                  feature=feature.reshape(feature.shape[0] * feature.shape[1], 1, feature.shape[2]),
                                  target=target.reshape(target.shape[0] * target.shape[1], 1), param_grid=param_grid)


def gru_parameter_gridsearch(train_feature=None, train_target=None, predict_term=None):
    """
    gru的模型超参数随机寻优
    :param train_feature: 训练集/目标集NWP特征量。参数类型为array。
    :param train_target: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return: grid_search.best_params_，返回最优参数的集合（字典）
    """
    # 初始化模型
    model = GRU()
    # 设置参数域
    param_grid = {
        "gru_alpha": [0.3, 0.35, 0.4],
    }
    # 切分数据
    feature, target = segmentation_of_data(train_feature, train_target, predict_term)
    # 将keras的神经网络转成sklearn的网络
    sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=model.build_grumodel)

    return parameter_girdsearch(model=sklearn_model,
                                feature=feature.reshape(feature.shape[0] * feature.shape[1], 1, feature.shape[2]),
                                target=target.reshape(target.shape[0] * target.shape[1], 1), param_grid=param_grid)


def bnn_parameter_selfsearch(train_feature=None, train_target=None, predict_term=None):
    """
    基于通用版网格寻优方法的bnn的模型参数网格寻优
    :param train_feature: 训练集/目标集NWP特征量。参数类型为array。
    :param train_target: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return: 调用通用版网格寻优方法输出模型最优参数集，类型为字典
    """
    # 设置参数域
    param_grid = {
        "hidden_units": [2, 4, 8, 16, 32, 64, 128],
        "prior_var": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    }
    # 切分样本集
    feature, target = segmentation_of_data(train_feature, train_target, predict_term)

    return parameter_common_gridsearch(optimization_model=BNN,
                                       feature=torch.tensor(
                                           feature.reshape(feature.shape[0] * feature.shape[1], 1, feature.shape[2])),
                                       target=torch.tensor(target.reshape(target.shape[0] * target.shape[1], 1)),
                                       param_grid=param_grid,
                                       input_size=feature.shape[2]
                                       )


def bpnn_parameter_Randomsearch(feature_data=None, target_data=None, predict_term='short'):
    """
    基于自定义的bpnn的模型参数网格寻优
    :param feature_data: 训练集/目标集NWP特征量。参数类型为array。
    :param target_data: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    """
    if predict_term == 'short':
        # 切分训练集
        n = int(feature_data.shape[0] / 4)
        train_feature_data = feature_data[:3 * n, :]
        test_feature_data = feature_data[3 * n:, :]
        train_target_data = target_data[:3 * n]
        test_target_data = target_data[3 * n:]

        B_hls = 100
        B_lri = 0.01
        model = MLPRegressor(hidden_layer_sizes=(B_hls,), activation="relu",
                             learning_rate_init=B_lri, max_iter=300,
                             random_state=21)
        model.fit(train_feature_data, train_target_data)
        result = model.predict(test_feature_data)
        delt_power_B = (numpy.square(result - test_target_data)).mean(axis=0)
        # 设置想要优化的超参数以及他们的取值分布
        hidden_layer_sizes_list = [70, 80, 90, 100]
        learning_rate_init_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
        numer_times = 0
        for hidden_layer_sizes in hidden_layer_sizes_list:
            for learning_rate_init in learning_rate_init_list:
                numer_times = numer_times + 1
                model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes,), activation="relu",
                                     learning_rate_init=learning_rate_init, max_iter=300,
                                     random_state=21)
                model.fit(train_feature_data, train_target_data)
                result = model.predict(test_feature_data)
                delt_power = (numpy.square(result - test_target_data)).mean(axis=0)
                if delt_power < delt_power_B:
                    B_hls = hidden_layer_sizes
                    B_lri = learning_rate_init
                    delt_power_B = delt_power
        return {'hidden_layer_sizes': B_hls, 'learning_rate_init': B_lri}
    # 使用带历史功率寻优
    else:
        # 切分训练集
        n = int(feature_data.shape[0] / 4)
        train_feature_data = feature_data[:3 * n, :]
        test_feature_data = feature_data[3 * n:, :]
        train_target_data = target_data[:3 * n]
        test_target_data = target_data[3 * n:]

        B_hls = 100
        B_lri = 0.01
        model = MLPRegressor(hidden_layer_sizes=(B_hls,), activation="relu",
                             learning_rate_init=B_lri, max_iter=300,
                             random_state=21)
        model.fit(numpy.hstack((train_feature_data[1:, :], train_target_data[:-1].reshape(1, -1).T)),
                  train_target_data[1:].reshape(1, -1).T)
        result = numpy.zeros((len(test_feature_data)))

        for i in range(int(len(test_feature_data)/16)):
            history_power = test_target_data[16 * i - 1]
            for j in range(16):
                result[16 * i + j] = model.predict(
                    numpy.hstack((test_feature_data[16 * i + j, :], history_power)).reshape(1, -1))
                history_power = 1 * result[i]
        delt_power_B = (numpy.square(result - test_target_data)).mean(axis=0)
        # 设置想要优化的超参数以及他们的取值分布
        hidden_layer_sizes_list = [85, 90, 95]
        learning_rate_init_list = [0.07, 0.09, 0.1]
        numer_times = 0
        for hidden_layer_sizes in hidden_layer_sizes_list:
            for learning_rate_init in learning_rate_init_list:
                numer_times = numer_times + 1
                model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes,), activation="relu",
                                     learning_rate_init=learning_rate_init, max_iter=300,
                                     random_state=21)
                model.fit(numpy.hstack((train_feature_data[1:, :], train_target_data[:-1].reshape(1, -1).T)),
                          train_target_data[1:].reshape(1, -1).T)
                result = numpy.zeros((len(test_feature_data)))

                for i in range(int(len(test_feature_data) / 16)):
                    history_power = test_target_data[16 * i - 1]
                    for j in range(16):
                        result[16 * i + j] = model.predict(
                            numpy.hstack((test_feature_data[16 * i + j, :], history_power)).reshape(1, -1))
                        history_power = 1 * result[i]
                delt_power = (numpy.square(result - test_target_data)).mean(axis=0)

                if delt_power < delt_power_B:
                    B_hls = hidden_layer_sizes
                    B_lri = learning_rate_init
                    delt_power_B = delt_power
        return {'hidden_layer_sizes': B_hls, 'learning_rate_init': B_lri}


def mffs_parameter_gridsearch(feature_data, target_data, predict_term):
    """
    基于自定义的mffs的模型参数网格寻优
    :param feature_data: 训练集/目标集NWP特征量。参数类型为array。
    :param target_data: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return: 调用自定义网格寻优方法输出模型最优参数域
    """
    if predict_term == 'short':
        # 归一化，并切分训练集
        ma = numpy.max(feature_data, axis=0)
        mi = numpy.min(feature_data, axis=0)
        input_train = feature_data.copy()
        n = len(feature_data)
        for i in range(n):
            input_train[i, :] = (input_train[i, :] - mi) / (ma - mi)
        n = int(feature_data.shape[0] / 4)
        train_feature_data = input_train[:3 * n, :]
        test_feature_data = input_train[3 * n:, :]
        train_target_data = target_data[:3 * n]
        test_target_data = target_data[3 * n:]
        # 切分训练集，获取训练集的特征输入feature_input_fit和历史功率power_input_fit
        B_gam = 0.1
        B_aph = 2.7
        delt_power_B = 1
        # 设置想要优化的超参数以及他们的取值分布
        gam_list = [0.05, 0.1, 0.15]
        aph_list = [2.7, 3]
        numer_times = 0
        for gam in gam_list:
            for aph in aph_list:
                numer_times = numer_times + 1

                m = len(test_feature_data)
                n = len(train_feature_data)
                result = numpy.zeros(m)
                for i in range(m):
                    a = numpy.ones((n, 1))
                    t = test_feature_data[i, :] * a
                    d = (numpy.square(train_feature_data - t)).mean(axis=1) ** 0.5
                    threshold = d.min() + gam * (d.max() - d.min())
                    p1 = numpy.zeros(0)
                    for j in range(n):
                        if d[j] < threshold:
                            p1 = numpy.append(p1, numpy.array([aph ** (-d[j]), train_target_data[j]]), axis=0)
                    p1 = p1.reshape(int(p1.size / 2), 2)
                    result[i] = (p1[:, 0] * p1[:, 1]).sum(axis=0) / p1[:, 0].sum(axis=0)
                delt_power = (numpy.square(result - test_target_data)).mean(axis=0) ** 0.5
                if delt_power > delt_power_B:
                    B_gam = gam
                    B_aph = aph
                    delt_power_B = delt_power
        return {'gam': B_gam, 'aph': B_aph}
    else:
        # 归一化，并切分训练集
        ma = numpy.max(feature_data, axis=0)
        mi = numpy.min(feature_data, axis=0)
        input_train = feature_data.copy()
        n = len(feature_data)
        for i in range(n):
            input_train[i, :] = (input_train[i, :] - mi) / (ma - mi)

        dpower = numpy.max(target_data, axis=0)
        target_data = target_data / dpower

        n = int(feature_data.shape[0] / 4)
        train_feature_data = input_train[:3 * n, :]
        test_feature_data = input_train[3 * n:, :]
        train_target_data = target_data[:3 * n]
        test_target_data = target_data[3 * n:]
        train_feature_data = numpy.hstack((train_feature_data[1:, :], train_target_data[:-1].reshape(-1, 1)))

        # 切分训练集，获取训练集的特征输入feature_input_fit和历史功率power_input_fit
        B_gam = 0.1
        B_aph = 2.7
        m = len(test_feature_data)
        n = len(train_feature_data)
        a = numpy.ones((n, 1))
        result = numpy.zeros(m)
        for i in range(int(m/16)):
            history_power = test_target_data[16 * i - 1]
            for k in range(16):
                t = numpy.hstack((test_feature_data[16*i+k, :], history_power)).reshape(1, -1) * a
                d = ((train_feature_data - t) ** 2).mean(axis=1) ** 0.5
                threshold = d.min() + B_gam * (d.max() - d.min())
                p1 = numpy.zeros(0)
                for j in range(n):
                    if d[j] < threshold:
                        p1 = numpy.append(p1, numpy.array([B_aph ** (-d[j]), train_target_data[j]]), axis=0)
                p1 = p1.reshape(int(p1.size / 2), 2)
                result[16*i+k] = (p1[:, 0] * p1[:, 1]).sum(axis=0) / p1[:, 0].sum(axis=0)
                history_power = 1 * result[i]
        result = result * dpower
        delt_power_B = (numpy.square(result - test_target_data)).mean(axis=0) ** 0.5

        # 设置想要优化的超参数以及他们的取值分布
        gam_list = [0.05, 0.1, 0.15]
        aph_list = [2.7, 3]
        numer_times = 0
        for gam in gam_list:
            for aph in aph_list:
                numer_times = numer_times + 1

                for i in range(int(m / 16)):
                    history_power = test_target_data[16 * i - 1]
                    for k in range(16):
                        t = numpy.hstack((test_feature_data[16 * i + k, :], history_power)).reshape(1, -1) * a
                        d = ((train_feature_data - t) ** 2).mean(axis=1) ** 0.5
                        threshold = d.min() + B_gam * (d.max() - d.min())
                        p1 = numpy.zeros(0)
                        for j in range(n):
                            if d[j] < threshold:
                                p1 = numpy.append(p1, numpy.array([B_aph ** (-d[j]), train_target_data[j]]), axis=0)
                        p1 = p1.reshape(int(p1.size / 2), 2)
                        result[16 * i + k] = (p1[:, 0] * p1[:, 1]).sum(axis=0) / p1[:, 0].sum(axis=0)
                        history_power = 1 * result[i]
                result = result * dpower
                delt_power = (numpy.square(result - test_target_data)).mean(axis=0) ** 0.5
                if delt_power < delt_power_B:
                    B_gam = gam
                    B_aph = aph
                    delt_power_B = delt_power
        return {'gam': B_gam, 'aph': B_aph}


def gbrt_parameter_randomsearch_for_sklearn(feature_data, target_data, predict_term):
    """
    gbrt的模型超参数随机寻优
    :param feature_data: 训练集/目标集NWP特征量。参数类型为array。
    :param target_data: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    """
    if predict_term == 'short':
        # 初始化模型
        optimizer_model = ensemble.GradientBoostingRegressor()
        # 设置参数域

        param_dist = {
            'n_estimators': range(100, 250, 50),  # 要执行的推进阶段的数量
            'max_depth': [3, 5, 10, 20],  # 单个回归估计量的最大深度
            'learning_rate': [0.01, 0.1],  # 学习率
            'min_samples_split': [2, 5]  # 分割一个内部节点所需的最小样本数
        }

        return parameter_gridsearch_for_sklearn(
            optimization_model=optimizer_model, feature=feature_data, target=target_data,
            param_grid=param_dist)
        # 超短期带历史功率寻优
    else:
        # 切分训练集
        n = int(feature_data.shape[0] / 4)
        train_feature_data = feature_data[:3 * n, :]
        test_feature_data = feature_data[3 * n:, :]
        train_target_data = target_data[:3 * n]
        test_target_data = target_data[3 * n:]
        best_params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 5, 'learning_rate': 0.01}  # 默认参数
        model = ensemble.GradientBoostingRegressor(**best_params)
        model.fit(numpy.hstack((train_feature_data[1:, :], train_target_data[:-1].reshape(1, -1).T)),
                  train_target_data[1:].reshape(1, -1).T)
        result = numpy.zeros((len(test_feature_data)))
        for i in range(int(len(test_feature_data) / 16)):
            history_power = test_target_data[16 * i - 1]
            for j in range(16):
                result[16 * i + j] = model.predict(
                    numpy.hstack((test_feature_data[16 * i + j, :], history_power)).reshape(1, -1))
                history_power = 1 * result[i]
        delt_power_B = (numpy.square(result - test_target_data)).mean(axis=0)
        # 设置想要优化的超参数以及他们的取值分布

        n_estimators_list = range(100, 250, 50)
        max_depth_list = [3, 5, 10, 20]
        learning_rate_list = [0.01, 0.1]
        min_samples_split_list = [2, 5]

        numer_times = 0
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for learning_rate in learning_rate_list:
                    for min_samples_split in min_samples_split_list:
                        numer_times = numer_times + 1
                        # ----------------------------------------------------------------------------------------------
                        params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                                       'min_samples_split': min_samples_split, 'learning_rate': learning_rate}
                        model = ensemble.GradientBoostingRegressor(**params)
                        model.fit(numpy.hstack((train_feature_data[1:, :], train_target_data[:-1].reshape(1, -1).T)),
                                  train_target_data[1:].reshape(1, -1).T)
                        result = numpy.zeros((len(test_feature_data)))
                        for i in range(int(len(test_feature_data) / 16)):
                            history_power = test_target_data[16 * i - 1]
                            for j in range(16):
                                result[16 * i + j] = model.predict(
                                    numpy.hstack((test_feature_data[16 * i + j, :], history_power)).reshape(1, -1))
                                history_power = 1 * result[i]
                        # ------------------------------------------------------------------------------------------------------
                        delt_power = (numpy.square(result - test_target_data)).mean(axis=0)
                        if delt_power < delt_power_B:
                            best_params = params
                            delt_power_B = delt_power
        return best_params


def decision_tree_parameter_randomsearch_for_sklearn(feature_data, target_data, predict_term):
    """
    决策树的模型超参数随机寻优
    :param feature_data: 训练集/目标集NWP特征量。参数类型为array。
    :param target_data: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    """
    # 短期不带历史功率寻优
    if predict_term == 'short':
        # 初始化模型
        optimizer_model = BaggingRegressor()
        # 设置参数域
        param_dist = {
            'n_estimators': range(100, 1000, 50),  # 要执行的推进阶段的数量
        }
        return parameter_gridsearch_for_sklearn(
            optimization_model=optimizer_model, feature=feature_data, target=target_data,
            param_grid=param_dist)
    # 超短期带历史功率寻优
    else:
        # 切分训练集
        n = int(feature_data.shape[0] / 4)
        train_feature_data = feature_data[:3 * n, :]
        test_feature_data = feature_data[3 * n:, :]
        train_target_data = target_data[:3 * n]
        test_target_data = target_data[3 * n:]
        best_params = {'n_estimators': 500}  # 默认参数
        model = BaggingRegressor()
        model.fit(numpy.hstack((train_feature_data[1:, :], train_target_data[:-1].reshape(1, -1).T)),
                  train_target_data[1:].reshape(1, -1).T)
        result = numpy.zeros((len(test_feature_data)))

        for i in range(int(len(test_feature_data) / 16)):
            history_power = test_target_data[16 * i - 1]
            for j in range(16):
                result[16 * i + j] = model.predict(
                    numpy.hstack((test_feature_data[16 * i + j, :], history_power)).reshape(1, -1))
                history_power = 1 * result[16 * i + j]
        delt_power_B = (numpy.square(result - test_target_data)).mean(axis=0)
        # 设置想要优化的超参数以及他们的取值分布
        the_parameters_list = range(100, 1000, 50)
        numer_times = 0
        for the_parameters in the_parameters_list:
            numer_times = numer_times + 1
            params = {'n_estimators': the_parameters}
            model = BaggingRegressor(**params)
            model.fit(numpy.hstack((train_feature_data[1:, :], train_target_data[:-1].reshape(1, -1).T)),
                      train_target_data[1:].reshape(1, -1).T)
            result = numpy.zeros((len(test_feature_data)))

            for i in range(int(len(test_feature_data)/16)):
                history_power = test_target_data[16*i-1]
                for j in range(16):
                    result[16*i+j] = model.predict(numpy.hstack((test_feature_data[16*i+j, :],
                                                                 history_power)).reshape(1, -1))
                    history_power = 1 * result[16*i+j]
            delt_power = (numpy.square(result - test_target_data)).mean(axis=0)
            if delt_power < delt_power_B:
                best_params = params
                delt_power_B = delt_power
        return best_params


def lstm_parameter_randomsearch(feature=None, target=None, predict_term = "short"):
    """
    lstm模型超参数随机寻优
    :param feature: 训练集/目标集NWP特征量。参数类型为array。
    :param target: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return: 输出模型最优参数域（字典）
    """
    # 寻优周期，短期或超短期
    if predict_term == "short":
        n = int(feature.shape[0] / 4)
        train_feature_data = feature[:3 * n, :]
        test_feature_data = feature[3 * n:, :]
        train_target_data = target[:3 * n]
        test_target_data = target[3 * n:]
        best_params = {'optimizer': 'adam', 'activation': 'relu'}
        model = LSTM(n_features=len(feature[1]))
        model.fit(train_feature_data, train_target_data, predict_term=predict_term, epochs=20, batch_size=1440,
                  without_history_power=None)
        result = model.predict(test_feature_data)
        delt_power_B = (numpy.square(result - test_target_data)).mean(axis=0)
        # 设置想要优化的超参数以及他们的取值分布
        optimizer_list = ['adam']
        activation_list = ['relu', 'hard_sigmoid']
        numer_times = 0
        for optimizer in optimizer_list:
            for activation in activation_list:
                numer_times = numer_times + 1
                params = {'optimizer': optimizer, 'activation': activation}
                model = LSTM(n_features=len(feature[1]), params=params)
                model.fit(train_feature_data, train_target_data, predict_term=predict_term, epochs=20, batch_size=1440,
                          without_history_power=None)
                result = model.predict(test_feature_data)
                delt_power = (numpy.square(result - test_target_data)).mean(axis=0)
                if delt_power < delt_power_B:
                    best_params = params
                    delt_power_B = delt_power
        return best_params
    else:
        n = int(feature.shape[0] / 4)
        train_feature_data = feature[:3 * n, :]
        test_feature_data = feature[3 * n:, :]
        train_target_data = target[:3 * n]
        test_target_data = target[3 * n:]
        best_params = {'optimizer': 'adam', 'activation': 'relu'}
        model = LSTM(n_features=len(feature[1]))
        model.fit(train_feature_data, train_target_data, predict_term=predict_term, epochs=20, batch_size=1440,
                  without_history_power=None)
        result = model.predict(test_feature_data)
        delt_power_B = (numpy.square(result - test_target_data)).mean(axis=0)
        # 设置想要优化的超参数以及他们的取值分布
        optimizer_list = ['adam']
        activation_list = ['relu', 'hard_sigmoid']
        numer_times = 0
        for optimizer in optimizer_list:
            for activation in activation_list:
                numer_times = numer_times + 1
                params = {'optimizer': optimizer, 'activation': activation}
                model = LSTM(n_features=len(feature[1]), params=params)
                model.fit(train_feature_data, train_target_data, predict_term=predict_term, epochs=20, batch_size=1440,
                          without_history_power=None)
                result = model.predict(test_feature_data)
                delt_power = (numpy.square(result - test_target_data)).mean(axis=0)
                if delt_power < delt_power_B:
                    best_params = params
                    delt_power_B = delt_power
        return best_params


def rf_parameter_randomsearch(train_feature, train_target, predict_term):
    """
    rf的模型超参数网格寻优
    :param train_feature: 训练集/目标集NWP特征量。参数类型为array。
    :param train_target: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return:grid_search.best_params_，返回最优参数的集合（字典）
    """
    # 初始化模型
    model = RandomForest()

    # 设置参数域
    if predict_term == 'ultra_short':
        grid = [{'n_estimators': [50, 100],
                 'max_features': ['sqrt'],
                 'max_depth': [20],
                 'min_samples_split': [10],
                 'min_samples_leaf': [4],
                 'bootstrap': [True]},
                {'n_estimators': [150],
                 'max_features': ['sqrt'],
                 'max_depth': [5],
                 'min_samples_split': [10],
                 'min_samples_leaf': [4],
                 'bootstrap': [True]},
                {'n_estimators': [200],
                 'max_features': ['sqrt'],
                 'max_depth': [10],
                 'min_samples_split': [10],
                 'min_samples_leaf': [4],
                 'bootstrap': [True]}
                ]
    else:
        grid = [{'n_estimators': [50, 150],
                 'max_features': ['sqrt'],
                 'max_depth': [10],
                 'min_samples_split': [10],
                 'min_samples_leaf': [4],
                 'bootstrap': [True]},
                {'n_estimators': [100],
                 'max_features': ['sqrt'],
                 'max_depth': [5],
                 'min_samples_split': [10],
                 'min_samples_leaf': [4],
                 'bootstrap': [True]},
                {'n_estimators': [200],
                 'max_features': ['sqrt'],
                 'max_depth': [20],
                 'min_samples_split': [10],
                 'min_samples_leaf': [4],
                 'bootstrap': [True]}]

    return parameter_gridsearch_for_sklearn(model.rf_model, train_feature, train_target, grid)


def xgb_parameter_randomsearch(train_feature, train_target, predict_term):
    """
    xgboost的模型超参数网格寻优
    :param train_feature: 训练集/目标集NWP特征量。参数类型为array。
    :param train_target: 加载训练集/目标集功率数据，参数类型为array。
    :param predict_term: 预测周期，分为短期’short‘和超短期‘ultra_short‘
    :return:grid_search.best_params_，返回最优参数的集合（字典）
    """
    # 初始化模型
    model = XGBoost()

    # 设置参数域
    if predict_term == 'ultra_short':
        grid = {
            'n_estimators': [20, 25, 30, 50],
            'learning_rate': [0.1, 0.13, 0.15],
            'reg_lambda': [1, 10],
            'gamma': [0, 100],
            'colsample_bynode': [0.7, 0.8, 0.9, 1],
            'colsample_bytree': [0.7, 0.8, 0.9, 1],
        }
    else:
        grid = {
            'n_estimators': [20, 25, 30, 50],
            'learning_rate': [0.1, 0.13, 0.15],
            'reg_lambda': [1, 10],
            'gamma': [0, 100],
            'colsample_bynode': [0.7, 0.8, 0.9, 1],
            'colsample_bytree': [0.7, 0.8, 0.9, 1],
        }

    return parameter_gridsearch_for_sklearn(model.xgb_model, train_feature, train_target, grid)
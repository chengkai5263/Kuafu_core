
import itertools
import numpy
import torch.optim as optim
from numpy import array
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow import keras


def segmentation_of_data(feature, target, predict_term, *args, **kwargs):
    """
    切分样本集，添加历史功率，非必要，有需要的可以使用。（暂时在blstm、gru、bnn中使用）
    :param feature: 特征集（即天气预报数据）或添加历史功率的特征集
    :param target: 目标集（即功率数据）
    :param predict_term: 预测周期，超短期或短期。当预测周期predict_term为超短期ultra_short时，
                       模型单次预测结果输出长度为predict_step = 16，预测间隔predict_inter = 1；
                       当预测周期predict_term为短期short时，
                       模型单次预测结果输出长度为predict_step = 288，预测间隔predict_inter = 96；
    :param args: 个数可变的参数集，类型为元组。
                 实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
    :param kwargs: 个数可变的参数集，类型为字典。
                   实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                   （for key, value in kwargs.items()）的方式来获取。
    :return: 切分后的feature，target
    """
    # 判断预测周期
    if predict_term == "ultra_short":
        predict_step = 16
        predict_inter = 1
    elif predict_term == "short":
        predict_step = 288
        predict_inter = 96
    else:
        print("预测周期输入错误")
        predict_step = 16
        predict_inter = 1

    # 切分并添加历史功率
    # 特征量输入和目标输出初始化为list
    input_data, output_data = list(), list()
    in_start = 0
    in_start = in_start + predict_inter
    in_end = in_start + predict_step
    while in_end in range(len(feature)):
        # 每一组都添加历史功率
        Power_the_step = numpy.c_[feature[in_start:in_end, :],
                                  target[in_start - 1] * numpy.ones(predict_step)]
        input_data.append(Power_the_step)
        output_data.append(target[in_start:in_end])
        in_start = in_start + predict_inter
        in_end = in_start + predict_step
    # 转换数据格式为array类型
    feature = array(input_data)
    target = array(output_data)
    return feature, target


def parameter_gridsearch_for_sklearn(optimization_model, feature, target, param_grid, *args, **kwargs):
    """
    模型参数网格寻优,适合rf、svm等sklearn库的模型
    :param optimization_model: 待寻优模型
    :param feature: 特征集（即天气预报数据）或添加历史功率的特征集
    :param target: 目标集（即功率数据）
    :param param_grid: 模型超参数域。
    :param args: 个数可变的参数集，类型为元组。
                 实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
    :param kwargs: 个数可变的参数集，类型为字典。
                   实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                   （for key, value in kwargs.items()）的方式来获取。
    :return:
    """
    # 生成网格寻优策略
    grid_search = GridSearchCV(optimization_model, param_grid, cv=2, scoring=make_scorer(r2_score,
                                                                                         greater_is_better=True),
                               refit=False)
    # 超参数网格寻优
    grid_search.fit(feature, target)
    return grid_search.best_params_


def parameter_common_gridsearch(optimization_model, param_grid, feature, target, input_size, *args, **kwargs):
    """
    模型参数网格寻优，适合除sklearn库之外的算法模型（此函数仅针对bnn）
    :param optimization_model: 参与参数寻优的模型，若要对不同模型寻优，只需要修改该模型即可。
    :param param_grid: 设定的超参数范围集，若要对不同模型的不同参数寻优，只需要修改该超参数范围集即可。这里已经对
    :param feature: 特征集（即天气预报数据）或添加历史功率的特征集
    :param target: 目标集（即功率数据）
    :param input_size: 输入的特征量个数
    :param args: 个数可变的参数集，类型为元组。
                 实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
    :param kwargs: 个数可变的参数集，类型为字典。
                   实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                   （for key, value in kwargs.items()）的方式来获取。
    :return:返回最优参数的集合（字典）
    """
    BNN_loss = []
    # 针对超参数域的排列组合
    param_grid_combination = list(itertools.product(*param_grid.values()))
    for i in range(len(param_grid_combination)):
        # 重新建立模型
        new_BNN = optimization_model(input_size=input_size, params={'hidden_units': param_grid_combination[i][0],
                                     'BNN_prior_var': param_grid_combination[i][1]})
        # 模型适配
        optimizer = optim.Adam(new_BNN.get_model().parameters(), lr=.1)
        epochs = 1
        new_BNN.get_model().double()
        for epoch in range(epochs):  # 在数据集上循环多次
            optimizer.zero_grad()
            # forward + backward + optimize
            loss = new_BNN.get_model().sample_elbo(feature, target, 1)
            loss.backward()
            optimizer.step()
        # 获取均方根误差
        BNN_loss.append(loss.item())
    # 找到最小的均方根误差对应的参数
    bestparameters = {
        "hidden_units": param_grid_combination[BNN_loss.index(min(BNN_loss))][0],
        "BNN_prior_var": param_grid_combination[BNN_loss.index(min(BNN_loss))][1]
    }

    return bestparameters


def parameter_randomsearch(model, feature, target, param_grid, *args, **kwargs):
    """
    模型参数随机寻优，适合lstm、blstm、gru等keras库的神经网络模型
    :param model: 参与参数寻优的模型。
    :param feature: 特征集（即天气预报数据）或添加历史功率的特征集
    :param target: 目标集（即功率数据）
    :param param_grid: 设定的超参数范围集。
    :param args: 个数可变的参数集，类型为元组。
                 实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
    :param kwargs: 个数可变的参数集，类型为字典。
                   实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                   （for key, value in kwargs.items()）的方式来获取。
    :return:bestparameters，返回最优参数的集合（字典）
    """
    bestparameters = []
    # 将神经网络模型转为sklearn模型
    callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
    # 生成随机寻优策略
    random_search_cv = RandomizedSearchCV(model,
                                          param_grid,
                                          n_iter=1,
                                          cv=3,
                                          n_jobs=1)
    # 添加神经网络的epochs开始随机寻优
    i = 0
    while i in range(len(param_grid['epochs'])):
        random_search_cv.fit(feature, target, epochs=param_grid['epochs'][i], callbacks=callbacks)
        random_search_cv.best_params_['epochs'] = param_grid['epochs']
        bestparameters.append(random_search_cv.best_params_)
        i = i + 1
    return bestparameters


def parameter_girdsearch(model, feature, target, param_grid, *args, **kwargs):
    """
    模型网格寻优，适合lstm、blstm、gru等keras库的神经网络模型
    :param model: 参与参数寻优的模型。
    :param feature: 特征集（即天气预报数据）或添加历史功率的特征集
    :param target: 目标集（即功率数据）
    :param param_grid: 设定的超参数范围集。
    :param args: 个数可变的参数集，类型为元组。
                 实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
    :param kwargs: 个数可变的参数集，类型为字典。
                   实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                   （for key, value in kwargs.items()）的方式来获取。
    :return:bestparameters，返回最优参数的集合（字典）
    """
    # 将神经网络模型转为sklearn模型
    callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
    # 生成随机寻优策略
    # 生成随机寻优策略
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=make_scorer(r2_score, greater_is_better=True),
                               refit=False)
    grid_search.fit(feature, target, callbacks=callbacks)
    return grid_search.best_params_

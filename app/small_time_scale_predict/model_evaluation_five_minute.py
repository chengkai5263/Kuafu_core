# _*_ coding: utf-8 _*_

from math import sqrt

import numpy
import numpy as np
from sklearn.metrics import mean_squared_error


def result_evaluation_five_minute_4h(predict_result, actual_result, online_capacity, predict_term="five_minute_ultra_short"):
    """
    评估预测精度-基于Q/CSG1211017-2018标准，计算单次预测的288个点；
    :param predict_result: 一段时间内每次预测结果集
    :param actual_result: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测类型，超短期"ultra_short"或短期"short"
    :return np.mean(evaluation_set): 日平均准确率为当日内全部超短期预测准确率的算术平均值
    """
    if predict_term == "ultra_short":
        interval = 16
    elif predict_term == "short":
        interval = 96
    else:
        interval = 48
    evaluation_set = []
    for j in range(len(actual_result)):
        s = 0
        for k in range(interval):
            y_true = actual_result[j][k]
            y_pred = predict_result[j][k]
            s += ((y_true - y_pred) / online_capacity) ** 2
        score = (1 - sqrt(s / interval)) * 100
        evaluation_set.append(score)
    return np.mean(evaluation_set)


def result_evaluation_five_minute(predict_result, actual_result, online_capacity, predict_term="five_minute_ultra_short"):
    """
    评估预测精度-基于Q/CSG1211017-2018标准，计算单次预测的第4个小时，即最后12个点
    :param predict_result: 一段时间内每次预测结果集
    :param actual_result: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测类型，超短期"ultra_short"或短期"short"
    :return np.mean(evaluation_set): 日平均准确率为当日内全部超短期预测准确率的算术平均值
    """
    if predict_term == "ultra_short":
        evaluation_term = 4
    elif predict_term == "short":
        evaluation_term = 96
    else:
        evaluation_term = 12

    evaluation_set = []
    for i in range(0, len(predict_result), 1):
        rmse = sqrt(mean_squared_error(
            numpy.array(actual_result[i][-evaluation_term:]), numpy.array(predict_result[i][-evaluation_term:]))
        )
        score = (1 - rmse / online_capacity) * 100
        evaluation_set.append(score)

    return np.mean(evaluation_set)
# _*_ coding: utf-8 _*_

import pandas
import numpy as np
import statsmodels.api as sm
import math
from scipy.stats import norm
from common.tools import save_model, load_model


class Scenario:
    """
    Probabilistic_Forecast模型，根据预测发电曲线生成区间预测曲线
    """
    def generate_forecast_bin(self, predict_data, actual_data, save_path, online_capacity, bin_num):
        """
        根据点预测结果及其相应的功率实测值，生成预测箱及对应的累积经验分布函数。
        :param predict_data:
        :param actual_data:
        :param save_path: 模型存储的位置
        :param online_capacity: 开机容量
        :param bin_num: 预测箱数量，数量越大预测准度越高，但是覆盖面积会降低。
        :return:None
        """
        # 生成所需的数据集，并进行清洗，去除Nan数据
        data_generate_bins = pandas.DataFrame(
            {'real': actual_data.tolist(), 'predicted': predict_data.tolist()})
        data_generate_bins = data_generate_bins.dropna(axis=0, how='any')
        pu_power = data_generate_bins/online_capacity

        b0 = 1 / bin_num
        b = 0
        bins = []
        bins_size = []
        for j in range(0, bin_num):
            one_bin = []
            count = 0
            for i in range(0, pu_power.shape[0]):
                if b < pu_power.loc[i, 'predicted'] < b + b0:
                    one_bin.append(pu_power.iloc[i].tolist())
                    count += 1
            if count == 0:
                count = 1
                one_bin.append([0, 0])
            bins_size.append(count)
            b += b0
            bins.append(one_bin)

        ECDF_Fn = []
        ECDF_Xn = []
        for j in range(0, bin_num):
            bink = []
            for i in range(0, bins_size[j]):
                bink.append(bins[j][i][0])
            ecdf = sm.distributions.ECDF(bink)
            bink = list(set(bink))
            xn = sorted(bink)
            fn = ecdf(xn)

            ECDF_Fn.append(fn)
            ECDF_Xn.append(xn)

        df = pandas.DataFrame(ECDF_Fn)
        save_model(df, save_path[:-4]+'ECDF_Fn.pkl')
        df = pandas.DataFrame(ECDF_Xn)
        save_model(df, save_path)
        return

    def generate_scenario(self, point_predict_data, model_path, online_capacity, bin_num, scenario_num):
        """
        根据点预测结果及训练的预测箱模型，生成区间预测的上下界。
        :param point_predict_data:
        :param model_path: 模型存储的位置
        :param online_capacity: 开机容量
        :param bin_num: 预测箱数量，数量越大预测准度越高，但是覆盖面积会降低。
        :param scenario_num:
        :return:None
        """
        # 加载模型文件
        model = load_model(model_path)
        df = np.array(model).tolist()
        ECDF_Xn = []
        for i in range(len(df)):
            cleanedList = [x for x in df[i] if x == x]
            ECDF_Xn.append(cleanedList)

        model = load_model(model_path[:-4]+'ECDF_Fn.pkl')
        df = np.array(model).tolist()
        ECDF_Fn = []
        for i in range(len(df)):
            cleanedList = [x for x in df[i] if x == x]
            ECDF_Fn.append(cleanedList)

        bin_n = np.linspace((1 / (2 * bin_num)), 0.99, bin_num).tolist()
        pu_power = np.array(point_predict_data) / online_capacity

        mu = [0 for i in range(0, pu_power.shape[0])]
        sigma = np.identity(pu_power.shape[0])
        for i in range(pu_power.shape[0]):
            for j in range(pu_power.shape[0]):
                sigma[i][j] = math.exp(-abs(i - j) / 3)
        X = np.random.multivariate_normal(mu, sigma, scenario_num)
        Y = norm.cdf(X)

        fc = []
        for i in range(pu_power.shape[0]):
            fc.append(pu_power[i])

        dataframe = {}

        sample = []
        for i in range(0, pu_power.shape[0]):
            min_location = np.argmin(abs(fc[i] - bin_n))
            y_temp = [Y[k][i] for k in range(scenario_num)]
            sample.append(inverse_ecdf(ECDF_Fn[min_location], ECDF_Xn[min_location], y_temp, scenario_num))

        for i in range(scenario_num):
            out_one = []
            for j in range(pu_power.shape[0]):
                out_one.append(sample[j][i] * online_capacity)

            scenario = np.array(out_one)
            dataframe["scenario" + str(i)] = scenario
        dataframe = pandas.DataFrame(dataframe)

        return dataframe


def model_train(config_cluster, bin_num, predict_data, actual_data, predict_term, model_name):
    """
    训练区间预测模型
    读取区间预测配置文件，对配置文件下所有场站的短期、超短期结果进行区间预测训练
    :param config_cluster: 配置文件
    :param predict_data:
    :param actual_data:
    :param bin_num: 预测箱数量
    :param predict_term:
    :param model_name:
    :return: None
    """
    online_capacity = config_cluster["capacity"]

    model_name, model_state = model_name.split('_', 1)

    model_save_path = "%s%s/%s/%s/" % (config_cluster["model_savepath"], config_cluster["id"], predict_term, model_name)
    model_save_path = model_save_path + predict_term + '_' + model_name + '_interval_predict.pkl'
    # 调用函数进行区间预测训练
    model = Scenario()
    model.generate_forecast_bin(predict_data, actual_data, model_save_path, online_capacity, bin_num=bin_num)

    return


def scenario_generation(point_predict_data, config_cluster, bin_num, predict_term, model_name):
    """
    区间预测
    读取区间预测配置文件，对配置文件下所有场站的短期、超短期结果进行区间预测
    :param point_predict_data:
    :param config_cluster: 配置文件
    :param bin_num: 预测箱数量
    :param predict_term:
    :param model_name:
    :return: None
    """
    # 容量
    online_capacity = config_cluster["capacity"]

    # 读取短期预测数据的CSV文件存放位置
    model_name, model_state = model_name.split('_', 1)

    model_save_path = "%s%s/%s/%s/" % (config_cluster["model_savepath"], config_cluster["id"], predict_term, model_name)
    model_save_path = model_save_path + predict_term + '_' + model_name + '_interval_predict.pkl'

    # 调用函数进行预测
    model = Scenario()
    dataframe = model.generate_scenario(point_predict_data, model_save_path, online_capacity,
                                        bin_num=bin_num, scenario_num=10)
    return dataframe


def inverse_ecdf(f, x, Y, N):
    X = []
    for i in range(N):
        minx = min(abs(f - Y[i]))
        minx_location = np.argmin(abs(f - Y[i]))
        print()
        if minx == 0 or minx_location + 1 > len(f) - 1 or minx_location - 1 < 0:
            X.append(x[minx_location])
        else:
            if f[minx_location] > Y[i] > f[minx_location - 1]:
                L = minx_location - 1
            if f[minx_location] < Y[i] < f[minx_location + 1]:
                L = minx_location + 1
            temp = (x[minx_location] * f[L] - x[L] * f[minx_location] + (x[L] - x[minx_location]) * Y[i]) / (
                    f[L] - f[minx_location])
            X.append(temp)
    return X

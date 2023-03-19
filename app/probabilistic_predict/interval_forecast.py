# _*_ coding: utf-8 _*_

import pandas
import numpy
import statsmodels.api as sm
import os
from common.tools import save_model
from common.tools import load_model
from numpy import where


class IntervalForecast:
    """
    Probabilistic_Forecast模型，根据预测发电曲线生成区间预测曲线
    """
    def generate_forecast_bin(self, predict_power, real_power,
                              predict_term, model_name, station_id, online_capacity, bin_num, model_savepath):
        """
            根据点预测结果及其相应的功率实测值，生成预测箱及对应的累积经验分布函数。
            :param predict_power: 预测功率
            :param real_power: 真实功率
            :param online_capacity: 开机容量
            :param predict_term: 短期/超短期的预测类型
            :param bin_num: 预测箱数量，数量越大预测准度越高，但是覆盖面积会降低。
            :param model_name: 模型名称
            :param station_id: 场站编号
            :param model_savepath: 模型保存位置
            :return:None
        """
        data_generate_bins = pandas.merge(predict_power, real_power, left_on='forecast_time', right_on='time',
                                          how='left')

        data_generate_bins = data_generate_bins.dropna(axis=0, how='any')
        data_generate_bins = data_generate_bins.loc[:, ['predicted', 'real']]
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
                bink.append(bins[j][i][1])
            ecdf = sm.distributions.ECDF(bink)
            bink = list(set(bink))
            xn = sorted(bink)
            fn = ecdf(xn)

            ECDF_Fn.append(fn)
            ECDF_Xn.append(xn)

        df = pandas.DataFrame(ECDF_Xn)
        model_name, model_state = model_name.split('_', 1)
        sub_dir_path = model_savepath + str(station_id) + '/' + predict_term + '/' + model_name + '/'
        os.makedirs(sub_dir_path, exist_ok=True)
        save_model(df, sub_dir_path + predict_term + '_' + model_name + '_interval_predict.pkl')

    def interval_prediction(self, station_id, predict_term, predict_type, point_predict_data, model_path, model_name,
                            online_capacity, bin_num, sr):
        """
            根据点预测结果及训练的预测箱模型，生成区间预测的上下界。
            :param point_predict_data: 点预测结果。
            :param model_path: 模型存储的位置
            :param online_capacity: 开机容量
            :param predict_type: 短期/超短期的预测类型
            :param bin_num: 预测箱数量，数量越大预测准度越高，但是覆盖面积会降低。
            :param station_id: 场站编号
            :param predict_term: 短期/超短期的预测类型
            :param sr: 辐照度所在列
            :param model_name: 模型名称
            :return:dataframe
        """
        # 加载模型文件
        model_name, model_state = model_name.split('_', 1)
        fitted_model = load_model(model_path + str(station_id) + '/' + predict_term + '/' + model_name + '/'
                                  + predict_term + '_' + model_name + '_interval_predict.pkl')

        df = numpy.array(fitted_model).tolist()
        ECDF_Xn = []
        for i in range(len(df)):
            cleanedList = [x for x in df[i] if x == x]
            ECDF_Xn.append(cleanedList)
        bin_n = numpy.linspace((1 / (2 * bin_num)), 0.99, bin_num).tolist()
        pu_power = numpy.array(point_predict_data)/online_capacity

        fc = []
        for i in range(pu_power.shape[0]):
            fc.append(pu_power[i])

        dataframe = {}
        for i in range(9, 4, -1):
            confidence_level = i / 10
            upper = []
            lower = []
            for j in range(pu_power.shape[0]):
                min_location = numpy.argmin(abs(fc[j] - bin_n))
                ECDF_Xn_arr = numpy.array(ECDF_Xn[min_location])
                upper.append(numpy.quantile(ECDF_Xn_arr, 0.5 + confidence_level / 2))
                lower.append(numpy.quantile(ECDF_Xn_arr, 0.5 - confidence_level / 2))

            upper = numpy.array(upper) * online_capacity
            lower = numpy.array(lower) * online_capacity
            if predict_type == 'solar' and sr is not None:
                lower[where(sr < 10)] = 0
            dataframe["upper bound at confidence level of " + str(i*10) + "%"] = upper
            dataframe["lower bound at confidence level of " + str(i*10) + "%"] = lower
        dataframe = pandas.DataFrame(dataframe)
        return dataframe

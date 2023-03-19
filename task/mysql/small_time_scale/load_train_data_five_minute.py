# _*_ coding: utf-8 _*_
import time

# import ksycopg2
import numpy
import pandas
from numpy import where

# from common.load_data_from_csv import load_csv_data
from abc import ABCMeta
import pymysql
from datetime import datetime, timedelta


class LoadFiveMinuteEnsembledata(metaclass=ABCMeta):
    def __init__(self):
        # 训练/测试时间集
        # 数据类型为ndarray，大小为m*1，即m行1列。列的含义是代表时刻。每个元素的内容是形如“2018-01-02 15:45:00”的字符串
        self._train_time_data = None

        # 训练/测试特征集（数据为原始数据，未作标准化处理）
        # 数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64
        self._train_feature_data = None

        # 训练/测试目标集
        # 数据类型为ndarray，大小为m*1，即m行1列。列的含义是代表功率值。每个元素的类型是float64
        self._train_target_data = None

        # 特征集文件数据（即天气预报数据）中，预报时间跨度
        # 为3时表示预报未来3天的天气数据；其他值表示预报未来1天的天气数据
        self.__forecast_time = 1

    def load_train_data(self, file_path, nrows=None, skiprows=None, usecols=None, rate=0.75, forecast_time=1,
                        predict_type="wind"):
        """
        加载训练集及测试集数据。此处加载的数据，均是预处理后的数据
        数据预处理：功率数据归一化，错误数据识别与修正，限电数据识别与修正
        :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
        :param nrows: 加载特征集/目标集文件时所要选取的记录条数，默认全加载
        :param skiprows: 加载特征集/目标集文件时的忽略记录条数（从文件头开始算起，表头也算一条数据），默认不忽略
        :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
                        为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
                        一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
        :param rate: 训练集数据量与（训练集+测试集数据量）的比值
        :param forecast_time: 所加载的特征集文件数据（即天气预报数据）中，预报时间跨度。为3时表示预报未来3天的天气数据；
                              其他值表示预报未来1天的天气数据。默认取值为1
        :param predict_type: 预测类型
        :return: self.__test_feature_data, self.__test_target_data 训练输入数据，训练输出数据
        """

        if forecast_time == 1:
            self.__forecast_time = 1
            self.__load_one_day_forecast_data_from_file(file_path, nrows, skiprows, usecols, rate, predict_type)
        else:
            self.__forecast_time = 3
            self.__load_three_day_forecast_data_from_file(file_path, nrows, skiprows, usecols, rate, predict_type)

        train_feature_data = self._train_feature_data
        train_target_data = self._train_target_data

        return train_feature_data, train_target_data

    def load_train_data_for_short(self, file_path, usecols=None, rate=0.75, forecast_time=1, predict_type="wind",
                                  nrows=None):
        """
        加载训练集及测试集数据。此处加载的数据，均是预处理后的数据
        数据预处理：功率数据归一化，错误数据识别与修正，限电数据识别与修正
        :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
        :param nrows: 加载特征集/目标集文件时所要选取的记录条数，默认全加载
        :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
                        为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
                        一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
        :param rate: 训练集数据量与（训练集+测试集数据量）的比值
        :param forecast_time: 所加载的特征集文件数据（即天气预报数据）中，预报时间跨度。为3时表示预报未来3天的天气数据；
                              其他值表示预报未来1天的天气数据。默认取值为1
        :param predict_type: 预测类型
        :return: self.__test_feature_data, self.__test_target_data 训练输入数据，训练输出数据
        """
        # 加载特征集文件

        data_set = None
        dataframe = None
        for file in file_path["feature"]:
            dataframe = pandas.read_csv(file, header=0, encoding="utf-8", nrows=nrows)
            nwp_data = dataframe.loc[:, usecols].values
            if predict_type == "solar":
                time = dataframe.loc[:, "forecast_time"]
                time_float = numpy.zeros((time.size, 1))
                for i in range(time.size):
                    time_float[i, 0] = float(time[i][-5:-3]) / 24 + float(time[i][-2:]) / 15 / 96
                data_set = numpy.hstack((nwp_data, time_float))
            else:
                data_set = nwp_data

        # 起报时间总天数。预报未来3天的天气预报数据时，起报时间为同一天的记录共有288个点
        start_days = int(len(data_set) / (96 * forecast_time))
        # 用于训练的起报时间天数
        train_start_days = int(start_days * rate)
        # 划分训练/测试特征集数据
        self._train_feature_data = data_set[0:train_start_days * (96 * forecast_time)]
        dataframe_train = dataframe.iloc[0:train_start_days * (96 * forecast_time)]

        data_set = None
        for file in file_path["target"]:
            data_set = pandas.read_csv(file, header=0, encoding="utf-8", nrows=nrows)

        train_data_merge = pandas.merge(data_set, dataframe_train, left_on='PyForecastTime', right_on='forecast_time',
                                        how='right')
        train_data_merge = train_data_merge.dropna(axis=0, how='any')
        self._train_target_data = train_data_merge.values[:, 1].reshape(-1, 1)

        # 将目标集的数据类型转换成float64
        self._train_target_data = self._train_target_data.astype("float64")

        train_feature_data = self._train_feature_data
        train_target_data = self._train_target_data[:, 0]

        return train_feature_data, train_target_data

    @staticmethod
    def load_train_data_for_five_minute_sql(host='localhost', user='root', password='123456', database='kuafu',
                                            charset='utf8', port=3306, file_path=None, usecols=None, rate=1,
                                            predict_type='wind', start_time_str='2021/1/1 00:00', end_time_str='2021/3/1 00:00', sql="mysql"):
        """
        加载测试集数据。加载测试的输入数据
        :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
        :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
                        为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
                        一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
        :param predict_type: 预测类型
        :return test_feature_data: 测试输入数据
        :return history_target_data: 测试历史功率数据
        :return time: 预报的时间
        :param host: 主机
        :param user: 用户名
        :param password: 密码
        :param database: 数据库名
        :param charset: 解码方式
        :param rate: 训练集数据量与（训练集+测试集数据量）的比值
        """
        end_time_datetime = datetime.strptime(end_time_str, '%Y/%m/%d %H:%M')
        start_time_float = datetime.strptime(start_time_str, '%Y/%m/%d %H:%M').timestamp()


        # 读取NWP数据----------------------------------------------------------------------------------------------------
        if sql == "kingbas":
            # 获取参与集成学习的模型名称
            db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
            c = db.cursor()
            c.execute(
                "select distinct start_time, forecast_time," + ','.join(iterm for iterm in usecols) + " from nwp_" +
                str(file_path["id"]) + " where start_time between %s and %s ORDER BY start_time,forecast_time asc;",
                (start_time_float, end_time_datetime))
            record = tuple(c.fetchall())
        else:
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
            # 定义游标
            c = db.cursor()
            c.execute(
                "select start_time, forecast_time," + ','.join(iterm for iterm in usecols) + " from nwp_" +
                str(file_path["id"]) + " where start_time between %s and %s ORDER BY forecast_time desc;",
                (start_time_float, end_time_datetime))
            record = c.fetchall()

        # # NWP数据补齐（时间太长，平常测试可以不运行）--------------------------------------------------------------------------
        # if record[0][1].timestamp() != record[0][0].timestamp():
        #     tem = tuple([tuple([record[0][0]]) + tuple([record[0][0]]) + record[0][2:]])
        #     record = tem + record
        #
        # if (record[-1][1].timestamp() - record[-1][0].timestamp()) < 4 * 24 * 3600:
        #     tem = tuple([tuple([record[-1][0]]) + tuple([record[-1][0] + timedelta(days=4)]) + record[-1][2:]])
        #     record = record + tem
        #
        # for i in range(len(record)-1, 0, -1):
        #     if record[i][0] != record[i-1][0]:
        #         if record[i][1].timestamp() != record[i][0].timestamp():
        #             tem = tuple([tuple([record[i][0]]) + tuple([record[i][0]]) + record[i][2:]])
        #             record = record[:i] + tem + record[i:]
        #
        # for i in range(len(record)-1, 0, -1):
        #     if record[i][0] != record[i-1][0]:
        #         if (record[i-1][1].timestamp() - record[i-1][0].timestamp()) < 4 * 24 * 3600:
        #             tem = tuple([tuple([record[i - 1][0]]) + tuple([record[i - 1][0] + timedelta(days=4)]) +
        #                          record[i - 1][2:]])
        #             record = record[:i] + tem + record[i:]
        #
        # for i in range(len(record)-1, 0, -1):
        #     if record[i][0] == record[i-1][0]:
        #         for j in range(int((record[i][1].timestamp() - record[i-1][1].timestamp()) / 900 - 1)):
        #             t = record[i][1].timestamp() - 900
        #             t_dt = datetime.strptime(time.strftime("%Y-%m-%d %H:%M", time.localtime(t)),
        #                                               '%Y-%m-%d %H:%M')
        #             flag = True
        #             for k in range(18816):
        #                 if record[i-k][1] == t_dt:
        #                     tem = tuple([tuple([record[i][0]]) + record[i-k][1:]])
        #                     record = record[:i] + tem + record[i:]
        #                     flag = False
        #                     break
        #             if flag:
        #                 tem = tuple([tuple([record[i][0]]) + tuple([t_dt]) + record[i][2:]])
        #                 record = record[:i] + tem + record[i:]

        # 超短期预测只取最近NWP数据-----------------------------------------------------------------------------------------
        train_feature_data = numpy.zeros((0, len(usecols) + 1))
        for i in range(len(record)):
            if (record[i][1].timestamp() - record[i][0].timestamp()) < 6 * 3600:
                nwp_row = ','.join(iterm for iterm in record[i][2:])
                nwp = eval(nwp_row)
                nwp_array = numpy.array(nwp)
                predict_time = record[i][1].timestamp()
                nwp_time = numpy.hstack((predict_time, nwp_array)).reshape(1, -1)
                train_feature_data = numpy.vstack((train_feature_data, nwp_time))

        # 读取功率数据----------------------------------------------------------------------------------------------------
        result = c.execute(
            "select time, power from real_power_" + str(file_path["id"]) + " where time between %s and %s ORDER BY time asc;",
            (start_time_float, end_time_datetime))
        record = c.fetchall()
        if sql == "kingbas":
            record = tuple(record)
            result = len(record)
        train_target_data = numpy.zeros((0, 2))
        for i in range(result):
            power = record[i][1]
            start_time = record[i][0].timestamp()
            power_time = numpy.hstack((start_time, power)).reshape(1, -1)
            train_target_data = numpy.vstack((train_target_data, power_time))

        # 判断历史功率分辨率是15分钟还是5分钟
        if train_target_data[1][0] - train_target_data[0][0] == 900:
            train_target_data = None
        else:
            # 补充历史功率缺失-------------------------------------------------------------------------------------------------
            for i in range(len(train_target_data) - 1, 0, -1):
                n = int((train_target_data[i, 0] - train_target_data[i - 1, 0]) / 300)
                d_p = (train_target_data[i, 1] - train_target_data[i-1, 1])
                for j in range(int(n - 1)):
                    t = train_target_data[i, 0] - 300
                    power_succedaneum = train_target_data[i, 1] - d_p/n
                    power_time = numpy.hstack((t, power_succedaneum)).reshape(1, -1)
                    train_target_data = numpy.vstack(
                        (train_target_data[:i, :], power_time, train_target_data[i:, :]))

            # 保证功率是从0：05开始的
            if train_target_data[0, 0] % 900 == 0:
                train_target_data = train_target_data[1:, :]
            if train_target_data[0, 0] % 900 == 600:
                train_target_data = train_target_data[2:, :]

            # 保证功率数据是3的倍数
            n = int(3*(len(train_target_data)//3))
            train_target_data = train_target_data[:n, :]
            power_table = train_target_data[:, 1].reshape(int(n/3), 3)
            time_table = train_target_data[:, 0].reshape(int(n/3), 3)

            train_target_data = pandas.DataFrame(numpy.hstack((time_table[:, -1].reshape(len(time_table), 1), power_table)))
            train_feature_data = pandas.DataFrame(train_feature_data)

            # 将15分钟分辨率的NWP与5分钟的功率数据按照功率数据的时间戳对齐
            train_data = pandas.merge(train_target_data, train_feature_data, left_on=0, right_on=0, how='right')
            train_data = train_data.dropna(axis=0, how='any')

            train_target_data = train_data.values[:, 1:4].reshape(len(train_data)*3, 1)
            train_feature_data = train_data.values[:, 4:]

            if predict_type == 'solar':
                t = ((train_data.values[:, 0] % 86400) / 86400).reshape(-1, 1)
                train_feature_data = numpy.hstack((train_feature_data, t))
            db.close()
        return train_feature_data, train_target_data

    @staticmethod
    def upload_predict_nwp_five_minute_for_cloud_predict(host='localhost', user='root', password='123456',
                                                         database='kuafu', charset='utf8', port=3306,
                                                         start_time=None, file_path=None, usecols=None, rate=1,
                                                         predict_type='wind', start_time_str='2021/1/1 00:00',
                                                         end_time_str='2021/3/1 00:00', sql="mysql", n=7):
        """
        加载测试集数据。加载测试的输入数据
        :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
        :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
                        为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
                        一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
        :param start_time: 功率预测开始时间
        :return test_feature_data: 测试输入数据
        :return history_target_data: 测试历史功率数据
        :return time: 预报的时间
        :param host: 主机
        :param user: 用户名
        :param password: 密码
        :param database: 数据库名
        :param charset: 解码方式
        :param port: 端口
        :param predict_type: 预测类型
        :param online_capacity: 装机容量
        """
        start_time_datetime = datetime.strptime(end_time_str, '%Y/%m/%d %H:%M')
        start_time_datetime1 = datetime.strptime(start_time_str, '%Y/%m/%d %H:%M')

        if sql == "kingbas":
            db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        else:
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        c = db.cursor()

        nwp_15min_row = numpy.zeros((16, 0))
        for i in range(n):
            for j in range(n):
                if i == int(n/2) and j == int(n/2):
                    table_name = "nwp_" + str(file_path["id"])
                else:
                    table_name = "nwp_" + str(file_path["id"]) + '_' + str(i) + '_' + str(j)
                if sql == "kingbas":
                    c.execute(
                        "select start_time, forecast_time," + ','.join(iterm for iterm in usecols) + " from " +
                        table_name + " where start_time between %s and %s ORDER BY start_time asc;",
                        (start_time_datetime1, start_time_datetime))
                    # 把数据变成元组形式
                    record = tuple(c.fetchall())
                    result = len(record)
                else:
                    result = c.execute(
                        "select start_time, forecast_time," + ','.join(iterm for iterm in usecols) + " from " +
                        table_name + " where start_time between %s and %s ORDER BY start_time asc;",
                        (start_time_datetime1, start_time_datetime))
                    # 把数据变成元组形式
                    record = c.fetchall()
                # 把nwp变成numpy的形式
                nwp = numpy.zeros((0, len(usecols)))
                # 时间那一列
                timeslist = numpy.zeros((0, 1))
                # 换个变量名
                predict_time = 1 * start_time

                train_feature_data = numpy.zeros((0, len(usecols) + 1))
                for i in range(len(record)):
                    if (record[i][1].timestamp() - record[i][0].timestamp()) < 6 * 3600:
                        nwp_row = ','.join(iterm for iterm in record[i][2:])
                        nwp = eval(nwp_row)
                        nwp_array = numpy.array(nwp)
                        predict_time = record[i][1].timestamp()
                        nwp_time = numpy.hstack((predict_time, nwp_array)).reshape(1, -1)
                        train_feature_data = numpy.vstack((train_feature_data, nwp_time))
                nwp = train_feature_data[:, :-1]
                timeslist = train_feature_data[:, -1].reshape(-1, 1)

                # nwp是预测的未来4小时的nwp
                if predict_type == 'solar':
                    nwp = numpy.hstack((nwp, timeslist))
                try:
                    nwp_15min_row = numpy.hstack((nwp_15min_row, nwp))
                except:
                    db.close()
                    return numpy.array([]), numpy.array([])

        if predict_type == 'solar':
            nwp_15min_row = nwp_15min_row.reshape(int(nwp_15min_row.shape[0]*nwp_15min_row.shape[1]), len(usecols) + 1)
        else:
            nwp_15min_row = nwp_15min_row.reshape(int(nwp_15min_row.shape[0]*nwp_15min_row.shape[1]), len(usecols))

        nwp_15min_tcc = numpy.zeros((16, 0))
        for i in range(n):
            for j in range(n):
                if i == int(n / 2) and j == int(n / 2):
                    table_name = "nwp_" + str(file_path["id"])
                else:
                    table_name = "nwp_" + str(file_path["id"]) + '_' + str(i) + '_' + str(j)
                if sql == "kingbas":
                    c.execute(
                        "select start_time, forecast_time," + ','.join(iterm for iterm in ['TCC']) + " from " +
                        table_name + " where start_time between %s and %s ORDER BY start_time asc;",
                        (start_time_datetime1, start_time_datetime))
                    # 把数据变成元组形式
                    record = tuple(c.fetchall())
                    result = len(record)
                else:
                    result = c.execute(
                        "select start_time, forecast_time," + ','.join(iterm for iterm in ['TCC']) + " from " +
                        table_name + " where start_time between %s and %s ORDER BY start_time asc;",
                        (start_time_datetime1, start_time_datetime))
                    # 把数据变成元组形式
                    record = c.fetchall()
                # 把nwp变成numpy的形式
                nwp = numpy.zeros((0, len(usecols)))
                # 换个变量名
                predict_time = 1 * start_time
                for i in range(16):
                    # 15分钟✖60=900秒
                    predict_time += 900
                    # 找离15分钟的预报时间最近的预报的数字天气预报
                    for j in range(result - 1, -1, -1):
                        if record[j][1].timestamp() == predict_time and record[j][0].timestamp() < start_time:
                            a = ','.join(iterm for iterm in record[j][2:])
                            b = eval(a)
                            c = numpy.array(b).reshape(1, -1)
                            nwp = numpy.vstack((nwp, c))
                            break
                # nwp是预测的未来4小时的nwp
                try:
                    nwp_15min_tcc = numpy.hstack((nwp_15min_tcc, nwp))
                except:
                    db.close()
                    return numpy.array([]), numpy.array([])
        nwp_15min_tcc = nwp_15min_tcc.reshape(int(nwp_15min_tcc.shape[0]*nwp_15min_tcc.shape[1]), 1)

        db.close()
        return nwp_15min_row, nwp_15min_tcc

    def load_five_minute_train_data(self, file_path, nrows=None, skiprows=None, usecols=None, rate=0.75,
                                    forecast_time=1, predict_type="wind"):
        """
        加载训练集及测试集数据。此处加载的数据，均是预处理后的数据
        数据预处理：功率数据归一化，错误数据识别与修正，限电数据识别与修正
        :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
        :param nrows: 加载特征集/目标集文件时所要选取的记录条数，默认全加载
        :param skiprows: 加载特征集/目标集文件时的忽略记录条数（从文件头开始算起，表头也算一条数据），默认不忽略
        :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
                        为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
                        一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
        :param rate: 训练集数据量与（训练集+测试集数据量）的比值
        :param forecast_time: 所加载的特征集文件数据（即天气预报数据）中，预报时间跨度。为3时表示预报未来3天的天气数据；
                              其他值表示预报未来1天的天气数据。默认取值为1
        :param predict_type: 预测类型
        :return: self.__test_feature_data, self.__test_target_data 训练输入数据，训练输出数据
        """
        if forecast_time == 1:
            self.__forecast_time = 1
            self.__load_one_day_forecast_data_from_file(file_path, nrows, skiprows, usecols, rate, predict_type)
        else:
            self.__forecast_time = 3
            self.__load_three_day_forecast_data_for_five_minute(file_path, nrows, skiprows, usecols, rate)

        train_feature_data = self._train_feature_data
        train_target_data = self._train_target_data
        return train_feature_data, train_target_data

    @staticmethod
    # def __load_three_day_forecast_data_for_five_minute(file_path, nrows, skiprows, usecols, rate):
    #     """
    #        加载训练集及测试集数据。此处加载的数据，均是预处理后的数据
    #        :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
    #        :param nrows: 加载特征集/目标集文件时所要选取的记录条数，默认全加载
    #        :param skiprows: 加载特征集/目标集文件时的忽略记录条数（从文件头开始算起，表头也算一条数据），默认不忽略。
    #                         该值必须是288的倍数+1
    #        :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
    #                        为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
    #                        一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
    #        :param rate: 训练集数据量与（训练集+测试集数据量）的比值
    #        :return:
    #        """
    #     # 加载特征集文件
    #     data_set = None
    #     for file in file_path["feature"]:
    #         if data_set is None:
    #             data_set = load_csv_data(file, nrows=nrows, skiprows=skiprows, usecols=usecols)
    #         else:
    #             data_set = numpy.concatenate((data_set,
    #                                           load_csv_data(file, nrows=nrows, skiprows=skiprows, usecols=usecols)
    #                                           ))
    #     # 起报时间总天数。预报未来3天的天气预报数据时，起报时间为同一天的记录共有288个点
    #     start_days = int(len(data_set) / 288)
    #     # 用于训练的起报时间天数
    #     train_start_days = int(start_days * rate)
    #     # 划分训练/测试特征集数据
    #     train_data = data_set[0:train_start_days * 288]
    #     test_feature_data = data_set[train_start_days * 288:start_days * 288]
    #
    #     # 处理训练特征集，仅取预报未来一天的预报数据
    #     train_feature_data = numpy.reshape(train_data, (train_start_days, -1))[:, :96 * train_data.shape[1]]
    #     train_feature_data = numpy.reshape(train_feature_data, (train_start_days * 96, -1))
    #
    #     data_set = None
    #     for file in file_path["target"]:
    #         if data_set is None:
    #             data_set = load_csv_data(file, nrows=nrows, skiprows=skiprows)
    #         else:
    #             data_set = numpy.concatenate((data_set,
    #                                           load_csv_data(file, nrows=nrows, skiprows=skiprows)
    #                                           ))
    #     test_data_end_index = start_days * 288
    #     train_data_len = train_start_days * 288
    #     train_target_data = data_set[:train_data_len, 1]
    #     test_target_data = data_set[train_data_len:test_data_end_index, 1]
    #
    #     # 将目标集的数据类型转换成float64
    #     train_target_data = train_target_data.astype("float64")
    #     test_target_data = test_target_data.astype("float64")
    #     return train_feature_data, train_target_data, test_feature_data, test_target_data

    def load_ensemble_data_for_five_minute_sql(self, host='localhost', user='root', password='123456', database='kuafu',
                                            charset='utf8', port=3306, file_path=None, usecols=None, rate=0.075,
                                            predict_type='wind'):
        """
        加载测试集数据。加载测试的输入数据
        :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
        :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
                        为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
                        一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
        :param predict_type: 预测类型
        :return test_feature_data: 测试输入数据
        :return history_target_data: 测试历史功率数据
        :return time: 预报的时间
        :param host: 主机
        :param user: 用户名
        :param password: 密码
        :param database: 数据库名
        :param charset: 解码方式
        :param port: 端口
        :param rate: 训练集数据量与（训练集+测试集数据量）的比值
        """
        length = 1
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        # 定义游标
        c = db.cursor()
        result = c.execute(
            "select start_time, forecast_time," + ','.join(iterm for iterm in usecols) + " from nwp_" +
            str(file_path["id"]) + " ORDER BY start_time asc;")
        record = c.fetchall()
        train_feature_data = numpy.zeros((0, len(usecols) + 1))
        for i in range(int(result * rate)):
            if (record[i][1].timestamp() - record[i][0].timestamp()) < 6 * 3600:
                nwp_row = ','.join(iterm for iterm in record[i][2:])
                nwp = eval(nwp_row)
                nwp_array = numpy.array(nwp)
                predict_time = record[i][1].timestamp()
                nwp_time = numpy.hstack((predict_time, nwp_array)).reshape(1, -1)
                train_feature_data = numpy.vstack((train_feature_data, nwp_time))

        result = c.execute("select time, power from real_power_five_minute_" + str(file_path["id"]) + " ORDER BY time asc;")
        record = c.fetchall()
        train_target_data = numpy.zeros((0, 2))
        for i in range(int(result * rate)):
            power = record[i][1]
            start_time = record[i][0].timestamp()
            power_time = numpy.hstack((start_time, power)).reshape(1, -1)
            train_target_data = numpy.vstack((train_target_data, power_time))

        train_target_data = pandas.DataFrame(train_target_data)
        train_feature_data = pandas.DataFrame(train_feature_data)

        # 将15分钟分辨率的NWP与5分钟的功率数据按照功率数据的时间戳对齐
        train_data = pandas.merge(train_target_data, train_feature_data, left_on=0, right_on=0, how='left')

        train_target_data = train_data.values[:, 1]
        train_feature_data = train_data.values[:, 2:]

        # 剔除NWP中的空数据
        train_feature_data = train_feature_data[~numpy.isnan(train_feature_data).any(axis=1), :]

        if predict_type == 'solar':
            t = ((train_data.values[:, 0] % 86400) / 86400).reshape(-1, 1)
            train_feature_data = numpy.hstack((train_feature_data, t))

        return train_feature_data, train_target_data

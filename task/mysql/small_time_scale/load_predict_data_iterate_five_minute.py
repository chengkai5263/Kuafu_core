# _*_ coding: utf-8 _*_
import numpy

from abc import ABCMeta
import pymysql
import time
import datetime

from common.logger import logs


class LoadPredictdataiterateFiveminute(metaclass=ABCMeta):
    def __init__(self):
        # 训练/测试时间集
        # 数据类型为ndarray，大小为m*1，即m行1列。列的含义是代表时刻。每个元素的内容是形如“2018-01-02 15:45:00”的字符串
        self._train_time_data = None
        self.__test_time_data = None

        # 训练/测试特征集（数据为原始数据，未作标准化处理）
        # 数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64
        self.__test_feature_data = None

        # 训练/测试目标集
        # 数据类型为ndarray，大小为m*1，即m行1列。列的含义是代表功率值。每个元素的类型是float64
        self.__test_target_data = None

        # 特征集文件数据（即天气预报数据）中，预报时间跨度
        # 为3时表示预报未来3天的天气数据；其他值表示预报未来1天的天气数据
        self.__forecast_time = 1

    def load_test_data_forpredict(self, file_path, nrows=None, skiprows=None, usecols=None, rate=0.75, forecast_time=1,
                                  predict_type="wind", predict_term="short", label=0):
        """
        加载测试集数据。加载测试的输入数据
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
        :param predict_term: 预测时长
        :param label: 计数序列
        :return test_feature_data: 测试输入数据
        :return history_target_data: 测试历史功率数据
        :return time: 预报的时间
        """
        if forecast_time == 1:
            self.__forecast_time = 1
            self.__load_one_day_forecast_data_from_file(file_path, nrows, skiprows, usecols, rate, predict_type)
        else:
            self.__forecast_time = 3
            self.__load_three_day_forecast_data_from_file(file_path, nrows, skiprows, usecols, rate, predict_type)

        test_feature_data = None

        feature_len = 16
        interval = 1
        if predict_term == "short":
            feature_len = 288
            interval = 96
            test_feature_data = self.__test_feature_data[label * feature_len: label * feature_len + feature_len, :]

        if predict_type != "short":
            start_days = int(len(self.__test_feature_data) / 288)
            self.__test_feature_data = \
                numpy.reshape(self.__test_feature_data, (start_days, -1))[:, :96 * self.__test_feature_data.shape[1]]
            self.__test_feature_data = numpy.reshape(self.__test_feature_data, (start_days * 96, -1))
            test_feature_data = self.__test_feature_data[label * interval: label * interval + feature_len, :]
        history_target_data = self.__test_target_data[label * interval - 1]
        time = self.__test_time_data[label * interval - 1]

        return test_feature_data, history_target_data, time

    def load_test_data_fortest(self, file_path, nrows=None, skiprows=None, usecols=None, rate=0.75, forecast_time=1,
                               predict_type="wind", predict_term="short", ita=iter(range(10000))):
        """
        加载测试集数据。加载测试的输出数据
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
        :param predict_term: 预测时长
        :param ita: 计数序列
        :return test_target_data: 测试验证数据
        :return time: 预报的时间
        """
        if forecast_time == 1:
            self.__forecast_time = 1
            self.__load_one_day_forecast_data_from_file(file_path, nrows, skiprows, usecols, rate, predict_type)
        else:
            self.__forecast_time = 3
            self.__load_three_day_forecast_data_from_file(file_path, nrows, skiprows, usecols, rate, predict_type)

        label = next(ita)

        feature_len = 16
        interval = 1
        if predict_term == "short":
            feature_len = 288
            interval = 96

        test_target_data = self.__test_target_data[label * interval: label * interval + feature_len]
        time = self.__test_time_data[label * interval - 1]

        return test_target_data, time

    def upload_predict_nwp_five_minute(self, host='localhost', user='root', password='123456', database='kuafu',
                                       charset='utf8', port=3306, start_time=None, file_path=None, usecols=None,
                                       predict_type='wind', online_capacity=20, sql="mysql"):
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
        # logs.info('upload_predict_nwp_five_minute')
        start_time = start_time - start_time % 900
        # 转换成datetime的格式，start_time传进来前已经把字符串转为float的数字。
        start_time_datetime = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time)),
                                                         '%Y-%m-%d %H:%M')
        # 过去7天起报的数值天气预报
        start_time_datetime1 = datetime.datetime.strptime(
            time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time - 7 * 24 * 3600)), '%Y-%m-%d %H:%M')
        if sql == "kingbas":
            # 获取参与集成学习的模型名称
            db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
            c = db.cursor()
            c.execute(
                "select start_time, forecast_time," + ','.join(iterm for iterm in usecols) + " from nwp_" +
                str(file_path["id"]) + " where start_time between %s and %s ORDER BY start_time asc;",
                (start_time_datetime1, start_time_datetime))
            # 把数据变成元组形式
            record = tuple(c.fetchall())
            result = len(record)
        else:
            # 连接数据库
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
            # 定义游标
            c = db.cursor()
            result = c.execute(
                "select start_time, forecast_time," + ','.join(iterm for iterm in usecols) + " from nwp_" +
                str(file_path["id"]) + " where start_time between %s and %s ORDER BY start_time asc;",
                (start_time_datetime1, start_time_datetime))
            # 把数据变成元组形式
            record = c.fetchall()
        # 把nwp变成numpy的形式
        nwp = numpy.zeros((0, len(usecols)))
        # 时间那一列
        timeslist = numpy.zeros((0, 1))
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
                    t = (predict_time % 86400) / 86400
                    timeslist = numpy.vstack((timeslist, t))
                    break
        # nwp是预测的未来4小时的nwp
        if predict_type == 'solar':
            nwp = numpy.hstack((nwp, timeslist))

        if sql == "kingbas":
            # 获取参与集成学习的模型名称
            db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        else:
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        # 定义游标
        c = db.cursor()
        # 取最近20天的功率，按秒计算。
        twenty_day_ago_datetime = datetime.datetime.strptime(
            time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time - 20 * 24 * 3600)), '%Y-%m-%d %H:%M')
        # 取20天前之后的所有历史功率
        if sql == "kingbas":
            c.execute(
                "select time, power from real_power_" + str(file_path["id"]) + " where time > %s ORDER BY time asc;",
                (twenty_day_ago_datetime,))
            record_real_power = tuple(c.fetchall())
            result_real_power = len(record_real_power)
            c.execute(
                "select forecast_time, predict_power from predict_power_" + str(
                    file_path["id"]) + " where predict_term=%s and start_time > %s ORDER BY start_time asc;",
                ("ultra_short", (twenty_day_ago_datetime,)))
            record_predict_power = tuple(c.fetchall())
            result_predict_power = len(record_predict_power)
        else:
            result_real_power = c.execute(
                "select time, power from real_power_" + str(file_path["id"]) + " where time > %s ORDER BY time asc;",
                twenty_day_ago_datetime)
            record_real_power = c.fetchall()
            result_predict_power = c.execute(
                "select forecast_time, predict_power from predict_power_" + str(
                    file_path["id"]) + " where predict_term=%s and start_time > %s ORDER BY start_time asc;",
                ("ultra_short", twenty_day_ago_datetime))
            record_predict_power = c.fetchall()
        # 拼接一起
        record = record_predict_power + record_real_power
        result = result_real_power + result_predict_power

        # 先找真实功率，找不到就用超短期预测的，如果都找不到的话就用前一天同一时段的来补充
        history_power = None
        for i in range(20):
            for j in range(result - 1, -1, -1):
                if record[j][0].timestamp() == start_time:
                    history_power = record[j][1]
                    break
            if history_power is None:
                start_time = start_time - 24 * 3600
            else:
                break

        if history_power is None:
            history_power = online_capacity / 2
        db.close()
        return nwp, history_power

    def upload_predict_nwp_five_minute_for_cloud_predict(self, host='localhost', user='root', password='123456',
                                                         database='kuafu', charset='utf8', port=3306,
                                                         start_time=None, file_path=None, usecols=None,
                                                         predict_type='wind', online_capacity=20, sql="mysql", n=7):
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
        # logs.info('upload_predict_nwp_five_minute')
        start_time = start_time - start_time % 900
        # 转换成datetime的格式，start_time传进来前已经把字符串转为float的数字。
        start_time_datetime = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time)),
                                                         '%Y-%m-%d %H:%M')
        # 过去7天起报的数值天气预报
        start_time_datetime1 = datetime.datetime.strptime(
            time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time - 7 * 24 * 3600)), '%Y-%m-%d %H:%M')
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
                            t = (predict_time % 86400) / 86400
                            timeslist = numpy.vstack((timeslist, t))
                            break
                # nwp是预测的未来4小时的nwp
                if predict_type == 'solar':
                    nwp = numpy.hstack((nwp, timeslist))
                if len(nwp) == 16:
                    nwp_15min_row = numpy.hstack((nwp_15min_row, nwp))
                else:
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
                if len(nwp) == 16:
                    nwp_15min_tcc = numpy.hstack((nwp_15min_tcc, nwp))
                else:
                    db.close()
                    return numpy.array([]), numpy.array([])
        nwp_15min_tcc = nwp_15min_tcc.reshape(int(nwp_15min_tcc.shape[0]*nwp_15min_tcc.shape[1]), 1)

        db.close()
        return nwp_15min_row, nwp_15min_tcc

    def upload_4_hours_ago_power(self, host='localhost', user='root', password='123456', database='kuafu',
                                 charset='utf8', port=3306, start_time=None, file_path=None, online_capacity=20, sql="mysql"):
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
        # logs.info('upload_4_hours_ago_power')
        if sql == "kingbas":
            # 获取参与集成学习的模型名称
            db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        else:
            # 连接数据库
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
            # 定义游标
        c = db.cursor()
        # 取最近20天的功率，按秒计算。
        twenty_day_ago_datetime = datetime.datetime.strptime(
            time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time - 20 * 24 * 3600)), '%Y-%m-%d %H:%M')
        # 取20天前之后的所有历史功率
        if sql == "kingbas":
            c.execute(
                "select time, power from real_power_" + str(file_path["id"]) + " where time > %s ORDER BY time asc;",
                (twenty_day_ago_datetime,))
            record_real_power = tuple(c.fetchall())
            result_real_power = len(record_real_power)
            c.execute(
                "select forecast_time, predict_power from predict_power_" + str(
                    file_path["id"]) + " where predict_term=%s and start_time > %s ORDER BY start_time asc;",
                ("ultra_short", (twenty_day_ago_datetime,)))
            record_predict_power = tuple(c.fetchall())
            result_predict_power = len(record_predict_power)
        else:
            result_real_power = c.execute(
                "select time, power from real_power_" + str(file_path["id"]) + " where time > %s ORDER BY time asc;",
                twenty_day_ago_datetime)
            record_real_power = c.fetchall()
            result_predict_power = c.execute(
                "select forecast_time, predict_power from predict_power_" + str(
                    file_path["id"]) + " where predict_term=%s and start_time > %s ORDER BY start_time asc;",
                ("ultra_short", twenty_day_ago_datetime))
            record_predict_power = c.fetchall()
        # 拼接一起
        record = record_predict_power + record_real_power
        result = result_real_power + result_predict_power
        # 先找真实功率，找不到就用超短期预测的，如果都找不到的话就用前一天同一时段的来补充
        history_power = []

        yyy_time = 1 * start_time - 900 * 48
        for i in range(48):
            # 15分钟✖60=900秒
            xxx_time = yyy_time + 300 * i
            flag = True
            for i in range(20):
                flag = True
                for j in range(result - 1, -1, -1):
                    if record[j][0].timestamp() == xxx_time:
                        history_power.append(record[j][1])
                        flag = False
                        break
                if flag:  # 判断也不对
                    xxx_time = xxx_time - 24 * 3600
                else:
                    break
            if flag:
                history_power.append(online_capacity / 2)
        db.close()
        return history_power

# _*_ coding: utf-8 _*_

import numpy
import pandas
from abc import ABCMeta
import pymysql
import time
import datetime
from common.logger import logs


class LoadTraindata(metaclass=ABCMeta):
    @staticmethod
    def load_train_data_for_ultra_short_sql(host='localhost', user='root', password='123456', database='kuafu',
                                            charset='utf8', port=3306, config=None, usecols=None, rate=0.75,
                                            predict_type='wind', start_time=None,
                                            end_time=None):
        """
        加载测试集数据。加载测试的输入数据
        :param config: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
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
        :param start_time: 训练集数据开始时间
        :param end_time: 训练集数据结束时间
        """
        # 获取数据时长，并分段
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        # 定义游标
        c = db.cursor()
        c.execute("select min(start_time) from nwp_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        start_time_float1 = record[0][0].timestamp()
        c.execute("select max(start_time) from nwp_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        end_time_float1 = record[-1][0].timestamp()
        c.execute("select min(time) from real_power_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        start_time_float2 = record[0][0].timestamp()
        c.execute("select max(time) from real_power_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        end_time_float2 = record[-1][0].timestamp()
        start_time_float_database = max(start_time_float1, start_time_float2)
        end_time_float_database = min(end_time_float1, end_time_float2)
        if start_time is None or end_time is None:
            start_time_float = start_time_float_database
            end_time_float = end_time_float_database
        else:
            start_time_float_input = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M').timestamp()
            end_time_float_input = datetime.datetime.strptime(end_time, '%Y/%m/%d %H:%M').timestamp()
            start_time_float = max(start_time_float_database, start_time_float_input)
            end_time_float = min(end_time_float_database, end_time_float_input)
        start_time_float = int((start_time_float - 57600) / 86400) * 86400 + 57600

        # ----------------------------------------------------------------------------------------------------------
        start_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time_float)), '%Y/%m/%d %H:%M')
        end_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float)), '%Y/%m/%d %H:%M')

        c.execute(
            "select distinct forecast_time from nwp_" + str(config["id"]) +
            " where forecast_time between %s and %s ORDER BY forecast_time asc;",
            (start_time_datetime, end_time_datetime))
        db.commit()
        record = c.fetchall()
        dataframe_nwp = pandas.DataFrame(record, columns=['forecast_time'])

        c.execute(
            "select distinct time from real_power_" + str(config["id"]) +
            " where time between %s and %s ORDER BY time asc;",
            (start_time_datetime, end_time_datetime))
        db.commit()
        record = c.fetchall()
        dataframe_real_power = pandas.DataFrame(record, columns=['time'])

        train_data = pandas.merge(dataframe_nwp, dataframe_real_power, left_on='forecast_time', right_on='time',
                                  how='right')
        train_data = train_data.dropna(axis=0, how='any')

        middle_time_float = int(
            (train_data.iloc[min(int(rate * len(train_data)),
                                 int(len(train_data) - 1)), 0].timestamp() - 57600) / 86400) * 86400 + 57600
        # ----------------------------------------------------------------------------------------------------------

        start_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time_float + 900)), '%Y/%m/%d %H:%M')
        middle_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float + 900)), '%Y/%m/%d %H:%M')

        thresha = 0.8
        length = 1

        # 修改读取NWP数据方式2023-3-16-------------------------------------------------------------------------------------
        sql = "select A.start_time, forecast_time, " + ','.join(iterm for iterm in usecols) + " from" \
              " (select DISTINCT start_time, forecast_time, " + ','.join(iterm for iterm in usecols) + " from nwp_" + \
              str(config["id"]) + " where %s <= forecast_time and forecast_time <= %s) A," \
              " (select max(forecast_time) max_forecast_time, max(start_time) max_start_time from nwp_" + \
              str(config["id"]) + \
              " where %s <= forecast_time and forecast_time <= %s group by forecast_time) B" \
              " where A.forecast_time = B.max_forecast_time and A.start_time = B.max_start_time" \
              " order by forecast_time asc;"
        c.execute(sql, (start_time_datetime, middle_time_datetime, start_time_datetime, middle_time_datetime))
        db.commit()

        record = c.fetchall()

        train_feature_data = numpy.zeros((0, len(usecols) + 1))
        # NWP每行的varchar解析为float都采用try，避免某一行出错，导致读取NWP失败，2023/3/16
        for i in range(len(record)):
            try:
                nwp_row = ','.join(iterm for iterm in record[i][2:])
                nwp = eval(nwp_row)
                nwp_array = numpy.array(nwp)
                predict_time = record[i][1].timestamp()
                nwp_time = numpy.hstack((predict_time, nwp_array)).reshape(1, -1)
                train_feature_data = numpy.vstack((train_feature_data, nwp_time))
            except Exception as err:
                logs.error(str(err), exc_info=True)

        # 读取功率数据----------------------------------------------------------------------------------------------------
        result = c.execute(
            "select distinct time, power from real_power_" + str(config["id"]) +
            " where time between %s and %s ORDER BY time asc;",
            (start_time_datetime, middle_time_datetime))
        db.commit()
        record = c.fetchall()
        c.close()
        db.close()
        train_target_data = numpy.zeros((0, 2))
        for i in range(result):
            power = record[i][1]
            start_time = record[i][0].timestamp()
            power_time = numpy.hstack((start_time, power)).reshape(1, -1)
            train_target_data = numpy.vstack((train_target_data, power_time))

        # 功率与NWP数据时间对齐--------------------------------------------------------------------------------------------
        train_target_data = pandas.DataFrame(train_target_data)
        train_feature_data = pandas.DataFrame(train_feature_data)

        train_data = pandas.merge(train_target_data, train_feature_data, left_on=0, right_on=0, how='right')
        train_data = train_data.dropna(axis=0, how='any')

        # 判断有多少比例的功率数据没有NWP对应---------------------------------------------------------------------------------
        if len(train_data) / len(train_target_data) / length < thresha:
            debug_log = '注意：' + str(config["id"]) + '有' + str(
                100 - 100 * len(train_data) / len(train_target_data) / length) + "%的历史数据没有NWP数据对应！！！"
            logs.debug(debug_log)

        # 如果是光伏，需要用时间作为输入-------------------------------------------------------------------------------------
        train_target_data = train_data.values[:, 1]
        train_feature_data = train_data.values[:, 2:]
        if predict_type == 'solar':
            t = ((train_data.values[:, 0] % 86400) / 86400).reshape(-1, 1)
            train_feature_data = numpy.hstack((train_feature_data, t))

        return train_feature_data, train_target_data

    @staticmethod
    def load_train_data_for_short_sql(host='localhost', user='root', password='123456', database='kuafu',
                                      charset='utf8', port=3306, config=None, usecols=None, rate=0.75,
                                      predict_type='wind', power_supplement_enable=False, start_time=None,
                                      end_time=None):
        """
        加载测试集数据。加载测试的输入数据
        :param config: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
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
        :param power_supplement_enable: 是否补齐历史功率
        :param start_time: 训练集数据开始时间
        :param end_time: 训练集数据结束时间
        """
        # 获取数据时长，并分段
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        # 定义游标
        c = db.cursor()
        c.execute("select min(start_time) from nwp_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        start_time_float1 = record[0][0].timestamp()
        c.execute("select max(start_time) from nwp_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        end_time_float1 = record[-1][0].timestamp()
        c.execute("select min(time) from real_power_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        start_time_float2 = record[0][0].timestamp()
        c.execute("select max(time) from real_power_" + str(config["id"]) + ";")
        db.commit()
        record = c.fetchall()
        end_time_float2 = record[-1][0].timestamp()
        start_time_float_database = max(start_time_float1, start_time_float2)
        end_time_float_database = min(end_time_float1, end_time_float2)
        if start_time is None or end_time is None:
            start_time_float = start_time_float_database
            end_time_float = end_time_float_database
        else:
            start_time_float_input = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M').timestamp()
            end_time_float_input = datetime.datetime.strptime(end_time, '%Y/%m/%d %H:%M').timestamp()
            start_time_float = max(start_time_float_database, start_time_float_input)
            end_time_float = min(end_time_float_database, end_time_float_input)

        start_time_float = int((start_time_float - 57600) / 86400) * 86400 + 57600

        # ----------------------------------------------------------------------------------------------------------
        start_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time_float)), '%Y/%m/%d %H:%M')
        end_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float)), '%Y/%m/%d %H:%M')

        c.execute(
            "select distinct forecast_time from nwp_" + str(config["id"]) +
            " where forecast_time between %s and %s ORDER BY forecast_time asc;",
            (start_time_datetime, end_time_datetime))
        db.commit()
        record = c.fetchall()
        dataframe_nwp = pandas.DataFrame(record, columns=['forecast_time'])

        c.execute(
            "select distinct time from real_power_" + str(config["id"]) +
            " where time between %s and %s ORDER BY time asc;",
            (start_time_datetime, end_time_datetime))
        db.commit()
        record = c.fetchall()
        dataframe_real_power = pandas.DataFrame(record, columns=['time'])

        train_data = pandas.merge(dataframe_nwp, dataframe_real_power, left_on='forecast_time', right_on='time',
                                  how='right')
        train_data = train_data.dropna(axis=0, how='any')

        middle_time_float = int(
            (train_data.iloc[min(int(rate * len(train_data)),
                                 int(len(train_data) - 1)), 0].timestamp() - 57600) / 86400) * 86400 + 57600
        # ----------------------------------------------------------------------------------------------------------

        start_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time_float + 900)), '%Y/%m/%d %H:%M')
        middle_time_datetime = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float + 900)), '%Y/%m/%d %H:%M')

        thresha = 0.8
        length = 1

        # 取功率数据------------------------------------------------------------------------------------------------------
        result = c.execute(
            "select time, power from real_power_" + str(config["id"]) +
            " where time between %s and %s ORDER BY time asc;",
            (start_time_datetime, middle_time_datetime))
        db.commit()
        record = c.fetchall()
        train_target_data = numpy.zeros((0, 2))
        for i in range(result):
            power = record[i][1]
            t = record[i][0].timestamp()
            power_time = numpy.hstack((t, power)).reshape(1, -1)
            train_target_data = numpy.vstack((train_target_data, power_time))

        # 补充历史功率缺失-------------------------------------------------------------------------------------------------
        if power_supplement_enable:
            for i in range(len(train_target_data)-1, 0, -1):
                for j in range(int((train_target_data[i, 0] - train_target_data[i-1, 0])/900 - 1)):
                    t = train_target_data[i, 0] - 900
                    for k in range(21):
                        r2 = numpy.argwhere(train_target_data[:, 0] == t-(k-10)*24*3600)
                        if len(r2) != 0:
                            break
                    if len(r2) != 0:
                        power_succedaneum = train_target_data[r2[0], 1]
                        power_time = numpy.hstack((t, power_succedaneum)).reshape(1, -1)
                        train_target_data = numpy.vstack(
                            (train_target_data[:i, :], power_time, train_target_data[i:, :]))

        # 修改读取NWP数据方式2023-03-16------------------------------------------------------------------------------------
        sql = "select A.start_time, forecast_time, " + ','.join(iterm for iterm in usecols) + " from" \
              " (select DISTINCT start_time, forecast_time, " + ','.join(iterm for iterm in usecols) + " from nwp_" + \
              str(config["id"]) + " where %s <= forecast_time and forecast_time <= %s) A," \
              " (select max(forecast_time) max_forecast_time, max(start_time) max_start_time from nwp_" + \
              str(config["id"]) + \
              " where %s <= forecast_time and forecast_time <= %s group by forecast_time) B" \
              " where A.forecast_time = B.max_forecast_time and A.start_time = B.max_start_time" \
              " order by forecast_time asc;"
        c.execute(sql, (start_time_datetime, middle_time_datetime, start_time_datetime, middle_time_datetime))
        db.commit()
        record = c.fetchall()
        c.close()
        db.close()
        # NWP每行的varchar解析为float都采用try，避免某一行出错，导致读取NWP失败，2023/3/16
        train_feature_data = numpy.zeros((0, len(usecols) + 1))
        for i in range(len(record)):
            try:
                nwp_row = ','.join(iterm for iterm in record[i][2:])
                nwp = eval(nwp_row)
                nwp_array = numpy.array(nwp)
                t = record[i][1].timestamp()
                nwp_time = numpy.hstack((t, nwp_array)).reshape(1, -1)
                train_feature_data = numpy.vstack((train_feature_data, nwp_time))
            except Exception as err:
                logs.error(str(err), exc_info=True)

        # 功率与NWP时间对齐-----------------------------------------------------------------------------------------------
        train_target_data = pandas.DataFrame(train_target_data)
        train_feature_data = pandas.DataFrame(train_feature_data)
        train_data = pandas.merge(train_target_data, train_feature_data, left_on=0, right_on=0, how='right')
        train_data = train_data.dropna(axis=0, how='any')

        # 判断有多少比例的功率数据没有NWP对应---------------------------------------------------------------------------------
        if len(train_data) / len(train_target_data) / length < thresha:
            debug_log = '注意：' + str(config["id"]) + '有' + str(
                100 - 100 * len(train_data) / len(train_target_data) / length) + "%的历史数据没有NWP数据对应！！！"
            logs.debug(debug_log)

        # 如果是光伏，需要用时间作为输入--------------------------------------------------------------------------------------
        train_target_data = train_data.values[:, 1]
        train_feature_data = train_data.values[:, 2:]
        if predict_type == 'solar':
            t = ((train_data.values[:, 0] % 86400) / 86400).reshape(-1, 1)
            train_feature_data = numpy.hstack((train_feature_data, t))

        return train_feature_data, train_target_data

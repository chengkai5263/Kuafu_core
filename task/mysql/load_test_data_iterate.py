# _*_ coding: utf-8 _*_

import numpy
from abc import ABCMeta
import pymysql
import time
import datetime


class LoadTestdataiterate(metaclass=ABCMeta):
    @staticmethod
    def upload_predict_nwp_short(host, user, password, database, port, charset, start_time, station_id, usecols,
                                 predict_type):
        """
        加载测试集数据。加载测试的输入数据
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
        :param station_id: 场站id
        """
        # 读取NWP数据（只取前7天起报的NWP）----------------------------------------------------------------------------------
        start_time_datetime2 = datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time)),
                                                          '%Y/%m/%d %H:%M')
        start_time_datetime1 = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time - 7 * 24 * 3600)), '%Y/%m/%d %H:%M')
        forecast_time_datetime1 = datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M",
                                                                           time.localtime(start_time+900)),
                                                             '%Y/%m/%d %H:%M')
        forecast_time_datetime2 = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time + 900*288)), '%Y/%m/%d %H:%M')
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        c = db.cursor()

        result = c.execute("SELECT t.* FROM (select start_time, forecast_time, " + ','.join(
            iterm for iterm in usecols) + " from nwp_" + str(
            station_id) + " where start_time BETWEEN %s and %s ORDER BY start_time desc LIMIT 18816) t"
                          " where t.forecast_time BETWEEN %s and %s group by t.forecast_time ORDER BY t.forecast_time;",
                           (start_time_datetime1, start_time_datetime2, forecast_time_datetime1,
                            forecast_time_datetime2))
        db.commit()
        record = c.fetchall()
        nwp = numpy.zeros((0, len(usecols)))
        timeslist = numpy.zeros((0, 1))
        predict_time = 1 * start_time
        for i in range(288):
            predict_time += 900
            for j in range(result - 1, -1, -1):
                if record[j][1].timestamp() == predict_time and record[j][0].timestamp() < start_time:
                    a = ','.join(iterm for iterm in record[j][2:])
                    b = eval(a)
                    tem = numpy.array(b).reshape(1, -1)
                    nwp = numpy.vstack((nwp, tem))

                    t = (predict_time % 86400) / 86400
                    timeslist = numpy.vstack((timeslist, t))
                    break
        if predict_type == 'solar':
            nwp = numpy.hstack((nwp, timeslist))

        start_time_datetime = datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time)),
                                                         '%Y/%m/%d %H:%M')
        c.execute("select DISTINCT power from real_power_" + str(station_id) +
                  " where time = %s ORDER BY time asc;", start_time_datetime)
        db.commit()
        record = c.fetchall()
        c.close()
        db.close()
        if len(record) == 0:
            history_power = None
        else:
            history_power = record[0][0]
        return nwp, history_power

    @staticmethod
    def upload_predict_nwp_ultrashort(host, user, password, database, charset, port,
                                      start_time, station_id, usecols, predict_type):
        """
        加载测试集数据。加载测试的输入数据
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
        :param station_id: 场站id
        """
        # 读取NWP数据，只取过去七天起报的------------------------------------------------------------------------------------
        start_time_datetime2 = datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time)),
                                                          '%Y/%m/%d %H:%M')
        start_time_datetime1 = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time - 7 * 24 * 3600)), '%Y/%m/%d %H:%M')
        forecast_time_datetime1 = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time + 900)),
            '%Y/%m/%d %H:%M')
        forecast_time_datetime2 = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time + 900 * 16)), '%Y/%m/%d %H:%M')
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        # 定义游标
        c = db.cursor()

        result = c.execute("SELECT t.* FROM (select start_time, forecast_time, " + ','.join(
            iterm for iterm in usecols) + " from nwp_" + str(
            station_id) + " where start_time BETWEEN %s and %s ORDER BY start_time desc LIMIT 18816) t"
                          " where t.forecast_time BETWEEN %s and %s group by t.forecast_time ORDER BY t.forecast_time;",
                           (start_time_datetime1, start_time_datetime2, forecast_time_datetime1,
                            forecast_time_datetime2))
        db.commit()

        record = c.fetchall()
        nwp = numpy.zeros((0, len(usecols)))
        timeslist = numpy.zeros((0, 1))
        predict_time = 1 * start_time
        for i in range(16):
            predict_time += 900
            for j in range(result - 1, -1, -1):
                if record[j][1].timestamp() == predict_time:
                    a = ','.join(iterm for iterm in record[j][2:])
                    b = eval(a)
                    tem = numpy.array(b).reshape(1, -1)
                    nwp = numpy.vstack((nwp, tem))

                    t = (predict_time % 86400) / 86400
                    timeslist = numpy.vstack((timeslist, t))
                    break
        if predict_type == 'solar':
            nwp = numpy.hstack((nwp, timeslist))

        start_time_datetime = datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time)),
                                                         '%Y/%m/%d %H:%M')
        c.execute("select DISTINCT power from real_power_" + str(station_id) +
                  " where time = %s ORDER BY time asc;", start_time_datetime)
        db.commit()
        record = c.fetchall()
        c.close()
        db.close()
        if len(record) == 0:
            history_power = None
        else:
            history_power = record[0][0]
        return nwp, history_power

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :   Kuafu_20230119
@File    :   hn_master_main_V2.py
@Contact :   zhoucc@csg.cn
@License :   (C)Copyright 2022, Green-Energy-Team-DGRI-CSG

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/1/30 9:51   ZhouCC              

"""
import platform
import sys
import numpy
import time
import logging
import pandas
import datetime
import pymysql
import os
import shutil
import setproctitle
from common.deamon import Daemon
from common.settings import settings, init_settings
from common.logger import logs
from common.logger import init_logger
from common.tools import catch_exception

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

from task.mysql.feature_selection import feature_select
from task.mysql.ensemble_learning import ensemble_learn
from task.mysql.transfer_learning import run_transfer_learning
from task.mysql.master_predict_hn import MultiPlantPredict
from task.mysql.create_table import create_table


@catch_exception("模型训练错误：")
def run_model_training(host, user, password, database, charset, port, rate=0.75,
                       scheduler=None, executor_name=None, csvrecord=True, figurerecord=True,
                       csv_path='./work_dir/csv/', figure_path='./work_dir/figure/'):
    """
    串行任务：特征工程+参数寻优+集成学习
    :param figurerecord: 是否需要绘图
    :param csvrecord: 是否需要打印csv记录
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param rate: 训练与测试比例划分
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :param csv_path: csv文件保存地址
    :param figure_path: 图片保存地址
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    if not os.path.exists(csv_path):
        os.makedirs(csv_path, exist_ok=True)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path, exist_ok=True)

    c.execute("select config_value from sys_config where config_key = 'operate_interval_time';")
    db.commit()
    record = c.fetchall()
    operate_interval_time = eval(record[0][0])

    now_time_float = datetime.datetime.now().timestamp()  # 当前时间float
    now_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(now_time_float))
    deadline_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(now_time_float - operate_interval_time * 86400))
    try:
        # 判断勾选的场站是否有对应的nwp、real_power、predict_power、predict_power_train表格，如果没有需要建表，或不参与测试
        result = c.execute("select DISTINCT id from configure where station_status = 2 and operation_time<%s;", deadline_time)
        db.commit()
        record_id = c.fetchall()
        if result == 0:
            logs.warning('被选中进行测试的场站数量为0，测试任务退出！！！')
            c.close()
            db.close()
            return
        for i in range(result):
            station_id = record_id[i][0]
            result_nwp = c.execute("select 1 from information_schema.tables where table_schema= 'kuafu'"
                                   " and table_name = 'nwp_%s';", station_id)
            db.commit()
            result_real_power = c.execute("select 1 from information_schema.tables where table_schema= 'kuafu'"
                                          " and table_name = 'real_power_%s';", station_id)
            db.commit()
            if result_nwp == 0 or result_real_power == 0:
                c.execute("update configure set station_status = 0 where id = %s;", station_id)
                db.commit()
                logs.warning("%s%s" % (str(station_id), '场站没有NWP或实际功率，无法参与测试！！！'))
                continue
            result_predict_power_train = c.execute("select 1 from information_schema.tables where table_schema= 'kuafu'"
                                                   " and table_name = 'predict_power_%s_train';", station_id)
            db.commit()
            if result_predict_power_train == 0:
                logs.warning("%s%s%s%s" % (str(station_id), "场站缺少predict_power_", str(station_id), "_train表，将被创建"))
                try:
                    c.execute("CREATE TABLE IF NOT EXISTS `predict_power_%s_train` ("
                              "`id` bigint NOT NULL AUTO_INCREMENT,"
                              "`predict_term` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,"
                              "`model_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,"
                              "`start_time` datetime NOT NULL,"
                              "`forecast_time` datetime NOT NULL,"
                              "`predict_power` float(10,4) NOT NULL,"
                              "`upper_bound_90` float(10,4) DEFAULT NULL,"
                              "`lower_bound_90` float(10,4) DEFAULT NULL,"
                              "`upper_bound_80` float(10,4) DEFAULT NULL,"
                              "`lower_bound_80` float(10,4) DEFAULT NULL,"
                              "`upper_bound_70` float(10,4) DEFAULT NULL,"
                              "`lower_bound_70` float(10,4) DEFAULT NULL,"
                              "`upper_bound_60` float(10,4) DEFAULT NULL,"
                              "`lower_bound_60` float(10,4) DEFAULT NULL,"
                              "`upper_bound_50` float(10,4) DEFAULT NULL,"
                              "`lower_bound_50` float(10,4) DEFAULT NULL,"
                              "PRIMARY KEY (`id`) USING BTREE);", station_id)
                    db.commit()
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    db.rollback()
                    logs.warning(
                        "%s%s%s%s" % (str(station_id), "表predict_power_", str(station_id), "_train创建失败，该场站无法参与测试！！！"))
                    c.execute("update configure set station_status = 0 where id  = %s;", station_id)
                    db.commit()
                    continue

        result = c.execute("select DISTINCT id from configure where station_status = 2;")
        db.commit()
        if result == 0:
            logs.warning('被选中进行测试且满足要求的场站数量为0，测试任务退出！！！')
            c.close()
            db.close()
            return
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()

    # 读取待测试场站id列表--------------------------------------------------------------------------------------------------
    c.execute("select id from configure where station_status = 2 and operation_time<%s;", deadline_time)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)

    # 场站id
    station_id_clusters = []
    for i in range(len(dataframe_config)):
        station_id_clusters.append(dataframe_config.loc[:, "id"][i])

    for station_id in station_id_clusters:
        c.execute("select model_savepath from configure where id = %s;", station_id)
        db.commit()
        record = c.fetchall()
        model_savepath = record[0][0]

        if not os.path.exists(model_savepath):
            os.makedirs(model_savepath, exist_ok=True)

        start_time = None
        end_time = None
        try:
            # 打印设置的时间范围
            timefile = open(csv_path + 'time.csv', 'a+')
            resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
            shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
            ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')
            dfcsv = pandas.DataFrame(['current time: ' + time.strftime("%Y/%m/%d %H:%M", time.localtime(
                int(datetime.datetime.now().timestamp())))])
            dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
            dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
            dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')
            dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')

            c.execute("select start_time from configure where id = %s;", station_id)
            db.commit()
            record_start_time = c.fetchall()
            c.execute("select end_time from configure where id = %s;", station_id)
            db.commit()
            record_end_time = c.fetchall()

            if record_start_time[0][0] is not None and record_end_time[0][0] is not None:
                start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record_start_time[0][0].timestamp()))
                end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record_end_time[0][0].timestamp()))

                dfcsv = pandas.DataFrame(
                    numpy.array(['setting test time: start_time: ' + start_time + '; end_time: ' + end_time]).reshape(1,
                                                                                                                      -1))
                logs.info(str(station_id) + '设置的测试数据时间：开始时间：' + start_time + '；结束时间：' + end_time)
            else:
                dfcsv = pandas.DataFrame(
                    numpy.array([str(station_id) + 'the test data time is not set, and the full data of the database'
                                                   ' will be applied to the test']).reshape(1, -1))
                logs.info(str(station_id) + '未设置测试数据时间，采用数据库的全量数据开展测试')
            dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
            dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf", mode='a+')
            dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf", mode='a+')
            dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf", mode='a+')
            timefile.close()
            resultfile.close()
            shortfile.close()
            ultrafile.close()
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.warning('写CSV失败！')

        try:
            # 如果设置了时间范围，计算该范围内可用数据天数和数据缺失率
            c.execute("select start_time from configure where id = %s;", station_id)
            db.commit()
            record_start_time = c.fetchall()
            c.execute("select end_time from configure where id = %s;", station_id)
            db.commit()
            record_end_time = c.fetchall()

            if record_start_time[0][0] is not None and record_end_time[0][0] is not None:
                start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record_start_time[0][0].timestamp()))
                end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record_end_time[0][0].timestamp()))
                start_time_float_input = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M').timestamp()
                end_time_float_input = datetime.datetime.strptime(end_time, '%Y/%m/%d %H:%M').timestamp()
                if end_time_float_input - start_time_float_input > 0:
                    start_time_datetime_input = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M')
                    end_time_datetime_input = datetime.datetime.strptime(end_time, '%Y/%m/%d %H:%M')
                    c.execute(
                        "select distinct forecast_time from nwp_" + str(station_id) +
                        " where forecast_time between %s and %s ORDER BY forecast_time asc;",
                        (start_time_datetime_input, end_time_datetime_input))
                    db.commit()
                    record = c.fetchall()
                    dataframe_nwp = pandas.DataFrame(record, columns=['forecast_time'])

                    c.execute(
                        "select distinct time from real_power_" + str(station_id) +
                        " where time between %s and %s ORDER BY time asc;",
                        (start_time_datetime_input, end_time_datetime_input))
                    db.commit()
                    record = c.fetchall()
                    dataframe_real_power = pandas.DataFrame(record, columns=['time'])

                    train_data = pandas.merge(dataframe_nwp, dataframe_real_power, left_on='forecast_time',
                                              right_on='time', how='right')
                    train_data = train_data.dropna(axis=0, how='any')

                    loss_rate = (((end_time_float_input - start_time_float_input + 900) / 900) - len(train_data)) / (
                            (end_time_float_input - start_time_float_input + 900) / 900)

                    dfcsv = pandas.DataFrame(numpy.array([str(station_id) +
                                                          'The actual data volume in the set period'
                                                          ' is: ' + str(int(len(train_data) / 96)) +
                                                          'days; missing data rate: ' + str(round(loss_rate * 100, 2)) +
                                                          '%']).reshape(1, -1))
                    logs.info(str(station_id) + '设置时段内实际数据量为：' + str(int(len(train_data) / 96)) + '天；数据缺失率为；'
                              + str(round(loss_rate * 100, 2)) + '%')
                    timefile = open(csv_path + 'time.csv', 'a+')
                    resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                    shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
                    ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')
                    dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                    dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                    dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')
                    dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')
                    timefile.close()
                    resultfile.close()
                    shortfile.close()
                    ultrafile.close()
                    if len(train_data) / 96 < 16:
                        logs.warning(str(station_id) + '场站在该时段可用数据不满足测试要求，该场站退出测试！！！')
                        continue
                else:
                    logs.info('设置的数据时长为0，无法满足测试要求，测试任务退出！！！')
                    continue
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.warning('写CSV失败！')

        try:
            # 实际的测试用数据范围
            c.execute("select min(start_time) from nwp_" + str(station_id) + ";")
            db.commit()
            record = c.fetchall()
            if len(record) == 0:
                logs.warning(str(station_id) + '场站在该时段可用数据不满足测试要求，该场站退出测试！！！')
                continue
            start_time_float1 = record[0][0].timestamp()
            c.execute("select max(start_time) from nwp_" + str(station_id) + ";")
            db.commit()
            record = c.fetchall()
            if len(record) == 0:
                logs.warning(str(station_id) + '场站在该时段可用数据不满足测试要求，该场站退出测试！！！')
                continue
            end_time_float1 = record[-1][0].timestamp()
            c.execute("select min(time) from real_power_" + str(station_id) + ";")
            db.commit()
            record = c.fetchall()
            if len(record) == 0:
                logs.warning(str(station_id) + '场站在该时段可用数据不满足测试要求，该场站退出测试！！！')
                continue
            start_time_float2 = record[0][0].timestamp()
            c.execute("select max(time) from real_power_" + str(station_id) + ";")
            db.commit()
            record = c.fetchall()
            if len(record) == 0:
                logs.warning(str(station_id) + '场站在该时段可用数据不满足测试要求，该场站退出测试！！！')
                continue
            end_time_float2 = record[-1][0].timestamp()
            start_time_float_database = max(start_time_float1, start_time_float2)
            end_time_float_database = min(end_time_float1, end_time_float2)

            c.execute("select start_time from configure where id = %s;", station_id)
            db.commit()
            record_start_time = c.fetchall()
            c.execute("select end_time from configure where id = %s;", station_id)
            db.commit()
            record_end_time = c.fetchall()

            if record_start_time[0][0] is None or record_end_time[0][0] is None:
                start_time_float = start_time_float_database
                end_time_float = end_time_float_database
            else:
                start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record_start_time[0][0].timestamp()))
                end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record_end_time[0][0].timestamp()))
                start_time_float_input = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M').timestamp()
                end_time_float_input = datetime.datetime.strptime(end_time, '%Y/%m/%d %H:%M').timestamp()

                start_time_float = max(start_time_float_database, start_time_float_input)
                end_time_float = min(end_time_float_database, end_time_float_input)

            start_time_float = int((start_time_float - 57600) / 86400) * 86400 + 57600
            end_time_float = int((end_time_float - 57600) / 86400) * 86400 + 57600

            start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time_float))
            end_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float))

            dfcsv = pandas.DataFrame(numpy.array([str(station_id) + 'actual test time: start_time: ' + start_time_str +
                                                  '; end_time: ' + end_time_str]).reshape(1, -1))
            logs.info(str(station_id) + '实际的测试数据时间：开始时间：' + start_time_str + '；结束时间：' + end_time_str)
            timefile = open(csv_path + 'time.csv', 'a+')
            resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
            shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
            ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')
            dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
            dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
            dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')
            dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')
            timefile.close()
            resultfile.close()
            shortfile.close()
            ultrafile.close()
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.warning('写CSV失败！')

        try:
            c.execute("select start_time from configure where id = %s;", station_id)
            db.commit()
            record_start_time = c.fetchall()
            c.execute("select end_time from configure where id = %s;", station_id)
            db.commit()
            record_end_time = c.fetchall()

            if record_start_time[0][0] is None or record_end_time[0][0] is None:
                c.execute("select min(start_time) from nwp_" + str(station_id) + ";")
                db.commit()
                record = c.fetchall()
                start_time_float1 = record[0][0].timestamp()
                c.execute("select max(start_time) from nwp_" + str(station_id) + ";")
                db.commit()
                record = c.fetchall()
                end_time_float1 = record[-1][0].timestamp()
                c.execute("select min(time) from real_power_" + str(station_id) + ";")
                db.commit()
                record = c.fetchall()
                start_time_float2 = record[0][0].timestamp()
                c.execute("select max(time) from real_power_" + str(station_id) + ";")
                db.commit()
                record = c.fetchall()
                end_time_float2 = record[-1][0].timestamp()
                start_time_float_database = max(start_time_float1, start_time_float2)
                end_time_float_database = min(end_time_float1, end_time_float2)

                start_time_datetime = datetime.datetime.strptime(
                    time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time_float_database)), '%Y/%m/%d %H:%M')
                end_time_datetime = datetime.datetime.strptime(
                    time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float_database)), '%Y/%m/%d %H:%M')
                c.execute(
                    "select distinct forecast_time from nwp_" + str(station_id) +
                    " where forecast_time between %s and %s ORDER BY forecast_time asc;",
                    (start_time_datetime, end_time_datetime))
                db.commit()
                record = c.fetchall()
                dataframe_nwp = pandas.DataFrame(record, columns=['forecast_time'])

                c.execute(
                    "select distinct time from real_power_" + str(station_id) +
                    " where time between %s and %s ORDER BY time asc;",
                    (start_time_datetime, end_time_datetime))
                db.commit()
                record = c.fetchall()
                dataframe_real_power = pandas.DataFrame(record, columns=['time'])

                train_data = pandas.merge(dataframe_nwp, dataframe_real_power, left_on='forecast_time', right_on='time',
                                          how='right')
                train_data = train_data.dropna(axis=0, how='any')

                loss_rate = (((end_time_float_database - start_time_float_database + 900) / 900) - len(train_data)) / (
                        (end_time_float_database - start_time_float_database + 900) / 900)

                dfcsv = pandas.DataFrame(numpy.array([str(station_id)
                                                      + 'The actual data volume in the set period is: '
                                                      + str(int(len(train_data) / 96)) + 'days; missing data rate: '
                                                      + str(round(loss_rate * 100, 2)) + '%']).reshape(1, -1))
                logs.info(str(station_id) + '测试时段内实际数据量为：' + str(int(len(train_data) / 96)) + '天；数据缺失率为；'
                          + str(round(loss_rate * 100, 2)) + '%')
                timefile = open(csv_path + 'time.csv', 'a+')
                resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
                ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')
                timefile.close()
                resultfile.close()
                shortfile.close()
                ultrafile.close()
                if len(train_data) / 96 < 16:
                    logs.warning(str(station_id) + '场站在该时段可用数据不满足测试要求，该场站退出测试！！！')
                    continue
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.warning('写CSV失败！')

        c.execute("select config_value from sys_config where config_key = 'task1_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            logs.info('--------------------------------任务1：仅集成学习：特征工程×，参数寻优×，集成学习√--------------------------------')
            logs.info('串行任务：仅集成学习 开始')
            try:
                dfcsv = pandas.DataFrame(['task1: only ensemble: feature N, parameter N, ensemble Y'])
                timefile = open(csv_path + 'time.csv', 'a+')
                resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
                ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')

                c.execute("select * from default_feature_and_parameters;")
                db.commit()
                des = c.description
                record = c.fetchall()
                coul = list(iterm[0] for iterm in des)
                dataframe_default = pandas.DataFrame(record, columns=coul)

                result = c.execute("select DISTINCT type from configure where id = %s;", station_id)
                db.commit()
                record = c.fetchall()

                if result != 0:
                    if record[0][0] == 'wind':
                        wind_ultra_short_usecols_default = eval(
                            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"][
                                'usecols'].values[0])
                        dfcsv = pandas.DataFrame(
                            ['default feature for ultra_short_wind: ' + str(wind_ultra_short_usecols_default)])
                        logs.info('风电超短期默认特征：' + str(wind_ultra_short_usecols_default))
                        dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                        wind_short_usecols_default = eval(
                            dataframe_default.loc[dataframe_default['term_type'] == "short_wind"]['usecols'].values[0])
                        dfcsv = pandas.DataFrame(
                            ['default feature for short_wind: ' + str(wind_short_usecols_default)])
                        logs.info('风电短期默认特征：' + str(wind_short_usecols_default))
                        dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                    else:
                        solar_ultra_short_usecols_default = eval(
                            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"][
                                'usecols'].values[0])
                        dfcsv = pandas.DataFrame(
                            ['default feature for ultra_short_solar: ' + str(solar_ultra_short_usecols_default)])
                        logs.info('光伏超短期默认特征：' + str(solar_ultra_short_usecols_default))
                        dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                        solar_short_usecols_default = eval(
                            dataframe_default.loc[dataframe_default['term_type'] == "short_solar"]['usecols'].values[0])
                        dfcsv = pandas.DataFrame(
                            ['default feature for short_solar: ' + str(solar_short_usecols_default)])
                        logs.info('光伏短期默认特征：' + str(solar_short_usecols_default))
                        dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')

                timefile.close()
                resultfile.close()
                shortfile.close()
                ultrafile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

            logs.info('串行任务：仅集成学习 开始')
            # 状态初始化
            c.execute("TRUNCATE TABLE parallel_tasks_station")
            db.commit()
            c.execute("INSERT INTO parallel_tasks_station (station_id, task_name, task_status)"
                      " VALUES (%s, %s, %s)", tuple([station_id, 'ensemble', str(station_id) + '集成学习开始']))
            db.commit()
            # 清除结果
            c.execute("select * from best_feature_parameters_and_model limit 1;")
            db.commit()
            des = c.description
            coul = list(iterm[0] for iterm in des)
            table_item_list = coul[4:-4]
            for table_item in table_item_list:
                c.execute("update best_feature_parameters_and_model set " + table_item + " = NULL where id = %s;",
                          station_id)
                db.commit()

            # 开始集成学习
            c.execute('TRUNCATE TABLE parallel_tasks;')
            db.commit()
            ensemble_learn(host=host, user=user, password=password, database=database, charset=charset, port=port,
                           start_time=start_time, end_time=end_time, rate=rate,
                           scheduler=scheduler, executor_name=executor_name, task=1, station_id=station_id,
                           csvrecord=csvrecord, figurerecord=figurerecord, csv_path=csv_path, figure_path=figure_path)

            c.execute("select * from best_feature_parameters_and_model limit 1;")
            db.commit()
            des = c.description
            coul = list(iterm[0] for iterm in des)
            a = ','.join(iterm for iterm in coul[1:])

            c.execute("INSERT INTO best_feature_parameters_and_model_ensemble (" + a + ")" +
                      " select " + a + " from best_feature_parameters_and_model where id=%s;", station_id)
            db.commit()

            path_name = 'task1_ensemble'
            try:
                if os.path.exists(model_savepath + path_name + '/' + str(station_id)):
                    shutil.rmtree(model_savepath + path_name + '/' + str(station_id))

                shutil.copytree(model_savepath + str(station_id),
                                model_savepath + path_name + '/' + str(station_id))
                shutil.rmtree(model_savepath + str(station_id))
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning(path_name + ' 训练好的模型保存时失败！！！')

            logs.info('串行任务：仅集成学习 完成')
        else:
            logs.info('任务1：仅集成学习 未参与执行！')

        c.execute("select config_value from sys_config where config_key = 'task2_feature_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            logs.info('-----------------------------任务2：特征工程+集成学习：特征工程√，参数寻优×，集成学习√-------------------------------')
            logs.info('串行任务：特征工程+集成学习 开始')
            try:
                dfcsv = pandas.DataFrame(['task2: feature and ensemble: feature Y, parameter N, ensemble Y'])
                timefile = open(csv_path + 'time.csv', 'a+')
                resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
                ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')
                timefile.close()
                resultfile.close()
                shortfile.close()
                ultrafile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')
            # 状态初始化
            c.execute("TRUNCATE TABLE parallel_tasks_station")
            c.execute("INSERT INTO parallel_tasks_station (station_id, task_name, task_status)"
                      " VALUES (%s, %s, %s)", tuple([station_id, 'feature', str(station_id) + '特征工程开始']))
            c.execute("INSERT INTO parallel_tasks_station (station_id, task_name, task_status)"
                      " VALUES (%s, %s, %s)", tuple([station_id, 'ensemble', str(station_id) + '集成学习开始']))
            db.commit()
            # 清除结果
            c.execute("select * from best_feature_parameters_and_model limit 1;")
            db.commit()
            des = c.description
            coul = list(iterm[0] for iterm in des)
            table_item_list = coul[4:-4]
            for table_item in table_item_list:
                c.execute("update best_feature_parameters_and_model set " + table_item + " = NULL where id = %s;",
                          station_id)
                db.commit()

            # 开始特征工程
            c.execute('TRUNCATE TABLE parallel_tasks;')
            db.commit()
            feature_select(host=host, user=user, password=password, database=database, charset=charset, port=port,
                           start_time=start_time, end_time=end_time,
                           scheduler=scheduler, executor_name=executor_name, station_id=station_id,
                           csvrecord=csvrecord, task=2, csv_path=csv_path)

            # 开始集成学习
            c.execute('TRUNCATE TABLE parallel_tasks;')
            db.commit()
            ensemble_learn(host=host, user=user, password=password, database=database, charset=charset, port=port,
                           start_time=start_time, end_time=end_time, rate=rate,
                           scheduler=scheduler, executor_name=executor_name, task=2, station_id=station_id,
                           csvrecord=csvrecord, figurerecord=figurerecord, csv_path=csv_path, figure_path=figure_path)

            c.execute("select * from best_feature_parameters_and_model limit 1;")
            db.commit()
            des = c.description
            coul = list(iterm[0] for iterm in des)
            a = ','.join(iterm for iterm in coul[1:])

            c.execute("INSERT INTO best_feature_parameters_and_model_feature_ensemble (" + a + ")" +
                      " select " + a + " from best_feature_parameters_and_model where id=%s;", station_id)
            db.commit()

            path_name = 'task2_feature_ensemble'
            try:
                if os.path.exists(model_savepath + path_name + '/' + str(station_id)):
                    shutil.rmtree(model_savepath + path_name + '/' + str(station_id))
                    os.makedirs(model_savepath + path_name + '/' + str(station_id))
                shutil.copytree(model_savepath + str(station_id),
                                model_savepath + path_name + '/' + str(station_id))
                shutil.rmtree(model_savepath + str(station_id))
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning(path_name + ' 训练好的模型保存时失败！！！')

            logs.info('串行任务：特征工程+集成学习 完成')
        else:
            logs.info('任务2：特征工程+集成学习 未参与执行！')

        # --------------------------------------------------------------------------------------------------------------
        try:
            c.execute("select * from best_feature_parameters_and_model limit 1;")
            db.commit()
            des = c.description
            coul = list(iterm[0] for iterm in des)
            a = ','.join(iterm for iterm in coul[1:])

            task_name_num_cluster = ['task1_ensemble', 'task2_feature_ensemble']

            for predict_term in ['short', 'ultra_short']:
                list_model_accuracy = []
                for task_name_num in task_name_num_cluster:
                    task_num, task_name = task_name_num.split('_', 1)
                    result = c.execute(
                        "select best_model, best_model_accuracy" +
                        " from best_feature_parameters_and_model_" + task_name +
                        " where id = %s and predict_term=%s;", (station_id, predict_term))
                    db.commit()
                    record = c.fetchall()
                    if result == 0 or record[0][1] is None:
                        list_model_accuracy.append([station_id, task_name, predict_term, None, 0])
                    else:
                        list_model_accuracy.append([station_id, task_name, predict_term, record[0][0],
                                                    sum(eval('[' + record[0][1].replace('%', '') + ']'))])

                nd_model_accuracy = numpy.array(list_model_accuracy)
                max_where = nd_model_accuracy[:, -1].argmax()
                if nd_model_accuracy[max_where, -1] == 0:
                    logs.warning(str(station_id) + predict_term + '未寻得最优模型！！！')
                    continue
                else:
                    task_name = nd_model_accuracy[max_where, 1]
                    task_name_num = task_name_num_cluster[max_where]

                    c.execute("DELETE FROM best_feature_parameters_and_model where id=%s and predict_term=%s;",
                              (station_id, predict_term))
                    db.commit()

                    c.execute("INSERT INTO best_feature_parameters_and_model (" + a + ")" +
                              " select " + a + " from best_feature_parameters_and_model_" + task_name +
                              " where id=%s and predict_term=%s;", (station_id, predict_term))
                    db.commit()

                    if os.path.exists(model_savepath + str(station_id) + '/' + predict_term):
                        shutil.rmtree(model_savepath + str(station_id) + '/' + predict_term)

                    shutil.copytree(model_savepath + '/' + task_name_num + '/' + str(station_id) + '/' + predict_term,
                                    model_savepath + str(station_id) + '/' + predict_term)
        except Exception as err:
            logs.error(str(err), exc_info=True)
            db.rollback()
            logs.warning('筛选最优模型失败')

        c.execute("update configure set operation_time = %s where id = %s;", (now_time, station_id))
        db.commit()
    c.close()
    db.close()
    logs.info('测试结束')


@catch_exception("短期预测错误：")
def run_short_term_predict(host, user, password, database, charset, port, use_cols_of_conf,
                           conf_table, predict_tag, model_table, scheduler, executor_name):
    model = MultiPlantPredict(host, user, password, database, charset, port, use_cols_of_conf,
                              conf_table, predict_tag, model_table)
    model.get_configure()
    model.init_plants_for_prediction()
    model.short_term_predict(predict_start_time=None, enable_interval_predict=True, scheduler=scheduler)


@catch_exception("超短期预测错误：")
def run_ultra_short_term_predict(host, user, password, database, charset, port, use_cols_of_conf,
                                 conf_table, predict_tag, model_table, scheduler, executor_name):
    model = MultiPlantPredict(host, user, password, database, charset, port, use_cols_of_conf,
                              conf_table, predict_tag, model_table)
    model.get_configure()
    model.init_plants_for_prediction()
    model.ultra_short_term_predict(predict_start_time=None, enable_interval_predict=True, scheduler=scheduler)


@catch_exception("中期预测错误：")
def run_medium_term_predict(host, user, password, database, charset, port, use_cols_of_conf,
                            conf_table, predict_tag, model_table, scheduler, executor_name):
    model = MultiPlantPredict(host, user, password, database, charset, port, use_cols_of_conf,
                              conf_table, predict_tag, model_table)
    model.get_configure()
    model.init_plants_for_prediction()
    model.medium_term_predict(predict_start_time=None, enable_interval_predict=True, scheduler=scheduler)


def init_common(*args, **kwargs):
    if len(args):
        main_conf = args[0].get("main", {})
        log_conf = args[0].get("log", {})
    else:
        main_conf = kwargs.get("main", {})
        log_conf = kwargs.get("log", {})

    init_logger(**log_conf)
    proc_title = main_conf.get("proc_title", None)
    if proc_title:
        if not log_conf.get("main_process", True):
            setproctitle.setproctitle("%s worker" % proc_title)

    # 重定向标准文件描述符
    class Logger(object):
        def __init__(self, filename='default.log', stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()
            os.fsync(self.log.fileno())

        def close(self):
            # self.terminal.close()
            self.log.close()

    if platform.system() == "Windows":
        stdout_file = log_conf.get("stdout_file", "/dev/null")
        stderr_file = log_conf.get("stderr_file", "/dev/null")
        if stdout_file and stdout_file != '/dev/null':
            os.makedirs(os.path.dirname(stdout_file), exist_ok=True)
            sys.stdout = Logger(stdout_file, sys.stdout)
        if stderr_file and stderr_file != '/dev/null':
            os.makedirs(os.path.dirname(stderr_file), exist_ok=True)
            sys.stderr = Logger(stderr_file, sys.stderr)

    # 设置apscheduler日志级别
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.processpool").setLevel(logging.WARNING)


class AppDaemon(Daemon):
    def __init__(self, name, pid_file="./daemon_class.pid", home_dir='.', umask=0o22,
                 stdout_file="/dev/null", stderr_file="/dev/null"):
        Daemon.__init__(self, name, pid_file, home_dir, umask, stdout_file, stderr_file)
        self.scheduler = None

    def init(self, *args, **kwargs):
        """
        完成程序启动时，正式开始处理业务之前的初始化操作
        :return:
        """
        # 初始化日志、进程名等通用设置
        init_common(**kwargs)

        # 初始化任务调度器
        self.init_schedule(**kwargs)

    def init_schedule(self, **kwargs):
        # 任务储存器
        jobstores = {
            'default': MemoryJobStore(),  # 默认是内存任务储存器
        }

        scheduler_conf = kwargs.get("scheduler", {})

        log_settings = {
            "main_process": False,
        }
        log_settings.update(kwargs.get("log", {}))
        init_common_conf = dict(log=log_settings, main=kwargs.get("main", {}))
        # 进程池ProcessPoolExecutor初始化参数（实际底层用于初始化concurrent.futures.ProcessPoolExecutor）
        process_pool_kwargs = {
            # 参数mp_context可以是一个多进程上下文或是 None。 它将被用来启动工作进程。
            # 如果 mp_context 为 None 或未给出，将使用默认的多进程上下文
            # "mp_context": None,
            # 参数initializer可以是一个函数或方法或None，在启用进程前会调用这个函数，默认值为None
            # "initializer": None,
            # 对进程池执行器中的所有进程重新进行日志初始化。使用QueueHandler，子进程将日志发往队列，主进程将日志持久化到磁盘文件
            "initializer": init_common,
            # 参数initargs表示以元组的方式给initializer中的函数传递参数，默认为空元组
            "initargs": (init_common_conf,),
        }
        # 执行器
        executors = {
            'default': ThreadPoolExecutor(scheduler_conf.get("thread_pool_size", 20)),  # 默认是线程执行器，最大线程20个
            # 名称“processpool”的进程执行器
            'processpool': ProcessPoolExecutor(scheduler_conf.get("process_pool_size", 10), process_pool_kwargs),
        }

        job_defaults = {
            # 是否为任务开启合并模式
            # True:是，当由于某种原因导致某个 job 积攒了好几次没有实际运行下次这个 job 被 submit 给 executor 时，只会执行 1 次
            # False:否，当由于某种原因导致某个 job 积攒了好几次没有实际运行下次这个 job 被 submit 给 executor 时，会结合参数
            #       misfire_grace_time来决定运行次数
            'coalesce': True,
            # misfire_grace_time：单位为秒，当某一 job 被调度时刚好线程池都被占满，调度器会选择将该 job 排队不运行，
            #                     misfire_grace_time 参数则是在线程池有可用线程时会比对该 job 的应调度时间跟当前时间的差值，
            #                     如果差值小于 misfire_grace_time 时，调度器会再次调度该 job；
            #                     反之该 job 的执行状态为 EVENTJOBMISSED 了，即错过运行
            #                     若设置为None时，则不进行时间差值比较，允许job被调度器调度
            'misfire_grace_time': None,
            'max_instances': scheduler_conf.get("job_max_instances", 5),  # 设置新任务的默认最大实例数为5
        }

        self.scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors,
                                             job_defaults=job_defaults, timezone='Asia/Shanghai')

    def set_startup_schedule_job(self, **kwargs):
        """
        设置程序启动时需执行一次的任务/服务
        """
        database_conf = kwargs.get("database", {})
        startup_conf = kwargs.get("execute_once_at_startup", {})
        if startup_conf.get("enable_create_table", False):
            self.scheduler.add_job(create_table, 'date', executor="default",
                                   kwargs={
                                       'host': database_conf.get('host', None),
                                       'user': database_conf.get('user', None),
                                       'password': database_conf.get('password', None),
                                       'database': database_conf.get('database', None),
                                       'charset': database_conf.get('charset', None),
                                       'port': database_conf.get('port', None)}
                                   )

    def set_normal_schedule_job(self, **kwargs):
        """
        设置常规的定时任务。
        即固定的，不需要根据外部条件都来判断是否需要执行的，必定执行的定时任务
        在自测时，可在本方法内，参考如下代码自行添加相应的测试代码
        self.scheduler.add_job(predict_short_power, 'interval', seconds=1, executor="default")
        此语句表示每隔1秒执行一次predict_short_power方法。测试自己的任务时，只要把predict_short_power改成自己的定时任务方法名即可
        :return: 无
        """
        # 对于耗时较短的定时任务，可以由线程池执行器来执行定时任务
        # 对于耗时较长或计算量较大（CPU密集型）的定时任务，建议由进程池执行器来执行定时任务
        database_conf = kwargs.get("database", {})

        # 根据配置项配置，模型训练任务：包括集成学习、特征工程+集成学习2向任务，串行
        model_training = kwargs.get("job_model_training", {})
        if model_training.get("enable", True):
            self.scheduler.add_job(run_model_training, executor="default",
                                   **model_training.get("trigger_args", {}),
                                   kwargs={
                                       'scheduler': self.scheduler,
                                       'executor_name': "processpool",
                                       'host': database_conf.get('host', None),
                                       'user': database_conf.get('user', None),
                                       'password': database_conf.get('password', None),
                                       'database': database_conf.get('database', None),
                                       'charset': database_conf.get('charset', None),
                                       'port': database_conf.get('port', None)}
                                   )

        # 根据配置项配置，设置中期预测定时任务
        medium_predict = kwargs.get("job_predict_medium_power", {})
        if medium_predict.get("enable", True):
            self.scheduler.add_job(run_medium_term_predict, executor="default",
                                   **medium_predict.get("triggers_args", {}),
                                   kwargs={
                                       'scheduler': self.scheduler,
                                       'executor_name': "processpool",
                                       'host': database_conf.get('host', None),
                                       'user': database_conf.get('user', None),
                                       'password': database_conf.get('password', None),
                                       'database': database_conf.get('database', None),
                                       'charset': database_conf.get('charset', None),
                                       'port': database_conf.get('port', None),
                                       'use_cols_of_conf': database_conf.get('use_cols_of_conf', None),
                                       'conf_table': database_conf.get('conf_table', None),
                                       'predict_tag': database_conf.get('predict_tag', None),
                                       'model_table': database_conf.get('model_table', None)
                                   })

        # 根据配置项配置，设置短期预测定时任务
        short_predict = kwargs.get("job_predict_short_power", {})
        if short_predict.get("enable", True):
            self.scheduler.add_job(run_short_term_predict, executor="default",
                                   **short_predict.get("triggers_args", {}),
                                   kwargs={
                                       'scheduler': self.scheduler,
                                       'executor_name': "processpool",
                                       'host': database_conf.get('host', None),
                                       'user': database_conf.get('user', None),
                                       'password': database_conf.get('password', None),
                                       'database': database_conf.get('database', None),
                                       'charset': database_conf.get('charset', None),
                                       'port': database_conf.get('port', None),
                                       'use_cols_of_conf': database_conf.get('use_cols_of_conf', None),
                                       'conf_table': database_conf.get('conf_table', None),
                                       'predict_tag': database_conf.get('predict_tag', None),
                                       'model_table': database_conf.get('model_table', None)
                                   })

        # 根据配置项配置，设置超短期预测定时任务
        ultra_short_predict = kwargs.get("job_predict_ultra_short_power", {})
        if ultra_short_predict.get("enable", True):
            self.scheduler.add_job(run_ultra_short_term_predict, executor="default",
                                   **ultra_short_predict.get("triggers_args", {}),
                                   kwargs={
                                       'scheduler': self.scheduler,
                                       'executor_name': "processpool",
                                       'host': database_conf.get('host', None),
                                       'user': database_conf.get('user', None),
                                       'password': database_conf.get('password', None),
                                       'database': database_conf.get('database', None),
                                       'charset': database_conf.get('charset', None),
                                       'port': database_conf.get('port', None),
                                       'use_cols_of_conf': database_conf.get('use_cols_of_conf', None),
                                       'conf_table': database_conf.get('conf_table', None),
                                       'predict_tag': database_conf.get('predict_tag', None),
                                       'model_table': database_conf.get('model_table', None)
                                   })

        # 根据配置项配置，设置迁移学习定时扫描任务
        transfer_learning = kwargs.get("job_transfer_learning", {})
        if transfer_learning.get("enable", True):
            self.scheduler.add_job(run_transfer_learning, executor="default",
                                   **transfer_learning.get("triggers_args", {}),
                                   kwargs={
                                       'host': database_conf.get('host', None),
                                       'user': database_conf.get('user', None),
                                       'password': database_conf.get('password', None),
                                       'database': database_conf.get('database', None),
                                       'charset': database_conf.get('charset', None),
                                       'port': database_conf.get('port', None)
                                   })

    def run(self, *args, **kwargs):
        proc_title = kwargs.get("main", {}).get("proc_title", None)
        if platform.system() == "Linux":
            # 设置多进程日志队列。队列只能在主进程中配置，且需在其他初始化工作之前完成配置设置
            if "log" in kwargs and "logs_queue" not in kwargs["log"]:
                if proc_title:
                    setproctitle.setproctitle("%s log queue worker" % proc_title)
                # 派生一个新的队列代理进程，用于维持多进程日志队列
                kwargs["log"]["logs_queue"] = multiprocessing.Manager().Queue()
        if proc_title:
            setproctitle.setproctitle("%s master" % proc_title)

        self.init(*args, **kwargs)
        logs.info("main process starts to set cron jobs and run....")
        self.set_startup_schedule_job(**kwargs)
        self.set_normal_schedule_job(**kwargs)
        self.scheduler.start()
        while self.alive:
            # 可处理某些非阻塞、短耗时的事件，一般是处理状态监控、健康检测等与具体业务无关的事宜
            logs.debug("main thread running...")
            time.sleep(1)
        # 关闭调度器
        self.scheduler.shutdown(wait=True)
        # logs.info("process stop...")
        logs.close()


if __name__ == '__main__':
    import multiprocessing

    if platform.system() != "Windows":
        init_settings("./configure/conf_hn.ini")
    else:
        multiprocessing.freeze_support()
        logs_queue = multiprocessing.Manager().Queue()
        update_settings = {
            "log": {"logs_queue": logs_queue},
        }

        init_settings("./configure/conf_hn.ini", update=update_settings)
    help_msg = 'Usage: python3 %s <start|stop|restart|status> processName' % sys.argv[0]

    opt = "start"
    if len(sys.argv) > 1:
        opt = sys.argv[1]
    elif platform.system() != "Windows":
        print(help_msg)
        sys.exit(0)

    p_name = settings.get("main", {}).get("proc_title", None)
    pid_fn = settings.get("main", {}).get("pid_file", './work_dir/data/daemon_class.pid')  # 守护进程pid文件的绝对路径
    if len(sys.argv) > 2:
        p_name = sys.argv[2]  # 守护进程名称

    daemon = AppDaemon(p_name, pid_fn,
                       stdout_file=settings.get("log", {}).get("stdout_file", "/dev/null"),
                       stderr_file=settings.get("log", {}).get("stderr_file", "/dev/null")
                       )

    if opt == 'start':
        # 此行代码在linux环境中，会使程序进入后台运行模式（守护模式），即使退出终端也不影响程序继续运行
        # 若对源码进行修改，屏蔽了此方法，但仍需继续保留“退出终端不影响程序继续运行”功能时，则需另行处理
        # 如使用 nohup 命令启动程序，参考命令为：nohup python3 main.py start   > nohup.log 2>&1 &
        daemon.start(**settings)
    elif opt == 'stop':
        daemon.stop()
    elif opt == 'restart':
        daemon.restart()
    elif opt == 'status':
        alive = daemon.is_running()
        if alive:
            print('process [%s] is running ......' % daemon.get_pid())
        else:
            print('daemon process [%s] stopped' % daemon.name)
    else:
        print('invalid argument!')
        print(help_msg)

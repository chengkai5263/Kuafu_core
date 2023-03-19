# _*_ coding: utf-8 _*_

from app.feature_selection import feature_selection
from common.settings import settings
import pandas
from task.mysql.model_evaluation import evaluate_for_station
import numpy
import time
from common.tools import save_model
import datetime
import pymysql
import math
from common.logger import logs
import os
from task.mysql.load_train_data import LoadTraindata
from common.tools import load_model
from common import data_preprocess
from common import training
from common.tools import catch_exception
import shutil


@catch_exception("run_feature_select error: ")
def run_feature_select(host, user, password, database, charset, port,
                       scheduler=None, executor_name=None, feature_engineering_stands=None,
                       csvrecord=False, task=2):
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    c.execute("select id from configure where station_status = 2;")
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

        result = c.execute("select start_time, end_time from configure where id = %s;", station_id)
        db.commit()
        if result == 0:
            logs.warning(str(station_id) + 'configure表中没有场站记录，退出特征工程!')
            continue
        else:
            record = c.fetchall()
            if record[0][0] is None or record[0][1] is None:
                logs.warning(str(station_id) + 'configure表中没有设置起始时间，退出特征工程!')
                continue
            else:
                start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record[0][0].timestamp()))
                end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record[0][1].timestamp()))

        result = c.execute("select station_id, task_name from parallel_tasks_station"
                           " where station_id = %s and task_name = %s;", (station_id, 'feature'))
        db.commit()
        if result == 0:
            c.execute("INSERT INTO parallel_tasks_station (station_id, task_name)"
                      " VALUES (%s, %s)", tuple([station_id, 'feature']))
            db.commit()

        c.execute("update parallel_tasks_station set task_status = %s where station_id = %s and task_name = %s;",
                  (str(station_id) + '特征工程开始', station_id, 'feature'))
        db.commit()

        feature_select(host, user, password, database, charset, port, start_time=start_time, end_time=end_time,
                       scheduler=scheduler, executor_name=executor_name, station_id=station_id,
                       feature_engineering_stands=feature_engineering_stands,
                       csvrecord=csvrecord, task=task)
    c.close()
    db.close()


@catch_exception("feature_select error: ")
def feature_select(host, user, password, database, charset, port, start_time=None, end_time=None,
                   scheduler=None, executor_name=None, station_id=None, feature_engineering_stands=None,
                   csvrecord=False, task=2, csv_path=None):
    """
    参数寻优
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param start_time: 集成学习开始时间
    :param end_time: 集成学习结束时间
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :param station_id: 场站id
    :param feature_engineering_stands: 评价标准
    :param csvrecord: 是否写csv
    :param task: 任务序号
    :param csv_path: csv文件保存地址
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    result = c.execute(
        "update parallel_tasks_station set task_status = '" + str(
            station_id) + "正在进行特征工程' where task_name = 'feature' and task_status = '" + str(
                        station_id) + "特征工程开始' and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        logs.info('特征工程开始')
        function_start_time = datetime.datetime.now().timestamp()
        # 如果当前是task4，且task2已经做了，且task2保存有结果，则不用再做特征工程了,直接用task2特征工程的结果
        copy_task2_result = False
        if task == 4:  # 判断当前是task4
            c.execute("select config_value from sys_config where config_key = 'task2_feature_ensemble';")
            db.commit()
            record = c.fetchall()
            do_task = record[0][0]
            if do_task == '1':  # 判断task2已经做了
                result_short = c.execute("select usecols from best_feature_parameters_and_model_feature_ensemble"
                                         " where id = %s and predict_term = %s;", (station_id, 'short'))
                db.commit()
                record_short = c.fetchall()
                result_ultra_short = c.execute(
                    "select usecols from best_feature_parameters_and_model_feature_ensemble"
                    " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
                db.commit()
                record_ultra_short = c.fetchall()
                if result_short == 0 or result_ultra_short == 0 or record_short[0][0] is None or \
                        len(record_short[0][0]) == 0 or record_ultra_short[0][0] is None or \
                        len(record_ultra_short[0][0]) == 0:
                    logs.info("当前为任务4，任务2已执行，但是结果保存不完整，无法调用任务2特征工程结果，仍需要做特征工程")
                    copy_task2_result = False
                else:
                    logs.info("当前为任务4，任务2已执行，且结果保存完整，不需要做特征工程,直接用任务2特征工程的结果")
                    copy_task2_result = True
            else:
                logs.info("当前为任务4，但任务2没有执行，无法调用任务2特征工程结果，仍需要做特征工程")
                copy_task2_result = False
        else:
            logs.debug("当前不是任务4，没有调用任务2结果的需求，仍需要做特征工程")
            copy_task2_result = False

        c.execute("select * from configure where id = %s;", station_id)
        db.commit()
        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        dataframe_config = pandas.DataFrame(record, columns=coul)

        c.execute("select * from default_feature_and_parameters;")
        db.commit()
        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        dataframe_default = pandas.DataFrame(record, columns=coul)

        wind_ultra_short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"]['usecols'].values[0])
        wind_short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_wind"]['usecols'].values[0])
        solar_ultra_short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"]['usecols'].values[0])
        solar_short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_solar"]['usecols'].values[0])

        # 配置信息
        if dataframe_config.loc[:, "type"][0] == 'wind':
            config_cluster = {station_id: {"id": dataframe_config.loc[:, 'id'][0],
                                           "name": dataframe_config.loc[:, 'name'][0],
                                           "type": dataframe_config.loc[:, "type"][0],
                                           "sr_col": dataframe_config.loc[:, "sr_col"][0],
                                           "online_capacity": dataframe_config.loc[:, "capacity"][0],
                                           "ultra_short_usecols": wind_ultra_short_usecols_default,
                                           "short_usecols": wind_short_usecols_default,
                                           "model_savepath": dataframe_config.loc[:, "model_savepath"][0]}}

        else:
            config_cluster = {station_id: {"id": dataframe_config.loc[:, 'id'][0],
                                           "name": dataframe_config.loc[:, 'name'][0],
                                           "type": dataframe_config.loc[:, "type"][0],
                                           "sr_col": dataframe_config.loc[:, "sr_col"][0],
                                           "online_capacity": dataframe_config.loc[:, "capacity"][0],
                                           "ultra_short_usecols": solar_ultra_short_usecols_default,
                                           "short_usecols": solar_short_usecols_default,
                                           "model_savepath": dataframe_config.loc[:, "model_savepath"][0]}}

        result = c.execute("select usecols from best_feature_parameters_and_model"
                           " where predict_term = 'short' and id=%s;", station_id)
        db.commit()
        if result > 0:
            record = c.fetchall()
            if record[0][0] is not None and len(record[0][0]) > 0:
                config_cluster[station_id]['short_usecols'] = eval(record[0][0])

        result = c.execute("select usecols from best_feature_parameters_and_model"
                           " where predict_term = 'ultra_short' and id=%s;", station_id)
        db.commit()
        if result > 0:
            record = c.fetchall()
            if record[0][0] is not None and len(record[0][0]) > 0:
                config_cluster[station_id]['ultra_short_usecols'] = eval(record[0][0])

        if copy_task2_result:
            c.execute("select usecols from best_feature_parameters_and_model_feature_ensemble"
                      " where id = %s and predict_term = %s;", (station_id, 'short'))
            db.commit()
            record_short = c.fetchall()
            short_best_feature = eval(record_short[0][0])

            c.execute("select usecols from best_feature_parameters_and_model_feature_ensemble"
                      " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
            db.commit()
            record_ultra_short = c.fetchall()

            ultra_short_best_feature = eval(record_ultra_short[0][0])
        else:
            if feature_engineering_stands is None:
                c.execute("select config_value from sys_config where config_key = 'feature_engineering_stands';")
                db.commit()
                record = c.fetchall()
                feature_engineering_stands = record[0][0]
            if feature_engineering_stands == 'GB':
                evaluation_method = 'GB'
            else:
                evaluation_method = 'Two_Detail'
            # ----------------------------------------------------------------------------------------------------------
            all_cols = feature_selection.get_all_features(config_cluster[station_id]['type'])
            train_data_load = LoadTraindata()
            train_feature_ultra_short, train_target_ultra_short = \
                train_data_load.load_train_data_for_ultra_short_sql(host=host, user=user, password=password,
                                                                    database=database, charset=charset, port=port,
                                                                    config=config_cluster[station_id], usecols=all_cols,
                                                                    predict_type=config_cluster[station_id]['type'],
                                                                    start_time=start_time, end_time=end_time, rate=1)
            if config_cluster[station_id]['type'] == 'solar':
                train_feature_ultra_short = pandas.DataFrame(train_feature_ultra_short, columns=all_cols + ['time'])
            else:
                train_feature_ultra_short = pandas.DataFrame(train_feature_ultra_short, columns=all_cols)
            train_feature_short, train_target_short = \
                train_data_load.load_train_data_for_short_sql(host=host, user=user, password=password,
                                                              database=database, charset=charset, port=port,
                                                              config=config_cluster[station_id], usecols=all_cols,
                                                              predict_type=config_cluster[station_id]['type'],
                                                              start_time=start_time, end_time=end_time, rate=1)
            if config_cluster[station_id]['type'] == 'solar':
                train_feature_short = pandas.DataFrame(train_feature_short, columns=all_cols + ['time'])
            else:
                train_feature_short = pandas.DataFrame(train_feature_short, columns=all_cols)

            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            ultra_short_best_feature, short_best_feature = feature_engineering(
                host=host, user=user, password=password, database=database,
                charset=charset, port=port, start_time=start_time, end_time=end_time,
                config=config_cluster[station_id], feature_type=config_cluster[station_id]['type'],
                rf_threshold=0.04, corr_threshold=0.3,
                train_feature_ultra_short=train_feature_ultra_short, train_target_ultra_short=train_target_ultra_short,
                train_feature_short=train_feature_short, train_target_short=train_target_short,
                evaluation_method=evaluation_method, scheduler=scheduler,
                executor_name=executor_name)
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return

        result = c.execute(
            "select id, predict_term from best_feature_parameters_and_model where id = %s and predict_term = %s;",
            (station_id, 'short'))
        db.commit()
        if result == 0:
            c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                      " VALUES (%s, %s, %s)", tuple([station_id, 'short', config_cluster[station_id]['name']]))
            db.commit()

        result = c.execute(
            "select id, predict_term from best_feature_parameters_and_model where id = %s and predict_term = %s;",
            (station_id, 'ultra_short'))
        db.commit()
        if result == 0:
            c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                      " VALUES (%s, %s, %s)", tuple([station_id, 'ultra_short', config_cluster[station_id]['name']]))
            db.commit()

        logs.info(str(config_cluster[station_id]['name']) + '超短期最优特征:' + str(ultra_short_best_feature))
        logs.info(str(config_cluster[station_id]['name']) + '短期最优特征:' + str(short_best_feature))
        c.execute("update best_feature_parameters_and_model set usecols = %s where id = %s and predict_term = %s;",
                  (str(ultra_short_best_feature), station_id, 'ultra_short'))
        db.commit()
        c.execute("update best_feature_parameters_and_model set usecols = %s where id = %s and predict_term = %s;",
                  (str(short_best_feature), station_id, 'short'))
        db.commit()

        if csvrecord is True:
            try:
                resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                dfcsv = pandas.DataFrame(numpy.array(
                    [str(station_id) + 'ultra_short best feature: ' + str(ultra_short_best_feature)]).reshape(1, -1))
                dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv = pandas.DataFrame(numpy.array(
                    [str(station_id) + 'short best feature: ' + str(short_best_feature)]).reshape(1, -1))
                dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                resultfile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

        # if os.path.exists(str(config_cluster[station_id]['model_savepath']) + 'feature/'):
        #     shutil.rmtree(str(config_cluster[station_id]['model_savepath']) + 'feature/')

        logs.info('特征工程完成')

        function_end_time = datetime.datetime.now().timestamp()
        logs.info('特征工程时间:' + str(int(function_end_time - function_start_time)) + '秒')
        hours = int((function_end_time - function_start_time) // 3600)
        mins = int(((function_end_time - function_start_time) % 3600) // 60)
        second = int((function_end_time - function_start_time) % 60)
        logs.info('特征工程时间:' + str(hours) + '时' + str(mins) + '分' + str(second) + '秒')

        if csvrecord is True:
            try:
                timefile = open(csv_path + 'time.csv', 'a+')
                dfcsv = pandas.DataFrame(numpy.array(['feature time: ' +
                                                      str(int(function_end_time - function_start_time)) +
                                                      's']).reshape(1, -1))
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                timefile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

        c.execute(
            "update parallel_tasks_station set task_status = '" + str(
                station_id) + "特征工程完成' where task_name = 'feature' and task_status = '"+str(
                station_id)+"正在进行特征工程' and station_id = %s;", station_id)
        db.commit()
        time.sleep(3)

    result = c.execute(
        "select task_status from parallel_tasks_station where task_name = 'feature'"
        " and task_status = '" + str(station_id) + "特征工程完成' and station_id = %s;", station_id)
    db.commit()

    while result == 0:
        time.sleep(1)
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        result = c.execute(
            "select task_status from parallel_tasks_station where task_name = 'feature'"
            " and task_status = '" + str(station_id) + "特征工程完成' and station_id = %s;", station_id)
        db.commit()

    c.close()
    db.close()


def combination_feature_filter(host, user, password, database, charset, port, start_time, end_time,
                               best_feature, feature_relevance, feature_pretreatment, config,
                               scheduler, executor_name):
    # 构造准确率集合
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    acr = numpy.zeros(shape=(4, 4))
    i = 0
    for combination_feature in [best_feature, feature_relevance, feature_pretreatment, config['ultra_short_usecols']]:
        # 使用集成学习模块输出国标与两个细则的短期、超短期准确率
        task_stop_sign = task_stop(db, c, config['id'])  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        accuracy_combination = feature_ensemble_learn(host, user, password, database, charset, port,
                                                      start_time=start_time, end_time=end_time, rate=0.75,
                                                      scheduler=scheduler, executor_name=executor_name,
                                                      combination_feature=combination_feature, config=config)
        # 收集在该特征量下的准确率
        acr[i, :] = accuracy_combination
        i = i + 1
    c.close()
    db.close()
    return acr


def combination_feature_filter_forshort(host, user, password, database, charset, port, start_time, end_time,
                                        best_feature, feature_relevance, feature_pretreatment, config, scheduler,
                                        executor_name):

    # 构造准确率集合
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    acr = numpy.zeros(shape=(4, 4))
    i = 0
    for combination_feature in [best_feature, feature_relevance, feature_pretreatment, config['short_usecols']]:
        # 使用集成学习模块输出国标与两个细则的短期、超短期准确率
        task_stop_sign = task_stop(db, c, config['id'])  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        accuracy_combination = feature_ensemble_learn(host, user, password, database, charset, port,
                                                      start_time=start_time, end_time=end_time, rate=0.75,
                                                      scheduler=scheduler, executor_name=executor_name,
                                                      combination_feature=combination_feature, config=config)
        # 收集在该特征量下的准确率
        acr[i, :] = accuracy_combination
        i = i + 1
    c.close()
    db.close()
    return acr


def feature_engineering(host, user, password, database, charset, port, start_time, end_time,
                        config, feature_type='wind', rf_threshold=0.04, corr_threshold=0.3,
                        train_feature_ultra_short=None, train_target_ultra_short=None,
                        train_feature_short=None, train_target_short=None,
                        evaluation_method="GB", scheduler=None, executor_name=None):
    """
    :param feature_type: 加载天气数据类型，'wind'为风电，'solar'为光伏
    :param rf_threshold: 随机森林模型的门槛值，低于门槛值的特征将会被舍弃，默认为0.05
    :param corr_threshold: 相关性分析的门槛值，与功率的相关性低于门槛值的特征将会被舍弃，默认风电为0.45，光伏为0.2
    :return: feature_RFE：经过特征工程筛选得到的特征集，用于进一步训练，类型为list，例如:['UU_hpa_700','VV_hpa_700','WS']
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    if evaluation_method == 'GB':
        logs.info('使用国标评价方法')
    else:
        logs.info('使用两个细则评价方法')
    all_cols = feature_selection.get_all_features(feature_type)
    if feature_type == 'solar':
        train_feature1 = train_feature_ultra_short.loc[:, all_cols + ['time']].values
    else:
        train_feature1 = train_feature_ultra_short.loc[:, all_cols].values
    train_target1 = train_target_ultra_short
    # # 得到相关性最大的前五个特征量,
    feature_relevance = feature_selection.select_feature_by_correlation_withinitialdata(usecols=all_cols,
                                                                                        feature_type=feature_type,
                                                                                        train_feature=train_feature1,
                                                                                        train_target=train_target1)
    if feature_type == 'wind':
        feature_default = ['WS', 'UU_hpa_700', 'WD', 'UU_hpa_500', 'RHU_meter']
    else:
        feature_default = ['SR', 'SWDDIF', 'SWDDIR', 'TCC', 'HCC']
    if feature_type == 'solar':
        train_feature2 = train_feature_ultra_short.loc[:, feature_default + ['time']].values
    else:
        train_feature2 = train_feature_ultra_short.loc[:, feature_default].values
    train_target2 = train_target_ultra_short

    train_feature3, train_target3 = data_preprocess.DataPreprocess.data_preprocess(train_feature2, train_target2,
                                                                                   online_capacity=config[
                                                                                     "online_capacity"],
                                                                                   record_restrict=None)
    train_feature = train_feature1  # 赋值全量NWP
    train_target = train_target3  # 赋值预处理后的功率
    # 随机森林特征选取
    feature_rf = feature_selection.select_feature_by_rf(usecols=all_cols, threshold=rf_threshold,
                                                        feature_type=feature_type, train_feature=train_feature,
                                                        train_target=train_target)
    use_cols = list(set(feature_rf).union(set(feature_relevance)))
    if feature_type == 'wind':
        rfe_feature_num = 5
    else:
        rfe_feature_num = 3
    # 利用预处理后的数据并基于随机森林+递归得到的特征组
    feature_pretreatment = feature_selection.select_feature_by_rfe(all_cols=use_cols, feature_type=feature_type,
                                                                   feature_num=rfe_feature_num,
                                                                   train_feature_ultra_short=train_feature_ultra_short,
                                                                   train_target=train_target)

    # 使用BPNN做组合特征筛选
    task_stop_sign = task_stop(db, c, config["id"])  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    acr_ultra_short = combination_feature_filter(host=host, user=user, password=password, database=database,
                                                 charset=charset, port=port, start_time=start_time, end_time=end_time,
                                                 best_feature=feature_default, feature_relevance=feature_relevance,
                                                 feature_pretreatment=feature_pretreatment, config=config,
                                                 scheduler=scheduler, executor_name=executor_name)

    if evaluation_method == 'GB':
        n = 3
        logs.debug('默认特征：' + str(feature_default) + '，国标超短期准确率：' + str(acr_ultra_short[0, n]))
        logs.debug('相关性最大特征：' + str(feature_relevance) + '，国标超短期准确率：' + str(acr_ultra_short[1, n]))
        logs.debug('随机森林+递归后的特征：' + str(feature_pretreatment) + '，国标超短期准确率：' + str(acr_ultra_short[2, n]))
        logs.debug('数据库特征：' + str(config['ultra_short_usecols']) + '，国标超短期准确率：' + str(acr_ultra_short[3, n]))
    else:
        n = 1
        logs.debug('默认特征：' + str(feature_default) + '，两个细则超短期准确率：' + str(acr_ultra_short[0, n]))
        logs.debug('相关性最大特征：' + str(feature_relevance) + '，两个细则超短期准确率：' + str(acr_ultra_short[1, n]))
        logs.debug('随机森林+递归后的特征：' + str(feature_pretreatment) + '，两个细则超短期准确率：' + str(acr_ultra_short[2, n]))
        logs.debug('数据库特征：' + str(config['ultra_short_usecols']) + '，两个细则超短期准确率：' + str(acr_ultra_short[3, n]))
    # 在四个特征量得到的准确率中选出超短期准确率最高的特征，如果都比较低则选用默认特征
    ultra_short_best_feature = feature_default
    if evaluation_method == "GB":
        if numpy.argmax(acr_ultra_short[:, 3]) == 0:
            ultra_short_best_feature = feature_default
        elif numpy.argmax(acr_ultra_short[:, 3]) == 1:
            ultra_short_best_feature = feature_relevance
        elif numpy.argmax(acr_ultra_short[:, 3]) == 2:
            ultra_short_best_feature = feature_pretreatment
        elif numpy.argmax(acr_ultra_short[:, 3]) == 3:
            ultra_short_best_feature = config['ultra_short_usecols']
    else:
        if numpy.argmax(acr_ultra_short[:, 1]) == 0:
            ultra_short_best_feature = feature_default
        elif numpy.argmax(acr_ultra_short[:, 1]) == 1:
            ultra_short_best_feature = feature_relevance
        elif numpy.argmax(acr_ultra_short[:, 1]) == 2:
            ultra_short_best_feature = feature_pretreatment
        elif numpy.argmax(acr_ultra_short[:, 1]) == 3:
            ultra_short_best_feature = config['ultra_short_usecols']
    # 短期
    # 加载全量NWP和功率，得到全量NWP
    if feature_type == 'solar':
        train_feature1 = train_feature_short.loc[:, all_cols + ['time']].values
    else:
        train_feature1 = train_feature_short.loc[:, all_cols].values
    train_target1 = train_target_short
    task_stop_sign = task_stop(db, c, config["id"])  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    # 得到相关性最大的前五个特征量,
    feature_relevance = feature_selection.select_feature_by_correlation_withinitialdata(usecols=all_cols,
                                                                                        feature_type=feature_type,
                                                                                        train_feature=train_feature1,
                                                                                        train_target=train_target1)

    # 使用BPNN做组合特征筛选
    task_stop_sign = task_stop(db, c, config["id"])  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    acr_short = combination_feature_filter_forshort(host=host, user=user, password=password, database=database,
                                                    charset=charset, port=port, start_time=start_time,
                                                    end_time=end_time, best_feature=feature_default,
                                                    feature_relevance=feature_relevance,
                                                    feature_pretreatment=feature_pretreatment, config=config,
                                                    scheduler=scheduler, executor_name=executor_name)
    if evaluation_method == 'GB':
        n = 2
        logs.debug('默认特征：' + str(feature_default) + '，国标短期准确率：' + str(acr_ultra_short[0, n]))
        logs.debug('相关性最大特征：' + str(feature_relevance) + '，国标短期准确率：' + str(acr_ultra_short[1, n]))
        logs.debug('随机森林+递归后的特征：' + str(feature_pretreatment) + '，国标短期准确率：' + str(acr_ultra_short[2, n]))
        logs.debug('数据库特征：' + str(config['short_usecols']) + '，国标短期准确率：' + str(acr_ultra_short[3, n]))
    else:
        n = 0
        logs.debug('默认特征：' + str(feature_default) + '，两个细则短期准确率：' + str(acr_ultra_short[0, n]))
        logs.debug('相关性最大特征：' + str(feature_relevance) + '，两个细则短期准确率：' + str(acr_ultra_short[1, n]))
        logs.debug('随机森林+递归后的特征：' + str(feature_pretreatment) + '，两个细则短期准确率：' + str(acr_ultra_short[2, n]))
        logs.debug('数据库特征：' + str(config['short_usecols']) + '，两个细则短期准确率：' + str(acr_ultra_short[3, n]))
    # 在四个特征量得到的准确率中选出短期准确率最高的特征，如果都比较低则选用默认特征
    short_best_feature = feature_default
    if evaluation_method == "GB":
        if numpy.argmax(acr_short[:, 2]) == 0:
            short_best_feature = feature_default
        elif numpy.argmax(acr_short[:, 2]) == 1:
            short_best_feature = feature_relevance
        elif numpy.argmax(acr_short[:, 2]) == 2:
            short_best_feature = feature_pretreatment
        elif numpy.argmax(acr_short[:, 2]) == 3:
            short_best_feature = config['short_usecols']
    else:
        if numpy.argmax(acr_short[:, 0]) == 0:
            short_best_feature = feature_default
        elif numpy.argmax(acr_short[:, 0]) == 1:
            short_best_feature = feature_relevance
        elif numpy.argmax(acr_short[:, 0]) == 2:
            short_best_feature = feature_pretreatment
        elif numpy.argmax(acr_short[:, 0]) == 3:
            short_best_feature = config['short_usecols']
    task_stop_sign = task_stop(db, c, config['id'])  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    shutil.rmtree(str(config['model_savepath'] + 'feature/') + str(config['id']))
    c.close()
    db.close()
    return ultra_short_best_feature, short_best_feature


def feature_ensemble_learn(host, user, password, database, charset, port, start_time=None, end_time=None, rate=0.75,
                           scheduler=None, executor_name=None, combination_feature=None, config=None):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param start_time: 集成学习开始时间
    :param end_time: 集成学习结束时间
    :param rate: 训练与测试比例划分
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_stop_sign = task_stop(db, c, config["id"])  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    config_cluster_train = feature_trainning_model(host=host, user=user, password=password, database=database,
                                                   charset=charset, port=port, combination_feature=combination_feature,
                                                   config=config)
    number_end = 0
    number_task = c.execute("select task_name from parallel_tasks"
                            " where task_type = 'data_load_preprocess_feature' and station_id=%s;", config["id"])
    db.commit()
    task_name_last = 'None'
    # 当测试任务全部完成，或者测试时间超过最大限值都会跳出循环
    while number_end < number_task:
        task_stop_sign = task_stop(db, c, config["id"])  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        number_finished = c.execute(
            "select task_name from parallel_tasks where task_status = '2'"
            " and task_type = 'data_load_preprocess_feature' and station_id=%s;", config["id"])
        db.commit()
        number_failed = c.execute(
            "select task_name from parallel_tasks where task_status = '3'"
            " and task_type = 'data_load_preprocess_feature' and station_id=%s;", config["id"])
        db.commit()
        number_end = number_failed + number_finished
        logs.debug('已结束的任务数：' + str(number_end))

        # 判断并分配任务 ----------------------------------------------------------------------------------------------
        result = c.execute(
            "select task_name from parallel_tasks where task_status = '0'"
            " and task_type = 'data_load_preprocess_feature' and station_id=%s limit 1;", config["id"])
        db.commit()
        if result == 0:
            time.sleep(2)
            continue
        record = c.fetchall()
        task_name = record[0][0]
        if task_name == task_name_last:
            time.sleep(2)
        task_name_last = task_name
        result = c.execute(
            "select task_name from parallel_tasks where task_status = '1'"
            " and task_type = 'data_load_preprocess_feature' and station_id=%s;", config["id"])
        db.commit()
        if result < settings['scheduler']['process_pool_size']:
            logs.debug('任务加入：' + task_name)
            station_id, term = task_name.split('_', 1)
            station_id = eval(station_id)

            # 多进程任务方式-------------------------------------------------------------------------------------------
            args_short = [station_id, term, config_cluster_train[station_id], host, user, password,
                          database, charset, port, start_time, end_time, rate]
            scheduler.add_job(data_load_preprocess, executor=executor_name,
                              args=args_short,
                              coalesce=True, misfire_grace_time=None)
            time.sleep(1)

    number_end = 0
    number_task = c.execute("select task_name from parallel_tasks where task_type = 'train_feature'"
                            " and station_id=%s;", config["id"])
    db.commit()
    task_name_last = 'None'
    # 当测试任务全部完成，或者测试时间超过最大限值都会跳出循环
    while number_end < number_task:
        task_stop_sign = task_stop(db, c, config["id"])  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        number_finished = c.execute(
            "select task_name from parallel_tasks where task_status = '2' and task_type = 'train_feature'"
            " and station_id=%s;", config["id"])
        db.commit()
        number_failed = c.execute(
            "select task_name from parallel_tasks where task_status = '3' and task_type = 'train_feature'"
            " and station_id=%s;", config["id"])
        db.commit()
        number_end = number_failed + number_finished
        logs.debug('已结束的任务数：' + str(number_end))

        # 判断并分配任务 ----------------------------------------------------------------------------------------------
        result = c.execute(
            "select task_name from parallel_tasks where task_status = '0' and task_type = 'train_feature'"
            " and station_id=%s limit 1;", config["id"])
        db.commit()
        if result == 0:
            time.sleep(2)
            continue
        record = c.fetchall()
        task_name = record[0][0]
        if task_name == task_name_last:
            time.sleep(2)
        task_name_last = task_name
        result = c.execute(
            "select task_name from parallel_tasks where task_status = '1' and task_type = 'train_feature'"
            " and station_id=%s;", config["id"])
        db.commit()
        if result < settings['scheduler']['process_pool_size']:
            logs.debug('任务加入：' + task_name)
            station, model_name = task_name.split('short_', 1)
            station_id, term = str(station + 'short').split('_', 1)
            station_id = eval(station_id)
            # 多进程任务方式-------------------------------------------------------------------------------------------
            args_short = [station_id, term, model_name, config_cluster_train[station_id], host, user, password,
                          database, charset, port]
            scheduler.add_job(save_a_fitted_model, executor=executor_name,
                              args=args_short,
                              coalesce=True, misfire_grace_time=None)
            time.sleep(1)

    model_name_cluster = ['BPNN']

    c.execute("DELETE FROM parallel_tasks where task_type = 'predict_feature' and station_id=%s;", config["id"])
    db.commit()

    c.execute("CREATE TABLE IF NOT EXISTS `predict_power_" + str(config["id"]) + "_train_feature` ("
              "`id` bigint NOT NULL AUTO_INCREMENT,"
              "`predict_term` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,"
              "`model_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,"
              "`start_time` datetime NOT NULL,"
              "`forecast_time` datetime NOT NULL,"
              "`predict_power` float(10,4) NOT NULL,"
              "PRIMARY KEY (`id`) USING BTREE"
              ");")
    db.commit()

    c.execute('TRUNCATE TABLE predict_power_' + str(config["id"]) + '_train_feature;')
    db.commit()

    # 场站id
    station_id = config["id"]

    # 配置信息
    config_cluster = {config["id"]: {"type": config["type"],
                                     "sr_col": 0,
                                     "online_capacity": config["online_capacity"],
                                     "model_path": config['model_savepath'] + 'feature/'
                                     }}

    # 读取默认特征和默认参数 -------------------------------------------------------------------------------------------
    c.execute("select * from default_feature_and_parameters;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_default = pandas.DataFrame(record, columns=coul)

    # 特征默认值
    wind_ultra_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"]['usecols'].values[0])
    wind_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "short_wind"]['usecols'].values[0])
    solar_ultra_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"]['usecols'].values[0])
    solar_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "short_solar"]['usecols'].values[0])

    # 超短期最优特征，如果不存在，则使用默认值
    if combination_feature == None:
        if config["type"] == 'wind':
            combination_feature = wind_ultra_short_usecols_default
        else:
            combination_feature = solar_ultra_short_usecols_default
    config_cluster[config["id"]]["ultra_short_usecols"] = combination_feature

    # 短期最优特征，如果不存在，则使用默认值
    if combination_feature == None:
        if config["type"] == 'wind':
            combination_feature = wind_short_usecols_default
        else:
            combination_feature = solar_short_usecols_default
    config_cluster[config["id"]]["short_usecols"] = combination_feature

    nwp_np_ultra_short = numpy.zeros((0, 0))
    predict_type = config_cluster[station_id]['type']
    ultra_short_usecols = config_cluster[station_id]['ultra_short_usecols']
    short_usecols = config_cluster[station_id]['short_usecols']
    online_capacity = config_cluster[station_id]['online_capacity']
    # ----------------------------------------------------------------------------------------------------------
    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    nwp_np_short, nwp_np_ultra_short = load_test_date_for_ensemble_learning(
        host=host, user=user, password=password, database=database, charset=charset, port=port,
        start_time_str=start_time, end_time_str=end_time, rate=rate, station_id=station_id,
        ultra_short_usecols=ultra_short_usecols, short_usecols=short_usecols,
        predict_type=predict_type)

    sub_dir_path = str(config_cluster[station_id]['model_path']) + str(station_id) + '/short/'
    os.makedirs(sub_dir_path, exist_ok=True)
    save_model(nwp_np_short, sub_dir_path + 'short_test_data.pkl')

    sub_dir_path = str(config_cluster[station_id]['model_path']) + str(station_id) + '/ultra_short/'
    os.makedirs(sub_dir_path, exist_ok=True)
    save_model(nwp_np_ultra_short, sub_dir_path + 'ultra_short_test_data.pkl')

    # ----------------------------------------------------------------------------------------------------------
    c.execute("select config_value from sys_config where config_key = 'mode_need_split_in_time';")
    db.commit()
    record = c.fetchall()
    mode_need_split_in_time, number_copies = record[0][0].split('/', 1)
    mode_need_split_in_time = eval(mode_need_split_in_time)
    number_copies = eval(number_copies)
    # ----------------------------------------------------------------------------------------------------------
    #  添加短期预测的任务2022/11/20
    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    for model_name in model_name_cluster:
        for model_state in ['_without_history_power']:  # model_state_cluster:
            if model_name in mode_need_split_in_time:
                for j in range(number_copies):
                    c.execute("INSERT INTO parallel_tasks "
                              "(task_name, task_status, distribution_times, task_type, station_id) "
                              "VALUES (%s, %s, %s, %s, %s) ",
                              tuple(
                                  [str(station_id) + '_short_' + model_name + model_state + '-' + str(j + 1) +
                                   '/' + str(number_copies)]) +
                              tuple(['0']) +
                              tuple([0]) + tuple(['predict_feature']) + tuple([station_id]))
            else:
                c.execute("INSERT INTO parallel_tasks "
                          "(task_name, task_status, distribution_times, task_type, station_id) "
                          "VALUES (%s, %s, %s, %s, %s) ",
                          tuple([str(station_id) + '_short_' + model_name + model_state]) +
                          tuple(['0']) +
                          tuple([0]) + tuple(['predict_feature']) + tuple([station_id]))
    db.commit()
    #  添加超短期预测的任务2022/11/20
    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    for model_name in model_name_cluster:
        for model_state in ['_with_history_power']:
            if model_name in mode_need_split_in_time:
                for j in range(number_copies):
                    c.execute("INSERT INTO parallel_tasks "
                              "(task_name, task_status, distribution_times, task_type, station_id) "
                              "VALUES (%s, %s, %s, %s, %s) ",
                              tuple(
                                  [str(station_id) + '_ultra_short_' + model_name + model_state + '-' +
                                   str(j + 1) + '/' + str(number_copies)]) +
                              tuple(['0']) +
                              tuple([0]) + tuple(['predict_feature']) + tuple([station_id]))
            else:
                c.execute("INSERT INTO parallel_tasks "
                          "(task_name, task_status, distribution_times, task_type, station_id) "
                          "VALUES (%s, %s, %s, %s, %s) ",
                          tuple([str(station_id) + '_ultra_short_' + model_name + model_state]) +
                          tuple(['0']) +
                          tuple([0]) + tuple(['predict_feature']) + tuple([station_id]))
    db.commit()

    number_end = 0
    number_task = c.execute("select task_name from parallel_tasks where task_type = 'predict_feature'"
                            " and station_id=%s;", station_id)
    db.commit()
    tast_name_last = 'None'

    evaluate_start_time = datetime.datetime.now().timestamp()  # 测试开始时间
    # if len(nwp_np_ultra_short) == 0:
    #     logs.warning('被选中进行集成学习的场站数量为0！！！')
    evaluate_time_maximum_limit = number_task * len(nwp_np_ultra_short) * 0.7  # 测试时间最大限值
    # 当测试任务全部完成，或者测试时间超过最大限值都会跳出循环
    while number_end < number_task and \
            (datetime.datetime.now().timestamp() - evaluate_start_time) < evaluate_time_maximum_limit:
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        number_finished = c.execute(
            "select task_name from parallel_tasks where task_status = '2' and task_type = 'predict_feature'"
            " and station_id=%s;", station_id)
        db.commit()
        number_failed = c.execute(
            "select task_name from parallel_tasks where task_status = '3' and task_type = 'predict_feature'"
            " and station_id=%s;", station_id)
        db.commit()
        number_end = number_failed + number_finished
        logs.debug('已结束的任务数：' + str(number_end))

        # 分布式并行任务处理时，对于超时或异常任务要进行重复捕捉 2022-09-22 --------------------------------------------------
        current_time = datetime.datetime.now().timestamp()
        result = c.execute("select task_name from parallel_tasks where "
                           "task_status = '1' and "
                           "(%s-task_start_time)>5400 and "
                           "distribution_times<2 and task_type = 'predict_feature' and station_id=%s limit 1;",
                           (current_time, station_id))
        db.commit()
        if result != 0:
            record = c.fetchall()
            task_name = record[0][0]
            c.execute(
                "update parallel_tasks set task_status = '0' where task_name = %s and task_status = '1' "
                "and task_type = 'predict_feature' and station_id=%s;",
                (task_name, station_id))
            db.commit()

        # 判断并分配任务 ----------------------------------------------------------------------------------------------
        result = c.execute(
            "select task_name from parallel_tasks where task_status = '0'"
            " and task_type = 'predict_feature' and station_id=%s limit 1;", station_id)
        db.commit()
        if result == 0:
            time.sleep(2)
            continue
        record = c.fetchall()
        task_name = record[0][0]
        if task_name == tast_name_last:
            time.sleep(2)
        tast_name_last = task_name

        result = c.execute(
            "select task_name from parallel_tasks where task_status = '1' and task_type = 'predict_feature'"
            " and station_id=%s;", station_id)
        db.commit()
        if result < settings['scheduler']['process_pool_size']:
            logs.debug('任务加入：' + task_name)
            station, model = task_name.split('short_', 1)
            station_id, term = str(station + 'short').split('_', 1)
            station_id = eval(station_id)
            model_name, model_state = model.split('_', 1)
            model_state = '_' + model_state

            # 多进程任务方式-------------------------------------------------------------------------------------------
            if term == 'short':
                # 短期预测
                args_short = [host, user, password, database, charset, port,
                              config_cluster[station_id]['type'], config_cluster[station_id]['model_path'],
                              model_name, model_state, station_id,
                              config_cluster[station_id]['sr_col'],
                              config_cluster[station_id]['online_capacity']]
                scheduler.add_job(ensemble_learning_evaluate_short, executor=executor_name,
                                  args=args_short,
                                  coalesce=True, misfire_grace_time=None)
            else:
                # 超短期预测
                args_ultra_short = [host, user, password, database, charset, port,
                                    config_cluster[station_id]['type'], config_cluster[station_id]['model_path'],
                                    model_name, model_state, station_id,
                                    config_cluster[station_id]['sr_col'],
                                    config_cluster[station_id]['online_capacity']]
                scheduler.add_job(ensemble_learning_evaluate_ultra_short, executor=executor_name,
                                  args=args_ultra_short,
                                  coalesce=True, misfire_grace_time=None)
            time.sleep(1)
    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    accuracy_combination = model_evaluate_save(
        host=host, user=user, password=password, database=database, charset=charset, port=port,
        model_name_cluster=model_name_cluster, config=config)

    c.close()
    db.close()
    return accuracy_combination


def feature_trainning_model(host='localhost', user='root', password='123456', database='kuafu', charset='utf8',
                            port=3306, combination_feature=None, config=None):
    """
    训练模型
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    # 从数据库读取需要集成学习的模型列表
    model_name_cluster = ['BPNN']

    # 读取配置-----------------------------------------------------------------------------------------------------------

    # 场站id
    station_id = config["id"]

    # 配置信息
    config_cluster = {config["id"]: {"id": config["id"],
                                     "name": config["name"],
                                     "type": config["type"],
                                     "sr_col": 0,
                                     "online_capacity": config["online_capacity"],
                                     "model_savepath": config["model_savepath"] + 'feature/'}
                      }

    # 读取默认特征和默认参数 -----------------------------------------------------------------------------------------------
    c.execute("select * from default_feature_and_parameters;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_default = pandas.DataFrame(record, columns=coul)

    # 模型参数默认值
    short_wind_best_parameter_default = {}
    ultra_short_wind_best_parameter_default = {}
    short_solar_best_parameter_default = {}
    ultra_short_solar_best_parameter_default = {}
    for i in range(len(model_name_cluster)):
        short_wind_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_wind"][model_name_cluster[i]].values[0])
        ultra_short_wind_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"][
                model_name_cluster[i]].values[0])
        short_solar_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_solar"][model_name_cluster[i]].values[0])
        ultra_short_solar_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"][
                model_name_cluster[i]].values[0])

    # 特征默认值
    wind_ultra_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"]['usecols'].values[0])
    wind_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "short_wind"]['usecols'].values[0])
    solar_ultra_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"]['usecols'].values[0])
    solar_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "short_solar"]['usecols'].values[0])

    # 读取最优特征和参数，如果不存在，则使用默认值 -----------------------------------------------------------------------------
    c.execute("select * from best_feature_parameters_and_model;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_best = pandas.DataFrame(record, columns=coul)

    # 模型短期最优参数，如果不存在，则使用默认值
    best_parameters_exist = False
    if config["type"] == 'wind':
        best_parameter_short = short_wind_best_parameter_default
    else:
        best_parameter_short = short_solar_best_parameter_default
    if not best_parameters_exist:
        logs.debug("最优参数路径未找到！！！采用默认参数训练")
    config_cluster[config["id"]]["best_parameter_short"] = best_parameter_short
    # 模型超短期最优参数，如果不存在，则使用默认值
    best_parameters_exist = False
    if config["type"] == 'wind':
        best_parameter_ultra_short = ultra_short_wind_best_parameter_default
    else:
        best_parameter_ultra_short = ultra_short_solar_best_parameter_default
    if not best_parameters_exist:
        logs.debug("最优参数路径未找到！！！采用默认参数训练")
    config_cluster[config["id"]]["best_parameter_ultra_short"] = best_parameter_ultra_short

    # 超短期最优特征，如果不存在，则使用默认值
    if combination_feature == None:
        if config["type"] == 'wind':
            combination_feature = wind_ultra_short_usecols_default
        else:
            combination_feature = solar_ultra_short_usecols_default
    config_cluster[config["id"]]["ultra_short_usecols"] = combination_feature

    # 短期最优特征，如果不存在，则使用默认值
    if combination_feature == None:
        if config["type"] == 'wind':
            combination_feature = wind_short_usecols_default
        else:
            combination_feature = solar_short_usecols_default
    config_cluster[config["id"]]["short_usecols"] = combination_feature

    c.execute("DELETE FROM parallel_tasks where task_type = 'data_load_preprocess_feature'"
              " and station_id=%s;", config["id"])
    db.commit()
    term_cluster = ["short", "ultra_short"]
    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    for term in term_cluster:
        c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                  "VALUES (%s, %s, %s, %s) ",
                  tuple([str(station_id) + '_' + term]) + tuple(['0']) + tuple(['data_load_preprocess_feature']) +
                  tuple([station_id]))
    db.commit()

    c.execute("DELETE FROM parallel_tasks where task_type = 'train_feature' and station_id=%s;", config["id"])
    db.commit()
    term_cluster = ["short", "ultra_short"]
    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    for term in term_cluster:
        for model_name in model_name_cluster:
            c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                      "VALUES (%s, %s, %s, %s) ",
                      tuple([str(station_id) + '_' + term + '_' + model_name]) + tuple(['0']) +
                      tuple(['train_feature']) + tuple([station_id]))
    db.commit()
    c.close()
    db.close()
    return config_cluster


def model_evaluate_save(host, user, password, database, charset, port,
                        model_name_cluster, config):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param model_name_cluster: 模型名称
    :param config: 模型配置参数
    :return: None
    """
    # ------------------------------------------------------------------------------------------------------------------
    if model_name_cluster is None or len(model_name_cluster) == 0:
        return
    # 结果评估
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 定义游标
    c = db.cursor()

    station_id = config['id']
    online_capacity = config['online_capacity']
    predict_type = config['type']

    c.execute("select time, power from real_power_" + str(station_id) + ";")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_real_power = pandas.DataFrame(record, columns=coul)
    actual_power_np = dataframe_real_power.values

    # 判断限电记录
    error_data_label = data_preprocess.DataPreprocess.data_preprocess_error_data_label(actual_power_np,
                                                                                   predict_type,
                                                                                   online_capacity)

    # --------------------------------------------------------------------------------------------------------------
    c.execute("select config_value from sys_config where config_key = 'NB_T_32011_2013';")
    db.commit()
    record = c.fetchall()
    config_value = eval(record[0][0])
    # NB_T_32011_2013
    if config_value == 1:
        short_accuracy = numpy.zeros((len(model_name_cluster), 1))
        n_model = 0
        model_state = '_without_history_power'
        for model_name in model_name_cluster:
            model_name_and_state = model_name + model_state
            # 计算准确率
            accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                           error_data_label=error_data_label, predict_term='short',
                                           host=host, user=user, password=password,
                                           database=database, charset=charset, port=port,
                                           model_name_and_state=model_name_and_state, scene='feature',
                                           sunrise_time='06:30:00', sunset_time='19:30:00',
                                           predict_type=predict_type, evaluation="capacity")
            if math.isnan(accrucy):
                short_accuracy[n_model, 0] = 0
            else:
                short_accuracy[n_model, 0] = accrucy
            n_model += 1

        ultra_short_accuracy = numpy.zeros((len(model_name_cluster), 1))
        n_model = 0
        model_state = '_with_history_power'
        for model_name in model_name_cluster:
            model_name_and_state = model_name + model_state
            # 计算准确率
            accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                           error_data_label=error_data_label, predict_term='ultra_short',
                                           host=host, user=user, password=password,
                                           database=database, charset=charset, port=port,
                                           model_name_and_state=model_name_and_state, scene='feature',
                                           sunrise_time='06:30:00', sunset_time='19:30:00',
                                           predict_type=predict_type, evaluation="capacity")
            if math.isnan(accrucy):
                ultra_short_accuracy[n_model, 0] = 0
            else:
                ultra_short_accuracy[n_model, 0] = accrucy
            n_model += 1
    # --------------------------------------------------------------------------------------------------------------
    config_value = 1
    # Two_Detailed_Rules
    if config_value == 1:
        short_accuracy = numpy.zeros((len(model_name_cluster), 1))
        n_model = 0
        model_state = '_without_history_power'
        for model_name in model_name_cluster:
            model_name_and_state = model_name + model_state
            # 计算准确率
            accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                           error_data_label=error_data_label, predict_term='short',
                                           host=host, user=user, password=password,
                                           database=database, charset=charset, port=port,
                                           model_name_and_state=model_name_and_state, scene='feature',
                                           evaluation="actual_power",
                                           predict_type=predict_type)
            if math.isnan(accrucy):
                short_accuracy[n_model, 0] = 0
            else:
                short_accuracy[n_model, 0] = accrucy
            n_model += 1

        ultra_short_accuracy = numpy.zeros((len(model_name_cluster), 1))
        n_model = 0
        model_state = '_with_history_power'
        for model_name in model_name_cluster:
            model_name_and_state = model_name + model_state
            # 计算准确率
            accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                           error_data_label=error_data_label, predict_term='ultra_short',
                                           host=host, user=user, password=password,
                                           database=database, charset=charset, port=port,
                                           model_name_and_state=model_name_and_state, scene='feature',
                                           evaluation="actual_power",
                                           predict_type=predict_type)
            if math.isnan(accrucy):
                ultra_short_accuracy[n_model, 0] = 0
            else:
                ultra_short_accuracy[n_model, 0] = accrucy
            n_model += 1
        # --------------------------------------------------------------------------------------------------------------
        n_model = 0
        for model_name in model_name_cluster:
            two_Detaile_short_accuracy = short_accuracy[n_model, 0]
            n_model += 1

        n_model = 0
        for model_name in model_name_cluster:
            two_Detaile_ultra_short_accuracy = ultra_short_accuracy[n_model, 0]
            n_model += 1
        # ----------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------------
    c.execute("select config_value from sys_config where config_key = 'Q_CSG1211017_2018';")
    db.commit()
    record = c.fetchall()
    config_value = eval(record[0][0])
    # Q_CSG1211017_2018
    if config_value == 1:
        short_accuracy = numpy.zeros((len(model_name_cluster), 1))
        n_model = 0
        model_state = '_without_history_power'
        for model_name in model_name_cluster:
            model_name_and_state = model_name + model_state
            # 计算准确率
            accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                           error_data_label=error_data_label, predict_term='short',
                                           host=host, user=user, password=password,
                                           database=database, charset=charset, port=port,
                                           model_name_and_state=model_name_and_state, scene='feature',
                                           standard='Q_CSG1211017_2018',
                                           predict_type=predict_type, evaluation="capacity")
            if math.isnan(accrucy):
                short_accuracy[n_model, 0] = 0
            else:
                short_accuracy[n_model, 0] = accrucy
            n_model += 1

        ultra_short_accuracy = numpy.zeros((len(model_name_cluster), 1))
        n_model = 0
        model_state = '_with_history_power'
        for model_name in model_name_cluster:
            model_name_and_state = model_name + model_state
            # 计算准确率
            accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                           error_data_label=error_data_label, predict_term='ultra_short',
                                           host=host, user=user, password=password,
                                           database=database, charset=charset, port=port,
                                           model_name_and_state=model_name_and_state, scene='feature',
                                           standard='Q_CSG1211017_2018',
                                           predict_type=predict_type, evaluation="capacity")
            if math.isnan(accrucy):
                ultra_short_accuracy[n_model, 0] = 0
            else:
                ultra_short_accuracy[n_model, 0] = accrucy
            n_model += 1
    # --------------------------------------------------------------------------------------------------------------
    short_accuracy = numpy.zeros((len(model_name_cluster), 1))
    n_model = 0
    model_state = '_without_history_power'
    for model_name in model_name_cluster:
        model_name_and_state = model_name + model_state
        # 计算准确率
        accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                       error_data_label=error_data_label, predict_term='short',
                                       host=host, user=user, password=password,
                                       database=database, charset=charset, port=port,
                                       model_name_and_state=model_name_and_state, scene='feature',
                                       predict_type=predict_type, evaluation="capacity")
        if math.isnan(accrucy):
            short_accuracy[n_model, 0] = 0
        else:
            short_accuracy[n_model, 0] = accrucy
        n_model += 1

    ultra_short_accuracy = numpy.zeros((len(model_name_cluster), 1))
    n_model = 0
    model_state = '_with_history_power'
    for model_name in model_name_cluster:
        model_name_and_state = model_name + model_state
        # 计算准确率
        accrucy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                       error_data_label=error_data_label, predict_term='ultra_short',
                                       host=host, user=user, password=password,
                                       database=database, charset=charset, port=port,
                                       model_name_and_state=model_name_and_state, scene='feature',
                                       predict_type=predict_type, evaluation="capacity")
        if math.isnan(accrucy):
            ultra_short_accuracy[n_model, 0] = 0
        else:
            ultra_short_accuracy[n_model, 0] = accrucy
        n_model += 1
    # --------------------------------------------------------------------------------------------------------------
    n_model = 0
    for model_name in model_name_cluster:
        GB_short_accuracy = short_accuracy[n_model, 0]
        n_model += 1

    n_model = 0
    for model_name in model_name_cluster:
        GB_ultra_accuracy = ultra_short_accuracy[n_model, 0]
        n_model += 1

    accuracy_combination = [two_Detaile_short_accuracy, two_Detaile_ultra_short_accuracy, GB_short_accuracy,
                            GB_ultra_accuracy]
    c.close()
    db.close()
    # 将特征与国标、两个细则的准确率合在一起

    return accuracy_combination


def data_load_preprocess(station_id, predict_term, config, host, user, password, database,
                         charset, port, start_time, end_time, rate):
    """
    针对特定的电场和预测类型，通过测试选择最优的模型并保存
    :param station_id: 场站名称
    :param predict_term: 预测类型
    :param config: 配置信息
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param start_time: 测试与训练的时间分解
    :param end_time: 训练集数据结束时间
    :param rate: 训练集数据量与（训练集+测试集数据量）的比值
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_name = str(station_id) + '_' + predict_term
    try:
        result = c.execute(
            "update parallel_tasks set task_status = '1' where task_status = '0' and task_name = %s and"
            " task_type = 'data_load_preprocess_feature' and station_id=%s;",
            (task_name, station_id))
        db.commit()
        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'data_load_preprocess_feature' and station_id=%s;",
                      (task_start_time, task_name, station_id))
            db.commit()

            ultra_short_usecols = config["ultra_short_usecols"]
            short_usecols = config["short_usecols"]
            predict_type = config["type"]
            online_capacity = config["online_capacity"]
            train_data_load = LoadTraindata()
            if predict_term == "ultra_short":
                train_feature, train_target = train_data_load.load_train_data_for_ultra_short_sql(host=host,
                                                                                                  user=user,
                                                                                                  password=password,
                                                                                                  database=database,
                                                                                                  charset=charset,
                                                                                                  port=port,
                                                                                                  config=config,
                                                                                                  usecols=ultra_short_usecols,
                                                                                                  predict_type=predict_type,
                                                                                                  start_time=start_time,
                                                                                                  end_time=end_time,
                                                                                                  rate=rate)

            else:
                train_feature, train_target = train_data_load.load_train_data_for_short_sql(host=host,
                                                                                            user=user,
                                                                                            password=password,
                                                                                            database=database,
                                                                                            charset=charset,
                                                                                            port=port,
                                                                                            config=config,
                                                                                            usecols=short_usecols,
                                                                                            predict_type=predict_type,
                                                                                            start_time=start_time,
                                                                                            end_time=end_time,
                                                                                            rate=rate)

            train_feature, train_target = data_preprocess.DataPreprocess.data_preprocess(train_feature, train_target,
                                                                                         online_capacity)

            sub_dir_path = config['model_savepath'] + str(station_id) + '/' + predict_term + '/'
            os.makedirs(sub_dir_path, exist_ok=True)
            save_model(train_feature, sub_dir_path + predict_term + '_train_feature.pkl')
            save_model(train_target, sub_dir_path + predict_term + '_train_target.pkl')

            c.execute(
                "update parallel_tasks set task_status = '2' where task_status = '1' and task_name = %s and"
                " task_type = 'data_load_preprocess_feature' and station_id=%s;",
                (task_name, station_id))
            db.commit()
            task_end_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                      " task_type = 'data_load_preprocess_feature' and station_id=%s;",
                      (task_end_time, task_name, station_id))
            db.commit()
        c.close()
        db.close()
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()
        logs.warning(task_name + '任务失败！')
        c.execute("update parallel_tasks set task_status = '3' where task_name = %s and"
                  " task_type = 'data_load_preprocess_feature' and station_id=%s;", (task_name, station_id))
        db.commit()
        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                  " task_type = 'data_load_preprocess_feature' and station_id=%s;",
                  (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


def save_a_fitted_model(station_id, predict_term, model_name, config, host, user, password, database, charset, port):
    """
    针对特定的电场和预测类型，通过测试选择最优的模型并保存
    :param station_id: 场站名称
    :param predict_term: 预测类型
    :param config: 配置信息
    :param model_name: 模型名称集合
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_name = str(station_id) + '_' + predict_term + '_' + model_name
    try:
        result = c.execute(
            "update parallel_tasks set task_status = '1' where task_status = '0' and task_name = %s and"
            " task_type = 'train_feature' and station_id=%s;", (task_name, station_id))
        db.commit()
        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'train_feature' and station_id=%s;",
                      (task_start_time, task_name, station_id))
            db.commit()
            if predict_term == 'ultra_short':
                fitted_model = training.fit_model(predict_term=predict_term, model_name=model_name,
                                                  config=config, record_restrict=None,
                                                  without_history_power=True, scene='ensemble_learn')
            else:
                fitted_model = training.fit_model(predict_term=predict_term, model_name=model_name,
                                                  config=config, record_restrict=None,
                                                  without_history_power=True, scene='ensemble_learn')

            sub_dir_path = config['model_savepath'] + str(station_id) + '/' + predict_term + '/' + model_name + '/'
            os.makedirs(sub_dir_path, exist_ok=True)
            save_model(fitted_model, sub_dir_path + predict_term + '_' + model_name + '.pkl')
            if predict_term == 'ultra_short':
                save_model(config["ultra_short_usecols"], sub_dir_path + 'ultra_short_usecols.pkl')
            else:
                save_model(config["short_usecols"], sub_dir_path + 'short_usecols.pkl')

            c.execute(
                "update parallel_tasks set task_status = '2' where task_status = '1' and task_name = %s and"
                " task_type = 'train_feature' and station_id=%s;",
                (task_name, station_id))
            db.commit()
            task_end_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_end_time = %s where task_name = %s"
                      " and task_type = 'train_feature' and station_id=%s;", (task_end_time, task_name, station_id))
            db.commit()
        c.close()
        db.close()
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()
        logs.warning(task_name + '任务失败！')
        c.execute("update parallel_tasks set task_status = '3' where task_name = %s and"
                  " task_type = 'train_feature' and station_id=%s;", (task_name, station_id))
        db.commit()
        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and task_type = 'train_feature'"
                  " and station_id=%s;", (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


def ensemble_learning_evaluate_short(host, user, password, database, charset, port, predict_type=None, model_path=None,
                                     model_name=None, model_state=None, station_id=None, sr_col=None,
                                     online_capacity=None):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param predict_type: 预测类型
    :param online_capacity: 开机容量
    :param model_name: 模型名称
    :param model_state: 预测方法
    :param model_path:
    :param station_id:
    :param sr_col:
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_name = str(station_id) + '_short_' + model_name + model_state
    try:
        result = c.execute(
            "update parallel_tasks set task_status = '1' where task_status = '0' and task_name = %s and"
            " task_type = 'predict_feature' and station_id=%s;", (task_name, station_id))
        db.commit()

        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute(
                "update parallel_tasks set distribution_times = distribution_times + 1 where task_status = '1' "
                "and task_name = %s and task_type = 'predict_feature' and station_id=%s;",
                (task_name, station_id))
            db.commit()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'predict_feature' and station_id=%s;", (task_start_time, task_name, station_id))
            db.commit()

            sub_dir_path = model_path + str(station_id) + '/short/'
            nwp_np_short = load_model(sub_dir_path + 'short_test_data.pkl')
            if '-' in model_state:
                model_state, evaluate_label = model_state.split('-', 1)
                evaluate_time, total_times = evaluate_label.split('/', 1)
                evaluate_time = eval(evaluate_time)
                total_times = eval(total_times)

                div = int((len(nwp_np_short) / 288) // total_times)  # 商
                mod = int((len(nwp_np_short) / 288) % total_times)  # 余数
                if evaluate_time == total_times:
                    nwp_np_short = nwp_np_short[
                                   ((evaluate_time - 1) * div + mod) * 288:, :]
                else:
                    if evaluate_time > mod:
                        nwp_np_short = nwp_np_short[
                                       ((evaluate_time - 1) * div + mod) * 288:
                                       (evaluate_time * div + mod) * 288, :]
                    else:
                        nwp_np_short = nwp_np_short[(evaluate_time - 1) * (div + 1) * 288:
                                                    evaluate_time * (div + 1) * 288, :]

            # 短期预测
            predict_result_cluster, time_predict_cluster = feature_selection.predict_short_power(
                model_name_state=model_name + model_state, sr_col=sr_col, online_capacity=online_capacity,
                predict_type=predict_type, model_path=model_path, station_id=station_id,
                nwp_np_short=nwp_np_short)
            result = predict_result_cluster.reshape(-1, 1)

            c.execute("select * from predict_power_" + str(station_id) + "_train_feature limit 1;")
            db.commit()
            des = c.description
            for j in range(int(len(result) / 288)):
                start_time_str = datetime.datetime.strptime(
                    time.strftime("%Y/%m/%d %H:%M", time.localtime(time_predict_cluster[288 * j].timestamp() - 900-8*3600)),
                    '%Y/%m/%d %H:%M')
                for ii in range(288):
                    forecast_time = time_predict_cluster[288 * j + ii]
                    value = (start_time_str, forecast_time) + tuple(result[288 * j + ii, :])
                    values = tuple(['short']) + tuple([model_name + model_state]) + value
                    c.execute("INSERT INTO predict_power_" + str(station_id) + '_train_feature (' + ','.join(
                        iterm[0] for iterm in des[1:]) + ')' + "VALUES(" + ("%s," * 5).strip(',') + ")", values)
                if j % 10 == 1:
                    db.commit()
                    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
                    if task_stop_sign == 1:
                        return
            db.commit()

            task_end_time = datetime.datetime.now().timestamp()

            try:
                c.execute(
                    "update parallel_tasks set task_status = '2' where task_status = '1' and task_name = %s and"
                    " task_type = 'predict_feature' and station_id=%s;",
                    (task_name, station_id))
                db.commit()
                c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                          " task_type = 'predict_feature' and station_id=%s;", (task_end_time, task_name, station_id))
                db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
            finally:
                c.close()
                db.close()
        else:
            logs.debug(task_name + '任务已被其他进程执行')
            c.close()
            db.close()
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()
        logs.warning(task_name + '任务失败！')
        c.execute("update parallel_tasks set task_status = '3' where task_name = %s and"
                  " task_type = 'predict_feature' and station_id=%s;", (task_name, station_id))
        db.commit()
        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                  " task_type = 'predict_feature' and station_id=%s;", (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


def ensemble_learning_evaluate_ultra_short(
        host='localhost', user='root', password='123456', database='kuafu', charset='utf8', port=3306,
        predict_type=None, model_path=None, model_name=None, model_state=None, station_id=None,
        sr_col=None, online_capacity=None):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param predict_type: 预测类型
    :param online_capacity: 开机容量
    :param model_name: 模型名称
    :param model_state: 预测方法
    :param model_path:
    :param station_id:
    :param sr_col:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_name = str(station_id) + '_ultra_short_' + model_name + model_state
    try:
        result = c.execute(
            "update parallel_tasks set task_status = '1' where task_status = '0' and task_name = %s and"
            " task_type = 'predict_feature' and station_id=%s;",
            (task_name, station_id))
        db.commit()

        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'predict_feature' and station_id=%s;",
                      (task_start_time, task_name, station_id))
            db.commit()
            c.execute(
                "update parallel_tasks set distribution_times = distribution_times + 1 where task_status = '1' "
                "and task_name = %s and task_type = 'predict_feature' and station_id=%s;",
                (task_name, station_id))
            db.commit()

            sub_dir_path = model_path + str(station_id) + '/ultra_short/'
            nwp_np_ultra_short = load_model(sub_dir_path + 'ultra_short_test_data.pkl')
            if '-' in model_state:
                model_state, evaluate_label = model_state.split('-', 1)
                evaluate_time, total_times = evaluate_label.split('/', 1)
                evaluate_time = eval(evaluate_time)
                total_times = eval(total_times)

                div = int((len(nwp_np_ultra_short)) // total_times)  # 商
                mod = int((len(nwp_np_ultra_short)) % total_times)  # 余数
                if evaluate_time == total_times:
                    nwp_np_ultra_short = nwp_np_ultra_short[((evaluate_time - 1) * div + mod):, :]
                else:
                    if evaluate_time > mod:
                        nwp_np_ultra_short = nwp_np_ultra_short[((evaluate_time - 1) * div + mod):
                                                                (evaluate_time * div + mod), :]
                    else:
                        nwp_np_ultra_short = nwp_np_ultra_short[(evaluate_time - 1) * (div + 1):
                                                                evaluate_time * (div + 1), :]

            # 超短期预测
            predict_result_cluster, time_predict_cluster = feature_selection.predict_ultra_short_power(
                model_name_state=model_name + model_state, predict_type=predict_type,
                model_path=model_path, station_id=station_id, sr_col=sr_col, online_capacity=online_capacity,
                nwp_np_ultra_short=nwp_np_ultra_short)
            result = predict_result_cluster.reshape(-1, 1)

            c.execute("select * from predict_power_" + str(station_id) + "_train_feature limit 1;")
            db.commit()
            des = c.description
            for j in range(int(len(result) / 16)):
                start_time_str = datetime.datetime.strptime(
                    time.strftime("%Y/%m/%d %H:%M", time.localtime(time_predict_cluster[16 * j].timestamp() - 900-8*3600)),
                    '%Y/%m/%d %H:%M')
                for ii in range(16):
                    forecast_time = time_predict_cluster[16 * j + ii]
                    value = (start_time_str, forecast_time) + tuple(result[16 * j + ii, :])
                    values = tuple(['ultra_short']) + tuple([model_name + model_state]) + value
                    c.execute("INSERT INTO predict_power_" + str(station_id) + '_train_feature (' + ','.join(
                        iterm[0] for iterm in des[1:]) + ')' + "VALUES(" + ("%s," * 5).strip(',') + ")", values)
                if j % 30 == 1:
                    db.commit()
                    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
                    if task_stop_sign == 1:
                        return
            db.commit()
            task_end_time = datetime.datetime.now().timestamp()

            try:
                c.execute(
                    "update parallel_tasks set task_status = '2' where task_status = '1' and task_name = %s and"
                    " task_type = 'predict_feature' and station_id=%s;",
                    (task_name, station_id))
                db.commit()
                c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                          " task_type = 'predict_feature' and station_id=%s;",
                          (task_end_time, task_name, station_id))
                db.commit()
            except Exception as e:
                logs.warning(e)
                db.rollback()
        else:
            logs.debug(task_name + '任务已被其他进程执行')
            c.close()
            db.close()
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()
        logs.warning(task_name + '任务失败！')
        c.execute("update parallel_tasks set task_status = '3' where task_name = %s and"
                  " task_type = 'predict_feature' and station_id=%s;", (task_name, station_id))
        db.commit()

        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                  " task_type = 'predict_feature' and station_id=%s;",
                  (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


def load_test_date_for_ensemble_learning(
        host='localhost', user='root', password='123456', database='kuafu', charset='utf8', port=3306,
        start_time_str=None, end_time_str=None, rate=None, station_id=None,
        ultra_short_usecols=None, short_usecols=None, predict_type=None):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param start_time_str: 集成学习开始时间
    :param end_time_str: 集成学习结束时间
    :param rate: 训练与测试比例划分
    :param station_id: 场站id
    :param ultra_short_usecols: 特征
    :param short_usecols: 特征
    :param predict_type: 预测类型
    :return: nwp_np_short, nwp_np_ultra_short
    """
    # 获取数据时长，并分段
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 定义游标
    c = db.cursor()
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
    if start_time_str is None or end_time_str is None:
        start_time_float = start_time_float_database
        end_time_float = end_time_float_database
    else:
        start_time_float_input = datetime.datetime.strptime(start_time_str, '%Y/%m/%d %H:%M').timestamp()
        end_time_float_input = datetime.datetime.strptime(end_time_str, '%Y/%m/%d %H:%M').timestamp()
        start_time_float = max(start_time_float_database, start_time_float_input)
        end_time_float = min(end_time_float_database, end_time_float_input)

    end_time_float = int((end_time_float - 57600) / 86400) * 86400 + 57600

    # ----------------------------------------------------------------------------------------------------------
    start_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time_float)), '%Y/%m/%d %H:%M')
    end_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float)), '%Y/%m/%d %H:%M')

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

    middle_time_float = int(
        (train_data.iloc[min(int(rate * len(train_data)),
                             int(len(train_data) - 1)), 0].timestamp() - 57600) / 86400) * 86400 + 57600
    # 短期---------------------------------------------------------------------------------------------------------------
    # 读取测试数据
    # NWP
    if middle_time_float % 86400 > 57600:
        middle_time_float = middle_time_float-(middle_time_float % 86400 - 57600) + 86400
    else:
        middle_time_float = middle_time_float - (middle_time_float % 86400 - 57600)

    if end_time_float % 86400 > 57600:
        end_time_float = end_time_float-(end_time_float % 86400 - 57600)
    else:
        end_time_float = end_time_float - (end_time_float % 86400 - 57600) - 86400

    middle_time_float = middle_time_float - 900
    end_time_float = end_time_float - 900

    middle_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float + 900)), '%Y/%m/%d %H:%M')
    end_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float)), '%Y/%m/%d %H:%M')

    c.execute(
        "select DISTINCT start_time, forecast_time," + ','.join(iterm for iterm in short_usecols) + " from nwp_" +
        str(station_id) + " where forecast_time between %s and %s"
                          " ORDER BY forecast_time, start_time desc;",
        (middle_time_datetime, end_time_datetime))
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    nwp_df = pandas.DataFrame(record, columns=coul)
    nwp_df = nwp_df.drop_duplicates(subset=['forecast_time'])

    # power
    middle_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float)), '%Y/%m/%d %H:%M')
    end_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float - 900)), '%Y/%m/%d %H:%M')

    c.execute(
        "select DISTINCT time, power from real_power_" + str(station_id) +
        " where time between %s and %s ORDER BY time asc;",
        (middle_time_datetime, end_time_datetime))
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    power_df = pandas.DataFrame(record, columns=coul)
    power_df = power_df.drop_duplicates(subset=['time'])

    time_list_power = []
    for j in range(middle_time_float, end_time_float, 900):
        time_list_power.append(
            datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(j)), '%Y/%m/%d %H:%M'))
    time_dic = {'time_list': time_list_power}
    time_dataframe = pandas.DataFrame(time_dic)
    data_base = pandas.merge(time_dataframe, power_df, left_on='time_list', right_on='time', how='left')
    data_base = data_base[['time_list', 'power']]
    data_base_power = data_base

    time_list = []
    time_list_float = []
    for j in range(middle_time_float + 900, end_time_float + 900, 900):
        time_list.append(
            datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(j)), '%Y/%m/%d %H:%M'))
        time_list_float.append(j % 86400 / 86400)
    time_dic = {'time': time_list, 'timelabel': time_list_float}
    time_dataframe = pandas.DataFrame(time_dic)
    data_base = pandas.merge(time_dataframe, nwp_df, left_on='time', right_on='forecast_time', how='left')

    if predict_type == 'wind':
        data_base.drop(['start_time', 'forecast_time', 'timelabel'], axis=1, inplace=True)
    else:
        data_base.drop(['start_time', 'forecast_time'], axis=1, inplace=True)
        cols = data_base.columns.tolist()
        cols = [cols[0]] + cols[2:] + [cols[1]]
        data_base = data_base[cols]
    data_base_np = numpy.hstack((data_base.values, data_base_power.values[:, 1].reshape(-1, 1)))

    n = data_base_np.shape[0]
    m = data_base_np.shape[1]

    nwp_np = data_base_np.reshape(int(n / 96), m * 96)
    nwp_np_short = numpy.hstack((nwp_np[:-2, :], nwp_np[1:-1, :], nwp_np[2:, :]))
    for i in range(int(n / 96)-2):
        for j in range(287):
            nwp_np_short[i, (j + 1) * m + m - 1] = nwp_np_short[i, m - 1]
    nwp_np_short = pandas.DataFrame(nwp_np_short).dropna(axis=0, how='any').values
    nwp_np_short = nwp_np_short.reshape(len(nwp_np_short) * 288, m)

    if predict_type == 'wind':
        for j in range(len(nwp_np_short)):
            for k in range(nwp_np_short.shape[1] - 2):
                nwp_np_short[j, k + 1] = eval(nwp_np_short[j, k + 1])

    else:
        for j in range(len(nwp_np_short)):
            for k in range(nwp_np_short.shape[1] - 3):
                nwp_np_short[j, k + 1] = eval(nwp_np_short[j, k + 1])

    # 超短期-------------------------------------------------------------------------------------------------------------
    # 读取测试数据
    # NWP
    middle_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float + 900)), '%Y/%m/%d %H:%M')
    end_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float)), '%Y/%m/%d %H:%M')

    a = "select DISTINCT max(start_time), forecast_time," + ','.join(iterm for iterm in ultra_short_usecols) + \
        " from nwp_" + str(station_id) + " where forecast_time between %s and %s group by forecast_time" + \
        " ORDER BY forecast_time asc;"

    c.execute(a, (middle_time_datetime, end_time_datetime))
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    nwp_df = pandas.DataFrame(record, columns=coul)

    # power
    middle_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float)), '%Y/%m/%d %H:%M')
    end_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float - 900)), '%Y/%m/%d %H:%M')

    c.execute(
        "select distinct time, power from real_power_" + str(station_id) +
        " where time between %s and %s ORDER BY time asc;",
        (middle_time_datetime, end_time_datetime))
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    power_df = pandas.DataFrame(record, columns=coul)
    c.close()
    db.close()

    time_list_power = []
    for j in range(middle_time_float, end_time_float, 900):
        time_list_power.append(
            datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(j)), '%Y/%m/%d %H:%M'))
    time_dic = {'time_list': time_list_power}
    time_dataframe = pandas.DataFrame(time_dic)
    data_base = pandas.merge(time_dataframe, power_df, left_on='time_list', right_on='time', how='left')
    data_base = data_base[['time_list', 'power']]
    data_base_power = data_base

    time_list = []
    time_list_float = []
    for j in range(middle_time_float + 900, end_time_float + 900, 900):
        time_list.append(
            datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(j)), '%Y/%m/%d %H:%M'))
        time_list_float.append(j % 86400 / 86400)
    time_dic = {'time': time_list, 'timelabel': time_list_float}
    time_dataframe = pandas.DataFrame(time_dic)
    data_base = pandas.merge(time_dataframe, nwp_df, left_on='time', right_on='forecast_time', how='left')

    if predict_type == 'wind':
        data_base.drop(['max(start_time)', 'forecast_time', 'timelabel'], axis=1, inplace=True)

    else:
        data_base.drop(['max(start_time)', 'forecast_time'], axis=1, inplace=True)
        cols = data_base.columns.tolist()
        cols = [cols[0]] + cols[2:] + [cols[1]]
        data_base = data_base[cols]

    data_base = data_base.values
    temp = data_base[:, 1:].copy()
    for i in range(15):
        data_base = numpy.hstack((data_base[:-1, :], temp[i + 1:, :]))

    data_base_np = numpy.hstack((data_base, data_base_power.values[:-15, 1].reshape(-1, 1)))

    data_base_df = pandas.DataFrame(data_base_np)

    nwp_np_ultra_short = data_base_df.dropna(axis=0, how='any').values

    if predict_type == 'wind':
        for j in range(len(nwp_np_ultra_short)):
            for k in range(nwp_np_ultra_short.shape[1] - 2):
                nwp_np_ultra_short[j, k + 1] = eval(nwp_np_ultra_short[j, k + 1])
    else:
        nu = len(ultra_short_usecols)
        for j in range(len(nwp_np_ultra_short)):
            for k in range(16):
                for p in range(nu):
                    nwp_np_ultra_short[j, k*(nu+1)+p + 1] = eval(nwp_np_ultra_short[j, k*(nu+1)+p + 1])

    return nwp_np_short, nwp_np_ultra_short


def task_stop(db, c, station_id):
    result = c.execute(
        "select id from parallel_tasks_station where task_name = 'feature' and task_status = 'task_stopped'"
        " and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        c.close()
        db.close()
    return result


import pandas
from task.mysql.model_evaluation import evaluate_for_station
import numpy
import time
from common.tools import save_model
from common.tools import load_model
import datetime
import pymysql
import math
from common.logger import logs
from common import data_preprocess
import os
from common.tools import catch_exception
import shutil
from app.ensemble_learning import ensemble_learning
from common.settings import settings
from task.mysql.load_train_data import LoadTraindata
from task.mysql.interval_forecast import interval_learning


@catch_exception("run_ensemble_learn error: ")
def run_ensemble_learn(host, user, password, database, charset, port, rate=0.75,
                       scheduler=None, executor_name=None, task=1, ensemble_learn_stands=None,
                       csvrecord=False, figurerecord=False):
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
            logs.warning(str(station_id) + 'configure表中没有场站记录，退出集成学习!')
            continue
        else:
            record = c.fetchall()
            if record[0][0] is None or record[0][1] is None:
                logs.warning(str(station_id) + 'configure表中没有设置起始时间，退出集成学习!')
                continue
            else:
                start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record[0][0].timestamp()))
                end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record[0][1].timestamp()))

        result = c.execute("select station_id, task_name from parallel_tasks_station"
                           " where station_id = %s and task_name = %s;", (station_id, 'ensemble'))
        db.commit()
        if result == 0:
            c.execute("INSERT INTO parallel_tasks_station (station_id, task_name)"
                      " VALUES (%s, %s)", tuple([station_id, 'ensemble']))
            db.commit()

        c.execute("update parallel_tasks_station set task_status = %s where station_id = %s and task_name = %s;",
                  (str(station_id) + '集成学习开始', station_id, 'ensemble'))
        db.commit()

        ensemble_learn(host, user, password, database, charset, port, start_time=start_time, end_time=end_time,
                       rate=rate, scheduler=scheduler, executor_name=executor_name, task=task, station_id=station_id,
                       ensemble_learn_stands=ensemble_learn_stands, csvrecord=csvrecord, figurerecord=figurerecord)
    c.close()
    db.close()


@catch_exception("ensemble_learn error: ")
def ensemble_learn(host, user, password, database, charset, port, start_time=None, end_time=None, rate=0.75,
                   scheduler=None, executor_name=None, task=1, station_id=None, ensemble_learn_stands=None,
                   csvrecord=False, figurerecord=False, csv_path=None, figure_path=None, model_name_cluster_setting=None):
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
    :param task: 任务序号
    :param station_id: 场站id
    :param ensemble_learn_stands: 评价标准
    :param csvrecord: 是否写csv
    :param figurerecord: 是否绘图
    :param csv_path: csv文件保存地址
    :param figure_path: 图片保存地址
    :param model_name_cluster_setting: 参与集成学习的模型列表
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    result = c.execute(
        "update parallel_tasks_station set task_status = '" + str(
            station_id) + "正在进行集成学习' where task_name = 'ensemble' and task_status = '" + str(
            station_id) + "集成学习开始' and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        logs.info('集成学习开始')
        function_start_time = datetime.datetime.now().timestamp()
        # --------------------------------------------------------------------------------------------------------------
        # 模型训练分配任务
        config_cluster_train = training_model(
            host=host, user=user, password=password, database=database, charset=charset, port=port, task=task,
            station_id=station_id, model_name_cluster_setting=model_name_cluster_setting)
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return

        # 训练数据获取与预处理
        logs.info('集成学习_数据获取与预处理开始')
        datapre_t1 = datetime.datetime.now().timestamp()  # 数据获取与预处理开始时间
        number_end = 0
        number_task = c.execute("select task_name from parallel_tasks"
                                " where task_type = 'data_load_preprocess' and station_id=%s;", station_id)
        db.commit()
        task_name_last = 'None'
        # 当测试任务全部完成，或者测试时间超过最大限值都会跳出循环
        while number_end < number_task:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            number_finished = c.execute(
                "select task_name from parallel_tasks where task_status = '2'"
                " and task_type = 'data_load_preprocess' and station_id=%s;", station_id)
            db.commit()
            number_failed = c.execute(
                "select task_name from parallel_tasks where task_status = '3'"
                " and task_type = 'data_load_preprocess' and station_id=%s;", station_id)
            db.commit()
            number_end = number_failed + number_finished
            logs.debug('已结束的任务数：' + str(number_end))

            # 判断并分配任务 ----------------------------------------------------------------------------------------------
            result = c.execute(
                "select task_name from parallel_tasks where task_status = '0'"
                " and task_type = 'data_load_preprocess' and station_id=%s limit 1;", station_id)
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
                " and task_type = 'data_load_preprocess' and station_id=%s;", station_id)
            db.commit()
            if result < settings['scheduler']['process_pool_size']:
                logs.debug('任务加入：' + task_name)
                task_station_id, term = task_name.split('_', 1)
                task_station_id = eval(task_station_id)

                # 多进程任务方式-------------------------------------------------------------------------------------------
                args_short = [task_station_id, term, config_cluster_train[station_id], host, user, password,
                              database, charset, port, start_time, end_time, rate]
                scheduler.add_job(data_load_preprocess, executor=executor_name,
                                  args=args_short,
                                  coalesce=True, misfire_grace_time=None)
                time.sleep(1)
        datapre_t2 = datetime.datetime.now().timestamp()  # 数据获取与预处理完成时间

        logs.info('集成学习_数据获取与预处理完成')
        logs.info('集成学习_数据获取与预处理时间:' + str(int(datapre_t2 - datapre_t1)) + '秒')

        if csvrecord is True:
            try:
                timefile = open(csv_path + 'time.csv', 'a+')
                dfcsv = pandas.DataFrame(
                    numpy.array(['ensemble data-load-time: ' + str(int(datapre_t2 - datapre_t1)) + 's']).reshape(1, -1))
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                timefile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

        # 模型训练
        logs.info('集成学习_模型训练开始')
        train_t1 = datetime.datetime.now().timestamp()  # 训练开始时间
        number_end = 0
        number_task = c.execute("select task_name from parallel_tasks where task_type = 'train' and station_id=%s;",
                                station_id)
        db.commit()
        task_name_last = 'None'
        # 当测试任务全部完成，或者测试时间超过最大限值都会跳出循环
        while number_end < number_task:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            number_finished = c.execute(
                "select task_name from parallel_tasks where task_status = '2' and task_type = 'train'"
                " and station_id=%s;", station_id)
            db.commit()
            number_failed = c.execute(
                "select task_name from parallel_tasks where task_status = '3' and task_type = 'train'"
                " and station_id=%s;", station_id)
            db.commit()
            number_end = number_failed + number_finished
            logs.debug('已结束的任务数：' + str(number_end))

            # 判断并分配任务 ----------------------------------------------------------------------------------------------
            result = c.execute(
                "select task_name from parallel_tasks where task_status = '0' and task_type = 'train'"
                " and station_id=%s limit 1;", station_id)
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
                "select task_name from parallel_tasks where task_status = '1' and task_type = 'train'"
                " and station_id=%s;", station_id)
            db.commit()
            if result < settings['scheduler']['process_pool_size']:
                logs.debug('任务加入：' + task_name)
                station, model_name = task_name.split('short_', 1)
                task_station_id, term = str(station + 'short').split('_', 1)
                task_station_id = eval(task_station_id)

                # 多进程任务方式-------------------------------------------------------------------------------------------
                args_short = [task_station_id, term, model_name, config_cluster_train[station_id], host, user, password,
                              database, charset, port]
                scheduler.add_job(save_a_fitted_model, executor=executor_name,
                                  args=args_short,
                                  coalesce=True, misfire_grace_time=None)
                time.sleep(1)
        train_t2 = datetime.datetime.now().timestamp()  # 训练完成时间
        logs.info('集成学习_模型训练完成')
        logs.info('集成学习_模型训练时间:' + str(int(train_t2 - train_t1)) + '秒')

        if csvrecord is True:
            try:
                timefile = open(csv_path + 'time.csv', 'a+')
                dfcsv = pandas.DataFrame(
                    numpy.array(['ensemble  train-time: ' + str(int(train_t2 - train_t1)) + 's']).reshape(1, -1))
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                timefile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

        # --------------------------------------------------------------------------------------------------------------
        # 预测
        logs.info('集成学习_功率预测开始')
        predict_t1 = datetime.datetime.now().timestamp()  # 预测开始时间

        if model_name_cluster_setting is None:
            c.execute("select config_value from sys_config where config_key = 'ensemble_learn_model_name';")
            db.commit()
            record = c.fetchall()
            model_name_cluster_setting = eval(record[0][0])

        c.execute("DELETE FROM parallel_tasks where task_type = 'predict' and station_id=%s;", station_id)
        db.commit()
        c.execute('TRUNCATE TABLE predict_power_' + str(station_id) + '_train;')
        db.commit()

        # 配置信息
        config_cluster = get_config_cluster(host, user, password, database, charset, port, station_id)

        # 读取默认特征和默认参数 -------------------------------------------------------------------------------------------
        ultra_short_usecols_default, short_usecols_default, \
        ultra_short_best_parameter_default, short_best_parameter_default = \
            get_default_feature_parameter(host=host, user=user, password=password, database=database,
                                          charset=charset, port=port,
                                          predict_type=config_cluster[station_id]["type"],
                                          model_name_cluster_setting=model_name_cluster_setting)

        # 读取最优特征和参数，如果不存在，则使用默认值 -------------------------------------------------------------------------
        config_cluster = get_best_feature_parameter(
            host=host, user=user, password=password, database=database, charset=charset, port=port,
            model_name_cluster_setting=model_name_cluster_setting, station_id=station_id, config_cluster=config_cluster,
            short_best_parameter_default=short_best_parameter_default,
            ultra_short_best_parameter_default=ultra_short_best_parameter_default,
            ultra_short_usecols_default=ultra_short_usecols_default, short_usecols_default=short_usecols_default)

        predict_type = config_cluster[station_id]['type']
        ultra_short_usecols = config_cluster[station_id]['ultra_short_usecols']
        short_usecols = config_cluster[station_id]['short_usecols']
        short_model_state = ['_without_history_power']
        if predict_type == 'solar':
            ultra_short_model_state = ['_with_history_power', '_without_history_power']
        else:
            ultra_short_model_state = ['_with_history_power']
        # ----------------------------------------------------------------------------------------------------------
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        nwp_np_short, nwp_np_ultra_short = load_test_date_for_ensemble_learning(
            host=host, user=user, password=password, database=database, charset=charset, port=port,
            start_time_str=start_time, end_time_str=end_time, rate=rate, station_id=station_id,
            ultra_short_usecols=ultra_short_usecols, short_usecols=short_usecols,
            predict_type=predict_type)
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        sub_dir_path = "%s%s%s" % (str(config_cluster[station_id]['model_savepath']), str(station_id), '/short/')
        os.makedirs(sub_dir_path, exist_ok=True)
        save_model(nwp_np_short, sub_dir_path + 'short_test_data.pkl')

        sub_dir_path = "%s%s%s" % (str(config_cluster[station_id]['model_savepath']), str(station_id), '/ultra_short/')
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
        # 重复任务的减并2022/12/9
        model_name_cluster_ultra_short, target_model_name_cluster_ultra_short, ultra_short_target_task, \
        model_name_cluster_short, target_model_name_cluster_short, short_target_task = \
            get_model_name(ultra_short_usecols_default, short_usecols_default,
                           ultra_short_best_parameter_default, short_best_parameter_default,
                           model_name_cluster_setting, host=host, user=user, password=password,
                           database=database, charset=charset, port=port, station_id=station_id,
                           task=task, do_logs=0)
        # ----------------------------------------------------------------------------------------------------------
        #  添加短期预测的任务2022/11/20
        result = c.execute(
            "update parallel_tasks_station set task_status = '" + str(
                station_id) + "正在写预测任务' where task_name = 'ensemble' and task_status = '"+str(
                station_id)+"写训练任务完成' and station_id = %s;", station_id)
        db.commit()
        if result == 1:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            for model_name in model_name_cluster_short:
                for model_state in short_model_state:
                    if model_name in mode_need_split_in_time:
                        for j in range(number_copies):
                            c.execute("INSERT INTO parallel_tasks "
                                      "(task_name, task_status, distribution_times, task_type, station_id) "
                                      "VALUES (%s, %s, %s, %s, %s) ",
                                      tuple(
                                          [str(station_id) + '_short_' + model_name + model_state + '-' + str(j + 1) +
                                           '/' + str(number_copies)]) +
                                      tuple(['0']) +
                                      tuple([0]) + tuple(['predict']) + tuple([station_id]))
                    else:
                        c.execute("INSERT INTO parallel_tasks "
                                  "(task_name, task_status, distribution_times, task_type, station_id)"
                                  " VALUES (%s, %s, %s, %s, %s) ",
                                  tuple([str(station_id) + '_short_' + model_name + model_state]) + tuple(['0']) +
                                  tuple([0]) + tuple(['predict']) + tuple([station_id]))
                db.commit()
            #  添加超短期预测的任务2022/11/20
            for model_name in model_name_cluster_ultra_short:
                for model_state in ultra_short_model_state:
                    if model_name in mode_need_split_in_time:
                        for j in range(number_copies):
                            c.execute("INSERT INTO parallel_tasks "
                                      "(task_name, task_status, distribution_times, task_type, station_id) "
                                      "VALUES (%s, %s, %s, %s, %s) ",
                                      tuple(
                                          [str(station_id) + '_ultra_short_' + model_name + model_state + '-' +
                                           str(j + 1) + '/' + str(number_copies)]) +
                                      tuple(['0']) +
                                      tuple([0]) + tuple(['predict']) + tuple([station_id]))
                    else:
                        c.execute("INSERT INTO parallel_tasks "
                                  "(task_name, task_status, distribution_times, task_type, station_id) "
                                  "VALUES (%s, %s, %s, %s, %s) ",
                                  tuple([str(station_id) + '_ultra_short_' + model_name + model_state]) +
                                  tuple(['0']) +
                                  tuple([0]) + tuple(['predict']) + tuple([station_id]))
                db.commit()

            c.execute(
                "update parallel_tasks_station set task_status = '" + str(
                    station_id) + "写预测任务完成' where task_name = 'ensemble' and task_status = '"+str(
                    station_id)+"正在写预测任务' and station_id = %s;", station_id)
            db.commit()

        result = c.execute(
            "select task_status from parallel_tasks_station where task_name = 'ensemble' and task_status = '" + str(
                station_id) + "写预测任务完成' and station_id = %s;", station_id)
        db.commit()

        while result == 0:
            time.sleep(1)
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            result = c.execute(
                "select task_status from parallel_tasks_station where task_name = 'ensemble' and task_status = '" + str(
                    station_id) + "写预测任务完成' and station_id = %s;", station_id)
            db.commit()

        number_end = 0
        number_task = c.execute("select task_name from parallel_tasks where task_type = 'predict'"
                                " and station_id=%s;", station_id)
        db.commit()
        task_name_last = 'None'

        evaluate_start_time = datetime.datetime.now().timestamp()  # 测试开始时间
        if len(nwp_np_ultra_short) == 0:
            logs.warning('选择时段的集成学习预测数据量为0，集成学习任务退出！！！')
            return
        evaluate_time_maximum_limit = number_task * len(nwp_np_ultra_short) * 0.7  # 测试时间最大限值
        # 当测试任务全部完成，或者测试时间超过最大限值都会跳出循环
        while number_end < number_task and \
                (datetime.datetime.now().timestamp() - evaluate_start_time) < evaluate_time_maximum_limit:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            number_finished = c.execute(
                "select task_name from parallel_tasks where task_status = '2' and task_type = 'predict'"
                " and station_id=%s;", station_id)
            db.commit()
            number_failed = c.execute(
                "select task_name from parallel_tasks where task_status = '3' and task_type = 'predict'"
                " and station_id=%s;", station_id)
            db.commit()
            number_end = number_failed + number_finished
            logs.debug('已结束的任务数：' + str(number_end))

            # 分布式并行任务处理时，对于超时或异常任务要进行重复捕捉 2022-09-22 --------------------------------------------------
            current_time = datetime.datetime.now().timestamp()
            result = c.execute("select task_name from parallel_tasks where "
                               "task_status = '1' and "
                               "(%s-task_start_time)>5400 and "
                               "distribution_times<2 and task_type = 'predict' and station_id=%s limit 1;",
                               (current_time, station_id))
            db.commit()
            if result != 0:
                record = c.fetchall()
                task_name = record[0][0]
                c.execute(
                    "update parallel_tasks set task_status = '0' where task_name = %s and task_status = '1' "
                    "and task_type = 'predict' and station_id=%s;",
                    (task_name, station_id))
                db.commit()

            # 判断并分配任务 ----------------------------------------------------------------------------------------------
            result = c.execute(
                "select task_name from parallel_tasks where task_status = '0'"
                " and task_type = 'predict' and station_id=%s limit 1;", station_id)
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
                "select task_name from parallel_tasks where task_status = '1' and task_type = 'predict'"
                " and station_id=%s;", station_id)
            db.commit()
            if result < settings['scheduler']['process_pool_size']:
                logs.debug('任务加入：' + task_name)
                station, model = task_name.split('short_', 1)
                task_station_id, term = str(station + 'short').split('_', 1)
                task_station_id = eval(task_station_id)
                model_name, model_state = model.split('_', 1)
                model_state = '_' + model_state

                # 多进程任务方式-------------------------------------------------------------------------------------------
                if term == 'short':
                    # 短期预测
                    args_short = [host, user, password, database, charset, port,
                                  config_cluster[station_id]['type'], config_cluster[station_id]['model_savepath'],
                                  model_name, model_state, task_station_id,
                                  config_cluster[station_id]['sr_col'],
                                  config_cluster[station_id]['online_capacity']]
                    scheduler.add_job(ensemble_learning_evaluate_short, executor=executor_name,
                                      args=args_short,
                                      coalesce=True, misfire_grace_time=None)
                else:
                    # 超短期预测
                    args_ultra_short = [host, user, password, database, charset, port,
                                        config_cluster[station_id]['type'],
                                        config_cluster[station_id]['model_savepath'],
                                        model_name, model_state, task_station_id,
                                        config_cluster[station_id]['sr_col'],
                                        config_cluster[station_id]['online_capacity']]
                    scheduler.add_job(ensemble_learning_evaluate_ultra_short, executor=executor_name,
                                      args=args_ultra_short,
                                      coalesce=True, misfire_grace_time=None)
                time.sleep(1)

        predict_t2 = datetime.datetime.now().timestamp()  # 预测结束时间
        logs.info('集成学习_功率预测完成')
        logs.info('集成学习_功率预测时间:' + str(int(predict_t2 - predict_t1)) + '秒')

        if csvrecord is True:
            try:
                shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
                ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')

                dfcsv = pandas.DataFrame(
                    [str(station_id) + 'short_usecols: ' + str(config_cluster_train[station_id]["short_usecols"])])
                dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')

                dfcsv = pandas.DataFrame([str(station_id) + 'ultra_short_usecols: ' +
                                          str(config_cluster_train[station_id]["ultra_short_usecols"])])
                dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')
                shortfile.close()
                ultrafile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

            try:
                timefile = open(csv_path + 'time.csv', 'a+')
                dfcsv = pandas.DataFrame(
                    numpy.array(['ensemble  predict-time: ' + str(int(predict_t2 - predict_t1)) + 's']).reshape(1, -1))
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                timefile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

        # --------------------------------------------------------------------------------------------------------------
        logs.info('集成学习_模型评价开始')
        result = c.execute(
            "update parallel_tasks_station set task_status = '" + str(
                station_id) + "正在模型评价' where task_name = 'ensemble' and task_status = '" + str(
                station_id) + "写预测任务完成' and station_id = %s;", station_id)
        db.commit()
        if result == 1:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            evaluate_t1 = datetime.datetime.now().timestamp()  # 模型评价开始时间
            model_evaluate_save(
                host=host, user=user, password=password, database=database, charset=charset, port=port,
                config_cluster_train=config_cluster_train, task=task, station_id=station_id,
                ensemble_learn_stands=ensemble_learn_stands, csvrecord=csvrecord, figurerecord=figurerecord,
                csv_path=csv_path, figure_path=figure_path, model_name_cluster_setting=model_name_cluster_setting)
            interval_learning(host, user, password, database, charset, port, station_id)
            evaluate_t2 = datetime.datetime.now().timestamp()  # 模型评价完成时间
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
            c = db.cursor()

            if csvrecord is True:
                try:
                    timefile = open(csv_path + 'time.csv', 'a+')
                    dfcsv = pandas.DataFrame(numpy.array(['ensemble evaluate-time: ' +
                                                          str(int(evaluate_t2 - evaluate_t1)) + 's']).reshape(1, -1))
                    dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                    timefile.close()
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.warning('写CSV失败！')

            logs.info('集成学习_模型评价完成')

            logs.info('集成学习完成')

            function_end_time = datetime.datetime.now().timestamp()
            logs.info('集成学习总时间:' + str(int(function_end_time - function_start_time)) + '秒')
            hours = int((function_end_time - function_start_time) // 3600)
            mins = int(((function_end_time - function_start_time) % 3600) // 60)
            second = int((function_end_time - function_start_time) % 60)
            logs.info('集成学习总时间:' + str(hours) + '时' + str(mins) + '分' + str(second) + '秒')

            if csvrecord is True:
                try:
                    timefile = open(csv_path + 'time.csv', 'a+')
                    dfcsv = pandas.DataFrame(numpy.array(
                        ['ensemble time: ' + str(int(function_end_time - function_start_time)) + 's']).reshape(1, -1))
                    dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                    timefile.close()
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.warning('写CSV失败！')

                try:
                    timefile = open(csv_path + 'time.csv', 'a+')

                    # 模型的训练时间
                    c.execute("select id, task_name, task_start_time, task_end_time"
                              " from parallel_tasks where task_type = 'train' and station_id=%s;", station_id)
                    db.commit()
                    des = c.description
                    record = c.fetchall()
                    coul = list(iterm[0] for iterm in des)
                    dataframe = pandas.DataFrame(record, columns=coul)
                    dataarray = dataframe.values

                    term_cl = ['short', 'ultra_short']
                    term_c = ['s', 'ultra_s']

                    train_time_model_cluster = numpy.zeros((2, len(model_name_cluster_setting)))
                    for i in range(len(model_name_cluster_setting)):
                        for j in range(len(term_cl)):
                            for k in range(len(dataarray)):
                                a, model_name = dataarray[k, 1].split('hort_', 1)
                                stationid, predict_term = a.split('_', 1)
                                if model_name == model_name_cluster_setting[i] and predict_term == term_c[j] and str(
                                        station_id) == stationid:
                                    if dataarray[k, 3] is not None:
                                        train_time_model_cluster[j, i] = \
                                            train_time_model_cluster[j, i] + dataarray[k, 3] - dataarray[k, 2]

                    dfcsv = pandas.DataFrame([str(station_id) + ' model train time:'])
                    dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                    short_predict_time_df = pandas.DataFrame(train_time_model_cluster,
                                                             index=term_cl, columns=model_name_cluster_setting)
                    short_predict_time_df.to_csv(timefile, index=True, header=True, encoding="utf_8", mode='a+')
                    # --------------------------------------------------------------------------------------------------
                    c.execute("select config_value from sys_config where config_key = 'mode_need_split_in_time';")
                    db.commit()
                    record = c.fetchall()
                    mode_need_split_in_time, number_copies = record[0][0].split('/', 1)
                    mode_need_split_in_time = eval(mode_need_split_in_time)
                    number_copies = eval(number_copies)

                    model_name_cluster_setting_split = model_name_cluster_setting.copy()
                    for model_name in model_name_cluster_setting:
                        if model_name in mode_need_split_in_time:
                            where_model = model_name_cluster_setting_split.index(model_name)
                            for i in range(number_copies):
                                model_name_cluster_setting_split.insert(where_model, model_name + '-' + str(
                                    int(number_copies - i)) + '/' + str(int(number_copies)))
                            model_name_cluster_setting_split.remove(model_name)
                    # --------------------------------------------------------------------------------------------------
                    c.execute("select id, task_name, task_start_time, task_end_time"
                              " from parallel_tasks where task_type = 'predict' and station_id=%s;", station_id)
                    db.commit()
                    des = c.description
                    record = c.fetchall()
                    coul = list(iterm[0] for iterm in des)
                    dataframe = pandas.DataFrame(record, columns=coul)
                    dataarray = dataframe.values

                    predict_type = config_cluster[station_id]['type']
                    short_model_s = ['without']
                    if predict_type == 'solar':
                        ultra_short_model_s = ['with', 'without']
                    else:
                        ultra_short_model_s = ['with']

                    short_predict_time_model_cluster = numpy.zeros(
                        (len(short_model_s), len(model_name_cluster_setting_split)))
                    for i in range(len(model_name_cluster_setting_split)):
                        for j in range(len(short_model_s)):
                            for k in range(len(dataarray)):
                                a, model_name = dataarray[k, 1].split('hort_', 1)
                                stationid, predict_term = a.split('_', 1)
                                model_name, b = model_name.split('_', 1)
                                model_state, num = b.split('_history_power', 1)
                                model_name = model_name + num
                                if model_name == model_name_cluster_setting_split[i] and predict_term == 's' and \
                                        str(station_id) == stationid and model_state == short_model_s[j]:
                                    if dataarray[k, 2] is not None and dataarray[k, 3] is not None:
                                        short_predict_time_model_cluster[j, i] = \
                                            short_predict_time_model_cluster[j, i] + dataarray[k, 3] - dataarray[k, 2]
                    dfcsv = pandas.DataFrame([str(station_id) + ' model short predict time:'])
                    dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')

                    short_predict_time_df = pandas.DataFrame(short_predict_time_model_cluster, index=short_model_s,
                                                             columns=model_name_cluster_setting_split)
                    short_predict_time_df.to_csv(timefile, index=True, header=True, encoding="utf_8", mode='a+')

                    ultra_short_predict_time_model_cluster = numpy.zeros(
                        (len(ultra_short_model_s), len(model_name_cluster_setting_split)))
                    for i in range(len(model_name_cluster_setting_split)):
                        for j in range(len(ultra_short_model_s)):
                            for k in range(len(dataarray)):
                                a, model_name = dataarray[k, 1].split('hort_', 1)
                                stationid, predict_term = a.split('_', 1)
                                model_name, b = model_name.split('_', 1)
                                model_state, num = b.split('_history_power', 1)
                                model_name = model_name + num
                                if model_name == model_name_cluster_setting_split[i] and predict_term == 'ultra_s' and \
                                        str(station_id) == stationid and model_state == ultra_short_model_s[j]:
                                    if dataarray[k, 2] is not None and dataarray[k, 3] is not None:
                                        ultra_short_predict_time_model_cluster[j, i] = \
                                            ultra_short_predict_time_model_cluster[j, i] + dataarray[k, 3] - \
                                            dataarray[k, 2]

                    dfcsv = pandas.DataFrame([str(station_id) + ' model ultra short predict time:'])
                    dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                    ultra_short_predict_time_df = pandas.DataFrame(ultra_short_predict_time_model_cluster,
                                                                   index=ultra_short_model_s,
                                                                   columns=model_name_cluster_setting_split)
                    ultra_short_predict_time_df.to_csv(timefile, index=True, header=True, encoding="utf_8", mode='a+')

                    timefile.close()
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.warning('写CSV失败！')

            c.execute(
                "update parallel_tasks_station set task_status = '" + str(
                    station_id) + "模型评价完成' where task_name = 'ensemble' and task_status = '" + str(
                    station_id) + "正在模型评价' and station_id = %s;", station_id)
            db.commit()
            time.sleep(3)

        result = c.execute(
            "select task_status from parallel_tasks_station where task_name = 'ensemble'"
            " and task_status = '" + str(station_id) + "模型评价完成' and station_id = %s;", station_id)
        db.commit()

        while result == 0:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            time.sleep(1)
            result = c.execute(
                "select task_status from parallel_tasks_station where task_name = 'ensemble'"
                " and task_status = '" + str(station_id) + "模型评价完成' and station_id = %s;", station_id)
            db.commit()

        c.execute(
            "update parallel_tasks_station set task_status = '" + str(
                station_id) + "集成学习完成' where task_name = 'ensemble'"
            " and task_status = '" + str(station_id) + "模型评价完成' and station_id = %s;", station_id)
        db.commit()

    c.close()
    db.close()


@catch_exception("get_config_cluster error: ", exc_info=True, default_return=None)
def get_config_cluster(host, user, password, database, charset, port, station_id):
    """
    读取配置
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param station_id: 场站id
    :return: config_cluster
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    c.execute("select id, name, type, sr_col, capacity, model_savepath from configure where id = %s;", station_id)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)

    # 配置信息
    config_cluster = {station_id: {"id": dataframe_config.loc[:, 'id'][0],
                                   "name": dataframe_config.loc[:, "name"][0],
                                   "type": dataframe_config.loc[:, "type"][0],
                                   "sr_col": dataframe_config.loc[:, "sr_col"][0],
                                   "online_capacity": dataframe_config.loc[:, "capacity"][0],
                                   "model_savepath": dataframe_config.loc[:, "model_savepath"][0]}}
    c.close()
    db.close()
    return config_cluster


@catch_exception("get_best_feature_parameter error: ", exc_info=True, default_return=None)
def get_best_feature_parameter(host, user, password, database, charset, port,
                               model_name_cluster_setting, station_id, config_cluster,
                               short_best_parameter_default, ultra_short_best_parameter_default,
                               ultra_short_usecols_default, short_usecols_default):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param station_id: 场站id
    :param config_cluster: 场站配置
    :param model_name_cluster_setting: 集成学习模型列表
    :param ultra_short_usecols_default: 超短期默认特征
    :param short_usecols_default: 短期默认特征
    :param ultra_short_best_parameter_default: 超短期默认参数
    :param short_best_parameter_default: 短期默认参数
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读取最优特征和参数，如果不存在，则使用默认值 -----------------------------------------------------------------------------
    c.execute("select * from best_feature_parameters_and_model where id=%s;", station_id)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_best = pandas.DataFrame(record, columns=coul)

    # 模型短期最优参数，如果不存在，则使用默认值
    config_cluster = ensemble_learning.get_best_feature_parameter(
        dataframe_best=dataframe_best, model_name_cluster_setting=model_name_cluster_setting, station_id=station_id,
        config_cluster=config_cluster, short_best_parameter_default=short_best_parameter_default,
        ultra_short_best_parameter_default=ultra_short_best_parameter_default,
        ultra_short_usecols_default=ultra_short_usecols_default, short_usecols_default=short_usecols_default)
    c.close()
    db.close()

    return config_cluster


@catch_exception("get_default_feature_parameter error: ", exc_info=True, default_return=None)
def get_default_feature_parameter(host, user, password, database, charset, port, predict_type=None,
                                  model_name_cluster_setting=None):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param predict_type: 预测类型
    :param model_name_cluster_setting: 集成学习模型列表
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读取默认特征和默认参数 -------------------------------------------------------------------------------------------
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
    for i in range(len(model_name_cluster_setting)):
        short_wind_best_parameter_default[model_name_cluster_setting[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_wind"][model_name_cluster_setting[i]].values[
                0])
        ultra_short_wind_best_parameter_default[model_name_cluster_setting[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"][
                model_name_cluster_setting[i]].values[0])
        short_solar_best_parameter_default[model_name_cluster_setting[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_solar"][
                model_name_cluster_setting[i]].values[0])
        ultra_short_solar_best_parameter_default[model_name_cluster_setting[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"][
                model_name_cluster_setting[i]].values[0])

    # 特征默认值
    if predict_type == 'wind':
        # 模型参数默认值
        short_best_parameter_default = {}
        ultra_short_best_parameter_default = {}
        for i in range(len(model_name_cluster_setting)):
            short_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "short_wind"][
                    model_name_cluster_setting[i]].values[
                    0])
            ultra_short_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"][
                    model_name_cluster_setting[i]].values[0])

        ultra_short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"]['usecols'].values[0])
        short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_wind"]['usecols'].values[0])
    else:
        # 模型参数默认值
        short_best_parameter_default = {}
        ultra_short_best_parameter_default = {}
        for i in range(len(model_name_cluster_setting)):
            short_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "short_solar"][
                    model_name_cluster_setting[i]].values[0])
            ultra_short_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"][
                    model_name_cluster_setting[i]].values[0])

        ultra_short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"]['usecols'].values[0])
        short_usecols_default = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_solar"]['usecols'].values[0])
    c.close()
    db.close()
    return ultra_short_usecols_default, short_usecols_default, ultra_short_best_parameter_default, \
           short_best_parameter_default


@catch_exception("get_model_name error: ", exc_info=True, default_return=None)
def get_model_name(ultra_short_usecols_default, short_usecols_default, ultra_short_best_parameter_default,
                   short_best_parameter_default, model_name_cluster_setting,
                   host, user, password, database, charset, port,
                   station_id, task=1, do_logs=0):
    """
    :param ultra_short_usecols_default: 超短期默认特征
    :param short_usecols_default: 短期默认特征
    :param ultra_short_best_parameter_default: 超短期默认参数
    :param short_best_parameter_default: 短期默认参数
    :param model_name_cluster_setting: 集成学习模型列表
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param station_id: 场站id
    :param task: 任务序号
    :param do_logs: 是否打印logs
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    # 重复任务的减并2022/12/9
    if task == 1:
        logs.debug('任务1：没有特征工程和参数寻优结果，正常执行集成学习。')
        model_name_cluster_short = model_name_cluster_setting.copy()
        target_model_name_cluster_short = []
        short_target_task = None
    elif task == 2:
        logs.debug('任务2：已执行特征工程，未执行参数寻优，需判断特征工程结果是否与默认值有区别！')
        c.execute("select config_value from sys_config where config_key = 'task1_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            result = c.execute("select usecols from best_feature_parameters_and_model"
                               " where id = %s and predict_term = %s;", (station_id, 'short'))
            db.commit()
            record = c.fetchall()
            if result == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                model_name_cluster_short = []
                target_model_name_cluster_short = []
                short_target_task = None

            elif record[0][0] is None or len(record[0][0]) == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                model_name_cluster_short = []
                target_model_name_cluster_short = []
                short_target_task = None
            elif record[0][0] == str(short_usecols_default):
                if do_logs == 1:
                    logs.info(str(station_id) + '短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务1结果')
                else:
                    logs.debug(str(station_id) + '短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务1结果')
                model_name_cluster_short = []
                target_model_name_cluster_short = model_name_cluster_setting.copy()
                short_target_task = 1
            else:
                logs.debug('如果特征工程有改变，正常执行集成学习。')
                model_name_cluster_short = model_name_cluster_setting.copy()
                target_model_name_cluster_short = []
                short_target_task = None
        else:
            logs.debug('未执行任务1，即使任务2的最优特征与默认特征一致，也无法使用任务1的结果，需正常执行集成学习。')
            model_name_cluster_short = model_name_cluster_setting.copy()
            target_model_name_cluster_short = []
            short_target_task = None
    elif task == 3:
        logs.debug('任务3：未执行特征工程，已执行参数寻优，需判断参数寻优结果是否与默认值有区别！')
        c.execute("select config_value from sys_config where config_key = 'task1_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            result = c.execute("select " + ','.join(iterm for iterm in model_name_cluster_setting) +
                               " from best_feature_parameters_and_model"
                               " where id = %s and predict_term = %s;", (station_id, 'short'))
            db.commit()
            if result == 0:
                if do_logs == 1:
                    logs.info('参数寻优失败，不再进行集成学习。')
                else:
                    logs.debug('参数寻优失败，不再进行集成学习。')
                model_name_cluster_short = []
                target_model_name_cluster_short = []
                short_target_task = None
            else:
                des = c.description
                record = c.fetchall()
                coul = list(iterm[0] for iterm in des)
                dataframe_best_parameter = pandas.DataFrame(record, columns=coul)
                model_name_cluster_short = model_name_cluster_setting.copy()
                target_model_name_cluster_short = []
                short_target_task = 1
                for model_name in model_name_cluster_setting:
                    if model_name == 'MLP':
                        model_name_cluster_short.remove(model_name)
                        target_model_name_cluster_short.append(model_name)
                    else:
                        if dataframe_best_parameter.loc[:, model_name][0] is None or len(
                                dataframe_best_parameter.loc[:, model_name][0]) == 0:
                            if do_logs == 1:
                                logs.info(str(station_id) + '短期，' + model_name + '参数寻优失败，不再进行集成学习')
                            else:
                                logs.debug(str(station_id) + '短期，' + model_name + '参数寻优失败，不再进行集成学习')
                            model_name_cluster_short.remove(model_name)
                        elif str(dataframe_best_parameter.loc[:, model_name][0]) == str(
                                short_best_parameter_default[model_name]):
                            if do_logs == 1:
                                logs.info(str(station_id) + '短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习')
                            else:
                                logs.debug(str(station_id) + '短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习')
                            model_name_cluster_short.remove(model_name)
                            target_model_name_cluster_short.append(model_name)
                        else:
                            logs.debug('如果参数寻优有改变，正常执行集成学习。')
        else:
            logs.debug('未执行任务1，即使任务3的最优参数与默认参数一致，也无法使用任务1的结果，需正常执行集成学习。')
            model_name_cluster_short = model_name_cluster_setting.copy()
            target_model_name_cluster_short = []
            short_target_task = None
    else:
        logs.debug('任务4：已执行特征工程，已执行参数寻优，需判断特征工程和参数寻优结果是否与默认值有区别！')
        c.execute("select config_value from sys_config where config_key = 'task3_parameter_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            result = c.execute("select usecols from best_feature_parameters_and_model"
                               " where id = %s and predict_term = %s;", (station_id, 'short'))
            db.commit()
            record = c.fetchall()
            if result == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                model_name_cluster_short = []
                target_model_name_cluster_short = []
                short_target_task = None
            elif record[0][0] is None or len(record[0][0]) == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '短期，特征工程失败，不再进行集成学习')
                model_name_cluster_short = []
                target_model_name_cluster_short = []
                short_target_task = None
            elif record[0][0] == str(short_usecols_default):
                if do_logs == 1:
                    logs.info(str(station_id) + '短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务3的结果！')
                else:
                    logs.debug(str(station_id) + '短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务3的结果！')
                model_name_cluster_short = []
                target_model_name_cluster_short = model_name_cluster_setting.copy()
                short_target_task = 3
                for model_name in model_name_cluster_setting:
                    if model_name != 'MLP':
                        result1 = c.execute(
                            "select " + model_name + " from best_feature_parameters_and_model_parameter_ensemble"
                                                     " where id=%s and predict_term=%s", (station_id, 'short'))
                        db.commit()
                        record1 = c.fetchall()
                        result2 = c.execute(
                            "select " + model_name + " from best_feature_parameters_and_model"
                                                     " where id=%s and predict_term=%s", (station_id, 'short'))
                        db.commit()
                        record2 = c.fetchall()
                        if result1 == 0:
                            if result2 == 0:
                                target_model_name_cluster_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            elif record2[0][0] is None or len(record2[0][0]) == 0:
                                target_model_name_cluster_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            else:
                                model_name_cluster_short.append(model_name)
                                target_model_name_cluster_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '短期，由于任务3中参数寻优失败了，仍需要集成学习！')
                                else:
                                    logs.debug(str(station_id) + model_name + '短期，由于任务3中参数寻优失败了，仍需要集成学习！')
                        elif record1[0][0] is None or len(record1[0][0]) == 0:
                            if result2 == 0:
                                target_model_name_cluster_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            elif record2[0][0] is None or len(record2[0][0]) == 0:
                                target_model_name_cluster_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            else:
                                model_name_cluster_short.append(model_name)
                                target_model_name_cluster_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '短期，由于任务3中参数寻优失败了，仍需要集成学习！')
                                else:
                                    logs.debug(str(station_id) + model_name + '短期，由于任务3中参数寻优失败了，仍需要集成学习！')
            else:
                logs.debug('如果特征工程有改变，判断参数寻优结果是否与默认值相同，如果相同，直接采用任务2结果。')
                c.execute("select config_value from sys_config where config_key = 'task2_feature_ensemble';")
                db.commit()
                record = c.fetchall()
                do_task = record[0][0]
                if do_task == '1':
                    result = c.execute("select " + ','.join(iterm for iterm in model_name_cluster_setting) +
                                       " from best_feature_parameters_and_model"
                                       " where id = %s and predict_term = %s;", (station_id, 'short'))
                    db.commit()
                    if result == 0:
                        if do_logs == 1:
                            logs.info('参数寻优失败，不再进行集成学习。')
                        else:
                            logs.debug('参数寻优失败，不再进行集成学习。')
                        model_name_cluster_short = []
                        target_model_name_cluster_short = []
                        short_target_task = 3
                        if 'MLP' in model_name_cluster_setting:
                            target_model_name_cluster_short.append('MLP')
                    else:
                        des = c.description
                        record = c.fetchall()
                        coul = list(iterm[0] for iterm in des)
                        dataframe_best_parameter = pandas.DataFrame(record, columns=coul)
                        model_name_cluster_short = model_name_cluster_setting.copy()
                        target_model_name_cluster_short = []
                        short_target_task = 2
                        for model_name in model_name_cluster_setting:
                            if model_name == 'MLP':
                                model_name_cluster_short.remove(model_name)
                                target_model_name_cluster_short.append(model_name)
                            else:
                                if dataframe_best_parameter.loc[:, model_name][0] is None or len(
                                        dataframe_best_parameter.loc[:, model_name][0]) == 0:
                                    if do_logs == 1:
                                        logs.info(str(station_id) + '短期，' + model_name + '参数寻优失败，不再进行集成学习')
                                    else:
                                        logs.debug(str(station_id) + '短期，' + model_name + '参数寻优失败，不再进行集成学习')
                                    model_name_cluster_short.remove(model_name)
                                elif str(dataframe_best_parameter.loc[:, model_name][0]) == str(
                                        short_best_parameter_default[model_name]):
                                    if do_logs == 1:
                                        logs.info(str(station_id) + '短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习，直接调用任务2结果')
                                    else:
                                        logs.debug(str(station_id) + '短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习，直接调用任务2结果')
                                    model_name_cluster_short.remove(model_name)
                                    target_model_name_cluster_short.append(model_name)
                                else:
                                    logs.debug('如果参数寻优有改变，正常执行集成学习。')
                else:
                    logs.debug('未执行任务2，即使任务4的最优参数与默认参数一致，也无法对任务简化合并。')
                    model_name_cluster_short = model_name_cluster_setting.copy()
                    target_model_name_cluster_short = []
                    short_target_task = None
        else:
            logs.debug('未执行任务3，即使任务4的最优特征与默认特征一致，也无法对任务简化合并。')
            model_name_cluster_short = model_name_cluster_setting.copy()
            target_model_name_cluster_short = []
            short_target_task = None
    # 超短期
    if task == 1:
        logs.debug('任务1：没有特征工程和参数寻优结果，正常执行集成学习。')
        model_name_cluster_ultra_short = model_name_cluster_setting.copy()
        target_model_name_cluster_ultra_short = []
        ultra_short_target_task = None
    elif task == 2:
        logs.debug('任务2：已执行特征工程，未执行参数寻优，需判断特征工程结果是否与默认值有区别！')
        c.execute("select config_value from sys_config where config_key = 'task1_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            result = c.execute("select usecols from best_feature_parameters_and_model"
                               " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
            db.commit()
            record = c.fetchall()
            if result == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                model_name_cluster_ultra_short = []
                target_model_name_cluster_ultra_short = []
                ultra_short_target_task = None

            elif record[0][0] is None or len(record[0][0]) == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                model_name_cluster_ultra_short = []
                target_model_name_cluster_ultra_short = []
                ultra_short_target_task = None
            elif record[0][0] == str(ultra_short_usecols_default):
                if do_logs == 1:
                    logs.info(str(station_id) + '超短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务1结果')
                else:
                    logs.debug(str(station_id) + '超短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务1结果')
                model_name_cluster_ultra_short = []
                target_model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                ultra_short_target_task = 1
            else:
                logs.debug('如果特征工程有改变，正常执行集成学习。')
                model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                target_model_name_cluster_ultra_short = []
                ultra_short_target_task = None
        else:
            logs.debug('未执行任务1，即使任务2的最优特征与默认特征一致，也无法使用任务1的结果，需正常执行集成学习。')
            model_name_cluster_ultra_short = model_name_cluster_setting.copy()
            target_model_name_cluster_ultra_short = []
            ultra_short_target_task = None
    elif task == 3:
        logs.debug('任务3：未执行特征工程，已执行参数寻优，需判断参数寻优结果是否与默认值有区别！')
        c.execute("select config_value from sys_config where config_key = 'task1_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            result = c.execute("select " + ','.join(iterm for iterm in model_name_cluster_setting) +
                               " from best_feature_parameters_and_model"
                               " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
            db.commit()
            if result == 0:
                if do_logs == 1:
                    logs.info('参数寻优失败，不再进行集成学习。')
                else:
                    logs.debug('参数寻优失败，不再进行集成学习。')
                model_name_cluster_ultra_short = []
                target_model_name_cluster_ultra_short = []
                ultra_short_target_task = None
            else:
                des = c.description
                record = c.fetchall()
                coul = list(iterm[0] for iterm in des)
                dataframe_best_parameter = pandas.DataFrame(record, columns=coul)
                model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                target_model_name_cluster_ultra_short = []
                ultra_short_target_task = 1
                for model_name in model_name_cluster_setting:
                    if model_name == 'MLP':
                        model_name_cluster_ultra_short.remove(model_name)
                        target_model_name_cluster_ultra_short.append(model_name)
                    else:
                        if dataframe_best_parameter.loc[:, model_name][0] is None or len(
                                dataframe_best_parameter.loc[:, model_name][0]) == 0:
                            if do_logs == 1:
                                logs.info(str(station_id) + '超短期，' + model_name + '参数寻优失败，不再进行集成学习')
                            else:
                                logs.debug(str(station_id) + '超短期，' + model_name + '参数寻优失败，不再进行集成学习')
                            model_name_cluster_ultra_short.remove(model_name)
                        elif str(dataframe_best_parameter.loc[:, model_name][0]) == str(
                                ultra_short_best_parameter_default[model_name]):
                            if do_logs == 1:
                                logs.info(str(station_id) + '超短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习')
                            else:
                                logs.debug(str(station_id) + '超短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习')
                            model_name_cluster_ultra_short.remove(model_name)
                            target_model_name_cluster_ultra_short.append(model_name)
                        else:
                            logs.debug('如果参数寻优有改变，正常执行集成学习。')
        else:
            logs.debug('未执行任务1，即使任务3的最优参数与默认参数一致，也无法使用任务1的结果，需正常执行集成学习。')
            model_name_cluster_ultra_short = model_name_cluster_setting.copy()
            target_model_name_cluster_ultra_short = []
            ultra_short_target_task = None
    else:
        logs.debug('任务4：已执行特征工程，已执行参数寻优，需判断特征工程和参数寻优结果是否与默认值有区别！')
        c.execute("select config_value from sys_config where config_key = 'task3_parameter_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            result = c.execute("select usecols from best_feature_parameters_and_model"
                               " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
            db.commit()
            record = c.fetchall()
            if result == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                model_name_cluster_ultra_short = []
                target_model_name_cluster_ultra_short = []
                ultra_short_target_task = None
            elif record[0][0] is None or len(record[0][0]) == 0:
                if do_logs == 1:
                    logs.info(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                else:
                    logs.debug(str(station_id) + '超短期，特征工程失败，不再进行集成学习')
                model_name_cluster_ultra_short = []
                target_model_name_cluster_ultra_short = []
                ultra_short_target_task = None
            elif record[0][0] == str(ultra_short_usecols_default):
                if do_logs == 1:
                    logs.info(str(station_id) + '超短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务3的结果！')
                else:
                    logs.debug(str(station_id) + '超短期，特征工程结果与默认特征相同，不再进行集成学习，直接调用任务3的结果！')
                model_name_cluster_ultra_short = []
                target_model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                ultra_short_target_task = 3
                for model_name in model_name_cluster_setting:
                    if model_name != 'MLP':
                        result1 = c.execute(
                            "select " + model_name + " from best_feature_parameters_and_model_parameter_ensemble"
                                                     " where id=%s and predict_term=%s",
                            (station_id, 'ultra_short'))
                        db.commit()
                        record1 = c.fetchall()
                        result2 = c.execute(
                            "select " + model_name + " from best_feature_parameters_and_model"
                                                     " where id=%s and predict_term=%s",
                            (station_id, 'ultra_short'))
                        db.commit()
                        record2 = c.fetchall()
                        if result1 == 0:
                            if result2 == 0:
                                target_model_name_cluster_ultra_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            elif record2[0][0] is None or len(record2[0][0]) == 0:
                                target_model_name_cluster_ultra_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            else:
                                model_name_cluster_ultra_short.append(model_name)
                                target_model_name_cluster_ultra_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '超短期，由于任务3中参数寻优失败了，仍需要集成学习！')
                                else:
                                    logs.debug(str(station_id) + model_name + '超短期，由于任务3中参数寻优失败了，仍需要集成学习！')
                        elif record1[0][0] is None or len(record1[0][0]) == 0:
                            if result2 == 0:
                                target_model_name_cluster_ultra_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            elif record2[0][0] is None or len(record2[0][0]) == 0:
                                target_model_name_cluster_ultra_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                                else:
                                    logs.debug(str(station_id) + model_name + '超短期，任务3、4参数寻优均失败，不再进行集成学习，也无法调用任务3结果！')
                            else:
                                model_name_cluster_ultra_short.append(model_name)
                                target_model_name_cluster_ultra_short.remove(model_name)
                                if do_logs == 1:
                                    logs.info(str(station_id) + model_name + '超短期，由于任务3中参数寻优失败了，仍需要集成学习！')
                                else:
                                    logs.debug(str(station_id) + model_name + '超短期，由于任务3中参数寻优失败了，仍需要集成学习！')

            else:
                logs.debug('如果特征工程有改变，判断参数寻优结果是否与默认值相同，如果相同，直接采用任务2结果。')
                c.execute("select config_value from sys_config where config_key = 'task2_feature_ensemble';")
                db.commit()
                record = c.fetchall()
                do_task = record[0][0]
                if do_task == '1':
                    result = c.execute("select " + ','.join(iterm for iterm in model_name_cluster_setting) +
                                       " from best_feature_parameters_and_model"
                                       " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
                    db.commit()
                    if result == 0:
                        if do_logs == 1:
                            logs.info('参数寻优失败，不再进行集成学习。')
                        else:
                            logs.debug('参数寻优失败，不再进行集成学习。')
                        model_name_cluster_ultra_short = []
                        target_model_name_cluster_ultra_short = []
                        ultra_short_target_task = None
                        if 'MLP' in model_name_cluster_setting:
                            target_model_name_cluster_ultra_short.append('MLP')
                            ultra_short_target_task = 2
                    else:
                        des = c.description
                        record = c.fetchall()
                        coul = list(iterm[0] for iterm in des)
                        dataframe_best_parameter = pandas.DataFrame(record, columns=coul)
                        model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                        target_model_name_cluster_ultra_short = []
                        ultra_short_target_task = 2
                        for model_name in model_name_cluster_setting:
                            if model_name == 'MLP':
                                model_name_cluster_ultra_short.remove(model_name)
                                target_model_name_cluster_ultra_short.append(model_name)
                            else:
                                if dataframe_best_parameter.loc[:, model_name][0] is None or len(
                                        dataframe_best_parameter.loc[:, model_name][0]) == 0:
                                    if do_logs == 1:
                                        logs.info(str(station_id) + '超短期，' + model_name + '参数寻优失败，不再进行集成学习')
                                    else:
                                        logs.debug(str(station_id) + '超短期，' + model_name + '参数寻优失败，不再进行集成学习')
                                    model_name_cluster_ultra_short.remove(model_name)
                                elif str(dataframe_best_parameter.loc[:, model_name][0]) == str(
                                        ultra_short_best_parameter_default[model_name]):
                                    if do_logs == 1:
                                        logs.info(str(station_id) + '超短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习，直接调用任务2结果')
                                    else:
                                        logs.debug(str(station_id) + '超短期，' + model_name + '参数寻优结果与默认值相同，不再进行集成学习，直接调用任务2结果')
                                    model_name_cluster_ultra_short.remove(model_name)
                                    target_model_name_cluster_ultra_short.append(model_name)
                                else:
                                    logs.debug('如果参数寻优有改变，正常执行集成学习。')
                else:
                    logs.debug('未执行任务2，即使任务4的最优参数与默认参数一致，也无法对任务简化合并。')
                    model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                    target_model_name_cluster_ultra_short = []
                    ultra_short_target_task = None
        else:
            logs.debug('未执行任务3，即使任务4的最优特征与默认特征一致，也无法对任务简化合并。')
            model_name_cluster_ultra_short = model_name_cluster_setting.copy()
            target_model_name_cluster_ultra_short = []
            ultra_short_target_task = None
    c.close()
    db.close()
    return model_name_cluster_ultra_short, target_model_name_cluster_ultra_short, ultra_short_target_task, \
           model_name_cluster_short, target_model_name_cluster_short, short_target_task


@catch_exception("ensemble_learning_evaluate_short error: ", exc_info=True, default_return=None)
def ensemble_learning_evaluate_short(host, user, password, database, charset, port,
                                     predict_type=None, model_path=None, model_name=None, model_state=None,
                                     station_id=None, sr_col=None, online_capacity=None):
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
            " task_type = 'predict' and station_id=%s;", (task_name, station_id))
        db.commit()

        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute(
                "update parallel_tasks set distribution_times = distribution_times + 1 where task_status = '1' "
                "and task_name = %s and task_type = 'predict' and station_id=%s;",
                (task_name, station_id))
            db.commit()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'predict' and station_id=%s;", (task_start_time, task_name, station_id))
            db.commit()

            time_predict_cluster, result, model_state = ensemble_learning.ensemble_learning_evaluate_short(
                predict_type=predict_type, model_path=model_path, model_name=model_name, model_state=model_state,
                station_id=station_id, sr_col=sr_col, online_capacity=online_capacity)

            c.execute("select * from predict_power_" + str(station_id) + "_train limit 1;")
            db.commit()
            des = c.description
            for j in range(int(len(result) / 288)):
                start_time_str = datetime.datetime.strptime(
                    time.strftime("%Y/%m/%d %H:%M",
                                  time.localtime(time_predict_cluster[288 * j].timestamp() - 900 - 8 * 3600)),
                    '%Y/%m/%d %H:%M')
                for ii in range(288):
                    forecast_time = time_predict_cluster[288 * j + ii]
                    value = (start_time_str, forecast_time) + tuple(result[288 * j + ii, :])
                    values = tuple(['short']) + tuple([model_name + model_state]) + value
                    c.execute("INSERT INTO predict_power_" + str(station_id) + '_train (' + ','.join(
                        iterm[0] for iterm in des[1:6]) + ')' + "VALUES(" + ("%s," * 5).strip(',') + ")", values)
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
                    " task_type = 'predict' and station_id=%s;",
                    (task_name, station_id))
                db.commit()
                c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                          " task_type = 'predict' and station_id=%s;", (task_end_time, task_name, station_id))
                db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
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
                  " task_type = 'predict' and station_id=%s;", (task_name, station_id))
        db.commit()
        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                  " task_type = 'predict' and station_id=%s;", (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


@catch_exception("ensemble_learning_evaluate_ultra_short error: ", exc_info=True, default_return=None)
def ensemble_learning_evaluate_ultra_short(host, user, password, database, charset, port,
                                           predict_type=None, model_path=None,
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
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_name = str(station_id) + '_ultra_short_' + model_name + model_state
    try:
        result = c.execute(
            "update parallel_tasks set task_status = '1' where task_status = '0' and task_name = %s and"
            " task_type = 'predict' and station_id=%s;", (task_name, station_id))
        db.commit()

        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'predict' and station_id=%s;", (task_start_time, task_name, station_id))
            db.commit()
            c.execute(
                "update parallel_tasks set distribution_times = distribution_times + 1 where task_status = '1' "
                "and task_name = %s and task_type = 'predict' and station_id=%s;",
                (task_name, station_id))
            db.commit()

            time_predict_cluster, result, model_state = ensemble_learning.ensemble_learning_evaluate_ultra_short(
                predict_type=predict_type, model_path=model_path, model_name=model_name, model_state=model_state,
                station_id=station_id, sr_col=sr_col, online_capacity=online_capacity)

            c.execute("select * from predict_power_" + str(station_id) + "_train limit 1;")
            db.commit()
            des = c.description
            for j in range(int(len(result) / 16)):
                start_time_str = datetime.datetime.strptime(
                    time.strftime("%Y/%m/%d %H:%M",
                                  time.localtime(time_predict_cluster[16 * j].timestamp() - 900 - 8 * 3600)),
                    '%Y/%m/%d %H:%M')
                for ii in range(16):
                    forecast_time = time_predict_cluster[16 * j + ii]
                    value = (start_time_str, forecast_time) + tuple(result[16 * j + ii, :])
                    values = tuple(['ultra_short']) + tuple([model_name + model_state]) + value
                    c.execute("INSERT INTO predict_power_" + str(station_id) + '_train (' + ','.join(
                        iterm[0] for iterm in des[1:6]) + ')' + "VALUES(" + ("%s," * 5).strip(',') + ")", values)
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
                    " task_type = 'predict' and station_id=%s;", (task_name, station_id))
                db.commit()
                c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                          " task_type = 'predict' and station_id=%s;", (task_end_time, task_name, station_id))
                db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
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
                  " task_type = 'predict' and station_id=%s;", (task_name, station_id))
        db.commit()

        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                  " task_type = 'predict' and station_id=%s;",
                  (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


@catch_exception("load_test_date_for_ensemble_learning error: ", exc_info=True, default_return=None)
def load_test_date_for_ensemble_learning(host, user, password, database, charset, port,
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
        middle_time_float = middle_time_float - (middle_time_float % 86400 - 57600) + 86400
    else:
        middle_time_float = middle_time_float - (middle_time_float % 86400 - 57600)

    if end_time_float % 86400 > 57600:
        end_time_float = end_time_float - (end_time_float % 86400 - 57600)
    else:
        end_time_float = end_time_float - (end_time_float % 86400 - 57600) - 86400

    middle_time_float = middle_time_float - 900
    end_time_float = end_time_float - 900

    middle_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float + 900)), '%Y/%m/%d %H:%M')
    end_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float)), '%Y/%m/%d %H:%M')

    # 修改读取NWP数据方式2023-3-16-------------------------------------------------------------------------------------
    sql = "select A.start_time, forecast_time, " + ','.join(iterm for iterm in short_usecols) + " from" \
          " (select DISTINCT start_time, forecast_time, " + ','.join(iterm for iterm in short_usecols) + \
          " from nwp_" + str(station_id) + " where %s <= forecast_time and forecast_time <= %s) A," \
          " (select max(forecast_time) max_forecast_time, max(start_time) max_start_time from nwp_" + \
          str(station_id) + \
          " where %s <= forecast_time and forecast_time <= %s group by forecast_time) B" \
          " where A.forecast_time = B.max_forecast_time and A.start_time = B.max_start_time" \
          " order by forecast_time asc;"
    c.execute(sql, (middle_time_datetime, end_time_datetime, middle_time_datetime, end_time_datetime))
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    nwp_df = pandas.DataFrame(record, columns=coul)
    nwp_df = nwp_df.drop_duplicates(subset=['forecast_time'])   # NWP数据

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
    data_base_power = data_base       # 功率数据，时间已标准化

    time_list = []
    time_list_float = []
    for j in range(middle_time_float + 900, end_time_float + 900, 900):
        time_list.append(
            datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(j)), '%Y/%m/%d %H:%M'))
        time_list_float.append(j % 86400 / 86400)
    time_dic = {'time': time_list, 'timelabel': time_list_float}
    time_dataframe = pandas.DataFrame(time_dic)     # 时间标签浮点数，时间已标准化
    data_base = pandas.merge(time_dataframe, nwp_df, left_on='time', right_on='forecast_time', how='left')
    # 测试输入，时间已标准化

    if predict_type == 'wind':
        data_base.drop(['start_time', 'forecast_time', 'timelabel'], axis=1, inplace=True)
    else:
        data_base.drop(['start_time', 'forecast_time'], axis=1, inplace=True)
        cols = data_base.columns.tolist()
        cols = [cols[0]] + cols[2:] + [cols[1]]
        data_base = data_base[cols]
    data_base_np = numpy.hstack((data_base.values, data_base_power.values[:, 1].reshape(-1, 1)))  # 测试输入+功率numpy形式
    data_base_np = data_base.values

    n = data_base_np.shape[0]
    m = data_base_np.shape[1]

    nwp_np = data_base_np.reshape(int(n / 96), m * 96)
    nwp_np_short = numpy.hstack((nwp_np[:-2, :], nwp_np[1:-1, :], nwp_np[2:, :]))

    nwp_np_short = pandas.DataFrame(nwp_np_short).dropna(axis=0, how='any').values
    nwp_np_short = nwp_np_short.reshape(len(nwp_np_short) * 288, m)
    # NWP每行的varchar解析为float都采用try，避免某一行出错，导致整个的读取NWP失败，2023/3/16--------------------------------------
    if predict_type == 'wind':
        for j in range(len(nwp_np_short)-1, -1, -1):
            try:
                for k in range(nwp_np_short.shape[1] - 1):
                    nwp_np_short[j, k + 1] = eval(nwp_np_short[j, k + 1])
            except Exception as err:
                nwp_np_short = numpy.delete(nwp_np_short, j, axis=0)
                logs.error(str(err), exc_info=True)

    else:
        for j in range(len(nwp_np_short)-1, -1, -1):
            try:
                for k in range(nwp_np_short.shape[1] - 2):
                    nwp_np_short[j, k + 1] = eval(nwp_np_short[j, k + 1])
            except Exception as err:
                nwp_np_short = numpy.delete(nwp_np_short, j, axis=0)
                logs.error(str(err), exc_info=True)

    # 超短期-------------------------------------------------------------------------------------------------------------
    # 读取测试数据
    # NWP
    middle_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(middle_time_float + 900)), '%Y/%m/%d %H:%M')
    end_time_datetime = datetime.datetime.strptime(
        time.strftime("%Y/%m/%d %H:%M", time.localtime(end_time_float)), '%Y/%m/%d %H:%M')

    # 修改读取NWP数据方式2023-3-16-------------------------------------------------------------------------------------
    sql = "select A.start_time, forecast_time, " + ','.join(iterm for iterm in ultra_short_usecols) + " from" \
          " (select DISTINCT start_time, forecast_time, " + ','.join(iterm for iterm in ultra_short_usecols) + \
          " from nwp_" + str(station_id) + " where %s <= forecast_time and forecast_time <= %s) A," \
          " (select max(forecast_time) max_forecast_time, max(start_time) max_start_time from nwp_" + \
          str(station_id) + \
          " where %s <= forecast_time and forecast_time <= %s group by forecast_time) B" \
          " where A.forecast_time = B.max_forecast_time and A.start_time = B.max_start_time" \
          " order by forecast_time asc;"
    c.execute(sql, (middle_time_datetime, end_time_datetime, middle_time_datetime, end_time_datetime))
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
        data_base.drop(['start_time', 'forecast_time', 'timelabel'], axis=1, inplace=True)

    else:
        data_base.drop(['start_time', 'forecast_time'], axis=1, inplace=True)
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

    # NWP每行的varchar解析为float都采用try，避免某一行出错，导致整个的读取NWP失败，2023/3/16---------------------------------------
    if predict_type == 'wind':
        for j in range(len(nwp_np_ultra_short)-1, -1, -1):
            try:
                for k in range(nwp_np_ultra_short.shape[1] - 2):
                    nwp_np_ultra_short[j, k + 1] = eval(nwp_np_ultra_short[j, k + 1])
            except Exception as err:
                nwp_np_ultra_short = numpy.delete(nwp_np_ultra_short, j, axis=0)
                logs.error(str(err), exc_info=True)
    else:
        nu = len(ultra_short_usecols)
        for j in range(len(nwp_np_ultra_short)-1, -1, -1):
            try:
                for k in range(16):
                    for p in range(nu):
                        nwp_np_ultra_short[j, k * (nu + 1) + p + 1] = eval(nwp_np_ultra_short[j, k * (nu + 1) + p + 1])
            except Exception as err:
                nwp_np_ultra_short = numpy.delete(nwp_np_ultra_short, j, axis=0)
                logs.error(str(err), exc_info=True)

    return nwp_np_short, nwp_np_ultra_short


@catch_exception("model_evaluate_save error: ", exc_info=True, default_return=None)
def model_evaluate_save(host, user, password, database, charset, port,
                        config_cluster_train=None, task=1, station_id=None, ensemble_learn_stands=None,
                        csvrecord=False, figurerecord=False,
                        csv_path=None, figure_path=None, model_name_cluster_setting=None):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param config_cluster_train: 模型配置参数
    :param station_id: 场站id
    :param task: 任务序号
    :param ensemble_learn_stands: 评价标准
    :param csvrecord: 是否写csv
    :param figurerecord: 是否绘图
    :param csv_path: csv文件保存地址
    :param figure_path: 图片保存地址
    :param model_name_cluster_setting: 参与集成学习的模型列表
    :return: None
    """
    # 结果评估
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 定义游标
    c = db.cursor()

    c.execute("select capacity, name, type from configure where id = %s;", station_id)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)
    online_capacity = dataframe_config.loc[:, 'capacity'].iloc[0]
    station_name = dataframe_config.loc[:, 'name'].iloc[0]
    predict_type = dataframe_config.loc[:, 'type'].iloc[0]

    if model_name_cluster_setting is None:
        c.execute("select config_value from sys_config where config_key = 'ensemble_learn_model_name';")
        db.commit()
        record = c.fetchall()
        model_name_cluster_setting = eval(record[0][0])

    short_model_state = ['_without_history_power']
    if predict_type == 'solar':
        ultra_short_model_state = ['_with_history_power', '_without_history_power']
    else:
        ultra_short_model_state = ['_with_history_power']

    # 读取默认特征和默认参数 -------------------------------------------------------------------------------------------
    ultra_short_usecols_default, short_usecols_default, \
    ultra_short_best_parameter_default, short_best_parameter_default = \
        get_default_feature_parameter(host=host, user=user, password=password, database=database, charset=charset,
                                      port=port, predict_type=predict_type,
                                      model_name_cluster_setting=model_name_cluster_setting)
    # ----------------------------------------------------------------------------------------------------------
    # 重复任务的减并2022/12/9
    model_name_cluster_ultra_short, target_model_name_cluster_ultra_short, ultra_short_target_task, \
    model_name_cluster_short, target_model_name_cluster_short, short_target_task = \
        get_model_name(ultra_short_usecols_default, short_usecols_default, ultra_short_best_parameter_default,
                       short_best_parameter_default, model_name_cluster_setting,
                       host=host, user=user, password=password, database=database, charset=charset, port=port,
                       station_id=station_id, task=task, do_logs=0)

    # ----------------------------------------------------------------------------------------------------------
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
    # 判断各模型成功预测的时长2022/11/28
    short_len_model_predict = numpy.zeros((len(model_name_cluster_setting), len(short_model_state)))
    n_model = 0
    for model_name in model_name_cluster_setting:
        m_model = 0
        for model_state in short_model_state:
            c.execute("select count(*) from predict_power_" + str(station_id) + "_train where model_name = %s and"
                                                                                " predict_term = 'short';",
                      model_name + model_state)
            db.commit()
            record = c.fetchall()
            if len(record) == 0:
                short_len_model_predict[n_model, m_model] = 0
            else:
                short_len_model_predict[n_model, m_model] = record[0][0]
            m_model += 1
        n_model += 1

    sub_dir_path = "%s%s/short/short_test_data.pkl" % (config_cluster_train[station_id]['model_savepath'],
                                                       str(station_id))
    nwp_np_short = load_model(sub_dir_path)
    short_len_model_predict_max = len(nwp_np_short)

    ultra_short_len_model_predict = numpy.zeros((len(model_name_cluster_setting), len(ultra_short_model_state)))
    n_model = 0
    for model_name in model_name_cluster_setting:
        m_model = 0
        for model_state in ultra_short_model_state:
            c.execute("select count(*) from predict_power_" + str(station_id) + "_train where model_name = %s and"
                                                                                " predict_term = 'ultra_short';",
                      model_name + model_state)
            db.commit()
            record = c.fetchall()
            if len(record) == 0:
                ultra_short_len_model_predict[n_model, m_model] = 0
            else:
                ultra_short_len_model_predict[n_model, m_model] = record[0][0]
            m_model += 1
        n_model += 1

    sub_dir_path = "%s%s/ultra_short/ultra_short_test_data.pkl" % (config_cluster_train[station_id]['model_savepath'],
                                                                   str(station_id))
    nwp_np_ultra_short = load_model(sub_dir_path)
    ultra_short_len_model_predict_max = len(nwp_np_ultra_short) * 16
    # --------------------------------------------------------------------------------------------------------------
    # 4种评价方法
    stand_list1 = []
    stand_list2 = []

    if ensemble_learn_stands is not None:
        if 'GB' in ensemble_learn_stands or 'gb' in ensemble_learn_stands:
            stand_list1.append('GB_T_40607_2021')
            stand_list2.append('GB_T_40607_2021')
        if 'TD' in ensemble_learn_stands or 'td' in ensemble_learn_stands:
            stand_list1.append('Two_Detailed_Rules')
            stand_list2.append('Two_Detailed_Rules')
    else:
        if predict_type == 'solar':
            c.execute("select config_value from sys_config where config_key = 'NB_T_32011_2013';")
            db.commit()
            record = c.fetchall()
            config_value_NB2013 = eval(record[0][0])
            if config_value_NB2013 == 1 or config_value_NB2013 == 2:
                stand_list1.append('NB_T_32011_2013')
                if config_value_NB2013 == 2:
                    stand_list2.append('NB_T_32011_2013')
        else:
            logs.debug(str(station_id) + "为风电场，不采用NB_T_32011_2013计算准确率")

        c.execute("select config_value from sys_config where config_key = 'Two_Detailed_Rules';")
        db.commit()
        record = c.fetchall()
        config_value_two_detailed = eval(record[0][0])
        if config_value_two_detailed == 1 or config_value_two_detailed == 2:
            stand_list1.append('Two_Detailed_Rules')
            if config_value_two_detailed == 2:
                stand_list2.append('Two_Detailed_Rules')

        c.execute("select config_value from sys_config where config_key = 'Q_CSG1211017_2018';")
        db.commit()
        record = c.fetchall()
        config_value_Q2018 = eval(record[0][0])
        if config_value_Q2018 == 1 or config_value_Q2018 == 2:
            stand_list1.append('Q_CSG1211017_2018')
            if config_value_Q2018 == 2:
                stand_list2.append('Q_CSG1211017_2018')

        c.execute("select config_value from sys_config where config_key = 'GB_T_40607_2021';")
        db.commit()
        record = c.fetchall()
        config_value_GB2021 = eval(record[0][0])
        if config_value_GB2021 == 1 or config_value_GB2021 == 2 or len(stand_list2) == 0:
            stand_list1.append('GB_T_40607_2021')
            if config_value_GB2021 == 2 or len(stand_list2) == 0:
                stand_list2.append('GB_T_40607_2021')

    short_accuracy_list = []
    ultra_short_accuracy_list = []
    short_accuracy = numpy.zeros((len(model_name_cluster_setting), len(short_model_state)))
    ultra_short_accuracy = numpy.zeros((len(model_name_cluster_setting), len(ultra_short_model_state)))
    for stand_name in stand_list1:
        short_accuracy_stand = numpy.zeros((len(model_name_cluster_setting), len(short_model_state)))
        n_model = 0
        for model_name in model_name_cluster_setting:
            m_model = 0
            if model_name in model_name_cluster_short:
                for model_state in short_model_state:
                    model_name_and_state = model_name + model_state
                    # 计算准确率
                    accuracy = 0
                    if stand_name == 'NB_T_32011_2013':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        sunrise_time='06:30:00', sunset_time='19:30:00',
                                                        predict_type=predict_type, evaluation="capacity")
                    if stand_name == 'Two_Detailed_Rules':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        evaluation="actual_power",
                                                        predict_type=predict_type)
                    if stand_name == 'Q_CSG1211017_2018':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        standard='Q_CSG1211017_2018',
                                                        predict_type=predict_type, evaluation="capacity")
                    if stand_name == 'GB_T_40607_2021':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        predict_type=predict_type, evaluation="capacity")

                    if math.isnan(accuracy):
                        short_accuracy_stand[n_model, m_model] = 0
                    else:
                        short_accuracy_stand[n_model, m_model] = accuracy
                    m_model += 1
            n_model += 1

        ultra_short_accuracy_stand = numpy.zeros((len(model_name_cluster_setting), len(ultra_short_model_state)))
        n_model = 0
        for model_name in model_name_cluster_setting:
            m_model = 0
            if model_name in model_name_cluster_ultra_short:
                for model_state in ultra_short_model_state:
                    model_name_and_state = model_name + model_state
                    # 计算准确率
                    accuracy = 0
                    if stand_name == 'NB_T_32011_2013':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='ultra_short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        sunrise_time='06:30:00', sunset_time='19:30:00',
                                                        predict_type=predict_type, evaluation="capacity")
                    if stand_name == 'Two_Detailed_Rules':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='ultra_short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        evaluation="actual_power",
                                                        predict_type=predict_type)
                    if stand_name == 'Q_CSG1211017_2018':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='ultra_short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        standard='Q_CSG1211017_2018',
                                                        predict_type=predict_type, evaluation="capacity")
                    if stand_name == 'GB_T_40607_2021':
                        accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                        error_data_label=error_data_label, predict_term='ultra_short',
                                                        host=host, user=user, password=password,
                                                        database=database, charset=charset, port=port,
                                                        model_name_and_state=model_name_and_state, scene='train',
                                                        predict_type=predict_type, evaluation="capacity")
                    if math.isnan(accuracy):
                        ultra_short_accuracy_stand[n_model, m_model] = 0
                    else:
                        ultra_short_accuracy_stand[n_model, m_model] = accuracy
                    m_model += 1
            n_model += 1

        if short_len_model_predict_max != 0:
            short_accuracy_stand = short_accuracy_stand / short_len_model_predict_max * short_len_model_predict
        if ultra_short_len_model_predict_max != 0:
            ultra_short_accuracy_stand = ultra_short_accuracy_stand / ultra_short_len_model_predict_max * \
                                          ultra_short_len_model_predict

        # ----------------------------------------------------------------------------------------------------------
        # 对被减并的模型拷贝 2022/12/11
        if task == 1:
            for j in range(len(short_model_state)):
                value = ('task'+str(task), station_id, 'short', short_model_state[j], stand_name)
                values = value + tuple(short_accuracy_stand[:, j])
                c.execute("INSERT INTO accuracy_statistic (task_name, station_id, term_type, model_state, stands, " +
                          ','.join(iterm for iterm in model_name_cluster_setting) + ")" +
                          " VALUES(" + ("%s," * (5 + len(model_name_cluster_setting))).strip(',') + ")", values)
                db.commit()
            for j in range(len(ultra_short_model_state)):
                value = ('task'+str(task), station_id, 'ultra_short', ultra_short_model_state[j], stand_name)
                values = value + tuple(ultra_short_accuracy_stand[:, j])
                c.execute("INSERT INTO accuracy_statistic (task_name, station_id, term_type, model_state, stands, " +
                          ','.join(iterm for iterm in model_name_cluster_setting) + ")" +
                          " VALUES(" + ("%s," * (5 + len(model_name_cluster_setting))).strip(',') + ")", values)
                db.commit()
        else:
            for k in range(len(model_name_cluster_setting)):
                for j in range(len(short_model_state)):
                    if model_name_cluster_setting[k] in target_model_name_cluster_short:
                        c.execute("select " + model_name_cluster_setting[k] +
                                  " from accuracy_statistic"
                                  " where task_name = %s and station_id=%s and term_type=%s"
                                  " and model_state=%s and stands=%s",
                                  ('task' + str(short_target_task), station_id, 'short', short_model_state[j],
                                   stand_name))
                        db.commit()
                        record = c.fetchall()
                        short_accuracy_stand[k, j] = record[0][0]
                for j in range(len(ultra_short_model_state)):
                    if model_name_cluster_setting[k] in target_model_name_cluster_ultra_short:
                        c.execute("select " + model_name_cluster_setting[k] +
                                  " from accuracy_statistic"
                                  " where task_name = %s and station_id=%s and term_type=%s"
                                  " and model_state=%s and stands=%s",
                                  ('task' + str(ultra_short_target_task), station_id, 'ultra_short',
                                   ultra_short_model_state[j], stand_name))
                        db.commit()
                        record = c.fetchall()
                        ultra_short_accuracy_stand[k, j] = record[0][0]
            for j in range(len(short_model_state)):
                value = ('task' + str(task), station_id, 'short', short_model_state[j], stand_name)
                values = value + tuple(short_accuracy_stand[:, j])
                c.execute("INSERT INTO accuracy_statistic (task_name, station_id, term_type, model_state, stands, " +
                          ','.join(iterm for iterm in model_name_cluster_setting) + ")" +
                          " VALUES(" + ("%s," * (5 + len(model_name_cluster_setting))).strip(',') + ")", values)
                db.commit()
            for j in range(len(ultra_short_model_state)):
                value = ('task' + str(task), station_id, 'ultra_short', ultra_short_model_state[j], stand_name)
                values = value + tuple(ultra_short_accuracy_stand[:, j])
                c.execute("INSERT INTO accuracy_statistic (task_name, station_id, term_type, model_state, stands, " +
                          ','.join(iterm for iterm in model_name_cluster_setting) + ")" +
                          " VALUES(" + ("%s," * (5 + len(model_name_cluster_setting))).strip(',') + ")", values)
                db.commit()
            # 拷贝图片
            if figurerecord is True:
                try:
                    path_figure = figure_path
                    path_figure_task = ['task1_ensemble', 'task2_feature_ensemble', 'task3_parameter_ensemble',
                                        'task4_feature_parameter_ensemble']
                    if len(target_model_name_cluster_short) > 0:
                        old_path_short = path_figure + path_figure_task[short_target_task-1] + '/' + str(station_id) + \
                                         '/short/'
                        new_path_short = path_figure + path_figure_task[task-1] + '/' + str(station_id) + '/short/'
                        os.makedirs(new_path_short, exist_ok=True)
                        for model_name in target_model_name_cluster_short:
                            for model_state in short_model_state:
                                model_name_state = model_name + model_state
                                old_path_short_file = old_path_short + model_name_state + '_' + str(station_id) + \
                                                      '.html'
                                new_path_short_file = new_path_short + model_name_state + '_' + str(station_id) + \
                                                      '.html'
                                if os.path.exists(old_path_short_file):
                                    shutil.copy(old_path_short_file, new_path_short_file)
                    if len(target_model_name_cluster_ultra_short) > 0:
                        old_path_ultra_short = path_figure + path_figure_task[ultra_short_target_task-1] + '/' + \
                                               str(station_id) + '/ultra_short/'
                        new_path_ultra_short = path_figure + path_figure_task[task-1] + '/' + str(station_id) + \
                                               '/ultra_short/'
                        os.makedirs(new_path_ultra_short, exist_ok=True)
                        for model_name in target_model_name_cluster_ultra_short:
                            for model_state in ultra_short_model_state:
                                model_name_state = model_name + model_state
                                old_path_ultra_short_file = old_path_ultra_short + model_name_state + '_' + \
                                                            str(station_id) + '.html'
                                new_path_ultra_short_file = new_path_ultra_short + model_name_state + '_' + \
                                                            str(station_id) + '.html'
                                if os.path.exists(old_path_ultra_short_file):
                                    shutil.copy(old_path_ultra_short_file, new_path_ultra_short_file)
                                old_path_ultra_short_files = old_path_ultra_short + model_name_state
                                new_path_ultra_short_files = new_path_ultra_short + model_name_state
                                if os.path.exists(old_path_ultra_short_files):
                                    if os.path.exists(new_path_ultra_short_files):
                                        shutil.rmtree(new_path_ultra_short_files)
                                    shutil.copytree(old_path_ultra_short_files, new_path_ultra_short_files)
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.debug('拷贝图形时失败！')
        # --------------------------------------------------------------------------------------------------------------
        if stand_name in stand_list2:
            short_accuracy = short_accuracy + short_accuracy_stand
            ultra_short_accuracy = ultra_short_accuracy + ultra_short_accuracy_stand

            short_accuracy_list.append(short_accuracy_stand)
            ultra_short_accuracy_list.append(ultra_short_accuracy_stand)

        if csvrecord is True:
            try:
                # 写csv
                shortfile = open(csv_path + 'short_accuracy.csv', 'a+')
                dfcsv = pandas.DataFrame([str(station_id) + ' ' + stand_name + ' short predict accuracy:'])
                dfcsv.to_csv(shortfile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv = pandas.DataFrame(short_accuracy_stand.T, index=short_model_state,
                                         columns=model_name_cluster_setting)
                dfcsv.to_csv(shortfile, index=True, header=True, encoding="utf_8", mode='a+')
                shortfile.close()

                ultrafile = open(csv_path + 'ultra_short_accuracy.csv', 'a+')
                dfcsv = pandas.DataFrame([str(station_id) + ' ' + stand_name + ' ultra_short predict accuracy:'])
                dfcsv.to_csv(ultrafile, index=False, header=False, encoding="utf_8", mode='a+')
                dfcsv = pandas.DataFrame(ultra_short_accuracy_stand.T, index=ultra_short_model_state,
                                         columns=model_name_cluster_setting)
                dfcsv.to_csv(ultrafile, index=True, header=True, encoding="utf_8", mode='a+')
                ultrafile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning('写CSV失败！')

        # --------------------------------------------------------------------------------------------------------------
        # 打logs
        n_model = 0
        for model_name in model_name_cluster_setting:
            m_model = 0
            for model_state in short_model_state:
                logs.info(station_name + model_name + model_state + ' ' + stand_name + ' 短期预测准确率为：' + str(
                    short_accuracy_stand[n_model, m_model]))
                m_model += 1
            n_model += 1

        n_model = 0
        for model_name in model_name_cluster_setting:
            m_model = 0
            for model_state in ultra_short_model_state:
                logs.info(station_name + model_name + model_state + ' ' + stand_name + ' 超短期预测准确率为：' + str(
                    ultra_short_accuracy_stand[n_model, m_model]))
                m_model += 1
            n_model += 1

    # ------------------------------------------------------------------------------------------------------------------
    # 短期 筛选最优模型并保存
    whe = short_accuracy.argmax()
    n_states = len(short_model_state)
    model_label = whe // n_states
    power_label = whe % n_states
    model_name = model_name_cluster_setting[model_label]
    model_state = short_model_state[power_label]

    result = c.execute(
        "select id, predict_term from best_feature_parameters_and_model where id = %s and predict_term = %s;",
        (station_id, 'short'))
    db.commit()
    if result == 0:
        c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                  " VALUES (%s, %s, %s)", tuple([station_id, 'short', config_cluster_train[station_id]['name']]))
        db.commit()

    c.execute("update best_feature_parameters_and_model set best_model = %s where id = %s and predict_term = %s;",
              (model_name + model_state, station_id, 'short'))
    db.commit()

    short_accuracy_str = ''
    for k in range(len(short_accuracy_list)):
        short_accuracy_str = short_accuracy_str + ', ' + str(
            round(short_accuracy_list[k][model_label, power_label], 2)) + '%'

    logs.info(station_name + 'short' + '基于标准：' + str(stand_list2) + '的最优模型是' + model_name + model_state +
              "准确率分别是：" + short_accuracy_str[2:])
    c.execute("update best_feature_parameters_and_model set best_model_accuracy = %s"
              " where id = %s and predict_term = %s;",
              (short_accuracy_str[2:], station_id, 'short'))
    db.commit()

    if len(model_name_cluster_setting) > 1:
        short_accuracy_second = short_accuracy.copy()
        for k in range(n_states):
            short_accuracy_second[model_label, k] = 0

        whe = short_accuracy_second.argmax()
        model_label = whe // n_states
        power_label = whe % n_states
        model_name_second = model_name_cluster_setting[model_label]
        model_state_second = short_model_state[power_label]
        c.execute("update best_feature_parameters_and_model set second_model = %s where id = %s and"
                  " predict_term = %s;",
                  (model_name_second + model_state_second, station_id, 'short'))
        db.commit()

        short_accuracy_str = ''
        for k in range(len(short_accuracy_list)):
            short_accuracy_str = short_accuracy_str + ', ' + str(
                round(short_accuracy_list[k][model_label, power_label], 2)) + '%'

        logs.info(station_name + 'short' + '基于标准：' + str(stand_list2) + '的次优模型是' + model_name_second +
                  model_state_second + "准确率分别是：" + short_accuracy_str[2:])
        c.execute("update best_feature_parameters_and_model set second_model_accuracy = %s where id = %s and"
                  " predict_term = %s;",
                  (short_accuracy_str[2:], station_id, 'short'))
        db.commit()
    # ------------------------------------------------------------------------------------------------------------------
    # 超短期 筛选最优模型并保存
    whe = ultra_short_accuracy.argmax()
    n_states = len(ultra_short_model_state)
    model_label = whe // n_states
    power_label = whe % n_states
    model_name = model_name_cluster_setting[model_label]
    model_state = ultra_short_model_state[power_label]

    result = c.execute(
        "select id, predict_term from best_feature_parameters_and_model where id = %s and predict_term = %s;",
        (station_id, 'ultra_short'))
    db.commit()
    if result == 0:
        c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                  " VALUES (%s, %s, %s)",
                  tuple([station_id, 'ultra_short', config_cluster_train[station_id]['name']]))
        db.commit()

    c.execute("update best_feature_parameters_and_model set best_model = %s where id = %s and predict_term = %s;",
              (model_name + model_state, station_id, 'ultra_short'))
    db.commit()

    ultra_short_accuracy_str = ''
    for k in range(len(ultra_short_accuracy_list)):
        ultra_short_accuracy_str = ultra_short_accuracy_str + ', ' + str(
            round(ultra_short_accuracy_list[k][model_label, power_label], 2)) + '%'

    logs.info(station_name + 'ultra_short' + '基于标准：' + str(stand_list2) + '的最优模型是' + model_name +
              model_state + "准确率分别是：" + ultra_short_accuracy_str[2:])

    c.execute("update best_feature_parameters_and_model set best_model_accuracy = %s"
              " where id = %s and predict_term = %s;",
              (ultra_short_accuracy_str[2:], station_id, 'ultra_short'))
    db.commit()

    if len(model_name_cluster_setting) > 1:
        ultra_short_accuracy_second = ultra_short_accuracy.copy()
        for k in range(n_states):
            ultra_short_accuracy_second[model_label, k] = 0

        whe = ultra_short_accuracy_second.argmax()
        model_label = whe // n_states
        power_label = whe % n_states
        model_name_second = model_name_cluster_setting[model_label]
        model_state_second = ultra_short_model_state[power_label]
        c.execute("update best_feature_parameters_and_model set second_model = %s where id = %s and"
                  " predict_term = %s;",
                  (model_name_second + model_state_second, station_id, 'ultra_short'))
        db.commit()

        ultra_short_accuracy_str = ''
        for k in range(len(ultra_short_accuracy_list)):
            ultra_short_accuracy_str = ultra_short_accuracy_str + ', ' + str(
                round(ultra_short_accuracy_list[k][model_label, power_label], 2)) + '%'

        logs.info(station_name + 'ultra_short' + '基于标准：' + str(stand_list2) + '的次优模型是' + model_name_second +
                  model_state_second + "准确率分别是：" + ultra_short_accuracy_str[2:])
        c.execute("update best_feature_parameters_and_model set second_model_accuracy = %s where id = %s and"
                  " predict_term = %s;",
                  (ultra_short_accuracy_str[2:], station_id, 'ultra_short'))
        db.commit()
    else:
        model_name_second = None
    # ------------------------------------------------------------------------------------------------------------------
    # 超短期模型不带历史功率再训练 2022/10/19
    ensemble_learning.ultra_short_best_model_retrain(config_cluster_train=config_cluster_train,
                                                     model_name_cluster_ultra_short=model_name_cluster_ultra_short,
                                                     station_id=station_id, predict_type=predict_type,
                                                     model_name=model_name, model_name_second=model_name_second)

    c.execute(
        "update configure set best_two_model_ultra_short_without_power_retrained = 1 where id = %s;",
        station_id)
    db.commit()

    c.execute("update configure set trained = 1 where id = %s;", station_id)
    db.commit()
    c.close()
    db.close()


@catch_exception("training_model error: ", exc_info=True, default_return=None)
def training_model(host, user, password, database, charset, port,
                   task=1, station_id=None, model_name_cluster_setting=None):
    """
    训练模型
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param task: 任务序号
    :param station_id: 场站id
    :param model_name_cluster_setting: 参与集成学习的模型列表
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    # 从数据库读取需要集成学习的模型列表
    if model_name_cluster_setting is None:
        c.execute("select config_value from sys_config where config_key = 'ensemble_learn_model_name';")
        db.commit()
        record = c.fetchall()
        model_name_cluster_setting = eval(record[0][0])

    # 读取配置-----------------------------------------------------------------------------------------------------------
    config_cluster = get_config_cluster(host, user, password, database, charset, port, station_id)

    # 读取默认特征和默认参数 -----------------------------------------------------------------------------------------------
    ultra_short_usecols_default, short_usecols_default, \
    ultra_short_best_parameter_default, short_best_parameter_default = \
        get_default_feature_parameter(host=host, user=user, password=password, database=database, charset=charset,
                                      port=port, predict_type=config_cluster[station_id]["type"],
                                      model_name_cluster_setting=model_name_cluster_setting)
    # ------------------------------------------------------------------------------------------------------------------
    # 重复任务的减并2022/12/9
    model_name_cluster_ultra_short, target_model_name_cluster_ultra_short, ultra_short_target_task, \
    model_name_cluster_short, target_model_name_cluster_short, short_target_task = \
        get_model_name(ultra_short_usecols_default, short_usecols_default, ultra_short_best_parameter_default,
                       short_best_parameter_default, model_name_cluster_setting,
                       host=host, user=user, password=password, database=database, charset=charset, port=port,
                       station_id=station_id, task=task, do_logs=1)

    # 读取最优特征和参数，如果不存在，则使用默认值 -----------------------------------------------------------------------------
    config_cluster = get_best_feature_parameter(
        host=host, user=user, password=password, database=database, charset=charset, port=port,
        model_name_cluster_setting=model_name_cluster_setting, station_id=station_id, config_cluster=config_cluster,
        short_best_parameter_default=short_best_parameter_default,
        ultra_short_best_parameter_default=ultra_short_best_parameter_default,
        ultra_short_usecols_default=ultra_short_usecols_default, short_usecols_default=short_usecols_default)

    c.execute("DELETE FROM parallel_tasks where task_type = 'data_load_preprocess' and station_id=%s;", station_id)
    db.commit()
    c.execute("DELETE FROM parallel_tasks where task_type = 'train' and station_id=%s;", station_id)
    db.commit()
    # ------------------------------------------------------------------------------------------------------------------
    result = c.execute("update parallel_tasks_station set task_status = '" + str(
        station_id) + "正在写训练任务' where task_name = 'ensemble' and task_status = '" + str(
        station_id) + "正在进行集成学习' and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        if len(model_name_cluster_short) > 0:
            c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                      "VALUES (%s, %s, %s, %s) ",
                      tuple([str(station_id) + '_' + 'short']) + tuple(['0']) + tuple(['data_load_preprocess']) +
                      tuple([station_id]))
            db.commit()

        if len(model_name_cluster_ultra_short) > 0:
            c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                      "VALUES (%s, %s, %s, %s) ",
                      tuple([str(station_id) + '_' + 'ultra_short']) + tuple(['0']) +
                      tuple(['data_load_preprocess']) + tuple([station_id]))
            db.commit()

        for model_name in model_name_cluster_short:
            c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                      "VALUES (%s, %s, %s, %s) ",
                      tuple([str(station_id) + '_short_' + model_name]) + tuple(['0']) + tuple(['train']) +
                      tuple([station_id]))
        db.commit()
        for model_name in model_name_cluster_ultra_short:
            c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                      "VALUES (%s, %s, %s, %s) ",
                      tuple([str(station_id) + '_ultra_short_' + model_name]) + tuple(['0']) + tuple(['train']) +
                      tuple([station_id]))
        db.commit()

        c.execute("update parallel_tasks_station set task_status = '" + str(
            station_id) + "写训练任务完成' where task_name = 'ensemble' and task_status = '"+str(
            station_id) + "正在写训练任务' and station_id = %s;", station_id)
        db.commit()

    result = c.execute("select task_status from parallel_tasks_station where task_name = 'ensemble'"
                       " and task_status = '" + str(station_id) + "写训练任务完成' and station_id = %s;", station_id)
    db.commit()

    while result == 0:
        time.sleep(1)
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        result = c.execute("select task_status from parallel_tasks_station where task_name = 'ensemble'"
                           " and task_status = '" + str(station_id) + "写训练任务完成' and station_id = %s;", station_id)
        db.commit()

    c.close()
    db.close()
    return config_cluster


@catch_exception("data_load_preprocess error: ", exc_info=True, default_return=None)
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
            " task_type = 'data_load_preprocess' and station_id=%s;",
            (task_name, station_id))
        db.commit()
        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'data_load_preprocess' and station_id=%s;",
                      (task_start_time, task_name, station_id))
            db.commit()

            ultra_short_usecols = config["ultra_short_usecols"]
            short_usecols = config["short_usecols"]
            predict_type = config["type"]
            online_capacity = config["online_capacity"]
            train_data_load = LoadTraindata()
            if predict_term == "ultra_short":
                train_feature, train_target = train_data_load.load_train_data_for_ultra_short_sql(
                    host=host, user=user, password=password, database=database, charset=charset, port=port,
                    config=config, usecols=ultra_short_usecols, predict_type=predict_type, start_time=start_time,
                    end_time=end_time, rate=rate)

            else:
                train_feature, train_target = train_data_load.load_train_data_for_short_sql(
                    host=host, user=user, password=password, database=database, charset=charset, port=port,
                    config=config, usecols=short_usecols, predict_type=predict_type, start_time=start_time,
                    end_time=end_time, rate=rate)

            train_feature, train_target = data_preprocess.DataPreprocess.data_preprocess(train_feature, train_target,
                                                                                         online_capacity)
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            sub_dir_path = "%s%s/%s/" % (config['model_savepath'], str(station_id), predict_term)
            os.makedirs(sub_dir_path, exist_ok=True)
            save_model(train_feature, sub_dir_path + predict_term + '_train_feature.pkl')
            save_model(train_target, sub_dir_path + predict_term + '_train_target.pkl')

            c.execute(
                "update parallel_tasks set task_status = '2' where task_status = '1' and task_name = %s and"
                " task_type = 'data_load_preprocess' and station_id=%s;", (task_name, station_id))
            db.commit()
            task_end_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                      " task_type = 'data_load_preprocess' and station_id=%s;",
                      (task_end_time, task_name, station_id))
            db.commit()
        c.close()
        db.close()
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()
        logs.warning(task_name + '任务失败！')
        c.execute("update parallel_tasks set task_status = '3' where task_name = %s and"
                  " task_type = 'data_load_preprocess' and station_id=%s;", (task_name, station_id))
        db.commit()
        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                  " task_type = 'data_load_preprocess' and station_id=%s;",
                  (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


@catch_exception("save_a_fitted_model error: ", exc_info=True, default_return=None)
def save_a_fitted_model(station_id, predict_term, model_name, config, host, user, password, database,
                        charset, port):
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
            " task_type = 'train' and station_id=%s;", (task_name, station_id))
        db.commit()
        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s and"
                      " task_type = 'train' and station_id=%s;", (task_start_time, task_name, station_id))
            db.commit()

            ensemble_learning.save_a_fitted_model(station_id, predict_term, model_name, config)

            c.execute(
                "update parallel_tasks set task_status = '2' where task_status = '1' and task_name = %s and"
                " task_type = 'train' and station_id=%s;", (task_name, station_id))
            db.commit()
            task_end_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and task_type = 'train'"
                      " and station_id=%s;", (task_end_time, task_name, station_id))
            db.commit()
        c.close()
        db.close()
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()
        logs.warning(task_name + '任务失败！')
        c.execute("update parallel_tasks set task_status = '3' where task_name = %s and"
                  " task_type = 'train' and station_id=%s;", (task_name, station_id))
        db.commit()
        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and task_type = 'train'"
                  " and station_id=%s;",
                  (task_end_time, task_name, station_id))
        db.commit()
        c.close()
        db.close()


def task_stop(db, c, station_id):
    result = c.execute(
        "select id from parallel_tasks_station where task_name = 'ensemble' and task_status = 'task_stopped'"
        " and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        c.close()
        db.close()
    return result

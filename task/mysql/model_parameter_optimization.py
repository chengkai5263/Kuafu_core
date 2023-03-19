
import pandas
import pymysql
import datetime
import time
from common.logger import logs
import numpy
from app.model_parameter_optimization import model_parameter_optimization
from task.mysql.load_train_data import LoadTraindata
from common import data_preprocess
from common.tools import catch_exception


@catch_exception("run_best_parameter_search error: ")
def run_best_parameter_search(host, user, password, database, charset, port, rate=1.0,
                              scheduler=None, executor_name=None, task=3,
                              csvrecord=False, model_name=None, scene=None):
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
            logs.warning(str(station_id) + 'configure表中没有场站记录，退出参数寻优!')
            continue
        else:
            record = c.fetchall()
            if record[0][0] is None or record[0][1] is None:
                logs.warning(str(station_id) + 'configure表中没有设置起始时间，退出参数寻优!')
                continue
            else:
                start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record[0][0].timestamp()))
                end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(record[0][1].timestamp()))

        result = c.execute("select station_id, task_name from parallel_tasks_station"
                           " where station_id = %s and task_name = %s;", (station_id, 'parameter'))
        db.commit()
        if result == 0:
            c.execute("INSERT INTO parallel_tasks_station (station_id, task_name)"
                      " VALUES (%s, %s)", tuple([station_id, 'parameter']))
            db.commit()

        c.execute("update parallel_tasks_station set task_status = %s where station_id = %s and task_name = %s;",
                  (str(station_id) + '参数寻优开始', station_id, 'parameter'))
        db.commit()

        best_parameter_search(host, user, password, database, charset, port, start_time=start_time, end_time=end_time,
                              rate=rate, scheduler=scheduler, executor_name=executor_name, task=task,
                              station_id=station_id, csvrecord=csvrecord, model_name=model_name, scene=scene)
    c.close()
    db.close()


@catch_exception("best_parameter_search error: ")
def best_parameter_search(host, user, password, database, charset, port, start_time=None, end_time=None, rate=1.0,
                          scheduler=None, executor_name=None, task=3, station_id=None,
                          csvrecord=False, model_name_cluster_setting=None, scene=None, csv_path=None):
    """
    模型参数寻优
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param start_time: 集成学习开始时间
    :param end_time: 集成学习结束时间
    :param rate: 训练与测试比例划分
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :param task: 任务序号
    :param station_id: 场站id
    :param csvrecord: 是否写csv
    :param model_name_cluster_setting: 参与寻优的模型列表
    :param scene: 运行场景
    :param csv_path: csv文件保存地址
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
    if task_stop_sign == 1:
        return
    result = c.execute(
        "update parallel_tasks_station set task_status = '" + str(
            station_id) + "正在进行参数寻优' where task_name = 'parameter' and task_status = '" + str(
            station_id) + "参数寻优开始' and station_id = %s;", station_id)
    db.commit()

    if result == 1:
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        logs.info('参数寻优开始')
        function_start_time = datetime.datetime.now().timestamp()

        # 读取配置-----------------------------------------------------------------------------------------------------------
        c.execute("select * from configure where id = %s;", station_id)
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

        # 读取默认特征和默认参数 -----------------------------------------------------------------------------------------------
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

        # 模型参数默认值
        if model_name_cluster_setting is None:
            c.execute("select config_value from sys_config where config_key = 'best_parameter_search_model_name';")
            db.commit()
            record = c.fetchall()
            model_name_cluster_setting = eval(record[0][0])

        short_wind_best_parameter_default = {}
        ultra_short_wind_best_parameter_default = {}
        short_solar_best_parameter_default = {}
        ultra_short_solar_best_parameter_default = {}
        for i in range(len(model_name_cluster_setting)):
            short_wind_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "short_wind"][
                    model_name_cluster_setting[i]].values[0])
            ultra_short_wind_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"][
                    model_name_cluster_setting[i]].values[0])
            short_solar_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "short_solar"][
                    model_name_cluster_setting[i]].values[0])
            ultra_short_solar_best_parameter_default[model_name_cluster_setting[i]] = eval(
                dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"][
                    model_name_cluster_setting[i]].values[0])

        # 读取最优特征和参数，如果不存在，则使用默认值 --------------------------------------------------------------------------
        c.execute("select * from best_feature_parameters_and_model;")
        db.commit()
        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        dataframe_best = pandas.DataFrame(record, columns=coul)

        # 超短期最优特征，如果不存在，则使用默认值
        for i in range(len(dataframe_config)):
            if dataframe_config.loc[:, "type"][i] == 'wind':
                ultra_short_usecols = wind_ultra_short_usecols_default.copy()
            else:
                ultra_short_usecols = solar_ultra_short_usecols_default.copy()
            for j in range(len(dataframe_best)):
                if dataframe_best.loc[:, 'id'][j] == dataframe_config.loc[:, 'id'][i] and \
                        dataframe_best.loc[:, 'predict_term'][j] == 'ultra_short' and \
                        dataframe_best.loc[:, 'usecols'][j] is not None and \
                        len(dataframe_best.loc[:, 'usecols'][j]) > 0:
                    ultra_short_usecols = eval(dataframe_best.loc[:, 'usecols'][j])
            config_cluster[dataframe_config.loc[:, "id"][i]]["ultra_short_usecols"] = ultra_short_usecols

        # 短期最优特征，如果不存在，则使用默认值
        for i in range(len(dataframe_config)):
            if dataframe_config.loc[:, "type"][i] == 'wind':
                short_usecols = wind_short_usecols_default.copy()
            else:
                short_usecols = solar_short_usecols_default.copy()
            for j in range(len(dataframe_best)):
                if dataframe_best.loc[:, 'id'][j] == dataframe_config.loc[:, 'id'][i] and \
                        dataframe_best.loc[:, 'predict_term'][j] == 'short' and \
                        dataframe_best.loc[:, 'usecols'][j] is not None and \
                        len(dataframe_best.loc[:, 'usecols'][j]) > 0:
                    short_usecols = eval(dataframe_best.loc[:, 'usecols'][j])
            config_cluster[dataframe_config.loc[:, "id"][i]]["short_usecols"] = short_usecols

        if model_name_cluster_setting is None:
            c.execute("select config_value from sys_config where config_key = 'best_parameter_search_model_name';")
            db.commit()
            record = c.fetchall()
            model_name_cluster_setting = eval(record[0][0])
        logs.info('参数寻优列表' + str(model_name_cluster_setting))

        c.execute("DELETE FROM parallel_tasks where task_type = 'best_parameter_search' and station_id=%s;", station_id)
        db.commit()
        # 在sql写数据
        result = c.execute("select id, predict_term, station_name from best_feature_parameters_and_model"
                           " where id = %s and predict_term = %s;", (station_id, 'short'))
        db.commit()

        if result == 0:
            c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                      " VALUES (%s, %s, %s)", tuple([station_id, 'short', config_cluster[station_id]['name']]))
            db.commit()

        result = c.execute("select id, predict_term, station_name from best_feature_parameters_and_model"
                           " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
        db.commit()
        if result == 0:
            c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                      " VALUES (%s, %s, %s)", tuple([station_id, 'ultra_short', config_cluster[station_id]['name']]))
            db.commit()
        # ----------------------------------------------------------------------------------------------------------
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        # 重复任务的减并2022/12/9
        predict_type = config_cluster[station_id]['type']
        if predict_type == 'solar':
            ultra_short_usecols_default = solar_ultra_short_usecols_default.copy()
            short_usecols_default = solar_short_usecols_default.copy()
            ultra_short_best_parameter_default = ultra_short_solar_best_parameter_default.copy()
            short_best_parameter_default = short_solar_best_parameter_default.copy()
        else:
            ultra_short_usecols_default = wind_ultra_short_usecols_default.copy()
            short_usecols_default = wind_short_usecols_default.copy()
            ultra_short_best_parameter_default = ultra_short_wind_best_parameter_default.copy()
            short_best_parameter_default = short_wind_best_parameter_default.copy()
        c.execute("select config_value from sys_config where config_key = 'task3_parameter_ensemble';")
        db.commit()
        record = c.fetchall()
        do_task = record[0][0]
        if do_task == '1':
            # 只有任务3已执行，且此时正在进行任务4才会对任务重复任务做简化合并
            if task == 4:
                logs.debug('任务4：已执行特征工程，已执行参数寻优，需判断特征工程和参数寻优结果是否与默认值有区别！')
                c.execute("select usecols from best_feature_parameters_and_model"
                          " where id = %s and predict_term = %s;", (station_id, 'short'))
                db.commit()
                record = c.fetchall()
                if record[0][0] == str(short_usecols_default):
                    logs.info(str(station_id) + '短期，特征工程结果与默认特征相同，不再进行参数寻优，直接调用任务3的结果！')
                    model_name_cluster_short = []
                    target_model_name_cluster_short = model_name_cluster_setting.copy()

                    for model_name in model_name_cluster_setting:
                        result = c.execute(
                            "select " + model_name + " from best_feature_parameters_and_model_parameter_ensemble"
                                                     " where id=%s and predict_term=%s",
                            (station_id, 'short'))
                        db.commit()
                        record = c.fetchall()
                        if result == 0:
                            logs.warning(str(config_cluster[station_id]['name']) + str(
                                model_name) + 'short 在任务3中未找到最优参数结果，需要重新寻优')
                            model_name_cluster_short.append(model_name)
                            target_model_name_cluster_short.remove(model_name)
                        elif record[0][0] is None or len(record[0][0]) == 0:
                            logs.warning(str(config_cluster[station_id]['name']) + str(
                                model_name) + 'short 在任务3中未找到最优参数结果，需要重新寻优')
                            model_name_cluster_short.append(model_name)
                            target_model_name_cluster_short.remove(model_name)
                        else:
                            logs.info(str(config_cluster[station_id]['name']) + str(
                                model_name) + '-' + 'short最优参数为：' + str(record[0][0]))

                            if csvrecord is True:
                                try:
                                    resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                                    dfcsv = pandas.DataFrame(
                                        numpy.array([str(station_id) + ' ' + model_name + ' ' + 'short' +
                                                     ' best parameter: ' + str(record[0][0])]).reshape(1, -1))
                                    dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                                    resultfile.close()
                                except Exception as err:
                                    logs.error(str(err), exc_info=True)
                                    logs.warning('写CSV失败！')
                else:
                    logs.debug('如果特征工程有改变，正常执行集成学习。')
                    model_name_cluster_short = model_name_cluster_setting.copy()
                    target_model_name_cluster_short = []
            else:
                model_name_cluster_short = model_name_cluster_setting.copy()
                target_model_name_cluster_short = []
            # 超短期
            if task == 4:
                logs.debug('任务4：已执行特征工程，已执行参数寻优，需判断特征工程和参数寻优结果是否与默认值有区别！')
                c.execute("select usecols from best_feature_parameters_and_model"
                          " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
                db.commit()
                record = c.fetchall()
                if record[0][0] == str(ultra_short_usecols_default):
                    logs.info(str(station_id) + '超短期，特征工程结果与默认特征相同，不再进行参数寻优，直接调用任务3的结果！')
                    model_name_cluster_ultra_short = []
                    target_model_name_cluster_ultra_short = model_name_cluster_setting.copy()

                    for model_name in model_name_cluster_setting:
                        result = c.execute(
                            "select " + model_name + " from best_feature_parameters_and_model_parameter_ensemble"
                                                     " where id=%s and predict_term=%s",
                            (station_id, 'ultra_short'))
                        db.commit()
                        record = c.fetchall()
                        if result == 0:
                            logs.warning(str(config_cluster[station_id]['name']) + str(
                                model_name) + 'ultra_short 在任务3中未找到最优参数结果，需要重新寻优')
                            model_name_cluster_ultra_short.append(model_name)
                            target_model_name_cluster_ultra_short.remove(model_name)
                        elif record[0][0] is None or len(record[0][0]) == 0:
                            logs.warning(str(config_cluster[station_id]['name']) + str(
                                model_name) + 'ultra_short 在任务3中未找到最优参数结果，需要重新寻优')
                            model_name_cluster_ultra_short.append(model_name)
                            target_model_name_cluster_ultra_short.remove(model_name)
                        else:
                            logs.info(str(config_cluster[station_id]['name']) + str(
                                model_name) + '-' + 'ultra_short最优参数为：' + str(record[0][0]))

                            if csvrecord is True:
                                try:
                                    resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                                    dfcsv = pandas.DataFrame(
                                        numpy.array([str(station_id) + ' ' + model_name + ' ' + 'ultra_short' +
                                                     ' best parameter: ' + str(record[0][0])]).reshape(1, -1))
                                    dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                                    resultfile.close()
                                except Exception as err:
                                    logs.error(str(err), exc_info=True)
                                    logs.warning('写CSV失败！')
                else:
                    logs.debug('如果特征工程有改变，正常执行集成学习。')
                    model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                    target_model_name_cluster_ultra_short = []
            else:
                model_name_cluster_ultra_short = model_name_cluster_setting.copy()
                target_model_name_cluster_ultra_short = []
        else:
            model_name_cluster_short = model_name_cluster_setting.copy()
            target_model_name_cluster_short = []
            model_name_cluster_ultra_short = model_name_cluster_setting.copy()
            target_model_name_cluster_ultra_short = []
        # ----------------------------------------------------------------------------------------------------------
        # 需要进程参数寻优的模型加入任务列表
        result = c.execute("update parallel_tasks_station set task_status = '" + str(
            station_id) + "正在写模型寻优任务列表' where task_name = 'parameter' and task_status = '" + str(
            station_id) + "正在进行参数寻优' and station_id = %s;", station_id)
        db.commit()
        if result == 1:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            for model_name in model_name_cluster_ultra_short:
                c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                          "VALUES (%s, %s, %s, %s) ",
                          tuple([str(station_id) + '_ultra_short_' + model_name]) + tuple(['0']) +
                          tuple(['best_parameter_search']) + tuple([station_id]))
            db.commit()
            for model_name in model_name_cluster_short:
                c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type, station_id) "
                          "VALUES (%s, %s, %s, %s) ",
                          tuple([str(station_id) + '_short_' + model_name]) + tuple(['0']) +
                          tuple(['best_parameter_search']) + tuple([station_id]))
            db.commit()

            # 不需要进程参数寻优的模型直接填入任务3的结果
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            if len(target_model_name_cluster_short) > 0:
                result = c.execute("select id, predict_term, station_name from best_feature_parameters_and_model"
                                   " where id = %s and predict_term = %s;", (station_id, 'short'))
                db.commit()
                if result == 0:
                    c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                              " VALUES (%s, %s, %s)", tuple([station_id, 'short', config_cluster[station_id]['name']]))
                    db.commit()

                for model_name in target_model_name_cluster_short:
                    result = c.execute("select " + model_name +
                                       " from best_feature_parameters_and_model_parameter_ensemble"
                                       " where id = %s and predict_term=%s",
                                       (station_id, 'short'))
                    db.commit()
                    record = c.fetchall()
                    if result == 0:
                        short_best_parameter_old = short_best_parameter_default[model_name]
                    elif record[0][0] is None or len(record[0][0]) == 0:
                        short_best_parameter_old = short_best_parameter_default[model_name]
                    else:
                        short_best_parameter_old = eval(record[0][0])
                    c.execute("update best_feature_parameters_and_model set " + model_name +
                              " = %s where id = %s and predict_term = %s;",
                              (str(short_best_parameter_old), station_id, 'short'))
                    db.commit()
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            if len(target_model_name_cluster_ultra_short) > 0:
                result = c.execute("select id, predict_term, station_name from best_feature_parameters_and_model"
                                   " where id = %s and predict_term = %s;", (station_id, 'ultra_short'))
                db.commit()
                if result == 0:
                    c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                              " VALUES (%s, %s, %s)",
                              tuple([station_id, 'ultra_short', config_cluster[station_id]['name']]))
                    db.commit()
                for model_name in target_model_name_cluster_ultra_short:
                    result = c.execute("select " + model_name +
                                       " from best_feature_parameters_and_model_parameter_ensemble"
                                       " where id = %s and predict_term=%s",
                                       (station_id, 'ultra_short'))
                    db.commit()
                    record = c.fetchall()
                    if result == 0:
                        ultra_short_best_parameter_old = ultra_short_best_parameter_default[model_name]
                    elif record[0][0] is None or len(record[0][0]) == 0:
                        ultra_short_best_parameter_old = ultra_short_best_parameter_default[model_name]
                    else:
                        ultra_short_best_parameter_old = eval(record[0][0])
                    c.execute("update best_feature_parameters_and_model set " + model_name +
                              " = %s where id = %s and predict_term = %s;",
                              (str(ultra_short_best_parameter_old), station_id, 'ultra_short'))
                    db.commit()

            c.execute(
                "update parallel_tasks_station set task_status = '" + str(
                    station_id) + "写模型寻优任务列表完成' where task_name = 'parameter' and task_status = '" + str(
                    station_id) + "正在写模型寻优任务列表' and station_id = %s;", station_id)
            db.commit()

        result = c.execute(
            "select task_status from parallel_tasks_station where task_name = 'parameter'"
            " and task_status = '" + str(station_id) + "写模型寻优任务列表完成' and station_id = %s;", station_id)
        db.commit()
        while result == 0:
            time.sleep(1)
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            result = c.execute(
                "select task_status from parallel_tasks_station where task_name = 'parameter'"
                " and task_status = '" + str(station_id) + "写模型寻优任务列表完成' and station_id = %s;", station_id)
            db.commit()

        train_data_load = LoadTraindata()
        # 从SQL读取训练数据
        train_feature_ultra_short, train_target_ultra_short = train_data_load.load_train_data_for_ultra_short_sql(
            host=host, user=user, password=password, database=database, charset=charset, port=port,
            config=config_cluster[station_id],
            usecols=config_cluster[station_id]['ultra_short_usecols'],
            predict_type=config_cluster[station_id]["type"], start_time=start_time, end_time=end_time,
            rate=rate)
        train_feature_short, train_target_short = \
            train_data_load.load_train_data_for_short_sql(host=host, user=user, password=password,
                                                          database=database, charset=charset, port=port,
                                                          config=config_cluster[station_id],
                                                          usecols=config_cluster[station_id]['short_usecols'],
                                                          predict_type=config_cluster[station_id]["type"],
                                                          start_time=start_time, end_time=end_time, rate=rate)
        # 数据预处理
        train_feature_short, train_target_short = data_preprocess.DataPreprocess.data_preprocess(
            train_feature_short, train_target_short, config_cluster[station_id]['online_capacity'])
        train_feature_ultra_short, train_target_ultra_short = data_preprocess.DataPreprocess.data_preprocess(
            train_feature_ultra_short, train_target_ultra_short,
            config_cluster[station_id]['online_capacity'])
        if len(train_target_short) > 39168:
            train_feature_short = train_feature_short[(len(train_target_short) - 39168):len(train_target_short), :]
            train_target_short = train_target_short[(len(train_target_short) - 39168):len(train_target_short)]
        if len(train_target_ultra_short) > 13056:
            train_feature_ultra_short = train_feature_ultra_short[
                                        (len(train_target_ultra_short) - 13056):len(train_target_ultra_short), :]
            train_target_ultra_short = train_target_ultra_short[
                                       (len(train_target_ultra_short) - 13056):len(train_target_ultra_short)]
        for model_name in model_name_cluster_short:
            scheduler.add_job(get_best_parameter, executor=executor_name,
                              args=[config_cluster[station_id]['name'], 'short',
                                    host, user, password, database, charset, port,
                                    model_name, train_feature_short, train_target_short,
                                    config_cluster[station_id], csvrecord, scene, csv_path])
        for model_name in model_name_cluster_ultra_short:
            scheduler.add_job(get_best_parameter, executor=executor_name,
                              args=[config_cluster[station_id]['name'], 'ultra_short',
                                    host, user, password, database, charset, port,
                                    model_name, train_feature_ultra_short, train_target_ultra_short,
                                    config_cluster[station_id], csvrecord, scene, csv_path])

        number_end = 0
        number_task = c.execute("select id from parallel_tasks where task_type = 'best_parameter_search'"
                                " and station_id=%s;", station_id)
        db.commit()

        task_start_time = datetime.datetime.now().timestamp()  # 测试开始时间
        # 当测试任务全部完成，或者测试时间超过最大限值都会跳出循环
        while number_end < number_task and \
                (datetime.datetime.now().timestamp() - task_start_time) < 3600 * 10:
            task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
            if task_stop_sign == 1:
                return
            time.sleep(2)
            number_finished = c.execute(
                "select id from parallel_tasks where task_status = '2' and task_type = 'best_parameter_search'"
                " and station_id=%s;", station_id)
            db.commit()
            number_failed = c.execute(
                "select id from parallel_tasks where task_status = '3' and task_type = 'best_parameter_search'"
                " and station_id=%s;", station_id)
            db.commit()
            number_end = number_failed + number_finished
        logs.info('参数寻优完成')

        function_end_time = datetime.datetime.now().timestamp()
        logs.info('参数寻优时间:' + str(int(function_end_time - function_start_time)) + '秒')
        hours = int((function_end_time - function_start_time) // 3600)
        mins = int(((function_end_time - function_start_time) % 3600) // 60)
        second = int((function_end_time - function_start_time) % 60)
        logs.info('参数寻优时间:' + str(hours) + '时' + str(mins) + '分' + str(second) + '秒')

        if csvrecord is True:
            try:
                timefile = open(csv_path + 'time.csv', 'a+')
                dfcsv = pandas.DataFrame(
                    numpy.array(['parameter time: ' + str(int(function_end_time - function_start_time)) + 's']).reshape(
                        1, -1))
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                c.execute("select id, task_name, task_start_time, task_end_time"
                          " from parallel_tasks where task_type = 'best_parameter_search' and station_id=%s;",
                          station_id)
                db.commit()
                des = c.description
                record = c.fetchall()
                coul = list(iterm[0] for iterm in des)
                dataframe = pandas.DataFrame(record, columns=coul)
                dataarray = dataframe.values

                parameter_time_model_cluster = numpy.zeros((2, len(model_name_cluster_setting)))
                term_cl = ['short', 'ultra_short']
                term_c = ['s', 'ultra_s']
                for i in range(len(model_name_cluster_setting)):
                    for j in range(len(term_cl)):
                        for k in range(len(dataarray)):
                            a, model_name = dataarray[k, 1].split('hort_', 1)
                            stationid, predict_term = a.split('_', 1)
                            if model_name == model_name_cluster_setting[i] and predict_term == term_c[j]:
                                parameter_time_model_cluster[j, i] = \
                                    parameter_time_model_cluster[j, i] + dataarray[k, 3] - dataarray[k, 2]
                dfcsv = pandas.DataFrame(['model parameter time:'])
                dfcsv.to_csv(timefile, index=False, header=False, encoding="utf_8", mode='a+')
                short_predict_time_df = pandas.DataFrame(parameter_time_model_cluster,
                                                         index=term_cl, columns=model_name_cluster_setting)
                short_predict_time_df.to_csv(timefile, index=True, header=True, encoding="utf_8",
                                             mode='a+')
                timefile.close()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                logs.warning('写CSV失败！')

        c.execute("update parallel_tasks_station set task_status = '" + str(
            station_id) + "参数寻优完成' where task_name = 'parameter' and task_status = '" + str(
            station_id) + "写模型寻优任务列表完成' and station_id = %s;", station_id)
        db.commit()

    c.close()
    db.close()


def get_best_parameter(power_station_name, predict_term, host, user, password, database,
                       charset, port, model_name, train_feature, train_target, config,
                       csvrecord, scene, csv_path):
    """
    获取单个场站模型最优参数，并保存
    :param power_station_name: 场站名称
    :param predict_term:预测周期，分为短期‘short’或超短期‘ultra_short’
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param csvrecord: 是否写csv
    :param model_name: 参与寻优的模型列表
    :param scene: 运行场景
    :param csv_path: csv文件保存地址
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    task_name = str(config['id']) + '_' + predict_term + '_' + model_name
    try:
        result = c.execute(
            "update parallel_tasks set task_status = '1' where task_status = '0' and task_name = %s"
            " and task_type='best_parameter_search' and station_id=%s;",
            (task_name, config['id']))
        db.commit()
        if result == 1:
            task_start_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_start_time = %s where task_name = %s"
                      " and task_type='best_parameter_search' and station_id=%s;",
                      (task_start_time, task_name, config['id']))
            db.commit()

            best_parameter = model_parameter_optimization.get_best_parameter(predict_term=predict_term,
                                                                             model_name=model_name,
                                                                             train_feature=train_feature,
                                                                             train_target=train_target)

            # 在sql写数据
            result = c.execute("select id, predict_term, station_name from best_feature_parameters_and_model"
                               " where id = %s and predict_term = %s;", (config['id'], predict_term))
            db.commit()
            if result == 0:
                c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                          " VALUES (%s, %s, %s)", tuple([config["id"], predict_term, power_station_name]))
                db.commit()

            c.execute("update best_feature_parameters_and_model set " + model_name +
                      " = %s where station_name = %s and predict_term = %s;",
                      (str(best_parameter), power_station_name, predict_term))
            db.commit()

            try:
                if scene == 'cloud':
                    if predict_term == 'ultra_short':
                        types = 'ultrashort'
                    elif predict_term == 'short':
                        types = 'short'
                    else:
                        types = 'medium'
                    result = c.execute("select station_id, type from syy_config_model"
                                       " where station_id = %s and type = %s;", (config['id'], types))
                    db.commit()
                    if result == 0:
                        c.execute("INSERT INTO syy_config_model (station_id, type)"
                                  " VALUES (%s, %s)", tuple([config["id"], types]))
                        db.commit()

                    c.execute("update syy_config_model set " + model_name +
                              " = %s where station_id = %s and type = %s;",
                              (str(best_parameter), config["id"], types))
                    db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                logs.warning(str(config['id']) + model_name + '云板写最优模型失败！')

            logs.info(power_station_name + model_name + '-' + predict_term + '最优参数为：' + str(best_parameter))

            if csvrecord is True:
                try:
                    resultfile = open(csv_path + 'feature_parameter_result.csv', 'a+')
                    dfcsv = pandas.DataFrame(numpy.array([str(config['id']) + ' ' + model_name + ' ' + predict_term +
                                                          ' best parameter: ' + str(best_parameter)]).reshape(1, -1))
                    dfcsv.to_csv(resultfile, index=False, header=False, encoding="utf_8", mode='a+')
                    resultfile.close()
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.warning('写CSV失败！')

            c.execute("update parallel_tasks set task_status = '2' where task_name = %s and"
                      " task_type='best_parameter_search' and station_id=%s;", (task_name, config['id']))
            db.commit()
            task_end_time = datetime.datetime.now().timestamp()
            c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                      " task_type='best_parameter_search' and station_id=%s;",
                      (task_end_time, task_name, config['id']))
            db.commit()
        c.close()
        db.close()
    except Exception as err:
        logs.error(str(err), exc_info=True)
        db.rollback()
        logs.warning(power_station_name + model_name + '-' + predict_term + '参数寻优失败')
        c.execute("update parallel_tasks set task_status = '3' where task_name = %s and"
                  " task_type='best_parameter_search' and station_id=%s;", (task_name, config['id']))
        db.commit()
        task_end_time = datetime.datetime.now().timestamp()
        c.execute("update parallel_tasks set task_end_time = %s where task_name = %s and"
                  " task_type='best_parameter_search' and station_id=%s;",
                  (task_end_time, task_name, config['id']))
        db.commit()
        c.close()
        db.close()


def task_stop(db, c, station_id):
    result = c.execute(
        "select id from parallel_tasks_station where task_name = 'parameter' and task_status = 'task_stopped'"
        " and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        c.close()
        db.close()
    return result

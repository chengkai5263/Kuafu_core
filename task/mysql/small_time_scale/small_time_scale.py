
import numpy
import time
import datetime

import pandas
import pymysql

from task.mysql.ensemble_learning import ensemble_learn, get_default_feature_parameter, get_config_cluster
from common.logger import logs
from task.mysql.small_time_scale.ensemble_learning_five_minute import one_station_ensemble_learn_five_minute
from task.mysql.small_time_scale.predict_five_minute import predict_five_minute_power
from task.mysql.small_time_scale.transfer_learning_five_minute import transfer_learning


def small_time_scale_ensemble_learn(host='localhost', user='root', password='123456', database='kuafu', charset='utf8',
                            port=3306, start_time_str='2021/8/1 00:00', end_time_str='2021/12/12 00:00',
                            rate=0.75, sql=None, station_id=None):
    """

    """
    # 配置信息
    config_cluster = get_config_cluster(host, user, password, database, charset, port, station_id)
    if sql == "kingbas":
        # 获取参与集成学习的模型名称
        db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        c = db.cursor()
        c.execute("select * from configure_model_default")
        des = c.description
        record = tuple(c.fetchall())
        coul = list(iterm[0] for iterm in des)
        dataframe_model = pandas.DataFrame(record, columns=coul)
    else:
        # 获取参与5分钟集成学习的模型名称
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        c = db.cursor()
    c.execute("select config_value from sys_config where config_key = 'ensemble_learn_model_name';")
    db.commit()
    record = c.fetchall()
    model_name_cluster_setting = eval(record[0][0])

    # 获取15分钟最优模型名称
    c.execute("select best_model from best_feature_parameters_and_model where id = %s and predict_term = 'ultra_short'", station_id)
    db.commit()
    record = c.fetchall()
    best_model_ultra_short = record[0][0]

    ultra_short_usecols_default, short_usecols_default, \
    ultra_short_best_parameter_default, short_best_parameter_default = \
        get_default_feature_parameter(host=host, user=user, password=password, database=database,
                                      charset=charset, port=port,
                                      predict_type=config_cluster[station_id]["type"],
                                      model_name_cluster_setting=model_name_cluster_setting)

    # 模型名称
    model_name_cluster = model_name_cluster_setting
    model_state_cluster = ['_without_history_power', '_with_history_power', '_mix_history_power']

    config_cluster[station_id].update({"usecols": ultra_short_usecols_default,
                                        "best_model_ultra_short": best_model_ultra_short,
                                        'best_paprameter_ultra_short': ultra_short_best_parameter_default})

    one_station_ensemble_learn_five_minute(host=host, user=user, password=password, database=database, charset=charset,
                                           port=port, start_time_str=start_time_str, end_time_str=end_time_str,
                                           rate=rate, sql=sql, station_id=station_id,
                                           config_cluster=config_cluster, model_name_cluster=model_name_cluster,
                                           model_state_cluster=model_state_cluster)

    db.close()


def add_ensemble_learning_task(host, user, password, database, charset, port):
    """
    添加集成学习任务
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
    # 读取状态
    result = c.execute("select id from syy_5_minute_ensemble_learning where command = 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()

        # 判断是否有场站需要集成学习
        if len(record_id_list) > 0:
            # 执行集成学习
            for record_id in record_id_list:
                result = c.execute("update syy_5_minute_ensemble_learning set command = 0 where command = 1 and "
                                   "id = %s;", record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    c.execute("update syy_5_minute_ensemble_learning set status = '正在学习' where id = %s;", record_id)
                    db.commit()
                    try:
                        c.execute("select station_id, start_time, end_time, standard, scale from "
                                  "syy_5_minute_ensemble_learning where id=%s;", record_id)
                        db.commit()
                        record = c.fetchall()
                        station_id = record[0][0]
                        start_time = record[0][1]
                        start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(
                            datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M').timestamp()))
                        end_time = record[0][2]
                        end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(
                            datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M').timestamp()))
                        if record[0][3] is None or len(record[0][3]) == 0:
                            standard = 'GB'
                        else:
                            standard = record[0][3]
                        if record[0][4] is None or len(record[0][4]) == 0:
                            rate = 0.75
                        else:
                            scale = record[0][4]
                            train_num, test_num = scale.split(':', 1)
                            rate = eval(train_num)/(eval(train_num) + eval(test_num))

                        c.execute("DELETE FROM parallel_tasks_station where station_id = %s and task_name = %s;",
                                  (station_id, 'ensemble'))
                        db.commit()
                        c.execute("INSERT INTO parallel_tasks_station  (station_id, task_name, task_status,"
                                  " start_time, end_time, standard, table_id, rate) "
                                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ",
                                  tuple([station_id]) + tuple(['ensemble']) + tuple(['0']) + tuple([start_time])
                                  + tuple([end_time]) + tuple([standard]) + tuple([record_id]) + tuple([rate]))
                        db.commit()
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
                        c.execute("update syy_5_minute_ensemble_learning set status = '学习失败' where id = %s;", record_id)
                        db.commit()
    c.close()
    db.close()


def start_ensemble_learning(host, user, password, database, charset, port, scheduler, executor_name):
    """
    开启集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读取状态
    result = c.execute("select station_id, table_id from parallel_tasks_station where task_status = '0'"
                       " and task_name = 'ensemble' limit 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        station_id = record[0][0]
        table_id = record[0][1]
        result = c.execute("update parallel_tasks_station set task_status = '" + str(
            station_id) + "集成学习开始' where task_status = '0' and task_name = 'ensemble' and station_id=%s"
                          " and table_id=%s;",
                           (station_id, table_id))
        db.commit()
        # 判断是否有场站需要集成学习
        if result > 0:
            c.execute("update syy_5_minute_ensemble_learning set status = '正在学习' where id=%s;", table_id)
            db.commit()
            # 执行集成学习
            try:
                task_start_time = datetime.datetime.now().timestamp()

                c.execute("select start_time, end_time, standard, rate from parallel_tasks_station"
                          " where task_name = 'ensemble' and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                record = c.fetchall()
                start_time = record[0][0]
                end_time = record[0][1]
                standard = record[0][2]
                rate = record[0][3]

                ensemble_learn(host=host, user=user, password=password, database=database, charset=charset,
                               port=port, start_time=start_time, end_time=end_time, rate=rate,
                               scheduler=scheduler, executor_name=executor_name, station_id=station_id,
                               ensemble_learn_stands=standard)
                small_time_scale_ensemble_learn(host=host, user=user, password=password, database=database,
                                           charset=charset, port=port, start_time_str=start_time, end_time_str=end_time,
                                           rate=rate, sql='mysql', station_id=station_id)

                task_end_time = datetime.datetime.now().timestamp()
                d_time = task_end_time - task_start_time
                if d_time > 10:
                    hours = int(d_time//3600)
                    mins = int((d_time % 3600)//60)
                    d_time_str = str(hours) + 'h' + str(mins)
                    # 修改标志位,已完成集成学习
                    c.execute("update syy_5_minute_ensemble_learning set status = '学习完成' where id = %s;", table_id)
                    db.commit()
                    c.execute("update syy_5_minute_ensemble_learning set study_duration = %s where id = %s;",
                              (d_time_str, table_id))
                    db.commit()
                    # 筛选模型和保存精度
                    record1 = c.fetchall()
                    c.execute(
                        "select best_model, best_model_accuracy" +
                        " from best_feature_parameters_and_model" +
                        " where id = %s and predict_term=%s;", (station_id, 'ultra_short'))
                    db.commit()
                    record2 = c.fetchall()

                    if standard == 'GB':
                        ultra_accuracy = {'国标': record2[0][1]}
                    elif standard == 'TD':
                        ultra_accuracy = {'两个细则': record2[0][1]}
                    elif standard == 'GBTD':
                        gb_a, td_a = record2[0][1].replace(' ', '').split(',', 1)
                        ultra_accuracy = {'国标': gb_a, '两个细则': td_a}
                        gb_a, td_a = record1[0][1].replace(' ', '').split(',', 1)
                    else:
                        ultra_accuracy = record2[0][1]

                    c.execute("update syy_5_minute_ensemble_learning set ultra_accuracy = %s where id = %s;",
                              (str(ultra_accuracy), table_id))
                    db.commit()

                    c.execute("update syy_5_minute_ensemble_learning set ultra_best_model = %s where id = %s;",
                              (record2[0][0], table_id))
                    db.commit()

                else:
                    # 修改标志位,学习失败
                    c.execute("update syy_5_minute_ensemble_learning set status = '学习失败' where id = %s;", table_id)
                    db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                c.execute("DELETE FROM parallel_tasks_station where task_name = 'ensemble'"
                          " and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                c.execute("update syy_5_minute_ensemble_learning set status = '学习失败' where id = %s;", table_id)
                db.commit()
    c.close()
    db.close()


def small_time_scale_predict(host='localhost', user='root', password='123456', database='kuafu', charset='utf8',
                                 port=3306, start_time=time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())),
                                 sql="mysql", scheduler=None, executor_name=None):
    if sql == "kingbas":
        # 获取参与集成学习的模型名称
        db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        c = db.cursor()
        c.execute("select * from syy_5_minute_ensemble_learning where ready_to_predict = 1;")
        des = c.description
        record = tuple(c.fetchall())
        coul = list(iterm[0] for iterm in des)
        todo_station_df = pandas.DataFrame(record, columns=coul)
    else:
        # 获取参与集成学习的模型名称
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        # 定义游标
        c = db.cursor()
        c.execute("select * from syy_5_minute_ensemble_learning where ready_to_predict = 1;")
        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        todo_station_df = pandas.DataFrame(record, columns=coul)
    if len(todo_station_df) != 0:
        try:
            # kingbas数据库
            # start_time_loop = '2022-01-01 00:00'
            # for i in range(8929):
            #     start_time_loop2 = datetime.datetime.strptime(start_time_loop, '%Y-%m-%d %H:%M').timestamp() + i * 5 * 60
            #     predict_five_minute_power(host=host, user=user, password=password, database=database, charset=charset,
            #                               port=port,
            #                               start_time=time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time_loop2)), sql=sql,
            #                               scheduler=scheduler, executor_name=executor_name, todo_station_df=todo_station_df)
            predict_five_minute_power(host=host, user=user, password=password, database=database, charset=charset,
                                      port=port,
                                      start_time=time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time())),
                                      sql=sql,
                                      scheduler=scheduler, executor_name=executor_name, todo_station_df=todo_station_df)
            # logs.info(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())) + '全部场站5分钟预测已完成')
            # logs.info(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())) + 'all station predict succeed')
        except Exception as err:
            logs.error(str(err), exc_info=True)
            # 运算错误
            todo_station_df.index = range(0, len(todo_station_df))
            for i in range(len(todo_station_df)):
                c.execute("update syy_5_minute_ensemble_learning set predict_status = %s where station_id = %s;", ('学习失败', str(todo_station_df['station_id'][i])))
            # logs.info('预测失败')
            logs.info('predict fail')
            db.commit()
    else:
        logs.info(start_time + 'no station need to predict')
    db.close()


def add_transfer_learning_task(host, user, password, database, charset, port):
    """
    小时间尺度开启迁移学习
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
    # 读取状态
    result = c.execute("select id from syy_transfer_study_5_minute where command = 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()

        # 判断是否有场站需要迁移学习
        if len(record_id_list) > 0:
            # 执行迁移学习
            for record_id in record_id_list:
                result = c.execute("update syy_transfer_study_5_minute set command = 0 where command = 1 and id = %s;",
                                   record_id)
                db.commit()
                if result == 1:
                    c.execute("update syy_transfer_study_5_minute set status = '正在迁移' where id = %s;", record_id)
                    db.commit()
                    try:
                        c.execute("select station_id from syy_transfer_study_5_minute where id=%s;", record_id)
                        db.commit()
                        record = c.fetchall()
                        station_id = record[0][0]

                        c.execute("DELETE FROM parallel_tasks_station where station_id = %s and task_name = %s;",
                                  (station_id, 'transfer'))
                        db.commit()
                        c.execute("INSERT INTO parallel_tasks_station  (station_id, task_name, task_status,"
                                  " table_id) "
                                  "VALUES (%s, %s, %s, %s) ",
                                  tuple([station_id]) + tuple(['transfer']) + tuple(['0']) + tuple([record_id]))
                        db.commit()
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
                        c.execute("update syy_transfer_study_5_minute set status = '迁移失败' where id = %s;", record_id)
                        db.commit()
    c.close()
    db.close()


def start_transfer_learning(host, user, password, database, charset, port):
    """
    小时间尺度开启迁移学习
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
    # 读取状态
    result = c.execute("select station_id, table_id from parallel_tasks_station where task_status = '0'"
                       " and task_name = 'transfer' limit 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        station_id = record[0][0]
        table_id = record[0][1]
        result = c.execute("update parallel_tasks_station set task_status = '" + str(
            station_id) + "迁移学习开始' where task_status = '0' and task_name = 'transfer' and station_id=%s"
                          " and table_id=%s;",
                           (station_id, table_id))
        db.commit()

        # 判断是否有场站需要迁移学习
        if result > 0:
            try:
                c.execute("select type from configure where id = %s;", station_id)
                db.commit()
                record = c.fetchall()
                predict_type = record[0][0]

                result = c.execute("select id from configure where type = %s and trained = 1;", predict_type)
                db.commit()

                if result > 0:
                    transfer_learning(host=host, user=user, password=password, database=database, charset=charset,
                                      port=port, station_id=station_id)
                    # 修改标志位,已完成迁移学习
                    c.execute("update syy_transfer_study_5_minute set status = '迁移完成' where id = %s;", table_id)
                    db.commit()
                    logs.info('已完成迁移学习')
                else:
                    c.execute("update syy_transfer_study_5_minute set status = '迁移失败' where id = %s;", table_id)
                    db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                c.execute("DELETE FROM parallel_tasks_station where task_name = 'transfer'"
                          " and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                c.execute("update syy_transfer_study_5_minute set status = '迁移失败' where id = %s;", table_id)
                db.commit()
    c.close()
    db.close()

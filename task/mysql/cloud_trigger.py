
import numpy
import time
import datetime

import pandas
import pymysql

from task.mysql.feature_selection import feature_select
from task.mysql.model_parameter_optimization import best_parameter_search
from task.mysql.ensemble_learning import ensemble_learn
from task.mysql.transfer_learning import transfer_learning
from common.logger import logs


def add_feature_select_task(host, user, password, database, charset, port):
    """
    云版开启特征工程
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
    result = c.execute("select id from syy_characteristic_engineer where command = 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()
        # 判断是否有场站需要特征工程
        if len(record_id_list) > 0:
            # 执行特征工程
            for record_id in record_id_list:
                result = c.execute("update syy_characteristic_engineer set command = 0 where command = 1 and id = %s;",
                                   record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    c.execute("update syy_characteristic_engineer set status = '进入任务列表序列' where id = %s;", record_id)
                    db.commit()
                    try:
                        c.execute("select station_id, start_time, end_time, standard from syy_characteristic_engineer"
                                  " where id = %s;", record_id)
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

                        c.execute("DELETE FROM parallel_tasks_station where station_id = %s and task_name = %s;",
                                  (station_id, 'feature'))
                        db.commit()

                        c.execute("INSERT INTO parallel_tasks_station "
                                  "(station_id, task_name, task_status, start_time, end_time, standard, table_id) "
                                  "VALUES (%s, %s, %s, %s, %s, %s, %s) ",
                                  tuple([station_id]) + tuple(['feature']) + tuple(['0']) + tuple([start_time])
                                  + tuple([end_time]) + tuple([standard]) + tuple([record_id]))
                        db.commit()
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
                        c.execute("update syy_characteristic_engineer set status = '筛选失败' where id = %s;", record_id)
                        db.commit()
    c.close()
    db.close()


def add_best_parameter_search_task(host, user, password, database, charset, port):
    """
    云版开启参数寻优
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
    result = c.execute("select id from syy_model_optimization where command = 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()
        # 判断是否有场站需要参数寻优
        if len(record_id_list) > 0:
            # 执行参数寻优
            for record_id in record_id_list:
                result = c.execute("update syy_model_optimization set command = 0 where command = 1 and id = %s;",
                                   record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    c.execute("update syy_model_optimization set status = 2 where id = %s;", record_id)
                    db.commit()
                    try:
                        c.execute("select station_id, start_time, end_time, model_name"
                                  " from syy_model_optimization where id=%s;", record_id)
                        db.commit()
                        record = c.fetchall()
                        station_id = record[0][0]
                        start_time = record[0][1]
                        start_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(
                            datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M').timestamp()))
                        end_time = record[0][2]
                        end_time = time.strftime("%Y/%m/%d %H:%M", time.localtime(
                            datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M').timestamp()))
                        model_name = record[0][3]
                        model_name = model_name.replace(' ', '')
                        model_name = model_name.replace('[', '')
                        model_name = model_name.replace(']', '')
                        model_name = model_name.replace("'", '')
                        model_name = model_name.split(",")

                        c.execute("DELETE FROM parallel_tasks_station where station_id = %s and task_name = %s;",
                                  (station_id, 'parameter'))
                        db.commit()
                        c.execute("INSERT INTO parallel_tasks_station  (station_id, task_name, task_status,"
                                  " start_time, end_time, table_id, model_name) "
                                  "VALUES (%s, %s, %s, %s, %s, %s, %s) ",
                                  tuple([station_id]) + tuple(['parameter']) + tuple(['0']) + tuple([start_time])
                                  + tuple([end_time]) + tuple([record_id]) + tuple([str(model_name)]))
                        db.commit()
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
                        c.execute("update syy_model_optimization set status = 4 where id = %s;", record_id)
                        db.commit()
    c.close()
    db.close()


def add_ensemble_learning_task(host, user, password, database, charset, port):
    """
    云版开启集成学习
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
    result = c.execute("select id from syy_integrated_study where command = 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()

        # 判断是否有场站需要集成学习
        if len(record_id_list) > 0:
            # 执行集成学习
            for record_id in record_id_list:
                result = c.execute("update syy_integrated_study set command = 0 where command = 1 and id = %s;",
                                   record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    c.execute("update syy_integrated_study set status = '进入任务列表序列' where id = %s;", record_id)
                    db.commit()
                    try:
                        c.execute("select station_id, start_time, end_time, standard, scale, model_name"
                                  " from syy_integrated_study where id=%s;", record_id)
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
                        model_name = record[0][5]
                        model_name = model_name.replace(' ', '')
                        model_name = model_name.replace('[', '')
                        model_name = model_name.replace(']', '')
                        model_name = model_name.replace("'", '')
                        model_name = model_name.split(",")

                        c.execute("DELETE FROM parallel_tasks_station where station_id = %s and task_name = %s;",
                                  (station_id, 'ensemble'))
                        db.commit()
                        c.execute("INSERT INTO parallel_tasks_station (station_id, task_name, task_status,"
                                  " start_time, end_time, standard, table_id, rate, model_name) "
                                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ",
                                  tuple([station_id]) + tuple(['ensemble']) + tuple(['0']) + tuple([start_time])
                                  + tuple([end_time]) + tuple([standard]) + tuple([record_id]) + tuple([rate])
                                  + tuple([str(model_name)]))
                        db.commit()
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
                        c.execute("update syy_integrated_study set status = '学习失败' where id = %s;", record_id)
                        db.commit()
    c.close()
    db.close()


def add_transfer_learning_task(host, user, password, database, charset, port):
    """
    云版开启迁移学习
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
    result = c.execute("select id from syy_transfer_study where command = 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()

        # 判断是否有场站需要迁移学习
        if len(record_id_list) > 0:
            # 执行迁移学习
            for record_id in record_id_list:
                result = c.execute("update syy_transfer_study set command = 0 where command = 1 and id = %s;",
                                   record_id)
                db.commit()
                if result == 1:
                    c.execute("update syy_transfer_study set status = '进入任务列表序列' where id = %s;", record_id)
                    db.commit()
                    try:
                        c.execute("select station_id from syy_transfer_study where id=%s;", record_id)
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
                        c.execute("update syy_transfer_study set status = '迁移失败' where id = %s;", record_id)
                        db.commit()
    c.close()
    db.close()


def start_feature_select(host, user, password, database, charset, port, scheduler, executor_name, instance_id):
    """
    云版开启特征工程
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :param instance_id: 实例编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读取状态
    result = c.execute("select station_id from parallel_tasks_station where instance_id = %s"
                       " and task_name = 'feature';", instance_id)
    db.commit()
    if result > 0:
        c.close()
        db.close()
        return
    result = c.execute("select station_id, table_id from parallel_tasks_station where task_status = '0'"
                       " and task_name = 'feature' limit 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        station_id = record[0][0]
        table_id = record[0][1]
        result = c.execute("update parallel_tasks_station set task_status = '" + str(
            station_id) + "特征工程开始' where task_status = '0' and task_name = 'feature' and station_id=%s"
                          " and table_id=%s;",
                           (station_id, table_id))
        db.commit()
        # 判断是否有场站需要特征工程
        if result > 0:
            c.execute("update parallel_tasks_station set instance_id = %s where task_name = 'feature' and station_id=%s"
                      " and table_id=%s;", (instance_id, station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int = %s where task_name = 'feature'"
                      " and station_id=%s and table_id=%s;",
                      (int(datetime.datetime.now().timestamp()), station_id, table_id))
            db.commit()
            c.execute("update syy_characteristic_engineer set status = '正在筛选' where id=%s;",
                      table_id)  # 状态改为'正在筛选'
            db.commit()
            # 执行特征工程
            try:
                task_start_time = datetime.datetime.now().timestamp()

                c.execute("select start_time, end_time, standard from parallel_tasks_station"
                          " where task_name = 'feature' and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                record = c.fetchall()
                start_time = record[0][0]
                end_time = record[0][1]
                standard = record[0][2]

                feature_select(host=host, user=user, password=password, database=database, charset=charset,
                               port=port, scheduler=scheduler, executor_name=executor_name, station_id=station_id,
                               start_time=start_time, end_time=end_time, feature_engineering_stands=standard)
                task_end_time = datetime.datetime.now().timestamp()
                d_time = task_end_time - task_start_time

                db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset,
                                     port=port)
                c = db.cursor()

                if d_time > 90:
                    result = c.execute("select station_id from syy_characteristic_engineer"
                                       " where status='已停止' and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        # 修改标志位,已完成特征工程
                        c.execute(
                            "select usecols from best_feature_parameters_and_model" +
                            " where id = %s and predict_term=%s;", (station_id, 'short'))
                        db.commit()
                        short_record = c.fetchall()
                        c.execute("select usecols from best_feature_parameters_and_model"
                                  " where id = %s and predict_term=%s;", (station_id, 'ultra_short'))
                        db.commit()
                        ultra_short_record = c.fetchall()
                        choose_character = {'超短期': ultra_short_record[0][0], '短期': short_record[0][0]}
                        c.execute("update syy_characteristic_engineer set choose_character = %s where id = %s;",
                                  (str(choose_character), table_id))
                        db.commit()

                        c.execute("select type from configure where id = %s;", station_id)
                        db.commit()
                        record = c.fetchall()
                        predict_type = record[0][0]

                        c.execute(
                            "select usecols from default_feature_and_parameters" +
                            " where term_type=%s;", 'short_'+predict_type)
                        db.commit()
                        short_record = c.fetchall()
                        c.execute(
                            "select usecols from default_feature_and_parameters" +
                            " where term_type=%s;", 'ultra_short_'+predict_type)
                        db.commit()
                        ultra_short_record = c.fetchall()
                        default_character = {'超短期': ultra_short_record[0][0], '短期': short_record[0][0]}
                        c.execute("update syy_characteristic_engineer set default_character = %s where id = %s;",
                                  (str(default_character), table_id))
                        db.commit()

                        c.execute("update syy_characteristic_engineer set time = %s where id = %s;",
                                  (d_time, table_id))
                        db.commit()
                        c.execute("update syy_characteristic_engineer set status = '筛选完成' where id = %s;", table_id)
                        db.commit()
                else:
                    result = c.execute("select station_id from syy_characteristic_engineer"
                                       " where status='已停止' and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        c.execute("update syy_characteristic_engineer set status = '筛选失败' where id = %s;", table_id)
                        db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                c.execute("DELETE FROM parallel_tasks_station where task_name = 'feature'"
                          " and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                result = c.execute("select station_id from syy_characteristic_engineer"
                                   " where status='已停止' and id = %s;", table_id)
                db.commit()
                if result == 0:
                    c.execute("update syy_characteristic_engineer set status = '筛选失败' where id = %s;", table_id)
                    db.commit()
            c.execute("update parallel_tasks_station set instance_id=Null where task_name ='feature' and station_id=%s"
                      " and table_id=%s;", (station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int=Null where task_name ='feature'"
                      " and station_id=%s and table_id=%s;", (station_id, table_id))
            db.commit()
    c.close()
    db.close()


def start_best_parameter_search(host, user, password, database, charset, port, scheduler, executor_name, instance_id):
    """
    云版开启参数寻优
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :param instance_id: 实例编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读取状态
    result = c.execute("select station_id from parallel_tasks_station where instance_id = %s"
                       " and task_name = 'parameter';", instance_id)
    db.commit()
    if result > 0:
        c.close()
        db.close()
        return
    result = c.execute("select station_id, table_id from parallel_tasks_station where task_status = '0'"
                       " and task_name = 'parameter' limit 1;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        station_id = record[0][0]
        table_id = record[0][1]
        result = c.execute("update parallel_tasks_station set task_status = '" + str(
            station_id) + "参数寻优开始' where task_status = '0' and task_name = 'parameter' and station_id=%s"
                          " and table_id=%s;", (station_id, table_id))
        db.commit()
        # 判断是否有场站需要参数寻优
        if result > 0:
            # 执行参数寻优
            c.execute("update parallel_tasks_station set instance_id = %s where task_name='parameter' and station_id=%s"
                      " and table_id=%s;", (instance_id, station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int = %s where task_name = 'parameter'"
                      " and station_id=%s and table_id=%s;",
                      (int(datetime.datetime.now().timestamp()), station_id, table_id))
            db.commit()
            c.execute("update syy_model_optimization set status = 2 where id = %s;", table_id)
            db.commit()
            try:
                task_start_time = datetime.datetime.now().timestamp()
                c.execute("select start_time, end_time, model_name from parallel_tasks_station"
                          " where task_name = 'parameter' and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                record = c.fetchall()
                start_time = record[0][0]
                end_time = record[0][1]
                model_name = eval(record[0][2])
                best_parameter_search(host=host, user=user, password=password, database=database, charset=charset,
                                      port=port, start_time=start_time, end_time=end_time,
                                      scheduler=scheduler, executor_name=executor_name, station_id=station_id,
                                      model_name_cluster_setting=model_name, scene='cloud')
                task_end_time = datetime.datetime.now().timestamp()
                d_time = task_end_time - task_start_time
                db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset,
                                     port=port)
                c = db.cursor()
                if d_time > 90:
                    result = c.execute("select station_id from syy_model_optimization"
                                       " where status=5 and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        # 修改标志位,已完成参数寻优
                        c.execute("update syy_model_optimization set status = 3 where id = %s;", table_id)
                        db.commit()
                        c.execute("update syy_model_optimization set time = %s where id = %s;",
                                  (d_time, table_id))
                        db.commit()
                else:
                    result = c.execute("select station_id from syy_model_optimization"
                                       " where status=5 and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        c.execute("update syy_model_optimization set status = 4 where id = %s;", table_id)
                        db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                c.execute("DELETE FROM parallel_tasks_station where task_name = 'parameter'"
                          " and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                result = c.execute("select station_id from syy_model_optimization"
                                   " where status=5 and id = %s;", table_id)
                db.commit()
                if result == 0:
                    c.execute("update syy_model_optimization set status = 4 where id = %s;", table_id)
                    db.commit()
            c.execute("update parallel_tasks_station set instance_id=Null where task_name='parameter' and station_id=%s"
                      " and table_id=%s;", (station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int=Null where task_name ='parameter'"
                      " and station_id=%s and table_id=%s;", (station_id, table_id))
            db.commit()
    c.close()
    db.close()


def start_ensemble_learning(host, user, password, database, charset, port, scheduler, executor_name, instance_id):
    """
    云版开启集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param scheduler: 调度程序
    :param executor_name: 执行器名称
    :param instance_id: 实例编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读取状态
    result = c.execute("select station_id from parallel_tasks_station where instance_id = %s"
                       " and task_name = 'ensemble';", instance_id)
    db.commit()
    if result > 0:
        c.close()
        db.close()
        return
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
            c.execute("update parallel_tasks_station set instance_id = %s where task_name='ensemble' and station_id=%s"
                      " and table_id=%s;", (instance_id, station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int = %s where task_name = 'ensemble'"
                      " and station_id=%s and table_id=%s;",
                      (int(datetime.datetime.now().timestamp()), station_id, table_id))
            db.commit()
            c.execute("update syy_integrated_study set status = '正在学习' where id=%s;", table_id)
            db.commit()
            # 执行集成学习
            try:
                task_start_time = datetime.datetime.now().timestamp()

                c.execute("select start_time, end_time, standard, rate, model_name from parallel_tasks_station"
                          " where task_name = 'ensemble' and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                record = c.fetchall()
                start_time = record[0][0]
                end_time = record[0][1]
                standard = record[0][2]
                rate = record[0][3]
                model_name = eval(record[0][4])

                ensemble_learn(host=host, user=user, password=password, database=database, charset=charset,
                               port=port, start_time=start_time, end_time=end_time, rate=rate,
                               scheduler=scheduler, executor_name=executor_name, station_id=station_id,
                               ensemble_learn_stands=standard, model_name_cluster_setting=model_name)
                task_end_time = datetime.datetime.now().timestamp()
                d_time = task_end_time - task_start_time
                db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset,
                                     port=port)
                c = db.cursor()
                if d_time > 60:
                    result = c.execute("select station_id from syy_integrated_study"
                                       " where status='已停止' and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        # 修改标志位,已完成集成学习
                        c.execute("update syy_integrated_study set time = %s where id = %s;",
                                  (d_time, table_id))
                        db.commit()
                        # 筛选最优模型和保存精度----------------------------------------------------------------------------
                        c.execute(
                            "select best_model, best_model_accuracy" +
                            " from best_feature_parameters_and_model" +
                            " where id = %s and predict_term=%s;", (station_id, 'short'))
                        db.commit()
                        record1 = c.fetchall()
                        c.execute(
                            "select best_model, best_model_accuracy" +
                            " from best_feature_parameters_and_model" +
                            " where id = %s and predict_term=%s;", (station_id, 'ultra_short'))
                        db.commit()
                        record2 = c.fetchall()
                        if standard == 'GB':
                            ultra_accuracy = {'国标': record2[0][1]}
                            short_accuracy = {'国标': record1[0][1]}
                            medium_accuracy = {'国标': record1[0][1]}
                        elif standard == 'TD':
                            ultra_accuracy = {'两个细则': record2[0][1]}
                            short_accuracy = {'两个细则': record1[0][1]}
                            medium_accuracy = {'两个细则': record1[0][1]}
                        elif standard == 'GBTD':
                            gb_a, td_a = record2[0][1].replace(' ', '').split(',', 1)
                            ultra_accuracy = {'国标': gb_a, '两个细则': td_a}
                            gb_a, td_a = record1[0][1].replace(' ', '').split(',', 1)
                            short_accuracy = {'国标': gb_a, '两个细则': td_a}
                            medium_accuracy = {'国标': gb_a, '两个细则': td_a}
                        else:
                            ultra_accuracy = record2[0][1]
                            short_accuracy = record1[0][1]
                            medium_accuracy = record1[0][1]
                        c.execute("update syy_integrated_study set ultra_accuracy = %s where id = %s;",
                                  (str(ultra_accuracy), table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set short_accuracy = %s where id = %s;",
                                  (str(short_accuracy), table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set medium_accuracy = %s where id = %s;",
                                  (str(medium_accuracy), table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set ultra_best_model = %s where id = %s;",
                                  (record2[0][0], table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set short_best_model = %s where id = %s;",
                                  (record1[0][0], table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set medium_best_model = %s where id = %s;",
                                  (record1[0][0], table_id))
                        db.commit()

                        # 筛选最优模型和保存精度----------------------------------------------------------------------------
                        c.execute(
                            "select second_model, second_model_accuracy" +
                            " from best_feature_parameters_and_model" +
                            " where id = %s and predict_term=%s;", (station_id, 'short'))
                        db.commit()
                        record1 = c.fetchall()
                        c.execute(
                            "select second_model, second_model_accuracy" +
                            " from best_feature_parameters_and_model" +
                            " where id = %s and predict_term=%s;", (station_id, 'ultra_short'))
                        db.commit()
                        record2 = c.fetchall()
                        if standard == 'GB':
                            ultra_accuracy = {'国标': record2[0][1]}
                            short_accuracy = {'国标': record1[0][1]}
                            medium_accuracy = {'国标': record1[0][1]}
                        elif standard == 'TD':
                            ultra_accuracy = {'两个细则': record2[0][1]}
                            short_accuracy = {'两个细则': record1[0][1]}
                            medium_accuracy = {'两个细则': record1[0][1]}
                        elif standard == 'GBTD':
                            gb_a, td_a = record2[0][1].replace(' ', '').split(',', 1)
                            ultra_accuracy = {'国标': gb_a, '两个细则': td_a}
                            gb_a, td_a = record1[0][1].replace(' ', '').split(',', 1)
                            short_accuracy = {'国标': gb_a, '两个细则': td_a}
                            medium_accuracy = {'国标': gb_a, '两个细则': td_a}
                        else:
                            ultra_accuracy = record2[0][1]
                            short_accuracy = record1[0][1]
                            medium_accuracy = record1[0][1]
                        c.execute("update syy_integrated_study set ultra_second_accuracy = %s where id = %s;",
                                  (str(ultra_accuracy), table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set short_second_accuracy = %s where id = %s;",
                                  (str(short_accuracy), table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set medium_second_accuracy = %s where id = %s;",
                                  (str(medium_accuracy), table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set ultra_second_model = %s where id = %s;",
                                  (record2[0][0], table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set short_second_model = %s where id = %s;",
                                  (record1[0][0], table_id))
                        db.commit()
                        c.execute("update syy_integrated_study set medium_second_model = %s where id = %s;",
                                  (record1[0][0], table_id))
                        db.commit()

                        c.execute("update syy_integrated_study set status = '学习完成' where id = %s;", table_id)
                        db.commit()
                else:
                    # 修改标志位,学习失败
                    result = c.execute("select station_id from syy_integrated_study"
                                       " where status='已停止' and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        c.execute("update syy_integrated_study set status = '学习失败' where id = %s;", table_id)
                        db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                c.execute("DELETE FROM parallel_tasks_station where task_name = 'ensemble'"
                          " and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                result = c.execute("select station_id from syy_integrated_study"
                                   " where status='已停止' and id = %s;", table_id)
                db.commit()
                if result == 0:
                    c.execute("update syy_integrated_study set status = '学习失败' where id = %s;", table_id)
                    db.commit()
            c.execute("update parallel_tasks_station set instance_id=Null where task_name='ensemble' and station_id=%s"
                      " and table_id=%s;", (station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int=Null where task_name ='ensemble'"
                      " and station_id=%s and table_id=%s;", (station_id, table_id))
            db.commit()
    c.close()
    db.close()


def start_transfer_learning(host, user, password, database, charset, port, instance_id):
    """
    云版开启迁移学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param instance_id: 实例编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读取状态
    result = c.execute("select station_id from parallel_tasks_station where instance_id = %s"
                       " and task_name = 'ensemble';", instance_id)
    db.commit()
    if result > 0:
        c.close()
        db.close()
        return
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
            c.execute("update parallel_tasks_station set instance_id = %s where task_name='transfer' and station_id=%s"
                      " and table_id=%s;", (instance_id, station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int = %s where task_name = 'transfer'"
                      " and station_id=%s and table_id=%s;",
                      (int(datetime.datetime.now().timestamp()), station_id, table_id))
            db.commit()
            c.execute("update syy_integrated_study set status = '正在迁移' where id=%s;", table_id)
            db.commit()
            try:
                c.execute("select type from configure where id = %s;", station_id)
                db.commit()
                record = c.fetchall()
                predict_type = record[0][0]

                result = c.execute("select id from configure where type = %s and trained = 1;", predict_type)
                db.commit()

                if result > 0:
                    task_start_time = datetime.datetime.now().timestamp()
                    transfer_learning(host=host, user=user, password=password, database=database, charset=charset,
                                      port=port, station_id=station_id)
                    task_end_time = datetime.datetime.now().timestamp()
                    d_time = task_end_time - task_start_time

                    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset,
                                         port=port)
                    c = db.cursor()
                    result = c.execute("select station_id from syy_transfer_study"
                                       " where status='已停止' and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        c.execute(
                            "select best_model" +
                            " from best_feature_parameters_and_model" +
                            " where id = %s and predict_term=%s;", (station_id, 'short'))
                        db.commit()
                        record1 = c.fetchall()
                        c.execute(
                            "select best_model" +
                            " from best_feature_parameters_and_model" +
                            " where id = %s and predict_term=%s;", (station_id, 'ultra_short'))
                        db.commit()
                        record2 = c.fetchall()

                        c.execute("update syy_transfer_study set ultra_best_model = %s where id = %s;",
                                  (record2[0][0], table_id))
                        db.commit()
                        c.execute("update syy_transfer_study set short_best_model = %s where id = %s;",
                                  (record1[0][0], table_id))
                        db.commit()
                        c.execute("update syy_transfer_study set medium_best_model = %s where id = %s;",
                                  (record1[0][0], table_id))
                        db.commit()

                        c.execute("update syy_transfer_study set status = '迁移完成' where id = %s;", table_id)
                        db.commit()

                        c.execute("update syy_transfer_study set time = %s where id = %s;",
                                  (d_time, table_id))
                        db.commit()
                else:
                    result = c.execute("select station_id from syy_transfer_study"
                                       " where status='已停止' and id = %s;", table_id)
                    db.commit()
                    if result == 0:
                        c.execute("update syy_transfer_study set status = '迁移失败' where id = %s;", table_id)
                        db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                c.execute("DELETE FROM parallel_tasks_station where task_name = 'transfer'"
                          " and station_id = %s and table_id=%s;", (station_id, table_id))
                db.commit()
                result = c.execute("select station_id from syy_transfer_study"
                                   " where status='已停止' and id = %s;", table_id)
                db.commit()
                if result == 0:
                    c.execute("update syy_transfer_study set status = '迁移失败' where id = %s;", table_id)
                    db.commit()
            c.execute("update parallel_tasks_station set instance_id=Null where task_name='transfer' and station_id=%s"
                      " and table_id=%s;", (station_id, table_id))
            c.execute("update parallel_tasks_station set task_start_time_int=Null where task_name ='transfer'"
                      " and station_id=%s and table_id=%s;", (station_id, table_id))
            db.commit()
    c.close()
    db.close()


def stop_task(host, user, password, database, charset, port):
    """
    云版开启特征工程
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
    # 关闭特征工程---------------------------------------------------------------------------------------------------------
    result = c.execute("select id from syy_characteristic_engineer where command = 2;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()
        if len(record_id_list) > 0:
            for record_id in record_id_list:
                result = c.execute("update syy_characteristic_engineer set command = 0 where command = 2 and id = %s;",
                                   record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    try:
                        c.execute("select station_id from syy_characteristic_engineer where id = %s;", record_id)
                        db.commit()
                        record = c.fetchall()
                        station_id = record[0][0]
                        c.execute("update parallel_tasks_station set task_status = 'task_stopped'"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'feature', record_id))
                        c.execute("update parallel_tasks_station set instance_id = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'feature', record_id))
                        c.execute("update parallel_tasks_station set task_start_time_int = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'feature', record_id))
                        db.commit()
                        for i in range(2):
                            c.execute("update parallel_tasks set task_status = 3 where station_id=%s and task_type=%s;",
                                      (station_id, 'data_load_preprocess_feature'))
                            c.execute("update parallel_tasks set task_status = 3 where station_id=%s and task_type=%s;",
                                      (station_id, 'train_feature'))
                            c.execute("update parallel_tasks set task_status = 3 where station_id=%s and task_type=%s;",
                                      (station_id, 'predict_feature'))
                            db.commit()
                            if i == 0:
                                time.sleep(3)
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
    # 关闭参数寻优---------------------------------------------------------------------------------------------------------
    result = c.execute("select id from syy_model_optimization where command = 2;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()
        if len(record_id_list) > 0:
            for record_id in record_id_list:
                result = c.execute(
                    "update syy_model_optimization set command = 0 where command = 2 and id = %s;",
                    record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    try:
                        c.execute("select station_id from syy_model_optimization where id = %s;", record_id)
                        db.commit()
                        record = c.fetchall()
                        station_id = record[0][0]
                        c.execute("update parallel_tasks_station set task_status = 'task_stopped'"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'parameter', record_id))
                        c.execute("update parallel_tasks_station set instance_id = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'parameter', record_id))
                        c.execute("update parallel_tasks_station set task_start_time_int = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'parameter', record_id))
                        db.commit()
                        for i in range(2):
                            c.execute("update parallel_tasks set task_status = 3 where station_id=%s and task_type=%s;",
                                      (station_id, 'best_parameter_search'))
                            db.commit()
                            if i == 0:
                                time.sleep(3)
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
    # 关闭集成学习---------------------------------------------------------------------------------------------------------
    result = c.execute("select id from syy_integrated_study where command = 2;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()
        if len(record_id_list) > 0:
            for record_id in record_id_list:
                result = c.execute(
                    "update syy_integrated_study set command = 0 where command = 2 and id = %s;",
                    record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    try:
                        c.execute("select station_id from syy_integrated_study where id = %s;", record_id)
                        db.commit()
                        record = c.fetchall()
                        station_id = record[0][0]
                        c.execute("update parallel_tasks_station set task_status = 'task_stopped'"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'ensemble', record_id))
                        c.execute("update parallel_tasks_station set instance_id = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'ensemble', record_id))
                        c.execute("update parallel_tasks_station set task_start_time_int = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'ensemble', record_id))
                        db.commit()
                        for i in range(2):
                            c.execute("update parallel_tasks set task_status = 3 where station_id=%s and task_type=%s;",
                                      (station_id, 'data_load_preprocess'))
                            c.execute("update parallel_tasks set task_status = 3 where station_id=%s and task_type=%s;",
                                      (station_id, 'train'))
                            c.execute("update parallel_tasks set task_status = 3 where station_id=%s and task_type=%s;",
                                      (station_id, 'predict'))
                            db.commit()
                            if i == 0:
                                time.sleep(3)
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
    # 关闭迁移学习---------------------------------------------------------------------------------------------------------
    result = c.execute("select id from syy_transfer_study where command = 2;")
    db.commit()
    if result > 0:
        record = c.fetchall()
        record_id_list = numpy.array(record)[:, 0].tolist()
        if len(record_id_list) > 0:
            for record_id in record_id_list:
                result = c.execute(
                    "update syy_transfer_study set command = 0 where command = 2 and id = %s;",
                    record_id)  # 指令矩阵归零
                db.commit()
                if result == 1:
                    try:
                        c.execute("select station_id from syy_transfer_study where id = %s;", record_id)
                        db.commit()
                        record = c.fetchall()
                        station_id = record[0][0]
                        c.execute("update parallel_tasks_station set task_status = 'task_stopped'"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'transfer', record_id))
                        c.execute("update parallel_tasks_station set instance_id = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'transfer', record_id))
                        c.execute("update parallel_tasks_station set task_start_time_int = Null"
                                  " where station_id=%s and task_name=%s"
                                  " and table_id=%s;", (station_id, 'transfer', record_id))
                        db.commit()
                    except Exception as err:
                        logs.error(str(err), exc_info=True)
                        db.rollback()
    c.close()
    db.close()


def timeout_task(host, user, password, database, charset, port, instance_id):
    """
    云版开启特征工程
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param instance_id: 实例编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 特征工程
    result = c.execute("select id, task_start_time_int from parallel_tasks_station where task_name=%s"
                       " and instance_id=%s;", ('feature', instance_id))
    db.commit()
    record = c.fetchall()
    for i in range(result):
        if record[i][1] is not None:
            if int(datetime.datetime.now().timestamp()) - record[i][1] > 3600 * 6:
                c.execute("update parallel_tasks_station set instance_id = Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'feature', record[i][1]))
                db.commit()
                c.execute("update parallel_tasks_station set task_start_time_int=Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'feature', record[i][1]))
                db.commit()
    # 参数寻优
    result = c.execute("select id, task_start_time_int from parallel_tasks_station where task_name=%s"
                       " and instance_id=%s;", ('parameter', instance_id))
    db.commit()
    record = c.fetchall()
    for i in range(result):
        if record[i][1] is not None:
            if int(datetime.datetime.now().timestamp()) - record[i][1] > 3600 * 12:
                c.execute("update parallel_tasks_station set instance_id = Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'parameter', record[i][1]))
                db.commit()
                c.execute("update parallel_tasks_station set task_start_time_int=Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'parameter', record[i][1]))
                db.commit()
    # 集成学习
    result = c.execute("select id, task_start_time_int from parallel_tasks_station where task_name=%s"
                       " and instance_id=%s;", ('ensemble', instance_id))
    db.commit()
    record = c.fetchall()
    for i in range(result):
        if record[i][1] is not None:
            if int(datetime.datetime.now().timestamp()) - record[i][1] > 3600 * 6:
                c.execute("update parallel_tasks_station set instance_id = Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'ensemble', record[i][1]))
                db.commit()
                c.execute("update parallel_tasks_station set task_start_time_int=Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'ensemble', record[i][1]))
                db.commit()
    # 迁移学习
    result = c.execute("select id, task_start_time_int from parallel_tasks_station where task_name=%s"
                       " and instance_id=%s;", ('feature', instance_id))
    db.commit()
    record = c.fetchall()
    for i in range(result):
        if record[i][1] is not None:
            if int(datetime.datetime.now().timestamp()) - record[i][1] > 3600:
                c.execute("update parallel_tasks_station set instance_id = Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'transfer', record[i][1]))
                db.commit()
                c.execute("update parallel_tasks_station set task_start_time_int=Null where id = %s and task_name = %s"
                          " and task_start_time_int = %s;", (record[i][0], 'transfer', record[i][1]))
                db.commit()
    c.close()
    db.close()


def instance_restart(host, user, password, database, charset, port, instance_id):
    """
    云版开启特征工程
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param instance_id: 实例编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    # 特征工程，feature，syy_characteristic_engineer
    c.execute("select table_id from parallel_tasks_station where task_name='feature' and instance_id=%s", instance_id)
    db.commit()
    record = c.fetchall()
    for i in range(len(record)):
        try:
            c.execute("update syy_characteristic_engineer set status = '进入任务列表序列' where id=%s;",
                      record[i][0])
            db.commit()
        except Exception as err:
            db.rollback()
            logs.error(str(err), exc_info=True)

    # 参数寻优，parameter，syy_model_optimization
    c.execute("select table_id from parallel_tasks_station where task_name='parameter' and instance_id=%s", instance_id)
    db.commit()
    record = c.fetchall()
    for i in range(len(record)):
        try:
            c.execute("update syy_model_optimization set status = '进入任务列表序列' where id=%s;",
                      record[i][0])
            db.commit()
        except Exception as err:
            db.rollback()
            logs.error(str(err), exc_info=True)

    # 集成学习，ensemble，syy_integrated_study
    c.execute("select table_id from parallel_tasks_station where task_name='ensemble' and instance_id=%s", instance_id)
    db.commit()
    record = c.fetchall()
    for i in range(len(record)):
        try:
            c.execute("update syy_integrated_study set status = '进入任务列表序列' where id=%s;",
                      record[i][0])
            db.commit()
        except Exception as err:
            db.rollback()
            logs.error(str(err), exc_info=True)

    # 迁移学习，transfer，syy_transfer_study
    c.execute("select table_id from parallel_tasks_station where task_name='transfer' and instance_id=%s", instance_id)
    db.commit()
    record = c.fetchall()
    for i in range(len(record)):
        try:
            c.execute("update syy_transfer_study set status = '进入任务列表序列' where id=%s;",
                      record[i][0])
            db.commit()
        except Exception as err:
            db.rollback()
            logs.error(str(err), exc_info=True)

    try:
        c.execute("update parallel_tasks_station set task_status = 0 where instance_id=%s", instance_id)
        db.commit()
    except Exception as err:
        db.rollback()
        logs.error(str(err), exc_info=True)
    try:
        c.execute("update parallel_tasks_station set task_start_time_int = Null where instance_id=%s", instance_id)
        db.commit()
    except Exception as err:
        db.rollback()
        logs.error(str(err), exc_info=True)
    try:
        c.execute("update parallel_tasks_station set instance_id = Null where instance_id=%s", instance_id)
        db.commit()
    except Exception as err:
        db.rollback()
        logs.error(str(err), exc_info=True)

    c.close()
    db.close()


def update_time(table_name, table_id, db):
    """
    云版开启特征工程
    :param table_name: 表名
    :param table_id: 表id
    :param db: 数据库连接
    :return: None
    """
    c = db.cursor()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        c.execute("update " + table_name + " set update_time = 0 where id=%s", (table_id, now_time))
        db.commit()
    except Exception as err:
        db.rollback()
        logs.error(str(err), exc_info=True)

    c.close()

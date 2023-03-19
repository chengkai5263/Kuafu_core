
import datetime
import time
from task.mysql.master_predict_hn import MultiPlantPredict
import pymysql
from task.mysql.model_evaluation import evaluate_for_station
import pandas
import numpy
from matplotlib import pyplot
import os
from task.mysql import draw_figure
from common.logger import logs


def predict(host, user, password, database, charset, port, scheduler, executor_name):
    """
    串行任务：特征工程+参数寻优+集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :return: None
    """
    use_cols_of_conf_hn = "['id', 'name', 'capacity', 'type', 'sr_col', 'bin_number', 'predict_pattern', " \
                          "'model_savepath', 'station_status', 'source_capacity_for_transfer'] "
    model = MultiPlantPredict(host, user, password, database, charset, port, use_cols_of_conf_hn)
    model.get_configure()
    model.init_plants_for_prediction()

    # 短期预测
    start_time_last = "2021-09-20 00:00:00"
    start_time_float = datetime.datetime.strptime("2021/10/01 12:00", '%Y/%m/%d %H:%M').timestamp()
    for i in range(273):
        predict_start_time_hn = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_float + i * 86400))
        for plant in model.plants_for_prediction:
            try:
                copy_nwp(host, user, password, database, charset, port, plant.plant_id, start_time_last, predict_start_time_hn)
            except Exception as err:
                logs.error(str(err), exc_info=True)
            plant.predict_interval_and_sql(predict_start_time_hn[:-8]+'23:45:00', enable_interval_predict=True, predict_term='short')
        start_time_last = predict_start_time_hn

    # for i in range(31):
    #     predict_start_time_hn = "2021-10-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         # args = [predict_start_time_hn, True, 'short']
    #         # scheduler.add_job(plant.predict_interval_and_sql,
    #         #                   args=args,
    #         #                   coalesce=True, misfire_grace_time=None)
    #         # time.sleep(10)
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(30):
    #     predict_start_time_hn = "2021-11-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(31):
    #     predict_start_time_hn = "2021-12-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(31):
    #     predict_start_time_hn = "2022-1-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(28):
    #     predict_start_time_hn = "2022-2-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(31):
    #     predict_start_time_hn = "2022-3-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(30):
    #     predict_start_time_hn = "2022-4-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(31):
    #     predict_start_time_hn = "2022-5-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')
    # for i in range(30):
    #     predict_start_time_hn = "2022-6-%s 23:45:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='short')

    # 超短期预测
    start_time_float = datetime.datetime.strptime("2021/10/01 00:00", '%Y/%m/%d %H:%M').timestamp()
    for i in range(273*96):
        predict_start_time_hn = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_float + i * 900))
        for plant in model.plants_for_prediction:
            plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='ultra_short')

    # # 中期预测
    # # start_time_last = "2022-03-20 00:00:00"
    # for i in range(30):
    #     predict_start_time_hn = "2022-04-%s 00:00:00" % str(i + 1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         # try:
    #         #     copy_nwp(host, user, password, database, charset, port, plant.plant_id, start_time_last, predict_start_time_hn)
    #         #     start_time_last = predict_start_time_hn
    #         # except Exception as err:
    #         #     logs.error(str(err), exc_info=True)
    #
    #         # args = [predict_start_time_hn, True, 'medium']
    #         # scheduler.add_job(plant.predict_interval_and_sql, executor=executor_name,
    #         #                   args=args,
    #         #                   coalesce=True, misfire_grace_time=None)
    #
    #         plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='medium')

    print('done')


def connect(host, user, password, database, charset, port):
    """
    串行任务：特征工程+参数寻优+集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :return: None
    """
    # 连接到MySQL数据库
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 创建游标对象
    cursor = db.cursor()

    name = 'predict_term, model_name, start_time, forecast_time, predict_power, upper_bound_90, lower_bound_90, ' \
           'upper_bound_80, lower_bound_80, upper_bound_70, lower_bound_70, upper_bound_60, lower_bound_60, ' \
           'upper_bound_50, lower_bound_50'

    # 选取源表中的数据
    select_query = "SELECT " + name + " FROM predict_power_52003_s_u"
    cursor.execute(select_query)

    # 将选取的数据插入到目标表中
    insert_query = "INSERT INTO predict_power_52003 (" + name + ") VALUES (" + ("%s," * 15).strip(',') + ")"
    for row in cursor.fetchall():
        cursor.execute(insert_query, row)
    db.commit()

    # 选取源表中的数据
    select_query = "SELECT " + name + " FROM predict_power_52003_m"
    cursor.execute(select_query)

    # 将选取的数据插入到目标表中
    insert_query = "INSERT INTO predict_power_52003 (" + name + ") VALUES (" + ("%s," * 15).strip(',') + ")"
    for row in cursor.fetchall():
        cursor.execute(insert_query, row)
    db.commit()

    # 关闭游标和连接
    cursor.close()
    db.close()


def test_evaluate(host, user, password, database, charset, port, station_id):
    """
    串行任务：特征工程+参数寻优+集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param station_id: 场站编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读预测功率
    c.execute('select capacity, type from configure where id = %s limit 1;', station_id)
    db.commit()

    record = c.fetchall()
    online_capacity = record[0][0]
    predict_type = record[0][1]

    short_td_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                             predict_term='short',
                                             host=host, user=user, password=password,
                                             database=database, charset=charset, port=port,
                                             model_name_and_state='no_nwp',
                                             scene='operation',
                                             evaluation="actual_power",
                                             predict_type=predict_type)
    short_gb_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                             predict_term='short',
                                             host=host, user=user, password=password,
                                             database=database, charset=charset, port=port,
                                             model_name_and_state='no_nwp',
                                             scene='operation',
                                             predict_type=predict_type, evaluation="capacity")
    ultra_short_td_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                   predict_term='ultra_short',
                                                   host=host, user=user, password=password,
                                                   database=database, charset=charset, port=port,
                                                   model_name_and_state='no_nwp',
                                                   scene='operation',
                                                   evaluation="actual_power",
                                                   predict_type=predict_type)
    ultra_short_gb_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                   predict_term='ultra_short',
                                                   host=host, user=user, password=password,
                                                   database=database, charset=charset, port=port,
                                                   model_name_and_state='no_nwp',
                                                   scene='operation',
                                                   predict_type=predict_type, evaluation="capacity")
    medium_td_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                              predict_term='medium',
                                              host=host, user=user, password=password,
                                              database=database, charset=charset, port=port,
                                              model_name_and_state='no_nwp',
                                              scene='operation',
                                              evaluation="actual_power",
                                              predict_type=predict_type)
    medium_gb_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                              predict_term='medium',
                                              host=host, user=user, password=password,
                                              database=database, charset=charset, port=port,
                                              model_name_and_state='no_nwp',
                                              scene='operation',
                                              predict_type=predict_type, evaluation="capacity")

    return {'短期两个细则': short_td_accuracy, '短期国标': short_gb_accuracy,
            '超短期两个细则': ultra_short_td_accuracy, '超短期国标': ultra_short_gb_accuracy,
            '中期两个细则': medium_td_accuracy, '中期国标': medium_gb_accuracy}


def test_figure_short(host, user, password, database, charset, port, station_id):
    """
    串行任务：特征工程+参数寻优+集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param station_id: 场站编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读预测功率
    c.execute('select predict_term, model_name, start_time, forecast_time, predict_power from predict_power_%s'
              ' where predict_term = %s ORDER BY start_time asc, forecast_time asc;', (station_id, 'short'))
    db.commit()

    record = c.fetchall()
    predict_power = pandas.DataFrame(record, columns=['predict_term', 'model_name', 'start_time', 'forecast_time',
                                                      'predict_power'])

    # 读真实功率
    c.execute('select time, power from real_power_%s where time between %s and %s ORDER BY time asc;', (station_id,
               predict_power.loc[:, 'forecast_time'].iloc[0], predict_power.loc[:, 'forecast_time'].iloc[-1]))
    db.commit()
    record = c.fetchall()
    real_power = pandas.DataFrame(record, columns=['time', 'power'])

    train_data_merge = pandas.merge(predict_power, real_power, left_on='forecast_time', right_on='time', how='left')
    day = int(len(train_data_merge) / 288)
    predict_power = train_data_merge.loc[:, 'predict_power'].values.reshape(day, 288)
    actual_power = train_data_merge.loc[:, 'power'].values.reshape(day, 288)

    power_cluster = numpy.hstack(
        (actual_power[:, :96].reshape(-1, 1), predict_power[:, :96].reshape(-1, 1)))
    power_cluster = pandas.DataFrame(power_cluster)
    power_cluster = power_cluster.dropna(axis=0, how='any').values

    pyplot.plot(power_cluster[:, 0], label='real_power')
    pyplot.plot(power_cluster[:, 1], label='predict_power')
    pyplot.show()
    pyplot.legend()


def figure(host, user, password, database, charset, port, station_id):
    """
    串行任务：特征工程+参数寻优+集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param station_id: 场站编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    path = './work_dir/figure/opration/' + str(station_id) + '/'
    os.makedirs(path + 'ultra_short/', exist_ok=True)
    os.makedirs(path + 'short/', exist_ok=True)
    os.makedirs(path + 'medium/', exist_ok=True)

    c.execute("select distinct model_name from predict_power_" + str(station_id) +
              " where predict_term = 'ultra_short';")
    db.commit()
    ultra_short_tuple = c.fetchall()

    c.execute("select distinct model_name from predict_power_" + str(station_id) +
              " where predict_term = 'short';")
    db.commit()
    short_tuple = c.fetchall()

    c.execute("select distinct model_name from predict_power_" + str(station_id) +
              " where predict_term = 'medium';")
    db.commit()
    medium_tuple = c.fetchall()

    for i in range(len(ultra_short_tuple)):
        model_name, model_state = ultra_short_tuple[i][0].split('_', 1)
        draw_figure.evaluate(host=host, user=user, password=password, database=database,
                             charset=charset,
                             port=port, model_name=model_name, model_state='_' + model_state,
                             term='ultra_short',
                             save_file_path=path + 'ultra_short/' + ultra_short_tuple[i][0],
                             station_id=station_id, scene='operation')

    for i in range(len(short_tuple)):
        model_name, model_state = short_tuple[i][0].split('_', 1)
        draw_figure.evaluate(host=host, user=user, password=password, database=database,
                             charset=charset,
                             port=port, model_name=model_name, model_state='_' + model_state,
                             term='short', save_file_path=path + 'short/' + short_tuple[i][0],
                             station_id=station_id, scene='operation')

    for i in range(len(medium_tuple)):
        model_name, model_state = medium_tuple[i][0].split('_', 1)
        draw_figure.evaluate(host=host, user=user, password=password, database=database,
                             charset=charset,
                             port=port, model_name=model_name, model_state='_' + model_state,
                             term='medium', save_file_path=path + 'medium/' + medium_tuple[i][0],
                             station_id=station_id, scene='operation')


def copy_nwp(host, user, password, database, charset, port, station_id, start_time_last, start_time_now):
    """
    串行任务：特征工程+参数寻优+集成学习
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param station_id: 场站编号
    :param start_time_last: 场站编号
    :param start_time_now: 场站编号
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 创建游标对象
    cursor = db.cursor()

    select_query = "SELECT * FROM nwp_%s limit 1"
    cursor.execute(select_query, station_id)
    des = cursor.description
    name = ','.join(iterm[0] for iterm in des[1:])

    # 选取源表中的数据
    select_query = "SELECT " + name + " FROM nwp_%s_copy where start_time between %s and %s"
    cursor.execute(select_query, (station_id, start_time_last, start_time_now))

    # 将选取的数据插入到目标表中
    insert_query = "INSERT INTO nwp_" + str(station_id) + " (" + name + ") VALUES (" + ("%s," * (len(des)-1)).strip(',') + ")"
    for row in cursor.fetchall():
        cursor.execute(insert_query, row)
    db.commit()

    # 关闭游标和连接
    cursor.close()
    db.close()


if __name__ == "__main__":
    host = 'localhost'
    user = 'root'
    password = '123456'
    database = 'kuafu'
    charset = 'utf8'
    port = 3306

    station_id = 51002

    # connect(host, user, password, database, charset, port)
    result = test_evaluate(host, user, password, database, charset, port, station_id)
    print(str(station_id) + str(result))
    # figure(host, user, password, database, charset, port, station_id)

    station_id = 52003

    # connect(host, user, password, database, charset, port)
    result = test_evaluate(host, user, password, database, charset, port, station_id)
    print(str(station_id) + str(result))
    # figure(host, user, password, database, charset, port, station_id)

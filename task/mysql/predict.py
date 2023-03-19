
import numpy
import time
import pandas
import datetime
import pymysql

from common import predict
from common.tools import catch_exception
from task.mysql.interval_forecast import IntervalForecastTask
from task.mysql import predict_without_nwp
from task.mysql.load_test_data_iterate import LoadTestdataiterate
from common.tools import load_model
from common.logger import logs


# 程凯
@catch_exception("predict_short_power error: ")
def predict_short_power(host, user, password, database, charset, port, enable_interval_predict=False,
                        model_name='BPNN_without_history_power', start_time=None):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param model_name: 预测的方法
    :param enable_interval_predict: 是否进行区间预测
    :param start_time: 开始预测时间
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    if start_time is None:
        short_time = '23:50'
        data_now = time.strftime("%Y/%m/%d", time.gmtime(datetime.datetime.now().timestamp()))
        start_time = datetime.datetime.strptime(data_now + ' ' + short_time, '%Y/%m/%d %H:%M').timestamp()
        now_time = datetime.datetime.now().timestamp()
        if now_time > start_time - 3600:
            start_time = start_time + 3600 * 24

    if isinstance(start_time, str):
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M').timestamp()

    result = c.execute("select * from configure where station_status = 1 or station_status = 2;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    df = pandas.DataFrame(record, columns=coul)

    # 循环预测
    probabilistic_model = IntervalForecastTask()
    for i in range(result):
        station_name = df.loc[:, 'name'].iloc[i]
        station_id = df.loc[:, 'id'].iloc[i]
        predict_type = df.loc[:, 'type'].iloc[i]
        sr_col = df.loc[:, 'sr_col'].iloc[i]
        model_path = df.loc[:, 'model_savepath'].iloc[i]
        online_capacity = df.loc[:, 'capacity'].iloc[i]
        bin_num = df.loc[:, "bin_number"].iloc[i]
        short_model_savepath = df.loc[:, 'model_savepath'].iloc[i]
        predict_pattern = df.loc[:, 'predict_pattern'].iloc[i]
        station_status = df.loc[:, 'station_status'].iloc[i]
        source_capacity_for_transfer = df.loc[:, 'source_capacity_for_transfer'].iloc[i]

        # 如果是迁移学习，rate_for_transfer为实际容量除以源场站容量；如果不为迁移学习，rate_for_transfer为1
        rate_for_transfer = 1
        if station_status == 1:
            rate_for_transfer = online_capacity/source_capacity_for_transfer

        result = c.execute("select best_model from best_feature_parameters_and_model where"
                           " id = %s and predict_term = %s;", (station_id, 'short'))
        db.commit()
        if result == 1:
            record = c.fetchall()
            best_model = record[0][0]
        else:
            best_model = model_name

        result = c.execute("select second_model from best_feature_parameters_and_model where"
                           " id = %s and predict_term = %s;", (station_id, 'short'))
        db.commit()
        if result == 1:
            record = c.fetchall()
            second_model = record[0][0]
        else:
            second_model = model_name

        test_data = LoadTestdataiterate()
        # 从MySQL读取测试数据
        ann_model_name, b = best_model.split('_', 1)

        try:
            usecols = load_model(model_path + str(station_id) + '/short/' + ann_model_name + '/' + 'short_usecols.pkl')
            test_feature, history_power = test_data.upload_predict_nwp_short(start_time=start_time, station_id=station_id,
                                                                             usecols=usecols,
                                                                             predict_type=predict_type, host=host,
                                                                             user=user, password=password,
                                                                             database=database, charset=charset, port=port)
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.info(station_name + ' short, 读NWP时失败')
            test_feature = numpy.array([])
            history_power = None

        forecast_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(start_time)))

        predict_result_without_nwp = predict_without_nwp.predict_without_nwp(
            station_name=station_name, station_id=station_id, station_type=predict_type,
            station_capacity=online_capacity, model_save_path=model_path, bin_num=bin_num, connection=db,
            predict_term='short', predict_start_time=forecast_time, predict_pattern=predict_pattern)

        predict_result, sr, model_used = predict.predict_short_power(
            station_name=station_name, station_id=station_id,
            model_path=model_path,
            predict_type=predict_type, sr_col=sr_col,
            online_capacity=online_capacity,
            capacity=online_capacity, forecast_time=forecast_time,
            best_model=best_model, second_model=second_model,
            predict_result_without_nwp=predict_result_without_nwp,
            test_feature=test_feature, rate_for_transfer=rate_for_transfer)
        if enable_interval_predict:
            interval_result = probabilistic_model.interval_prediction(station_id=station_id, predict_term="short",
                                                                      predict_type=predict_type,
                                                                      point_predict_data=predict_result.values[0, :].
                                                                      tolist(),
                                                                      model_path=short_model_savepath,
                                                                      model_name=best_model,
                                                                      online_capacity=online_capacity,
                                                                      bin_num=bin_num, sr=sr)

            result = numpy.hstack((predict_result.values.T, interval_result.values))
            start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))

            list_predict_power = ','.join(
                ['predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power', 'upper_bound_90',
                 'lower_bound_90', 'upper_bound_80', 'lower_bound_80', 'upper_bound_70', 'lower_bound_70',
                 'upper_bound_60', 'lower_bound_60', 'upper_bound_50', 'lower_bound_50'])
            for ii in range(len(result)):
                forecast_time_float = start_time + (ii + 1) * 15 * 60
                forecast_time_struct = time.localtime(forecast_time_float)
                forecast_time = time.strftime("%Y/%m/%d %H:%M", forecast_time_struct)
                value = (start_time_str, forecast_time) + tuple(result[ii, :])
                values = tuple(['short']) + tuple([model_used]) + value
                c.execute("INSERT INTO predict_power_" + str(station_id) + '(' + list_predict_power + ')' + "VALUES("
                          + ("%s," * 15).strip(',') + ")", values)
            db.commit()
        else:
            result = predict_result.values.T
            start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))

            list_predict_power = ','.join(
                ['predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power'])
            for ii in range(len(result)):
                forecast_time_float = start_time + (ii + 1) * 15 * 60
                forecast_time_struct = time.localtime(forecast_time_float)
                forecast_time = time.strftime("%Y/%m/%d %H:%M", forecast_time_struct)
                value = (start_time_str, forecast_time) + tuple(result[ii, :])
                values = tuple(['short']) + tuple([model_used]) + value
                c.execute("INSERT INTO predict_power_" + str(station_id) + '(' + list_predict_power + ')' +
                          "VALUES(" + ("%s," * 5).strip(',') + ")", values)
            db.commit()
    c.close()
    db.close()


# 程凯
@catch_exception("predict_ultra_short_power error: ")
def predict_ultra_short_power(host, user, password, database, charset, port, enable_interval_predict=False,
                              model_name='BPNN_without_history_power', start_time=None):
    """
    为结果存储文件撰写表头，初始化时调用
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param model_name: 预测的方法
    :param enable_interval_predict: 是否进行区间预测
    :param start_time: 开始预测时间
    :return:
    """
    if start_time is None:
        now_time = datetime.datetime.now().timestamp()
        d = now_time % 900
        start_time = now_time - d
    if isinstance(start_time, str):
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M').timestamp()

    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 定义游标
    c = db.cursor()
    result = c.execute("select * from configure where station_status = 1 or station_status = 2;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    df = pandas.DataFrame(record, columns=coul)

    # --------------------------------------------------------------------------------------------------------------
    # 循环预测
    probabilistic_model = IntervalForecastTask()
    for i in range(result):
        station_name = df.loc[:, 'name'].iloc[i]
        station_id = df.loc[:, 'id'].iloc[i]
        predict_type = df.loc[:, 'type'].iloc[i]
        sr_col = df.loc[:, 'sr_col'].iloc[i]
        model_path = df.loc[:, 'model_savepath'].iloc[i]
        online_capacity = df.loc[:, 'capacity'].iloc[i]
        bin_num = df.loc[:, "bin_number"].iloc[i]
        ultra_short_model_savepath = df.loc[:, 'model_savepath'].iloc[i]
        predict_pattern = df.loc[:, 'predict_pattern'].iloc[i]
        station_status = df.loc[:, 'station_status'].iloc[i]
        source_capacity_for_transfer = df.loc[:, 'source_capacity_for_transfer'].iloc[i]

        # 如果是迁移学习，rate_for_transfer为实际容量除以源场站容量；如果不为迁移学习，rate_for_transfer为1
        rate_for_transfer = 1
        if station_status == 1:
            rate_for_transfer = online_capacity / source_capacity_for_transfer

        result = c.execute("select best_model from best_feature_parameters_and_model where"
                           " id = %s and predict_term = %s;", (station_id, 'ultra_short'))
        db.commit()
        if result == 1:
            record = c.fetchall()
            best_model = record[0][0]
        else:
            best_model = model_name

        result = c.execute("select second_model from best_feature_parameters_and_model where"
                           " id = %s and predict_term = %s;", (station_id, 'ultra_short'))
        db.commit()
        if result == 1:
            record = c.fetchall()
            second_model = record[0][0]
        else:
            second_model = model_name

        test_data = LoadTestdataiterate()
        # 从MySQL读取测试数据
        ann_model_name, b = best_model.split('_', 1)

        try:
            usecols = load_model(
                model_path + str(station_id) + '/ultra_short/' + ann_model_name + '/' + 'ultra_short_usecols.pkl')
            test_feature, history_power = test_data.upload_predict_nwp_ultrashort(start_time=start_time,
                                                                                  station_id=station_id, usecols=usecols,
                                                                                  predict_type=predict_type, host=host,
                                                                                  user=user, password=password,
                                                                                  database=database, charset=charset,
                                                                                  port=port)
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.info(station_name + ' ultra_short, 读NWP时失败')
            test_feature = numpy.array([])
            history_power = None

        forecast_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(start_time)))

        predict_result_without_nwp = predict_without_nwp.predict_without_nwp(
            station_name=station_name, station_id=station_id, station_type=predict_type,
            station_capacity=online_capacity, model_save_path=model_path, bin_num=bin_num, connection=db,
            predict_term='ultra_short', predict_start_time=forecast_time, predict_pattern=predict_pattern)

        predict_result, sr, model_used = predict.predict_ultra_short_power(
            station_name=station_name, station_id=station_id,
            predict_type=predict_type, model_path=model_path,
            sr_col=sr_col, online_capacity=online_capacity,
            capacity=online_capacity, forecast_time=forecast_time,
            best_model=best_model, second_model=second_model,
            predict_result_without_nwp=predict_result_without_nwp,
            test_feature=test_feature, history_power=history_power,
            rate_for_transfer=rate_for_transfer)
        if enable_interval_predict:
            interval_result = probabilistic_model.interval_prediction(station_id=station_id, predict_term="ultra_short",
                                                                      predict_type=predict_type,
                                                                      point_predict_data=predict_result.values[0, :].
                                                                      tolist(),
                                                                      model_path=ultra_short_model_savepath,
                                                                      model_name=best_model,
                                                                      online_capacity=online_capacity,
                                                                      bin_num=bin_num, sr=sr)

            result = numpy.hstack((predict_result.values.T, interval_result.values))
            start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))

            list_predict_power = ','.join(
                ['predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power', 'upper_bound_90',
                 'lower_bound_90', 'upper_bound_80', 'lower_bound_80', 'upper_bound_70', 'lower_bound_70',
                 'upper_bound_60', 'lower_bound_60', 'upper_bound_50', 'lower_bound_50'])

            for ii in range(len(result)):
                forecast_time_float = start_time + (ii + 1) * 15 * 60
                forecast_time_struct = time.localtime(forecast_time_float)
                forecast_time = time.strftime("%Y/%m/%d %H:%M", forecast_time_struct)
                value = (start_time_str, forecast_time) + tuple(result[ii, :])
                values = tuple(['ultra_short']) + tuple([model_used]) + value
                c.execute("INSERT INTO predict_power_" + str(station_id) + '(' + list_predict_power + ')'
                          + "VALUES(" + ("%s," * 15).strip(',') + ")", values)
            db.commit()
        else:
            result = predict_result.values.T
            start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))

            list_predict_power = ','.join(
                ['predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power'])

            for ii in range(len(result)):
                forecast_time_float = start_time + (ii + 1) * 15 * 60
                forecast_time_struct = time.localtime(forecast_time_float)
                forecast_time = time.strftime("%Y/%m/%d %H:%M", forecast_time_struct)
                value = (start_time_str, forecast_time) + tuple(result[ii, :])
                values = tuple(['ultra_short']) + tuple([model_used]) + value
                c.execute("INSERT INTO predict_power_" + str(station_id) + '(' + list_predict_power + ')'
                          + "VALUES(" + ("%s," * 5).strip(',') + ")", values)
            db.commit()
    c.close()
    db.close()

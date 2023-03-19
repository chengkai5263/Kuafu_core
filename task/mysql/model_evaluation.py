# _*_ coding: utf-8 _*_

import pandas
import datetime
import pymysql
from common.logger import logs
from common import model_evaluation


# 主站版误差评价函数
def evaluate_for_master(host, user, password, database, charset, port,
                        error_data_label=None, predict_term='short', save_path=None,
                        sunrise_time=None, sunset_time=None):
    """
    为结果存储文件撰写表头，初始化时调用
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param error_data_label: 限电记录
    :param predict_term: 预测类型:
    :param save_path: 结果存放位置:
    :param sunrise_time: 日出时间:
    :param sunset_time: 日落时间:
    :return:
    """
    # 读取数据库
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 定义游标
    c = db.cursor()
    c.execute("select * from configure where station_status = 2;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe = pandas.DataFrame(record, columns=coul)

    # 场站名称
    station_name_cluster = []
    for i in range(len(dataframe)):
        station_name_cluster.append(dataframe.loc[:, "name"][i])

    # 配置信息
    config_cluster = {}
    for i in range(len(dataframe)):
        config_cluster[dataframe.loc[:, "name"][i]] = {"id": dataframe.loc[:, 'id'][i],
                                                       "type": dataframe.loc[:, "type"][i],
                                                       "sr_col": 0,
                                                       "online_capacity": dataframe.loc[:, "capacity"][i],
                                                       "usecols": eval(dataframe.loc[:, "usecols"][i]),
                                                       "model_savepath": dataframe.loc[:, "model_savepath"][i]}

    # 循环计算各场站的准确率
    accuracy_cluster = {}
    for name in station_name_cluster:
        c.execute('select start_time, forecast_time, predict_power from predict_power_' + str(
            config_cluster[name]['id']) + ';')
        db.commit()
        des = c.description
        record = c.fetchall()
        list(iterm[0] for iterm in des)
        predict_power = pandas.DataFrame(record, columns=['start_time', 'forecast_time', 'predict_power'])

        if save_path is not None:
            path = f'{save_path}{name}_{datetime.datetime.now().strftime("%Y_%m%d_%H%M")}.csv'
            predict_power.to_csv(path, index=False, encoding="utf_8_sig", mode='a+', header=False)
            c.execute('DELETE FROM predict_power_' + str(id) + ' WHERE id > %s', 0)
            db.commit()

        c.execute('select time, power from real_power_' + str(config_cluster[name]['id']) + ';')
        db.commit()
        des = c.description
        record = c.fetchall()
        list(iterm[0] for iterm in des)
        real_power = pandas.DataFrame(record, columns=['time', 'power'])

        if sunrise_time is None or sunset_time is None:
            accuracy_cluster[name] = model_evaluation.evaluate_GB_T_40607_2021_withtimetag(
                predict_power, real_power, config_cluster[name]['online_capacity'], predict_term, error_data_label)
        else:
            accuracy_cluster[name] = model_evaluation.evaluate_NB_T_32011_2013_withtimetag_solar_without_night(
                predict_power, real_power, config_cluster[name]['online_capacity'], 'short', error_data_label,
                sunrise_time=sunrise_time, sunset_time=sunset_time)

    c.close()
    db.close()
    return accuracy_cluster


# 子站版误差评价函数
def evaluate_for_station(name=None, id=None, online_capacity=None, error_data_label=None, predict_term=None, host=None,
                         user=None, password=None, database=None, charset=None, port=None, save_path=None,
                         model_name_and_state=None, scene='operation', sunrise_time=None, sunset_time=None,
                         evaluation=None, predict_type='wind', standard=None):
    """
    为结果存储文件撰写表头，初始化时调用
    :param name: 场站名（打印日志用，不影响程序执行）
    :param id: id
    :param online_capacity:
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param error_data_label: 限电记录
    :param predict_term: 预测类型:
    :param save_path: 结果存放位置:
    :param model_name_and_state: 结果存放位置:
    :param scene: 运行场景‘train’ or ‘operation’:
    :param sunrise_time: 日出时间:
    :param sunset_time: 日落时间:
    :return:
    """
    # 读取数据库
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)

    # 定义游标
    c = db.cursor()

    # 读预测功率
    if model_name_and_state is not None:
        if scene == 'train':
            # 集成学习评价，表为 predict_power_id_train
            c.execute('select predict_term, model_name, start_time, forecast_time, predict_power from predict_power_' +
                      str(id) + '_train where model_name = %s and predict_term = %s'
                                ' ORDER BY start_time asc, forecast_time asc;', (model_name_and_state, predict_term))
        elif scene == 'feature':
            # 特征工程对BPNN模型的评价，表为 predict_power_id_train_feature
            c.execute('select predict_term, model_name, start_time, forecast_time, predict_power from predict_power_' +
                      str(id) + '_train_feature where model_name = %s and predict_term = %s'
                                ' ORDER BY start_time asc, forecast_time asc;', (model_name_and_state, predict_term))
        else:
            # 上线运行时的评价，表为 predict_power_id
            c.execute('select predict_term, model_name, start_time, forecast_time, predict_power from predict_power_' +
                      str(id) + ' where predict_term = %s ORDER BY start_time asc, forecast_time asc;', predict_term)
    else:
        # 中期预测评价，表为 predict_power_id_train，不区分预测模型
        c.execute('select predict_term, model_name, start_time, forecast_time, predict_power from predict_power_' +
                  str(id) + '_train where predict_term = %s ORDER BY start_time asc, forecast_time asc;', predict_term)
    db.commit()

    record = c.fetchall()
    predict_power = pandas.DataFrame(record, columns=['predict_term', 'model_name', 'start_time', 'forecast_time',
                                                      'predict_power'])

    if save_path is not None:
        path = f'{save_path}{name}_{datetime.datetime.now().strftime("%Y_%m%d_%H%M")}.csv'
        predict_power.to_csv(path, index=False, encoding="utf_8_sig", mode='a+', header=False)
        c.execute('DELETE FROM predict_power_' + str(id) + ' WHERE id > %s', 0)
        db.commit()

    # 读真实功率
    try:
        c.execute('select time, power from real_power_' + str(id) + ' where time between %s and %s ORDER BY time asc;',
                  (predict_power.loc[:, 'forecast_time'].iloc[0], predict_power.loc[:, 'forecast_time'].iloc[-1]))
        db.commit()
        record = c.fetchall()
        real_power = pandas.DataFrame(record, columns=['time', 'power'])

        # 计算准确率
        if standard == 'Q_CSG1211017_2018':
            # 如果标准名称为'Q_CSG1211017_2018'
            accuracy_station = model_evaluation.result_evaluation_Q_CSG1211017_2018_without_time_tag(
                predict_power, real_power, online_capacity, predict_term=predict_term, evaluation=evaluation)
        else:
            if sunrise_time is not None and sunset_time is not None and predict_type == 'solar':
                # 如果入参有日出日落时间
                accuracy_station = model_evaluation.evaluate_NB_T_32011_2013_withtimetag_solar_without_night(
                    predict_power, real_power, online_capacity, predict_term, error_data_label,
                    sunrise_time=sunrise_time, sunset_time=sunset_time)
            else:
                if evaluation == 'capacity':
                    # 如果以开机容量作为考核
                    accuracy_station = model_evaluation.evaluate_GB_T_40607_2021_withtimetag(
                        predict_power, real_power, online_capacity, predict_term, error_data_label)
                else:
                    accuracy_station = model_evaluation.result_evaluation_Two_Detailed_Rules_without_time_tag(
                        predict_power, real_power, online_capacity, predict_term, error_data_label)
    except Exception as err:
        logs.error(str(err), exc_info=True)
        accuracy_station = 0
        logs.info('场站，编号:' + str(id) + '，时间尺度:' + predict_term + '，模型:' + model_name_and_state + '评价无效')
    c.close()
    db.close()
    return accuracy_station

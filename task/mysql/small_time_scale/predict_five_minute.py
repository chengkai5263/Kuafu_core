import datetime

import pymysql

from app.small_time_scale_predict.predict_five_minute_withoutsql import predict_five_minute_power_withoutsql
from common.logger import logs
from common.tools import load_model
import pandas
import time

from task.mysql.ensemble_learning import get_default_feature_parameter


def predict_five_minute_power(host, user, password, database, charset, port, start_time=None, sql="mysql", scheduler=None,
                              executor_name=None, todo_station_df=pandas.DataFrame):
    """
    为结果存储文件撰写表头，初始化时调用
    :param model_set_path: 配置文件路径:
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :param data_resource: 数据来源是‘CSV’或‘SQL’
    :param start_time: 预报的时间
    :param model_name: 预测的方法
    :return:
    """
    if isinstance(start_time, str):
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M').timestamp()
    # ------------------------------------------------------------------------------------------------------------------
    if sql == "kingbas":
        # 获取参与集成学习的模型名称
        db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        c = db.cursor()
        c.execute("select * from configure")
        des = c.description
        record = tuple(c.fetchall())
        coul = list(iterm[0] for iterm in des)
        df = pandas.DataFrame(record, columns=coul)
    else:
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        # 定义游标
        c = db.cursor()
        c.execute("select * from configure;")
        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        df = pandas.DataFrame(record, columns=coul)
    # 云版中添加的，只执行id在do_list列表里的场站
    if len(todo_station_df) != 0:
        df = df[df['id'].isin(todo_station_df['station_id'])]
        df.index = range(0, len(df))
    result = len(df)
    db.close()
    # --------------------------------------------------------------------------------------------------------------
    # 循环预测
    logs.info('start loop predict')
    for i in range(result):
        # one_station_predict(df, i, start_time, host, user, password, database, charset, port, sql)
        # 并行运算
        scheduler.add_job(one_station_predict, executor=executor_name, coalesce=True, misfire_grace_time=None,
                          args=[df, i, start_time, host, user, password, database, charset, port, sql])


def one_station_predict(df, i, start_time, host, user, password, database, charset, port, sql):
    """
    单场预测
    """
    logs.info('start one station predict')
    try:
        if sql == "kingbas":
            # 获取参与集成学习的模型名称
            db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
            c = db.cursor()
        else:
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
            # 定义游标
            c = db.cursor()
        station_id = df.loc[:, 'id'].iloc[i]
        file_path = {"id": df.loc[:, 'id'].iloc[i]}
        predict_type = df.loc[:, 'type'].iloc[i]

        ultra_short_usecols_default, short_usecols_default, \
        ultra_short_best_parameter_default, short_best_parameter_default = \
            get_default_feature_parameter(host=host, user=user, password=password, database=database,
                                          charset=charset, port=port,
                                          predict_type=predict_type,
                                          model_name_cluster_setting=['BPNN'])

        usecols = ultra_short_usecols_default
        sr_col = df.loc[:, 'sr_col'].iloc[i]
        model_path = df.loc[:, 'model_savepath'].iloc[i]
        online_capacity = df.loc[:, 'capacity'].iloc[i]
        # 加在5分钟和15分钟模型
        fifth_model = load_model(
            model_path + str(station_id) + "/five_minute/" + "five_minute_best_model.pkl")
        fifteen_minute_model = load_model(
            model_path + str(station_id) + "/five_minute/" + "fifteen_minute_best_model.pkl")
        predict_result, model_name = predict_five_minute_power_withoutsql(
            station_id=station_id,
            file_path=file_path, predict_type=predict_type, model_path=model_path, sr_col=sr_col,
            online_capacity=online_capacity, capacity=online_capacity, predict_time=start_time, fifth_model=fifth_model,
            fifteen_minute_model=fifteen_minute_model, usecols=usecols, data_resource='SQL', day=None, label=None,
            host=host, user=user, password=password, database=database, charset=charset, port=port, sql=sql)

        start_time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(start_time))

        if sql == "kingbas":
            c.execute("select * from predict_power_" + str(file_path["id"]) + ";")
        else:
            c.execute("select * from predict_power_" + str(file_path["id"]) + ";")
        des = c.description

        list_predict_power = ','.join(
            ['predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power'])

        for ii in range(len(predict_result)):
            forecast_time_float = start_time + (ii + 1) * 5 * 60
            forecast_time_struct = time.localtime(forecast_time_float)
            forecast_time = time.strftime("%Y-%m-%d %H:%M", forecast_time_struct)
            value = (start_time_str, forecast_time) + tuple(predict_result[ii, :])
            values = tuple(['five_minute']) + tuple([model_name]) + value
            c.execute("INSERT INTO predict_power_" + str(station_id) + '(' + list_predict_power + ')'
                      + "VALUES(" + ("%s," * 5).strip(',') + ")", values)
        db.commit()
        db.close()
        logs.info(start_time_str + ": " + str(station_id) + ' predict succeed', exc_info=True)
    except:
        logs.debug(start_time_str + 'one station predict fail', exc_info=True)







if __name__ == '__main__':
    print(__name__)

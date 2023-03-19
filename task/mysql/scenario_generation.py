# _*_ coding: utf-8 _*_

from app.probabilistic_predict.scenario_generation import model_train, scenario_generation
import pymysql
import pandas


def model_train_sql(host, user, password, database, charset, port,
                    station_id, bin_num, predict_term, model_name):
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    c.execute("select id, name, type, sr_col, capacity, model_savepath from configure where id = %s;",
              station_id)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)

    # 配置信息
    config_cluster = {"id": dataframe_config.loc[:, 'id'][0],
                      "name": dataframe_config.loc[:, "name"][0],
                      "capacity": dataframe_config.loc[:, "capacity"][0],
                      "model_savepath": dataframe_config.loc[:, "model_savepath"][0]}

    c.execute('select distinct start_time, forecast_time, predict_power from predict_power_' +
              str(station_id) + '_train where model_name = %s and predict_term = %s'
                                ' ORDER BY start_time asc, forecast_time asc;',
              (model_name, predict_term))
    db.commit()
    record = c.fetchall()
    predict_power = pandas.DataFrame(record, columns=['start_time', 'forecast_time', 'predict_power'])

    c.execute('select distinct time, power from real_power_' + str(station_id) +
              ' where time between %s and %s ORDER BY time asc;',
              (predict_power.loc[:, 'forecast_time'].iloc[0], predict_power.loc[:, 'forecast_time'].iloc[-1]))
    db.commit()
    record = c.fetchall()
    real_power = pandas.DataFrame(record, columns=['time', 'power'])

    train_data_merge = pandas.merge(predict_power, real_power, left_on='forecast_time', right_on='time', how='left')
    train_data_merge = train_data_merge.dropna(axis=0, how='any')
    predict_power = train_data_merge.loc[:, 'predict_power'].values
    actual_power = train_data_merge.loc[:, 'power'].values

    c.close()
    db.close()
    model_train(config_cluster, bin_num, predict_power, actual_power, predict_term, model_name)


def scenario_generation_sql(host, user, password, database, charset, port,
                            station_id, bin_num, predict_term, model_name):
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    c.execute("select id, name, type, sr_col, capacity, model_savepath from configure where id = %s;",
              station_id)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)

    # 配置信息
    config_cluster = {"id": dataframe_config.loc[:, 'id'][0],
                      "name": dataframe_config.loc[:, "name"][0],
                      "capacity": dataframe_config.loc[:, "capacity"][0],
                      "model_savepath": dataframe_config.loc[:, "model_savepath"][0]}

    c.execute('select distinct predict_power from predict_power_' +
              str(station_id) + ' where model_name = %s and predict_term = %s'
                                ' ORDER BY start_time asc, forecast_time asc;',
              (model_name, predict_term))
    db.commit()
    record = c.fetchall()
    point_predict_data = pandas.DataFrame(record, columns=['predict_power']).values[:, 0]
    dataframe = scenario_generation(point_predict_data, config_cluster, bin_num, predict_term, model_name)

    c.close()
    db.close()
    return dataframe, point_predict_data


import pymysql
from task.mysql.predict import predict_short_power, predict_ultra_short_power
from task.mysql.model_evaluation import evaluate_for_station
import pandas
import numpy
from matplotlib import pyplot


def test_predict(host, user, password, database, charset, port):
    # 短期预测
    for i in range(30):           # days
        start_time = '2021-11-%s 00:00' % str(i+1)
        predict_short_power(host, user, password, database, charset, port, start_time=start_time)

    # 超短期预测
    for i in range(0):                         # days
        for j in range(24):                     # hours
            for k in ['00', '15', '30', '45']:  # minutes
                start_time = '2022-04-%s %s:%s' % (str(i + 1), str(j), k)
                predict_ultra_short_power(host, user, password, database, charset, port, start_time=start_time)


def test_evaluate(host, user, password, database, charset, port, station_id):

    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读预测功率
    c.execute('select capacity, type from configure where id = %s limit 1;', station_id)
    db.commit()

    record = c.fetchall()
    online_capacity = record[0][0]
    predict_type = record[0][1]

    short_TD_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                             predict_term='short',
                                             host=host, user=user, password=password,
                                             database=database, charset=charset, port=port,
                                             model_name_and_state='no_NWP',
                                             scene='operation',
                                             evaluation="actual_power",
                                             predict_type=predict_type)
    short_GB_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                             predict_term='short',
                                             host=host, user=user, password=password,
                                             database=database, charset=charset, port=port,
                                             model_name_and_state='no_NWP',
                                             scene='operation',
                                             predict_type=predict_type, evaluation="capacity")
    ultra_short_TD_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                   predict_term='ultra_short',
                                                   host=host, user=user, password=password,
                                                   database=database, charset=charset, port=port,
                                                   model_name_and_state='no_NWP',
                                                   scene='operation',
                                                   evaluation="actual_power",
                                                   predict_type=predict_type)
    ultra_short_GB_accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                                   predict_term='ultra_short',
                                                   host=host, user=user, password=password,
                                                   database=database, charset=charset, port=port,
                                                   model_name_and_state='no_NWP',
                                                   scene='operation',
                                                   predict_type=predict_type, evaluation="capacity")

    return {'短期两个细则': short_TD_accuracy, '短期国标': short_GB_accuracy,
            '超短期两个细则': ultra_short_TD_accuracy, '超短期国标': ultra_short_GB_accuracy}


def test_figure_short(host, user, password, database, charset, port, station_id):
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


if __name__ == "__main__":
    host = 'localhost'
    user = 'root'
    password = '123456'
    database = 'kuafu'
    charset = 'utf8'
    port = 3306

    station_id = 19663

    test_predict(host, user, password, database, charset, port)
    # accuracy = test_evaluate(host, user, password, database, charset, port, station_id)
    # print(accuracy)
    # test_figure_short(host, user, password, database, charset, port, station_id)

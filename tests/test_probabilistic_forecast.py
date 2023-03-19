
from task.mysql.interval_forecast import interval_learning
from task.mysql.predict import predict_short_power, predict_ultra_short_power
import pymysql
import pandas
import numpy
from matplotlib import pyplot


def test_predict(host, user, password, database, charset, port):
    # 短期预测
    for i in range(30):           # days
        start_time = '2022-04-%s 00:00' % str(i+1)
        predict_short_power(host, user, password, database, charset, port,
                            enable_interval_predict=True, start_time=start_time)

    # 超短期预测
    for i in range(30):                         # days
        for j in range(24):                     # hours
            for k in ['00', '15', '30', '45']:  # minutes
                start_time = '2022-04-%s %s:%s' % (str(i + 1), str(j), k)
                predict_ultra_short_power(host, user, password, database, charset, port,
                                          enable_interval_predict=True, start_time=start_time)


def test_evaluate(host, user, password, database, charset, port, station_id, predict_term):
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    c.execute("select * from predict_power_%s where  UNIX_TIMESTAMP(forecast_time) - UNIX_TIMESTAMP(start_time) < 86400;", station_id)
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_predict_power = pandas.DataFrame(record, columns=coul)

    c.execute("select time, power from real_power_%s;", station_id)
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_real_power = pandas.DataFrame(record, columns=coul)

    predict_power_short = dataframe_predict_power.loc[dataframe_predict_power['predict_term'] == predict_term]
    data_set = pandas.merge(predict_power_short, dataframe_real_power, left_on='forecast_time', right_on='time',
                            how='left')
    data_set = data_set.dropna(axis=0, how='any')

    predict_result = data_set.loc[:, 'predict_power'].values
    actual_result = numpy.maximum(data_set.loc[:, 'power'].values, 0)
    qujian = data_set.iloc[:, 6:16].values

    flag = numpy.zeros((5, 1))
    for i in range(predict_result.shape[0]):
        if actual_result[i] > qujian[i, 0] or actual_result[i] < qujian[i, 1]:
            flag[0] += 1
        if actual_result[i] > qujian[i, 2] or actual_result[i] < qujian[i, 3]:
            flag[1] += 1
        if actual_result[i] > qujian[i, 4] or actual_result[i] < qujian[i, 5]:
            flag[2] += 1
        if actual_result[i] > qujian[i, 6] or actual_result[i] < qujian[i, 7]:
            flag[3] += 1
        if actual_result[i] > qujian[i, 8] or actual_result[i] < qujian[i, 9]:
            flag[4] += 1
    flag = 1 - flag / predict_result.shape[0]

    wit = numpy.zeros((5, 1))
    for i in range(5):
        wit[i] = numpy.mean(qujian[:, 2 * i] - qujian[:, 2 * i + 1])
    wit = wit / 188

    ai = numpy.zeros((5, 1))
    for i in range(5):
        ai[i] = average_interval_score(qujian[:, 2 * i + 1], qujian[:, 2 * i], actual_result, 0.5 + 0.1 * i)
    return flag, wit, ai


def average_interval_score(lower, upper, actual_value, cl):
    score = 0
    for i in range(len(actual_value)):
        coverage = upper[i] - lower[i]
        if actual_value[i] < lower[i]:
            score += -2*cl*coverage - 4*(lower[i]-actual_value[i])
        if actual_value[i] > upper[i]:
            score += -2*cl*coverage - 4*(actual_value[i]-upper[i])
        else:
            score += -2*cl*coverage

    return score / len(actual_value)


def test_figure_short(host, user, password, database, charset, port, station_id):
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    # 读预测功率
    c.execute('select forecast_time, predict_power,'
              ' upper_bound_90, lower_bound_90, upper_bound_80, lower_bound_80, upper_bound_70,'
              ' lower_bound_70, upper_bound_60, lower_bound_60, upper_bound_50, lower_bound_50 from predict_power_%s'
              ' where predict_term = %s and  UNIX_TIMESTAMP(forecast_time) - UNIX_TIMESTAMP(start_time) < 86400'
              ' ORDER BY start_time asc, forecast_time asc;', (station_id, 'short'))
    db.commit()

    record = c.fetchall()
    predict_power = pandas.DataFrame(record, columns=['forecast_time',
                                                      'predict_power', 'upper_bound_90', 'lower_bound_90',
                                                      'upper_bound_80', 'lower_bound_80', 'upper_bound_70',
                                                      'lower_bound_70', 'upper_bound_60', 'lower_bound_60',
                                                      'upper_bound_50', 'lower_bound_50'])

    # 读真实功率
    c.execute('select time, power from real_power_%s where time between %s and %s ORDER BY time asc;', (station_id,
               predict_power.loc[:, 'forecast_time'].iloc[0], predict_power.loc[:, 'forecast_time'].iloc[-1]))
    db.commit()
    record = c.fetchall()
    real_power = pandas.DataFrame(record, columns=['time', 'power'])

    train_data_merge = pandas.merge(predict_power, real_power, left_on='forecast_time', right_on='time', how='left')

    power_cluster = train_data_merge.dropna(axis=0, how='any').values

    pyplot.plot(power_cluster[:, 1], label='real_power')
    n = 1
    pyplot.plot(power_cluster[:, 2 * n])
    pyplot.plot(power_cluster[:, 2 * n + 1])
    pyplot.show()
    pyplot.legend()


if __name__ == '__main__':
    host = 'localhost'
    user = 'root'
    password = '123456'
    database = 'kuafu'
    charset = 'utf8'
    port = 3306

    station_id_cluster = [51002, 52003]  # 峨蔓风电场, 神华光伏电站

    # 提取已完成对应场站的集成学习（2021-01-01 00:00——2022-04-01 00:00）

    # 区间预测训练
    for station_id in station_id_cluster:
        interval_learning(host, user, password, database, charset, port, station_id)

    # 功率预测+区间预测（2022-04-01 00:00—往后）
    test_predict(host, user, password, database, charset, port)

    # 评价
    # PICP(PI coverage probability) ：实际值落在预测区间内的比率（0~100%，越接近100% 指标性能越好）
    # PINAW(PI normalized averaged width) ：区间的狭窄程度（0~1，越接近0指标性能越好）
    # AIS(average interval score) ：区间分数（数值越大指标性能越好）

    for station_id in station_id_cluster:
        flag, wit, ai = test_evaluate(host, user, password, database, charset, port, station_id, predict_term='short')
        print(str(station_id) + '短期：' + '置信度: [90%, 80%, 70%, 60%, 50%]')
        print(str(station_id) + '短期：' + 'PICP: ' + str(flag.T))
        print(str(station_id) + '短期：' + 'PINAW: ' + str(wit.T))
        print(str(station_id) + '短期：' + 'AIS: ' + str(ai.T))

        flag, wit, ai = test_evaluate(host, user, password, database, charset, port,
                                      station_id, predict_term='ultra_short')
        print(str(station_id) + '超短期：' + '置信度: [90%, 80%, 70%, 60%, 50%]')
        print(str(station_id) + '超短期：' + 'PICP: ' + str(flag.T))
        print(str(station_id) + '超短期：' + 'PINAW: ' + str(wit.T))
        print(str(station_id) + '超短期：' + 'AIS: ' + str(ai.T))

    # for station_id in station_id_cluster:
    #     test_figure_short(host, user, password, database, charset, port, station_id)

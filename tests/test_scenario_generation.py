# _*_ coding: utf-8 _*_

from task.mysql.scenario_generation import model_train_sql, scenario_generation_sql
from matplotlib import pyplot
from task.mysql.predict import predict_short_power, predict_ultra_short_power
from task.mysql.model_evaluation import evaluate_for_station
import pymysql


def test_predict(host, user, password, database, charset, port):
    # 短期预测
    for i in range(30):           # days
        start_time = '2022-4-%s 00:00' % str(i+1)
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
    if len(record) == 0:  # 读数据库记录的时候，判断是否为空，如果是空要提前退出，2023/3/19
        return
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


if __name__ == '__main__':
    host = 'localhost'
    user = 'root'
    password = '123456'
    database = 'kuafu'
    charset = 'utf8'
    port = 3306

    station_id = 52003
    bin_num = 50

    predict_term = 'short'
    model_name = 'XGBoost_without_history_power'

    # 场景生成模型训练
    model_train_sql(host, user, password, database, charset, port, station_id, bin_num, predict_term, model_name)

    # 点预测
    test_predict(host, user, password, database, charset, port)

    # 场景预测结果
    dataframe, point_predict_data = scenario_generation_sql(host, user, password, database, charset, port,
                                                            station_id, bin_num, predict_term, model_name)

    # 点预测精度
    accuracy = test_evaluate(host, user, password, database, charset, port, station_id)
    print(accuracy)

    # 场景生成绘图

    for i in range(dataframe.shape[1]):
        pyplot.plot(dataframe.values[:, i], color='b')
    pyplot.plot(point_predict_data, LineWidth=2, color='r')
    pyplot.show()
    pyplot.legend()

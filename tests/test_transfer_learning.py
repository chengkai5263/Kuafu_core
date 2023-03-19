
import pymysql
import pandas
from task.mysql.transfer_learning import run_transfer_learning
from task.mysql.predict import predict_short_power, predict_ultra_short_power
from task.mysql.model_evaluation import evaluate_for_station
from common import data_preprocess


def test_transfer_learning_train(host, user, password, database, charset, port):
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    station_id_clusters = run_transfer_learning(host, user, password, database, charset, port)
    c.close()
    db.close()
    return station_id_clusters


def test_transfer_learning_predict(host, user, password, database, charset, port):
    # 短期预测
    for i in range(30):           # days
        start_time = '2022-04-%s 00:00' % str(i+1)
        predict_short_power(host, user, password, database, charset, port, start_time=start_time)

    # 超短期预测
    for i in range(10):                         # days
        for j in range(24):                     # hours
            for k in ['00', '15', '30', '45']:  # minutes
                start_time = '2022-04-%s %s:%s' % (str(i + 1), str(j), k)
                predict_ultra_short_power(host, user, password, database, charset, port, start_time=start_time)


def test_transfer_learning_evaluate(host, user, password, database, charset, port, station_id):
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    c.execute("select id,capacity,type from configure where id = %s;" %station_id)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)
    online_capacity = dataframe_config.loc[:, "capacity"][0]
    predict_type = dataframe_config.loc[:, "type"][0]

    c.execute("select time, power from real_power_" + str(station_id) + ";")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_real_power = pandas.DataFrame(record, columns=coul)
    actual_power_np = dataframe_real_power.values
    error_data_label = data_preprocess.DataPreprocess.data_preprocess_error_data_label(actual_power_np,
                                                                                       predict_type,
                                                                                       online_capacity)

    result = c.execute("select best_model from best_feature_parameters_and_model where"
                       " id = %s and predict_term = %s;", (station_id, 'short'))
    db.commit()
    if result == 1:
        record = c.fetchall()
        best_model = record[0][0]
    else:
        best_model = 'BPNN_without_history_power'
    short_best_model_and_state = best_model
    accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                    error_data_label=error_data_label, predict_term='short',
                                    host=host, user=user, password=password,
                                    database=database, charset=charset, port=port,
                                    model_name_and_state=short_best_model_and_state, scene='operation',
                                    sunrise_time='06:30:00', sunset_time='19:30:00',
                                    predict_type=predict_type, evaluation="capacity")
    print('短期最优模型' + best_model + '的国标预测精度是' + str(accuracy))
    accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                    error_data_label=error_data_label, predict_term='short',
                                    host=host, user=user, password=password,
                                    database=database, charset=charset, port=port,
                                    model_name_and_state=short_best_model_and_state, scene='operation',
                                    sunrise_time=None, sunset_time=None,
                                    predict_type=predict_type, evaluation=None)
    print('短期最优模型' + best_model + '的两个细则预测精度是' + str(accuracy))

    # 超短期评价
    result = c.execute("select best_model from best_feature_parameters_and_model where"
                       " id = %s and predict_term = %s;", (station_id, 'ultra_short'))
    db.commit()
    if result == 1:
        record = c.fetchall()
        best_model = record[0][0]
    else:
        best_model = 'BPNN_without_history_power'
    ultra_short_best_model_and_state = best_model
    accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                    error_data_label=error_data_label, predict_term='ultra_short',
                                    host=host, user=user, password=password,
                                    database=database, charset=charset, port=port,
                                    model_name_and_state=ultra_short_best_model_and_state, scene='operation',
                                    sunrise_time='06:30:00', sunset_time='19:30:00',
                                    predict_type=predict_type, evaluation="capacity")
    print('超短期最优模型' + best_model + '的国标预测精度是' + str(accuracy))
    accuracy = evaluate_for_station(id=station_id, online_capacity=online_capacity,
                                    error_data_label=error_data_label, predict_term='ultra_short',
                                    host=host, user=user, password=password,
                                    database=database, charset=charset, port=port,
                                    model_name_and_state=ultra_short_best_model_and_state, scene='operation',
                                    sunrise_time=None, sunset_time=None,
                                    predict_type=predict_type, evaluation=None)
    print('超短期最优模型' + best_model + '的两个细则预测精度是' + str(accuracy))

    c.close()
    db.close()


if __name__ == "__main__":
    host = 'localhost'
    user = 'root'
    password = '520linyulu'
    database = 'test0112'
    charset = 'utf8'
    port = 3306

    station_id_clusters = test_transfer_learning_train(host, user, password, database, charset, port)
    test_transfer_learning_predict(host, user, password, database, charset, port)
    for station_id in station_id_clusters:
        test_transfer_learning_evaluate(host, user, password, database, charset, port, station_id)

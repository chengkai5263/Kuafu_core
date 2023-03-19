import numpy as np
import pandas as pd
from app.long_term_predict.long_term_predict import LongTermPred
from common.logger import logs
import pymysql
from pymysql import cursors, connections
from common.tools import catch_exception


def connect_mysql(host, user, password, database, charset, port):
    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=password,
                                 database=database,
                                 charset=charset,
                                 port=port,
                                 cursorclass=cursors.DictCursor)
    return connection


def load_database_from_mysql(connection: connections.Connection):
    table_name_list = ['long_term_database_power', 'nwp_data_base_solar', 'nwp_data_base_wind',
                       'long_term_database_solar', 'long_term_database_wind']
    result = list()
    cursor = connection.cursor()
    for tn in table_name_list:
        sql = f"SELECT * FROM {tn}"
        cursor.execute(sql)
        result.append(pd.DataFrame(cursor.fetchall()))
    cursor.close()
    return result[0], result[1], result[2], result[3], result[4]


def save_result_to_mysql(predict_result: pd.DataFrame, connection: connections.Connection):
    table_name = "long_term_power_generation_prediction"
    cursor = connection.cursor()
    try:
        sql = f'''
                        CREATE TABLE IF NOT EXISTS {table_name}(
                        id INT,
                        area VARCHAR(255),
                        name VARCHAR(255),
                        type VARCHAR(255),
                        year INT,
                        month INT,
                        predict_power_generation FLOAT(10),
                        match_target VARCHAR(255),
                        pred_hours_of_utilization FLOAT(10))
                        '''
        cursor.execute(sql)
        start_id = cursor.execute(f"SELECT * FROM {table_name}") + 1
        predict_result['id'] = start_id + np.arange(predict_result.shape[0])
        predict_result['id'] = predict_result['id'].astype(int)
        des = cursor.description
        placeholder = "%s," * len(des)
        placeholder = placeholder.strip(',')
        use_cols = [des[i][0] for i in range(len(des))]
        des = ", ".join(str(item[0]) for item in cursor.description)
        predict_result = predict_result[use_cols]
        predict_result['match_target'].fillna(0, inplace=True)
        for i in range(predict_result.shape[0]):
            sql = f"INSERT INTO {table_name}({des}) VALUES({placeholder})"
            values = tuple(predict_result.values[i])
            cursor.execute(sql, values)
        connection.commit()
    except Exception as err:
        logs.error(err, exc_info=True)
        connection.rollback()


@catch_exception("long_term_predict error: ")
def long_term_predict(long_term_predict_model: LongTermPred, station_name: str, station_type: str,
                      station_capacity: float or list, area: str, predict_start_time: str,
                      feature_weight: dict = None, predict_length: int = 12):
    predict_result = long_term_predict_model.long_term_predict(station_name=station_name,
                                                               station_type=station_type,
                                                               station_capacity=station_capacity,
                                                               area=area,
                                                               predict_start_time=predict_start_time,
                                                               feature_weight=feature_weight,
                                                               predict_length=predict_length)
    predict_result.rename(columns={'pred_mix': 'predict_power_generation'}, inplace=True)
    return predict_result

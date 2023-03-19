#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :   Kuafu20230112
@File    :   test_mid_term_20230118.py
@Contact :   zhoucc@csg.cn
@License :   (C)Copyright 2022, Green-Energy-Team-DGRI-CSG

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/1/18 0:04   ZhouCC              

"""

from task.mysql.mid_term_prediction import MultiMidTermPred, mid_term_predict
import pandas as pd
import pymysql
from pymysql import Connection, cursors
from common.model_evaluation import evaluate_GB_T_40607_2021_withtimetag as evaluate_I
from common.model_evaluation import result_evaluation_Two_Detailed_Rules_without_time_tag as evaluate_II

host = 'localhost'
user = 'root'
password = '123456'
database = 'kuafu'
charset = 'utf8'
port = 3306


def test_predict():
    predict_start_time_list = ["2021-01-01 00:00:00"]
    predict_end_time_list = ["2021-02-01 00:00:00"]
    for i in range(len(predict_start_time_list)):
        predict_start_time = predict_start_time_list[i]
        predict_end_time = predict_end_time_list[i]
        predict_time_range = pd.date_range(start=predict_start_time, end=predict_end_time)
        for predict_time in predict_time_range:
            mid_term_predict(host=host, user=user, password=password, database=database, charset=charset,
                             port=port, predict_start_time=predict_time)


def test_mid_term_evaluate():
    station_id_list = [51002, 52003]
    station_cap_list = [99, 100]
    conn = pymysql.connect(host=host,
                           user=user,
                           password=password,
                           database=database,
                           charset=charset,
                           port=port,
                           cursorclass=pymysql.cursors.DictCursor)
    cursor = conn.cursor()
    df_accuracy = pd.DataFrame(columns=['id', 'GB_accuracy', 'Two_Detailed_Rules_accuracy'])
    merge_result_I = []
    merge_result_II = []
    for i in range(len(station_id_list)):
        station_id = station_id_list[i]
        station_cap = station_cap_list[i]
        pred_table_name = "predict_power_" + str(station_id)
        real_table_name = "real_power_" + str(station_id)

        sql = f"SELECT start_time, forecast_time, predict_power FROM {pred_table_name}"
        cursor.execute(sql)
        df_pred_power = pd.DataFrame(cursor.fetchall())

        sql = f"SELECT time, power FROM {real_table_name}"
        cursor.execute(sql)
        df_real_power = pd.DataFrame(cursor.fetchall())

        merge_result_I.append(merge_power(df_pred_power, df_real_power, mode='GB'))
        merge_result_II.append(merge_power(df_pred_power, df_real_power, mode='Two_Detailed_Rules'))

        df_accuracy.loc[i, 'id'] = station_id
        df_accuracy.loc[i, 'GB_accuracy'] = evaluate_I(predict_power=df_pred_power.values,
                                                       actual_power=df_real_power.values,
                                                       online_capacity=station_cap,
                                                       predict_term='medium')
        df_accuracy.loc[i, 'Two_Detailed_Rules_accuracy'] = evaluate_II(predict_power=df_pred_power.values,
                                                                        actual_power=df_real_power.values,
                                                                        online_capacity=station_cap,
                                                                        predict_term='medium')
    cursor.close()
    conn.close()
    return df_accuracy, merge_result_I, merge_result_II


def merge_power(df_pred_power, df_real_power, mode='GB'):
    df_real_pred_power = pd.merge(left=df_pred_power, right=df_real_power, left_on='forecast_time', right_on='time',
                                  how='inner')
    df_merge_power = df_real_pred_power[['start_time', 'forecast_time', 'predict_power', 'power']]
    df_merge_power.sort_values(by=['start_time', 'forecast_time'], inplace=True)
    df_merge_power.rename(columns={'power': 'real_power'}, inplace=True)
    df_result = pd.DataFrame(columns=['forecast_time', 'predict_power', 'real_power'])
    start_time_list = df_merge_power['start_time'].unique()
    for i in range(len(start_time_list)):
        df_temp = df_merge_power.loc[
            df_merge_power['start_time'] == start_time_list[i], ['forecast_time', 'predict_power', 'real_power']]
        if mode == 'GB':
            df_result = df_result.append(df_temp.tail(96), ignore_index=True)
        else:
            df_result = df_result.append(df_temp.head(4*96).tail(96))
    return df_result


def start_test():
    test_predict()
    res = test_mid_term_evaluate()
    return res


if __name__ == "__main__":
    # test_predict()
    res = test_mid_term_evaluate()

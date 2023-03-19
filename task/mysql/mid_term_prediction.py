from app.mid_term_predict_model.mid_term_prediction import MidTermPred, generate_start_predict_time, \
    prepare_nwp_data, prepare_power_data
from pymysql import cursors, connections
import pymysql
import pandas as pd
import numpy as np
from common.tools import catch_exception


@catch_exception("mid_term_predict error: ")
def mid_term_predict(host, user, password, database, charset, port, predict_start_time, limit_days=10):
    total_predict_length = 960
    model = MultiMidTermPred()
    model.set_mysql_info(host=host,
                         user=user,
                         password=password,
                         database=database,
                         charset=charset,
                         port=port,
                         total_predict_length=total_predict_length,
                         limit_days=limit_days,
                         predict_start_time=predict_start_time,
                         )
    model.predict()


class MultiMidTermPred:
    def __init__(self):
        self.station_classes = list()
        self.host = None
        self.user = None
        self.password = None
        self.database = None
        self.charset = None
        self.port = None
        self.feature_for_predict = None
        self.power_now = None
        self.total_predict_length = None
        self.limit_days = None
        self.predict_start_time = None

    def set_mysql_info(self,
                       host='172.16.129.89',
                       user='KuafuCore',
                       password='KuaCore#220520',
                       database='kuafu_zhoucc',
                       charset='utf8',
                       port=15000,
                       total_predict_length=960,
                       limit_days=800,
                       predict_start_time=None,
                       ):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.port = port
        self.total_predict_length = total_predict_length
        self.limit_days = limit_days
        self.predict_start_time = predict_start_time

    def predict(self):
        connection = self.__connect_mysql()
        with connection:
            self._load_config_from_mysql(connection)
            self._load_data_for_prediction_from_mysql(connection)
            predict_result_dict = dict()
            for i in range(len(self.station_classes)):
                predict_result = self.station_classes[i].predict()
                self.prediction_result_to_mysql(i, connection, False)
                predict_result_dict[self.station_classes[i].name] = predict_result
            return predict_result_dict

    def _load_config_from_mysql(self, connection: connections.Connection):
        with connection.cursor() as cursor:
            sql = "SELECT * FROM configure where station_status = 2"
            cursor.execute(sql)
            conf = pd.DataFrame(cursor.fetchall())
        predict_start_time = self.predict_start_time
        for i in range(conf.shape[0]):
            station_data = MidTermPred()
            fitted_nwp_based_model_name = None
            with connection.cursor() as cursor:
                station_type = conf.loc[i, 'type']
                if station_type == 1:
                    station_type = 'wind'
                if station_type == 2:
                    station_type = 'solar'
                type_term = 'short_' + station_type
                sql = f"SELECT * FROM default_feature_and_parameters WHERE term_type=%s"
                cursor.execute(sql, type_term)
                df_default_feature_param = pd.DataFrame(cursor.fetchall())
                short_usecols = df_default_feature_param.loc[0, 'usecols']

                sql = f"SELECT * FROM best_feature_parameters_and_model WHERE id=%s AND predict_term='short'"
                cursor.execute(sql, conf.loc[i, 'id'])
                df_best_param_model = pd.DataFrame(cursor.fetchall())
                if not df_best_param_model.empty:
                    best_model = df_best_param_model.loc[0, 'best_model']
                    if isinstance(best_model, str):
                        fitted_nwp_based_model_name = best_model

            station_data.set_configure(id=conf.loc[i, 'id'], name=conf.loc[i, 'name'],
                                       capacity=conf.loc[i, 'capacity'],
                                       type=conf.loc[i, 'type'],
                                       sr_col=conf.loc[i, 'sr_col'],
                                       usecols=str(short_usecols),
                                       model_savepath=conf.loc[i, 'model_savepath'],
                                       fitted_nwp_based_model_name=fitted_nwp_based_model_name,
                                       predict_start_time=predict_start_time,
                                       total_predict_length=self.total_predict_length,
                                       bin_num=conf.loc[i, 'bin_number'],
                                       )
            self.station_classes.append(station_data)

    def _load_data_for_prediction_from_mysql(self, connection: connections.Connection):
        predict_start_time = self.predict_start_time
        limit_days = self.limit_days
        for i in range(len(self.station_classes)):
            forecast_time = generate_start_predict_time(predict_start_time)
            limit_rows = limit_days * 4 * 7 * 96
            cursor = connection.cursor()
            # 如果需要用nwp预测，才需要读nwp数据，如果前置环节判断模型或特征pkl文件不存在的话，直接就不花时间读取nwp数据了
            if self.station_classes[i].use_nwp_based_model:
                select_cols = ['start_time', 'forecast_time'] + self.station_classes[i].use_cols
                select_cols = ", ".join(select_cols)
                nwp_table_name = "nwp_" + str(self.station_classes[i].id)
                nwp_sql = f"SELECT DISTINCT {select_cols} " \
                          f"FROM {nwp_table_name} " \
                          f"WHERE (start_time <= %s AND forecast_time >= %s) " \
                          f"LIMIT %s "
                cursor.execute(nwp_sql, (predict_start_time, forecast_time, limit_rows))
                df_feature = pd.DataFrame(cursor.fetchall())
                if self.station_classes[i].type == 'solar':
                    use_cols = self.station_classes[i].use_cols + ['timelabel']
                else:
                    use_cols = self.station_classes[i].use_cols
                df_feature = prepare_nwp_data(df_data=df_feature,
                                              predict_start_time=forecast_time,
                                              generation_type=self.station_classes[i].type)
                if not df_feature.empty:
                    self.station_classes[i].nwp_for_predict = df_feature.loc[
                                                              :, use_cols].values[
                                                              :self.station_classes[i].max_nwp_length]
                else:
                    self.station_classes[i].nwp_for_predict = None

            # 不管用什么预测，历史功率肯定是要读取的
            power_select_cols = "time, power"
            power_table_name = "real_power_" + str(self.station_classes[i].id)
            power_sql = f"SELECT {power_select_cols} FROM {power_table_name} WHERE time < %s"
            cursor.execute(power_sql, forecast_time)
            df_target = pd.DataFrame(cursor.fetchall())
            cursor.close()
            df_target = prepare_power_data(df_data=df_target, predict_start_time=forecast_time)

            self.station_classes[i].h_power_for_prediction = df_target.iloc[
                                                             -self.station_classes[
                                                                 i].max_historical_power_length_for_prediction:, ]

    def prediction_result_to_mysql(self, i: int, connection: connections.Connection, for_test=False):
        if for_test:
            table_name = "predict_power_" + str(self.station_classes[i].id) + "_train"
        else:
            table_name = "predict_power_" + str(self.station_classes[i].id)
        cursor = connection.cursor()
        try:
            sql = f'''
            CREATE TABLE IF NOT EXISTS {table_name}(
            id BIGINT,
            predict_term VARCHAR(255),
            model_name VARCHAR(255),
            start_time DATETIME,
            forecast_time DATETIME,
            predict_power FLOAT(10),
            upper_bound_90 FLOAT(10),
            lower_bound_90 FLOAT(10),
            upper_bound_80 FLOAT(10),
            lower_bound_80 FLOAT(10),
            upper_bound_70 FLOAT(10),
            lower_bound_70 FLOAT(10),
            upper_bound_60 FLOAT(10),
            lower_bound_60 FLOAT(10),
            upper_bound_50 FLOAT(10),
            lower_bound_50 FLOAT(10))
            '''
            cursor.execute(sql)
            start_id = cursor.execute(f"SELECT * FROM {table_name}") + 1
            self.station_classes[i].prediction_result['id'] = start_id + np.arange(
                self.station_classes[i].total_predict_length)
            self.station_classes[i].prediction_result['id'] = self.station_classes[i].prediction_result['id'].astype(
                int)
            des = cursor.description[0:6]
            placeholder = "%s," * len(des)
            placeholder = placeholder.strip(',')
            des = ", ".join(str(item[0]) for item in des)
            for j in range(self.station_classes[i].prediction_result.shape[0]):
                sql = f"INSERT INTO {table_name}({des}) VALUES ({placeholder})"
                values = tuple(self.station_classes[i].prediction_result.values[j, 0:6])
                cursor.execute(sql, values)
            connection.commit()
            cursor.close()
        except Exception as err:
            print(err)
            connection.rollback()

    def __connect_mysql(self):
        connection = pymysql.connect(host=self.host,
                                     user=self.user,
                                     password=self.password,
                                     database=self.database,
                                     charset=self.charset,
                                     port=self.port,
                                     cursorclass=pymysql.cursors.DictCursor)
        return connection

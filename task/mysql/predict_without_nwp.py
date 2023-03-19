
import pandas
import datetime
import pymysql
from pymysql import cursors
from common.logger import logs
from app.predict_without_nwp.PredictWithoutNwp import PredictWithoutNwp
from common.tools import catch_exception


class MainPredictWithoutNwp:
    def __init__(self, host, user, password, database, charset, port):
        self.conn = self.__connect_mysql(host, user, password, database, charset, port)
        self.config = self.__set_config()
        self.predict_result = dict()

    @staticmethod
    def __connect_mysql(host, user, password, database, charset, port):
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     database=database,
                                     charset=charset,
                                     port=port,
                                     cursorclass=cursors.DictCursor)
        return connection

    def __set_config(self):
        cursor = self.conn.cursor()
        sql = f"SELECT * FROM configure"
        cursor.execute(sql)
        cursor.close()
        return pandas.DataFrame(cursor.fetchall())

    def predict(self, predict_term, predict_start_time: str):
        if predict_start_time is None:
            predict_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if predict_term == 'short':
            predict_length = 288
        else:
            predict_length = 16
        columns = ['id', 'name', 'type', 'capacity', 'model_savepath', 'bin_number', 'predict_pattern']
        for i in range(self.config.shape[0]):
            try:
                station_config = dict()
                for key in columns:
                    station_config[key] = self.config.loc[i, key]
                model = PredictWithoutNwp(station_name=station_config['name'],
                                          station_id=station_config['id'],
                                          station_type=station_config['type'],
                                          station_capacity=float(station_config['capacity']),
                                          model_save_path=station_config['model_savepath'],
                                          bin_num=int(station_config['bin_number']),
                                          # connection=self.conn,
                                          predict_start_time=predict_start_time,
                                          predict_pattern=station_config['predict_pattern']
                                          )

                table_name = "real_power_" + str(model.station_id)
                cursor = self.conn.cursor()
                sql = f"SELECT time, power FROM {table_name} WHERE (time <= %s) LIMIT %s"
                cursor.execute(sql, (model.predict_start_time.strftime("%Y-%m-%d %H:%M:%S"), 3000))
                cursor.close()

                model.clean_power_data(cursor.fetchall())
                model.predict(predict_length=predict_length)
                self.predict_result[station_config['id']] = model.predict_result
            except Exception as err:
                logs.error(err, exc_info=True)


@catch_exception("predict_without_nwp error: ")
def predict_without_nwp(station_name, station_id, station_type, station_capacity,
                        model_save_path, bin_num, connection, predict_term,
                        predict_start_time=None, predict_pattern='super-fast'):
    """
    备胎计划
    :param station_name:
    :param station_id:
    :param station_type:
    :param station_capacity:
    :param model_save_path:
    :param bin_num:
    :param connection:
    :param predict_term:
    :param predict_start_time:
    :param predict_pattern:
    :return: None
    """
    if predict_start_time is None:
        predict_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    model = PredictWithoutNwp(station_name=station_name, station_id=station_id, station_type=station_type,
                              station_capacity=station_capacity, model_save_path=model_save_path, bin_num=bin_num,
                              predict_start_time=predict_start_time,
                              predict_pattern=predict_pattern)

    table_name = "real_power_" + str(model.station_id)
    cursor = connection.cursor()
    sql = f"SELECT time, power FROM {table_name} WHERE (time <= %s) ORDER BY time desc LIMIT %s;"
    cursor.execute(sql, (model.predict_start_time.strftime("%Y-%m-%d %H:%M:%S"), 3000))
    cursor.close()

    model.clean_power_data(cursor.fetchall())
    if predict_term == 'short':
        predict_length = 288
    else:
        predict_length = 16
    model.predict(predict_length=predict_length)

    return model.predict_result.values[:, -1]

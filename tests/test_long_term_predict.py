
import numpy
import pandas
from app.long_term_predict.long_term_predict import LongTermPred
from common.logger import logs
from task.mysql.long_term_predict import connect_mysql, load_database_from_mysql, save_result_to_mysql, \
    long_term_predict


def test():
    host = 'localhost'
    user = 'root'
    password = '123456'
    database = 'kuafu'
    charset = 'utf8'
    port = 3306

    predict_start_time = '2021-1-10 09:00:00'

    connection = connect_mysql(host, user, password, database, charset, port)

    sql = "SELECT * FROM long_term_database_power"
    cursor = connection.cursor()
    cursor.execute(sql)
    test_data = pandas.DataFrame(cursor.fetchall())

    df_real_power_generation = test_data[['name', 'year', 'month', 'power_generation', 'hours_of_utilization']]

    sql = "SELECT name, capacity, area FROM long_term_database_solar"
    cursor.execute(sql)
    df_solar_stations = pandas.DataFrame(cursor.fetchall())
    df_solar_stations['type'] = "光伏"

    sql = "SELECT name, capacity, area FROM long_term_database_wind"
    cursor.execute(sql)
    df_wind_stations = pandas.DataFrame(cursor.fetchall())
    df_wind_stations['type'] = "风电"

    df_all_stations = pandas.concat([df_solar_stations, df_wind_stations]).drop_duplicates(ignore_index=True)
    cursor.close()

    df_result = pandas.DataFrame()
    long_term_predict_model = LongTermPred()
    database = load_database_from_mysql(connection)
    long_term_predict_model.load_database(database[0], database[1], database[2], database[3], database[4])
    for i in range(df_all_stations.shape[0]):
        try:
            station_name = df_all_stations.loc[i, 'name']
            station_type = df_all_stations.loc[i, 'type']
            station_capacity = df_all_stations.loc[i, 'capacity']
            area = df_all_stations.loc[i, 'area']
            res = long_term_predict(long_term_predict_model, station_name, station_type, station_capacity, area,
                                    predict_start_time)
            res_real = df_real_power_generation[df_real_power_generation['name'] == station_name]
            res = res.merge(res_real, on=['name', 'year', 'month'], how='left')
            res['error'] = (
                    (res['predict_power_generation'] - res['power_generation']) / res['power_generation'] * 100).round(
                2)
            res = res.replace([numpy.inf, -numpy.inf], numpy.nan)
            if df_result.empty:
                df_result = res
            else:
                df_result = pandas.concat([df_result, res], ignore_index=True)
        except Exception as err:
            logs.error(err, exc_info=True)
            print(i)
    save_result_to_mysql(predict_result=df_result, connection=connection)
    connection.close()
    return df_result


if __name__ == "__main__":
    res = test()

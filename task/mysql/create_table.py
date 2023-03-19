
import numpy
import time
import pandas
import datetime
import pymysql

from common import predict
from common.tools import catch_exception
from task.mysql.interval_forecast import IntervalForecastTask
from task.mysql import predict_without_nwp
from task.mysql.load_test_data_iterate import LoadTestdataiterate
from common.tools import load_model
from common.logger import logs


@catch_exception("初始建表 出错: ")
def create_table(host, user, password, database, charset, port):
    """
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :return:
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    result = c.execute("select DISTINCT id from configure;")
    db.commit()
    record_id = c.fetchall()
    for i in range(result):
        station_id = record_id[i][0]
        # 建predict_power_id
        result_predict_power = c.execute("select 1 from information_schema.tables where table_schema= 'kuafu'"
                                         " and table_name = 'predict_power_%s';", station_id)
        db.commit()
        if result_predict_power == 0:
            logs.warning("%s%s%s%s" % (str(station_id), "场站缺少predict_power_", str(station_id), "表，将被创建"))
            try:
                c.execute("CREATE TABLE IF NOT EXISTS `predict_power_%s` ("
                          "`id` bigint NOT NULL AUTO_INCREMENT,"
                          "`predict_term` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,"
                          "`model_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,"
                          "`start_time` datetime NOT NULL,"
                          "`forecast_time` datetime NOT NULL,"
                          "`predict_power` float(10,4) NOT NULL,"
                          "`upper_bound_90` float(10,4) DEFAULT NULL,"
                          "`lower_bound_90` float(10,4) DEFAULT NULL,"
                          "`upper_bound_80` float(10,4) DEFAULT NULL,"
                          "`lower_bound_80` float(10,4) DEFAULT NULL,"
                          "`upper_bound_70` float(10,4) DEFAULT NULL,"
                          "`lower_bound_70` float(10,4) DEFAULT NULL,"
                          "`upper_bound_60` float(10,4) DEFAULT NULL,"
                          "`lower_bound_60` float(10,4) DEFAULT NULL,"
                          "`upper_bound_50` float(10,4) DEFAULT NULL,"
                          "`lower_bound_50` float(10,4) DEFAULT NULL,"
                          "PRIMARY KEY (`id`) USING BTREE);", station_id)
                db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                logs.warning("%s%s%s%s" % (str(station_id), "表predict_power_", str(station_id), "创建失败！！！"))
                continue
    c.close()
    db.close()

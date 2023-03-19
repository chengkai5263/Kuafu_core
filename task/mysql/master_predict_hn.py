#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :   Kuafu_20230119
@File    :   master_predict_hn.py
@Contact :   zhoucc@csg.cn
@License :   (C)Copyright 2022, Green-Energy-Team-DGRI-CSG

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2023/2/10 12:36   ZhouCC              
本文件主要实现面向海南新能源功率预测主站的短期预测、超短期预测、中期预测、概率预测等功能
"""

import datetime
import time
import numpy
import pymysql
import pandas as pd

from pymysql import cursors
from apscheduler.schedulers.background import BackgroundScheduler
from app.mid_term_predict_model.mid_term_prediction import MidTermPred
from app.mid_term_predict_model.mid_term_prediction import generate_start_predict_time
from app.mid_term_predict_model.mid_term_prediction import prepare_nwp_data

from task.mysql.interval_forecast import IntervalForecastTask
from task.mysql import predict_without_nwp
from common.tools import load_model
from common.logger import logs
from common import predict


# 这是一个python操作mysql数据库增删查改、查询连接是否存在的类，作为数据库操作的父类
class MySQL:
    def __init__(self, host, user, password, database, charset, port, autocommit=True):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.port = port
        self.autocommit = autocommit
        self.connection = self.__connect()

    def __connect(self):
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            port=self.port,
            autocommit=self.autocommit,
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection

    def check_connection(self):
        if not self.connection.ping():
            self.__connect()

    def execute(self, query, args=None):
        self.check_connection()
        with self.connection.cursor() as cursor:
            cursor.execute(query, args)
            return cursor

    def select(self, query, args=None):
        cursor = self.execute(query, args)
        return cursor.fetchall()

    def insert(self, query, args=None):
        cursor = self.execute(query, args)
        return cursor.lastrowid

    def update(self, query, args=None):
        cursor = self.execute(query, args)
        return cursor.rowcount

    def delete(self, query, args=None):
        return self.update(query, args)

    def close(self):
        self.connection.close()


# 将海南主站功率预测系统抽象成一个类，
# 包括以下属性：
# (1)所要预测的场站集合，这是一个list，形如[plant1, plant2, ..., plantn]，每一个元素是一个新能源场站预测的类对象
# (2)数据库的信息
# (3)场站的最优模型、次优模型信息

# 包括以下行为：
# （1）根据数据库的configure表格实例化所有场站的类
# （2）批量开始中期、短期、超短期预测任务

class MultiPlantPredict(MySQL):
    def __init__(self, host, user, password, database, charset, port, use_cols_of_conf,
                 conf_table='configure', predict_tag=0, model_table='best_feature_parameters_and_model'):
        """
        初始化所需参数，在主函数中调用时应均从配置项读取
        :param host:
        :param user:
        :param password:
        :param database:
        :param charset:
        :param port:
        :param use_cols_of_conf:
        :param conf_table:
        :param predict_tag:
        :param model_table:
        """
        super().__init__(host, user, password, database, charset, port)
        self.plants_for_prediction = None
        self.best_feature_parameters_and_model = None
        self.use_cols_of_conf = eval(use_cols_of_conf)
        self.conf_table = conf_table
        self.predict_tag = predict_tag
        self.model_table = model_table
        self.conf = self.get_configure()

    def get_configure(self):
        query = "SELECT DISTINCT * FROM {} WHERE station_status > %s".format(self.conf_table)
        conf = pd.DataFrame(self.select(query, self.predict_tag))
        conf = conf[self.use_cols_of_conf]

        query = "SELECT DISTINCT id, predict_term, best_model, second_model FROM {}".format(self.model_table)
        df_model = pd.DataFrame(self.select(query))

        if df_model.empty:
            conf[['short_best_model', 'short_second_model',
                  'ultra_short_best_model', 'ultra_short_second_model']] = None
            logs.info("{}无数据！".format(self.model_table))

        else:
            df_pivot = df_model.pivot(index='id', columns='predict_term', values=['best_model', 'second_model'])
            df_pivot.columns = ['_'.join([col2, col1]) for col1, col2 in df_pivot.columns]
            df_pivot.reset_index(inplace=True)
            conf = pd.merge(conf, df_pivot, on='id', how='left')
        return conf

    def init_plants_for_prediction(self):
        """
        :return:
        """
        self.plants_for_prediction = list()
        conf = self.conf
        if not conf.empty:
            for i in range(conf.shape[0]):
                try:
                    plant = SinglePlantPredict(host=self.host,
                                               user=self.user,
                                               password=self.password,
                                               database=self.database,
                                               charset=self.charset,
                                               port=self.port,
                                               autocommit=self.autocommit,
                                               **conf.loc[i].to_dict()
                                               )
                    self.plants_for_prediction.append(plant)
                except Exception as err:
                    logs.warning("场站{}({})初始化失败，自动跳过该场站，开始加载下一个场站的预测服务！".format(id, type))
                    logs.error(err, exc_info=True)
            self.close()

    def short_term_predict(self, predict_start_time, enable_interval_predict, scheduler: BackgroundScheduler):
        if predict_start_time is None:
            start_time = '23:45:00'
            date_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
            predict_start_time = date_now + ' ' + start_time
        for plant in self.plants_for_prediction:
            scheduler.add_job(plant.predict_interval_and_sql,
                              args=[predict_start_time, enable_interval_predict, 'ultra_short'],
                              coalesce=True, misfire_grace_time=None)

    def ultra_short_term_predict(self, predict_start_time, enable_interval_predict, scheduler: BackgroundScheduler):
        if predict_start_time is None:
            now_time = datetime.datetime.now().timestamp()
            d = now_time % 900
            predict_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time - d))
        for plant in self.plants_for_prediction:
            scheduler.add_job(plant.predict_interval_and_sql,
                              args=[predict_start_time, enable_interval_predict, 'short'],
                              coalesce=True, misfire_grace_time=None)

    def medium_term_predict(self, predict_start_time, enable_interval_predict, scheduler: BackgroundScheduler):
        if predict_start_time is None:
            start_time = '23:45:00'
            date_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
            predict_start_time = date_now + ' ' + start_time
        for plant in self.plants_for_prediction:
            scheduler.add_job(plant.predict_interval_and_sql,
                              args=[predict_start_time, enable_interval_predict, 'medium'],
                              coalesce=True, misfire_grace_time=None)


# 将每一个新能源场站的预测抽象成一个类，它包括以下属性：
# （1）数据库的信息：host, user, password, database, charset, port；
# （2）中/短期预测模型的本地文件路径；
# （3）超短期预测模型的本地文件路径；
# （4）场站信息：id、名称、类型；

class SinglePlantPredict(MySQL):
    def __init__(self, host, user, password, database, charset, port, autocommit, **kwargs):
        """
        初始化函数
        """
        super().__init__(host, user, password, database, charset, port, autocommit)

        self.plant_id = kwargs['id']
        self.plant_name = kwargs['name']
        self.plant_capacity = kwargs['capacity']
        if isinstance(kwargs['type'], int):
            if kwargs['type'] == 1:
                self.plant_type = 'wind'
            else:
                self.plant_type = 'solar'
        else:
            self.plant_type = kwargs['type']

        self.model_savepath = kwargs['model_savepath']
        self.short_best_model_name = kwargs['short_best_model']
        self.short_second_model_name = kwargs['short_second_model']
        self.ultra_short_best_model_name = kwargs['ultra_short_best_model']
        self.ultra_short_second_model_name = kwargs['ultra_short_second_model']
        self.sr_col = kwargs['sr_col']
        self.bin_number = kwargs['bin_number']
        self.station_status = kwargs['station_status']
        self.source_capacity_for_transfer = kwargs['source_capacity_for_transfer']
        self.conn = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        self.predict_pattern = kwargs['predict_pattern']

    def _load_nwp_for_short_predict(self, start_time, short_nwp_feature):
        """
        从数据库中加载短期预测的数据并校验
        :return: 如果数据长度符合要求，返回相应的np.array，否则返回None
        """
        forecast_time_datetime1 = datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M",
                                                                           time.localtime(start_time + 900)),
                                                             '%Y/%m/%d %H:%M')
        forecast_time_datetime2 = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time + 900 * 288)), '%Y/%m/%d %H:%M')

        query = "SELECT t.* FROM (select start_time, forecast_time, " + \
                ','.join(iterm for iterm in short_nwp_feature) + " from nwp_%s" \
                " where forecast_time BETWEEN %s and %s ORDER BY start_time desc LIMIT 18816) t" \
                " group by t.forecast_time ORDER BY t.forecast_time;"
        args = (self.plant_id, forecast_time_datetime1, forecast_time_datetime2)
        record = pd.DataFrame(self.select(query, args))

        nwp = numpy.zeros((0, len(short_nwp_feature)))
        timeslist = numpy.zeros((0, 1))
        predict_time = 1 * start_time
        for j in range(len(record)):
            predict_time += 900
            a = ','.join(iterm for iterm in record.iloc[j, 2:].tolist())
            b = eval(a)
            tem = numpy.array(b).reshape(1, -1)
            nwp = numpy.vstack((nwp, tem))
            t = (predict_time % 86400) / 86400
            timeslist = numpy.vstack((timeslist, t))
        if self.plant_type == 'solar':
            nwp = numpy.hstack((nwp, timeslist))
        return nwp

    def _load_nwp_for_ultra_short_predict(self, start_time, ultra_short_nwp_feature):
        """
        从数据库中加载超短期预测的数据并校验
        :param conn: mysql的数据库连接
        :return: 如果数据长度符合要求，返回相应的np.array，否则返回None
        """
        forecast_time_start = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time + 900)),
            '%Y/%m/%d %H:%M')
        forecast_time_end = datetime.datetime.strptime(
            time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time + 900 * 16)), '%Y/%m/%d %H:%M')

        query = "SELECT t.* FROM (select start_time, forecast_time, " + \
                ','.join(iterm for iterm in ultra_short_nwp_feature) + " from nwp_%s" \
                " where forecast_time BETWEEN %s and %s ORDER BY start_time desc LIMIT 18816) t"\
                " group by t.forecast_time ORDER BY t.forecast_time;"
        args = (self.plant_id, forecast_time_start, forecast_time_end)
        record = pd.DataFrame(self.select(query, args))

        nwp = numpy.zeros((0, len(ultra_short_nwp_feature)))
        timeslist = numpy.zeros((0, 1))
        predict_time = 1 * start_time
        for j in range(len(record)):
            predict_time += 900
            a = ','.join(iterm for iterm in record.iloc[j, 2:].tolist())
            b = eval(a)
            tem = numpy.array(b).reshape(1, -1)
            nwp = numpy.vstack((nwp, tem))
            t = (predict_time % 86400) / 86400
            timeslist = numpy.vstack((timeslist, t))
        if self.plant_type == 'solar':
            nwp = numpy.hstack((nwp, timeslist))

        start_time_datetime = datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time)),
                                                         '%Y/%m/%d %H:%M')
        query = "select DISTINCT power from real_power_%s where time = %s ORDER BY time asc;"
        args = (self.plant_id, start_time_datetime)
        record = pd.DataFrame(self.select(query, args))
        if len(record) == 0:
            history_power = None
        else:
            history_power = record.iloc[0, 0]
        return nwp, history_power

    def _load_nwp_for_medium_predict(self, predict_start_time, limit_days, use_cols,
                                     plant_id, plant_type, max_nwp_length):
        """
        从数据库中加载中期预测的数据并校验
        :return: 如果数据长度符合要求，返回相应的np.array，否则返回None
        """

        # 设置开始预测的时间至整15分钟
        forecast_time = generate_start_predict_time(predict_start_time)
        limit_rows = limit_days * 4 * 7 * 96
        select_cols = ['start_time', 'forecast_time'] + use_cols
        select_cols = ", ".join(select_cols)
        nwp_table_name = "nwp_" + str(plant_id)
        nwp_sql = f"SELECT DISTINCT {select_cols} " \
                  f"FROM {nwp_table_name} " \
                  f"WHERE (start_time <= %s AND forecast_time >= %s) " \
                  f"LIMIT %s "
        df_feature = pd.DataFrame(self.select(nwp_sql, (predict_start_time, forecast_time, limit_rows)))
        if plant_type == 'solar':
            use_cols = use_cols + ['timelabel']
        df_feature = prepare_nwp_data(df_data=df_feature,
                                      predict_start_time=forecast_time,
                                      generation_type=plant_type)
        if not df_feature.empty:
            nwp_for_predict = df_feature.loc[:, use_cols].values[:max_nwp_length]
        else:
            nwp_for_predict = None

        return nwp_for_predict

    def _load_real_power_for_medium_predict(self, plant_id, predict_start_time,
                                            max_historical_power_length_for_prediction):
        power_select_cols = "time, power"
        power_table_name = "real_power_" + str(plant_id)
        power_sql = f"SELECT {power_select_cols} FROM {power_table_name} WHERE time < %s"
        df_power = pd.DataFrame(self.select(power_sql, predict_start_time))
        h_power_for_prediction = df_power.iloc[max_historical_power_length_for_prediction:, ]
        return h_power_for_prediction

    def short_term_predict(self, start_time):
        """
        先加载NWP数据，如果NWP数据是合格的，那么就再来加载最优模型
        try：最优模型是否可以成果获得预测结果
        except：
            try：次优模型是否可以获得预测结果
            except：转备胎
        判断一下是不是可以用模型预测，如果不行，用备胎预测
        预测结果直接写回数据库
        如果有异常，写到对应log里面：
        log.info：用最优、次优模型预测的
        log.warn：用备胎预测的
        log.error：没预测出来
        把station_status这个参数打印到日志里，logs.warn只打迁移学习的
        :return:
        """
        if self.station_status == 1:
            logs.warning(self.plant_name + ' | ' + start_time + ' | 短期预测 | 采用的迁移模型！')
        if isinstance(start_time, str):
            start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp() + 900

        # 如果是迁移学习，rate_for_transfer为实际容量除以源场站容量；如果不为迁移学习，rate_for_transfer为1
        rate_for_transfer = 1
        if self.station_status == 1:
            rate_for_transfer = self.plant_capacity / self.source_capacity_for_transfer

        # 从MySQL读取测试数据
        ann_model_name, b = self.short_best_model_name.split('_', 1)

        try:
            usecols = load_model(self.model_savepath + str(self.plant_id) + '/short/' + ann_model_name + '/' + 'short_usecols.pkl')
            test_feature = self._load_nwp_for_short_predict(start_time=start_time, short_nwp_feature=usecols)
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.info(self.plant_name + ' short, 读NWP时失败')
            test_feature = numpy.array([])

        forecast_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(start_time)))

        predict_result_without_nwp = predict_without_nwp.predict_without_nwp(
            station_name=self.plant_name, station_id=self.plant_id, station_type=self.plant_type,
            station_capacity=self.plant_capacity, model_save_path=self.model_savepath, bin_num=self.bin_number,
            connection=self.conn, predict_term='short', predict_start_time=forecast_time,
            predict_pattern=self.predict_pattern)

        predict_result, sr, model_used = predict.predict_short_power(
            station_name=self.plant_name, station_id=self.plant_id,
            model_path=self.model_savepath,
            predict_type=self.plant_type, sr_col=self.sr_col,
            online_capacity=self.plant_capacity,
            capacity=self.plant_capacity, forecast_time=forecast_time,
            best_model=self.short_best_model_name, second_model=self.short_second_model_name,
            predict_result_without_nwp=predict_result_without_nwp,
            test_feature=test_feature, rate_for_transfer=rate_for_transfer)
        return predict_result, sr, model_used

    def ultra_short_term_predict(self, start_time):
        """
        先加载模型，
        再加载数据，
        判断一下是不是可以用模型预测，如果不行，用备胎预测
        预测结果直接写回数据库
        :return:
        """
        if self.station_status == 1:
            logs.warning(self.plant_name + ' | ' + start_time + ' | 超短期预测 | 采用的迁移模型！')
        if isinstance(start_time, str):
            start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp()

        # 如果是迁移学习，rate_for_transfer为实际容量除以源场站容量；如果不为迁移学习，rate_for_transfer为1
        rate_for_transfer = 1
        if self.station_status == 1:
            rate_for_transfer = self.plant_capacity / self.source_capacity_for_transfer

        # 从MySQL读取测试数据
        ann_model_name, b = self.ultra_short_best_model_name.split('_', 1)

        try:
            usecols = load_model(self.model_savepath + str(self.plant_id) + '/ultra_short/' + ann_model_name + '/' + 'ultra_short_usecols.pkl')
            test_feature, history_power = self._load_nwp_for_ultra_short_predict(start_time=start_time, ultra_short_nwp_feature=usecols)
        except Exception as err:
            logs.error(str(err), exc_info=True)
            logs.info(self.plant_name + ' ultra_short, 读NWP时失败')
            test_feature = numpy.array([])
            history_power = None

        forecast_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(start_time)))

        predict_result_without_nwp = predict_without_nwp.predict_without_nwp(
            station_name=self.plant_name, station_id=self.plant_id, station_type=self.plant_type,
            station_capacity=self.plant_capacity, model_save_path=self.model_savepath, bin_num=self.bin_number, connection=self.conn,
            predict_term='ultra_short', predict_start_time=forecast_time, predict_pattern=self.predict_pattern)

        predict_result, sr, model_used = predict.predict_ultra_short_power(
            station_name=self.plant_name, station_id=self.plant_id,
            predict_type=self.plant_type, model_path=self.model_savepath,
            sr_col=self.sr_col, online_capacity=self.plant_capacity,
            capacity=self.plant_capacity, forecast_time=forecast_time,
            best_model=self.ultra_short_best_model_name, second_model=self.ultra_short_second_model_name,
            predict_result_without_nwp=predict_result_without_nwp,
            test_feature=test_feature, history_power=history_power,
            rate_for_transfer=rate_for_transfer)

        return predict_result, sr, model_used

    def predict_interval_and_sql(self, start_time, enable_interval_predict, predict_term):
        """
        对中期预测的补充：
        1、进行区间预测
        2、结果写入数据库
        """
        if predict_term == 'medium':
            predict_term_name = '中期预测'
            try:
                predict_result, model_applied = self.medium_term_predict(start_time)
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.info(self.plant_name + ' | ' + start_time + ' | 中期预测 | 预测失败')
            sr = None
            interval_term = 'short'
            interval_best_model = self.short_best_model_name
        elif predict_term == 'short':
            predict_term_name = '短期预测'
            try:
                predict_result, sr, model_applied = self.short_term_predict(start_time)
                predict_result = predict_result.values[0, :]
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.error(self.plant_name + ' | ' + start_time + ' | 短期预测 | 预测失败')
            interval_term = predict_term
            interval_best_model = self.short_best_model_name
        elif predict_term == 'ultra_short':
            predict_term_name = '超短期预测'
            try:
                predict_result, sr, model_applied = self.ultra_short_term_predict(start_time)
                predict_result = predict_result.values[0, :]
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.error(self.plant_name + ' | ' + start_time + ' | 超短期预测 | 预测失败')
            interval_term = predict_term
            interval_best_model = self.ultra_short_best_model_name
        else:
            logs.warning(self.plant_name + ' | ' + start_time + ' | predict_term未按要求的格式输入！')
            return

        if self.station_status == 1:
            logs.warning(self.plant_name + ' | ' + start_time + ' | ' + predict_term_name + ' | 采用的迁移模型！')
        if isinstance(start_time, str):
            start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp()
        start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))
        if enable_interval_predict:
            try:
                probabilistic_model = IntervalForecastTask()
                interval_result = probabilistic_model.interval_prediction(station_id=self.plant_id,
                                                                          predict_term=interval_term,
                                                                          predict_type=self.plant_type,
                                                                          point_predict_data=predict_result.tolist(),
                                                                          model_path=self.model_savepath,
                                                                          model_name=interval_best_model,
                                                                          online_capacity=self.plant_capacity,
                                                                          bin_num=self.bin_number, sr=sr)

                result = numpy.hstack((predict_result.reshape(-1, 1), interval_result.values))
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning(self.plant_name + ' | ' + start_time_str + ' | ' + predict_term_name + ' | ，区间预测出错！')
                enable_interval_predict = False
            else:
                list_predict_power = ','.join(
                    ['predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power', 'upper_bound_90',
                     'lower_bound_90', 'upper_bound_80', 'lower_bound_80', 'upper_bound_70', 'lower_bound_70',
                     'upper_bound_60', 'lower_bound_60', 'upper_bound_50', 'lower_bound_50'])
                for ii in range(len(result)):
                    forecast_time_float = start_time + (ii + 1) * 15 * 60
                    if predict_term == 'short':
                        forecast_time_float = forecast_time_float + 900
                    forecast_time_struct = time.localtime(forecast_time_float)
                    forecast_time = time.strftime("%Y/%m/%d %H:%M", forecast_time_struct)
                    value = (start_time_str, forecast_time) + tuple(result[ii, :])
                    values = tuple([predict_term]) + tuple([model_applied]) + value

                    query = "INSERT INTO predict_power_" + str(self.plant_id) + '(' + list_predict_power + ')' + \
                            "VALUES(" + ("%s," * 15).strip(',') + ");"
                    args = values
                    lastrowid = self.insert(query, args)
        if not enable_interval_predict:
            result = predict_result.reshape(-1, 1)
            start_time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime(start_time))

            list_predict_power = ','.join(
                ['predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power'])

            for ii in range(len(result)):
                forecast_time_float = start_time + (ii + 1) * 15 * 60
                if predict_term == 'short':
                    forecast_time_float = forecast_time_float + 900
                forecast_time_struct = time.localtime(forecast_time_float)
                forecast_time = time.strftime("%Y/%m/%d %H:%M", forecast_time_struct)
                value = (start_time_str, forecast_time) + tuple(result[ii, :])
                values = tuple([predict_term]) + tuple([model_applied]) + value
                query = "INSERT INTO predict_power_" + str(self.plant_id) + '(' + list_predict_power + ')' + \
                        "VALUES(" + ("%s," * 5).strip(',') + ");"
                args = values
                lastrowid = self.insert(query, args)

    def medium_term_predict(self, predict_start_time):
        """
        先加载模型，
        再加载数据，
        判断一下是不是可以用模型预测，如果不行，用备胎预测
        预测结果直接写回数据库
        :return:
        """
        predict_result = None
        predict_model = MidTermPred()
        if self.plant_type == 'wind':
            default_feature = "['WS', 'UU_hpa_700', 'WD', 'UU_hpa_500', 'RHU_meter']"
        else:
            default_feature = "['SR', 'RHU_meter', 'SWDDIR', 'TCC', 'HCC']"
        predict_model.set_configure(
            id=self.plant_id,
            name=self.plant_name,
            capacity=self.plant_capacity,
            type=self.plant_type,
            sr_col=self.sr_col,
            usecols=default_feature,
            model_savepath=self.model_savepath,
            fitted_nwp_based_model_name=self.short_best_model_name,
            predict_start_time=predict_start_time
        )
        predict_model.h_power_for_prediction = self._load_real_power_for_medium_predict(
            predict_model.id, predict_start_time, predict_model.max_historical_power_length_for_prediction)
        if predict_model.use_nwp_based_model:
            # 尝试用最优模型预测
            try:
                predict_model.nwp_for_predict = self._load_nwp_for_medium_predict(
                    predict_start_time, 800, predict_model.use_cols, predict_model.id, predict_model.type,
                    predict_model.max_nwp_length)
                predict_result = predict_model.predict_with_nwp(predict_model.nwp_for_predict,
                                                                predict_model.h_power_for_prediction)
                model_applied = self.short_best_model_name  # ----------------------------------------------------------
            # 最优模型预测失败，转次优模型
            except Exception as err:
                logs.warning("场站{}({})无法使用最优模型{}预测，将尝试采用次优模型……".format(
                    self.plant_name, self.plant_id, self.short_best_model_name))
                logs.error(err, exc_info=True)
                predict_model.use_nwp_based_model = False
                predict_model.nwp_for_predict = None
                predict_model.fitted_nwp_based_model_name, predict_model.fitted_nwp_based_model_state = \
                    self.short_second_model_name.split('_', 1)
                predict_model.check_nwp_based_model_available()
                if predict_model.use_nwp_based_model:
                    # 尝试用次优模型预测
                    try:
                        predict_model.nwp_for_predict = self._load_nwp_for_medium_predict(
                            predict_start_time, 800, predict_model.use_cols, predict_model.id, predict_model.type,
                            predict_model.max_nwp_length)
                        predict_result = predict_model.predict_with_nwp(predict_model.nwp_for_predict,
                                                                        predict_model.h_power_for_prediction)
                        model_applied = self.short_second_model_name  # ------------------------------------------------
                    except Exception as err:
                        # 次优模型也失败了，用备胎预测
                        logs.warning("场站{}({})无法使用次优模型{}预测，将使用备胎预测……".format(
                            self.plant_name, self.plant_id, self.short_best_model_name))
                        logs.error(err, exc_info=True)
                        predict_result = predict_model.predict_without_nwp(predict_model.h_power_for_prediction)
                        model_applied = 'no_NWP'  # --------------------------------------------------------------------
                else:
                    logs.warning("场站{}({})无法使用次优模型{}预测，将使用备胎预测……".format(
                        self.plant_name, self.plant_id, self.short_best_model_name))
                    predict_result = predict_model.predict_without_nwp(predict_model.h_power_for_prediction)
                    model_applied = 'no_NWP'  # ------------------------------------------------------------------------

        else:
            predict_model.use_nwp_based_model = False
            predict_model.nwp_for_predict = None
            predict_model.fitted_nwp_based_model_name, predict_model.fitted_nwp_based_model_state = \
                self.short_second_model_name.split('_', 1)
            predict_model.check_nwp_based_model_available()
            if predict_model.use_nwp_based_model:
                # 尝试用次优模型预测
                try:
                    predict_model.nwp_for_predict = self._load_nwp_for_medium_predict(
                        predict_start_time, 800, predict_model.use_cols, predict_model.id, predict_model.type,
                        predict_model.max_nwp_length)
                    predict_result = predict_model.predict_with_nwp(predict_model.nwp_for_predict,
                                                                    predict_model.h_power_for_prediction)
                    model_applied = 'no_NWP'  # ------------------------------------------------------------------------
                except Exception as err:
                    # 次优模型也失败了，用备胎预测
                    logs.warning("场站{}({})无法使用次优模型{}预测，将使用备胎预测……".format(
                        self.plant_name, self.plant_id, self.short_best_model_name))
                    logs.error(err, exc_info=True)
                    predict_result = predict_model.predict_without_nwp(predict_model.h_power_for_prediction)
                    model_applied = 'no_NWP'  # ------------------------------------------------------------------------
            else:
                logs.warning("场站{}({})无法使用次优模型{}预测，将使用备胎预测……".format(
                    self.plant_name, self.plant_id, self.short_best_model_name))
                predict_result = predict_model.predict_without_nwp(predict_model.h_power_for_prediction)
                model_applied = 'no_NWP'  # ----------------------------------------------------------------------------
        return predict_result, model_applied

    # TODO
    def _medium_term_predict_with_best_model(self):
        predict_model = MidTermPred()
        predict_model.set_configure(
            id=self.plant_id,
            name=self.plant_name,
            capacity=self.plant_capacity,
            type=self.plant_type,
            sr_col=self.sr_col,
            usecols="",
            model_savepath=self.model_savepath,
            fitted_nwp_based_model_name=self.short_best_model_name
        )


if __name__ == '__main__':
    host_hn = 'localhost'
    user_hn = 'root'
    password_hn = '123456'
    database_hn = 'kuafu'
    charset_hn = 'utf8'
    port_hn = 3306

    use_cols_of_conf_hn = "['id', 'name', 'capacity', 'type', 'sr_col', 'bin_number', 'predict_pattern', " \
                          "'model_savepath', 'station_status', 'source_capacity_for_transfer'] "
    conf_table_hn = 'configure'
    predict_tag_hn = 0
    model_table_hn = 'best_feature_parameters_and_model'
    model = MultiPlantPredict(host_hn, user_hn, password_hn, database_hn, charset_hn, port_hn, use_cols_of_conf_hn)
    model.get_configure()
    model.init_plants_for_prediction()

    # predict_start_time_hn = "2022-04-01 00:00:00"
    # result = []

    # # 短期预测
    # for i in range(30):
    #     predict_start_time_hn = "2022-04-%s 00:00:00" % str(i+1)
    #     if predict_start_time_hn is None:
    #         short_time = '23:45:00'
    #         data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
    #         predict_start_time_hn = data_now + ' ' + short_time
    #     for plant in model.plants_for_prediction:
    #         plant.short_term_predict(predict_start_time_hn, enable_interval_predict=True)

    # # 超短期预测
    # for i in range(30):  # days
    #     for j in range(24):  # hours
    #         for k in ['00', '15', '30', '45']:  # minutes
    #             predict_start_time_hn = '2022-04-%s %s:%s:00' % (str(i + 1), str(j), k)
    #             if predict_start_time_hn is None:
    #                 now_time = datetime.datetime.now().timestamp()
    #                 d = now_time % 900
    #                 predict_start_time_hn = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_time - d))
    #             for plant in model.plants_for_prediction:
    #                 plant.ultra_short_term_predict(predict_start_time_hn, enable_interval_predict=True)

    # 中期预测
    for i in range(30):
        predict_start_time_hn = "2022-04-%s 00:00:00" % str(i+1)
        if predict_start_time_hn is None:
            short_time = '23:45:00'
            data_now = time.strftime("%Y-%m-%d", time.gmtime(datetime.datetime.now().timestamp()))
            predict_start_time_hn = data_now + ' ' + short_time
        for plant in model.plants_for_prediction:
            plant.predict_interval_and_sql(predict_start_time_hn, enable_interval_predict=True, predict_term='medium')

    print('done')

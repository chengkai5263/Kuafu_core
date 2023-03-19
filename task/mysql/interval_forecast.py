
import pandas
from app.probabilistic_predict.interval_forecast import IntervalForecast
import pymysql
from common.tools import catch_exception
from common.logger import logs


class IntervalForecastTask:
    """
    Probabilistic_Forecast模型，根据预测发电曲线生成区间预测曲线
    """
    def generate_forecast_bin(self, host, user, password, database, charset, port,
                              predict_term, model_name, station_id, online_capacity, bin_num, model_savepath):
        """
        根据点预测结果及其相应的功率实测值，生成预测箱及对应的累积经验分布函数。
        :param host: 主机
        :param user: 用户名
        :param password: 密码
        :param database: 数据库名
        :param charset: 解码方式
        :param port: 端口号
        :param station_id: 场站id
        :param model_name: 参与寻优的模型列表
        :return:None
        """
        # 加载特征集文件
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        c = db.cursor()
        c.execute("select distinct start_time, forecast_time, predict_power from predict_power_" + str(station_id) +
                  "_train where predict_term = %s and model_name = %s"
                  " ORDER BY start_time asc, forecast_time asc;",
                  (predict_term, model_name))
        db.commit()
        record = c.fetchall()
        predict_power = pandas.DataFrame(record, columns=['start_time', 'forecast_time', 'predicted'])

        c.execute("select distinct time, power from real_power_" + str(station_id) +
                  " ORDER BY time asc;")
        db.commit()
        record = c.fetchall()
        real_power = pandas.DataFrame(record, columns=['time', 'real'])
        c.close()
        db.close()

        probabilistic_model = IntervalForecast()
        probabilistic_model.generate_forecast_bin(predict_power=predict_power, real_power=real_power,
                                                  predict_term=predict_term, model_name=model_name,
                                                  station_id=station_id, online_capacity=online_capacity,
                                                  bin_num=bin_num, model_savepath=model_savepath)

    def interval_prediction(self, station_id, predict_term, predict_type, point_predict_data, model_path, model_name,
                            online_capacity, bin_num, sr):
        """
            根据点预测结果及训练的预测箱模型，生成区间预测的上下界。
            :param point_predict_data: 点预测结果。
            :param model_path: 模型存储的位置
            :param online_capacity: 开机容量
            :param predict_type: 短期/超短期的预测类型
            :param bin_num: 预测箱数量，数量越大预测准度越高，但是覆盖面积会降低。
            :return:dataframe
        """
        probabilistic_model = IntervalForecast()
        dataframe = probabilistic_model.interval_prediction(station_id=station_id, predict_term=predict_term,
                                                            predict_type=predict_type,
                                                            point_predict_data=point_predict_data,
                                                            model_path=model_path, model_name=model_name,
                                                            online_capacity=online_capacity, bin_num=bin_num, sr=sr)
        return dataframe


# 子昊
@catch_exception("interval_learning error: ")
def interval_learning(host, user, password, database, charset, port, station_id, model_name=None):
    """
    训练区间预测模型
    读取区间预测配置文件，对配置文件下所有场站的短期、超短期结果进行区间预测训练
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param model_name: 预测的方法
    :return: None
    """
    logs.info('开始区间预测的训练')
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 定义游标
    c = db.cursor()
    c.execute("select * from configure where id = %s;", station_id)
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe = pandas.DataFrame(record, columns=coul)

    config_cluster = {station_id: {"id": dataframe.loc[:, 'id'][0],
                                   "online_capacity": dataframe.loc[:, "capacity"][0],
                                   "bin_number": dataframe.loc[:, "bin_number"][0],
                                   "model_savepath": dataframe.loc[:, 'model_savepath'][0]
                                   }
                      }

    result_short = c.execute(
        "select best_model from best_feature_parameters_and_model where id=%s and predict_term='short';", station_id)
    db.commit()
    record_short = c.fetchall()

    result_ultra_short = c.execute(
        "select best_model from best_feature_parameters_and_model"
        " where id=%s and predict_term='ultra_short';", station_id)
    db.commit()
    record_ultra_short = c.fetchall()

    # 调用函数进行区间预测训练
    probabilistic_model = IntervalForecastTask()
    model_name_short = model_name
    if result_short > 0 and record_short[0][0] is not None and len(record_short[0][0]) > 0:
        model_name_short = record_short[0][0]
    probabilistic_model.generate_forecast_bin(host=host, user=user, password=password,
                                              database=database, charset=charset, port=port,
                                              predict_term='short',
                                              model_name=model_name_short,
                                              station_id=station_id,
                                              online_capacity=config_cluster[station_id]["online_capacity"],
                                              bin_num=config_cluster[station_id]["bin_number"],
                                              model_savepath=config_cluster[station_id]["model_savepath"])
    model_name_ultra_short = model_name
    if result_ultra_short > 0 and record_ultra_short[0][0] is not None and len(record_ultra_short[0][0]) > 0:
        model_name_ultra_short = record_ultra_short[0][0]
    probabilistic_model.generate_forecast_bin(host=host, user=user, password=password,
                                              database=database, charset=charset, port=port,
                                              predict_term='ultra_short',
                                              model_name=model_name_ultra_short,
                                              station_id=station_id,
                                              online_capacity=config_cluster[station_id]["online_capacity"],
                                              bin_num=config_cluster[station_id]["bin_number"],
                                              model_savepath=config_cluster[station_id]["model_savepath"])
    logs.info('区间预测的训练完成')
    c.close()
    db.close()

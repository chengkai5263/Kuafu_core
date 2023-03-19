
import pandas
import pymysql
from common import training
from common.logger import logs


# 程凯
def trainning_model(host, user, password, database, charset, port, record_restrict=None, without_history_power=None):
    """
    训练模型
    :param record_restrict: 限电记录
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param without_history_power: 是否训练带历史功率的模型
    :return: None
    """
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    # 从数据库读取需要集成学习的模型列表
    c.execute("select config_value from sys_config where config_key = 'ensemble_learn_model_name';")
    db.commit()
    record = c.fetchall()
    if len(record) == 0:  # 读数据库记录的时候，判断是否为空，如果是空要提前退出，2023/3/19
        model_name_cluster = []
    else:
        model_name_cluster = eval(record[0][0])

    # 读取配置-----------------------------------------------------------------------------------------------------------
    c.execute("select * from configure where station_status = 2;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)

    # 场站id
    station_id_cluster = []
    for i in range(len(dataframe_config)):
        station_id_cluster.append(dataframe_config.loc[:, "id"][i])

    # 配置信息
    config_cluster = {}
    for i in range(len(dataframe_config)):
        config_cluster[dataframe_config.loc[:, "id"][i]] = {"id": dataframe_config.loc[:, 'id'][i],
                                                            "name": dataframe_config.loc[:, "name"][i],
                                                            "type": dataframe_config.loc[:, "type"][i],
                                                            "sr_col": 0,
                                                            "online_capacity": dataframe_config.loc[:, "capacity"][i],
                                                            "model_savepath":
                                                                dataframe_config.loc[:, "model_savepath"][i]}

    # 读取默认特征和默认参数 -----------------------------------------------------------------------------------------------
    c.execute("select * from default_feature_and_parameters;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_default = pandas.DataFrame(record, columns=coul)

    # 模型参数默认值
    short_wind_best_parameter_default = {}
    ultra_short_wind_best_parameter_default = {}
    short_solar_best_parameter_default = {}
    ultra_short_solar_best_parameter_default = {}
    for i in range(len(model_name_cluster)):
        short_wind_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_wind"][model_name_cluster[i]].values[0])
        ultra_short_wind_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"][
                model_name_cluster[i]].values[0])
        short_solar_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "short_solar"][model_name_cluster[i]].values[0])
        ultra_short_solar_best_parameter_default[model_name_cluster[i]] = eval(
            dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"][
                model_name_cluster[i]].values[0])

    # 特征默认值
    wind_ultra_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_wind"]['usecols'].values[0])
    wind_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "short_wind"]['usecols'].values[0])
    solar_ultra_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "ultra_short_solar"]['usecols'].values[0])
    solar_short_usecols_default = eval(
        dataframe_default.loc[dataframe_default['term_type'] == "short_solar"]['usecols'].values[0])

    # 读取最优特征和参数，如果不存在，则使用默认值 -----------------------------------------------------------------------------
    c.execute("select * from best_feature_parameters_and_model;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_best = pandas.DataFrame(record, columns=coul)

    # 模型短期最优参数，如果不存在，则使用默认值
    for i in range(len(dataframe_config)):
        best_parameters_exist = False
        if dataframe_config.loc[:, "type"][i] == 'wind':
            best_parameter_short = short_wind_best_parameter_default.copy()
        else:
            best_parameter_short = short_solar_best_parameter_default.copy()
        for j in range(len(dataframe_best)):
            if dataframe_best.loc[:, 'id'][j] == dataframe_config.loc[:, 'id'][i] and \
                    dataframe_best.loc[:, 'predict_term'][j] == 'short':
                best_parameters_exist = True
                for name in model_name_cluster:
                    if dataframe_best.loc[:, name][j] is not None and len(
                            dataframe_best.loc[:, name][j]) > 0:
                        best_parameter_short[name] = eval(dataframe_best.loc[:, name][j])
        if not best_parameters_exist:
            logs.debug("最优参数路径未找到！！！采用默认参数训练")
        config_cluster[dataframe_config.loc[:, "id"][i]]["best_parameter_short"] = best_parameter_short
    # 模型超短期最优参数，如果不存在，则使用默认值
    for i in range(len(dataframe_config)):
        best_parameters_exist = False
        if dataframe_config.loc[:, "type"][i] == 'wind':
            best_parameter_ultra_short = ultra_short_wind_best_parameter_default.copy()
        else:
            best_parameter_ultra_short = ultra_short_solar_best_parameter_default.copy()
        for j in range(len(dataframe_best)):
            if dataframe_best.loc[:, 'id'][j] == dataframe_config.loc[:, 'id'][i] and \
                    dataframe_best.loc[:, 'predict_term'][j] == 'ultra_short':
                best_parameters_exist = True
                for name in model_name_cluster:
                    if dataframe_best.loc[:, name][j] is not None and len(
                            dataframe_best.loc[:, name][j]) > 0:
                        best_parameter_ultra_short[name] = eval(dataframe_best.loc[:, name][j])
        if not best_parameters_exist:
            logs.debug("最优参数路径未找到！！！采用默认参数训练")
        config_cluster[dataframe_config.loc[:, "id"][i]]["best_parameter_ultra_short"] = best_parameter_ultra_short

    # 超短期最优特征，如果不存在，则使用默认值
    for i in range(len(dataframe_config)):
        if dataframe_config.loc[:, "type"][i] == 'wind':
            ultra_short_usecols = wind_ultra_short_usecols_default.copy()
        else:
            ultra_short_usecols = solar_ultra_short_usecols_default.copy()
        for j in range(len(dataframe_best)):
            if dataframe_best.loc[:, 'id'][j] == dataframe_config.loc[:, 'id'][i] and \
                    dataframe_best.loc[:, 'predict_term'][j] == 'ultra_short' and \
                    dataframe_best.loc[:, 'usecols'][j] is not None and \
                    len(dataframe_best.loc[:, 'usecols'][j]) > 0:
                ultra_short_usecols = eval(dataframe_best.loc[:, 'usecols'][j])
        config_cluster[dataframe_config.loc[:, "id"][i]]["ultra_short_usecols"] = ultra_short_usecols

    # 短期最优特征，如果不存在，则使用默认值
    for i in range(len(dataframe_config)):
        if dataframe_config.loc[:, "type"][i] == 'wind':
            short_usecols = wind_short_usecols_default.copy()
        else:
            short_usecols = solar_short_usecols_default.copy()
        for j in range(len(dataframe_best)):
            if dataframe_best.loc[:, 'id'][j] == dataframe_config.loc[:, 'id'][i] and \
                    dataframe_best.loc[:, 'predict_term'][j] == 'short' and \
                    dataframe_best.loc[:, 'usecols'][j] is not None and \
                    len(dataframe_best.loc[:, 'usecols'][j]) > 0:
                short_usecols = eval(dataframe_best.loc[:, 'usecols'][j])
        config_cluster[dataframe_config.loc[:, "id"][i]]["short_usecols"] = short_usecols

    c.execute("DELETE FROM parallel_tasks where task_type = 'data_load_preprocess';")
    db.commit()
    for station_id in station_id_cluster:
        term_cluster = ["short", "ultra_short"]
        for term in term_cluster:
            c.execute("INSERT INTO parallel_tasks (task_name, task_status, task_type) "
                      "VALUES (%s, %s, %s) ",
                      tuple([str(station_id) + '_' + term]) + tuple(['0']) + tuple(['data_load_preprocess']))
        db.commit()

    # 调用函数
    training.save_all_fitted_model(station_id_cluster=station_id_cluster, config_cluster=config_cluster,
                                   model_name_cluster=model_name_cluster,
                                   record_restrict=record_restrict,
                                   without_history_power=without_history_power)
    c.close()
    db.close()


import numpy
import time
from common.tools import save_model
from common.tools import load_model
import datetime
from common import training
from common.logger import logs
from common import data_postprocess
import os
from common.tools import catch_exception


@catch_exception("ensemble_learning_evaluate_short error: ", exc_info=True, default_return=None)
def ensemble_learning_evaluate_short(predict_type=None, model_path=None, model_name=None, model_state=None,
                                     station_id=None, sr_col=None, online_capacity=None):
    """
    :param predict_type: 预测类型
    :param online_capacity: 开机容量
    :param model_name: 模型名称
    :param model_state: 预测方法
    :param model_path:
    :param station_id:
    :param sr_col:
    :return:
    """
    sub_dir_path = "%s%s/short/" % (model_path, str(station_id))
    nwp_np_short = load_model(sub_dir_path + 'short_test_data.pkl')
    if '-' in model_state:
        model_state, evaluate_label = model_state.split('-', 1)
        evaluate_time, total_times = evaluate_label.split('/', 1)
        evaluate_time = eval(evaluate_time)
        total_times = eval(total_times)

        div = int((len(nwp_np_short) / 288) // total_times)  # 商
        mod = int((len(nwp_np_short) / 288) % total_times)  # 余数
        if evaluate_time == total_times:
            nwp_np_short = nwp_np_short[
                           ((evaluate_time - 1) * div + mod) * 288:, :]
        else:
            if evaluate_time > mod:
                nwp_np_short = nwp_np_short[
                               ((evaluate_time - 1) * div + mod) * 288:
                               (evaluate_time * div + mod) * 288, :]
            else:
                nwp_np_short = nwp_np_short[(evaluate_time - 1) * (div + 1) * 288:
                                            evaluate_time * (div + 1) * 288, :]

    # 短期预测
    predict_result_cluster, time_predict_cluster = predict_short_power(
        model_name_state=model_name + model_state, sr_col=sr_col, online_capacity=online_capacity,
        predict_type=predict_type, model_path=model_path, station_id=station_id,
        nwp_np_short=nwp_np_short)
    result = predict_result_cluster.reshape(-1, 1)

    return time_predict_cluster, result, model_state


@catch_exception("ensemble_learning_evaluate_ultra_short error: ", exc_info=True, default_return=None)
def ensemble_learning_evaluate_ultra_short(predict_type=None, model_path=None, model_name=None, model_state=None,
                                           station_id=None, sr_col=None, online_capacity=None):
    """
    :param predict_type: 预测类型
    :param online_capacity: 开机容量
    :param model_name: 模型名称
    :param model_state: 预测方法
    :param model_path:
    :param station_id:
    :param sr_col:
    """
    sub_dir_path = "%s%s/ultra_short/" % (model_path, str(station_id))
    nwp_np_ultra_short = load_model(sub_dir_path + 'ultra_short_test_data.pkl')
    if '-' in model_state:
        model_state, evaluate_label = model_state.split('-', 1)
        evaluate_time, total_times = evaluate_label.split('/', 1)
        evaluate_time = eval(evaluate_time)
        total_times = eval(total_times)

        div = int((len(nwp_np_ultra_short)) // total_times)  # 商
        mod = int((len(nwp_np_ultra_short)) % total_times)  # 余数
        if evaluate_time == total_times:
            nwp_np_ultra_short = nwp_np_ultra_short[((evaluate_time - 1) * div + mod):, :]
        else:
            if evaluate_time > mod:
                nwp_np_ultra_short = nwp_np_ultra_short[((evaluate_time - 1) * div + mod):
                                                        (evaluate_time * div + mod), :]
            else:
                nwp_np_ultra_short = nwp_np_ultra_short[(evaluate_time - 1) * (div + 1):
                                                        evaluate_time * (div + 1), :]

    # 超短期预测
    predict_result_cluster, time_predict_cluster = predict_ultra_short_power(
        model_name_state=model_name + model_state, predict_type=predict_type,
        model_path=model_path, station_id=station_id, sr_col=sr_col, online_capacity=online_capacity,
        nwp_np_ultra_short=nwp_np_ultra_short)
    result = predict_result_cluster.reshape(-1, 1)
    return time_predict_cluster, result, model_state


@catch_exception("ultra_short_best_model_retrain error: ", exc_info=True, default_return=None)
def ultra_short_best_model_retrain(config_cluster_train=None,
                                   model_name_cluster_ultra_short=None, station_id=None, predict_type=None,
                                   model_name=None, model_name_second=None):
    """
    :param config_cluster_train: 模型配置参数
    :param model_name_cluster_ultra_short: 超短期集成学习列表
    :param station_id: 场站id
    :param predict_type: 场站类型
    :param model_name: 最优模型名称
    :param model_name_second: 次优模型名称
    :return: None
    """
    # 超短期模型不带历史功率再训练 2022/10/19
    try:
        if len(model_name_cluster_ultra_short) > 0 and predict_type == 'wind':
            for i in range(2):
                try:
                    if model_name in ['BPNN',  'LSTM', 'MLP', 'GBRT', 'DecisionTreeRegressor', 'RandomForest', 'SVM']:
                        fitted_model = training.fit_model(predict_term='ultra_short', model_name=model_name,
                                                          config=config_cluster_train[station_id], record_restrict=None,
                                                          without_history_power=True, scene='ensemble_learn')

                        sub_dir_path = "%s%s%s%s%s" % (config_cluster_train[station_id]['model_savepath'],
                                                       str(station_id), '/ultra_short/', model_name, '/')
                        os.makedirs(sub_dir_path, exist_ok=True)
                        save_model(fitted_model, sub_dir_path + 'ultra_short_' + model_name + '.pkl')
                    break
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.warning('超短期最优模型不带历史功率模型再训练保存时失败'+str(i+1)+'次！')
            # 超短期次优模型的再训练与保存
            for i in range(2):
                try:
                    if model_name_second is not None:
                        if model_name_second in ['BPNN',  'LSTM', 'MLP', 'GBRT', 'DecisionTreeRegressor',
                                                 'RandomForest', 'SVM']:
                            fitted_model = training.fit_model(predict_term='ultra_short', model_name=model_name_second,
                                                              config=config_cluster_train[station_id],
                                                              record_restrict=None, without_history_power=True,
                                                              scene='ensemble_learn')

                            sub_dir_path = "%s%s%s%s%s" % (config_cluster_train[station_id]['model_savepath'],
                                                           str(station_id), '/ultra_short/', model_name_second, '/')
                            os.makedirs(sub_dir_path, exist_ok=True)
                            save_model(fitted_model, sub_dir_path + 'ultra_short_' + model_name_second + '.pkl')
                    break
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.warning('超短期次优模型不带历史功率模型再训练保存时失败'+str(i+1)+'次！')
    except Exception as err:
        logs.error(str(err), exc_info=True)
        logs.warning('超短期最优及次优模型不带历史功率模型再训练保存时失败！')


@catch_exception("save_a_fitted_model error: ", exc_info=True, default_return=None)
def save_a_fitted_model(station_id, predict_term, model_name, config):
    """
    针对特定的电场和预测类型，通过测试选择最优的模型并保存
    :param station_id: 场站名称
    :param predict_term: 预测类型
    :param config: 配置信息
    :param model_name: 模型名称集合
    :return:
    """
    if predict_term == 'ultra_short':
        if config['type'] == 'solar':
            fitted_model = training.fit_model(predict_term=predict_term, model_name=model_name,
                                              config=config, record_restrict=None,
                                              without_history_power=True, scene='ensemble_learn')
        else:
            fitted_model = training.fit_model(predict_term=predict_term, model_name=model_name,
                                              config=config, record_restrict=None,
                                              without_history_power=False, scene='ensemble_learn')
    else:
        fitted_model = training.fit_model(predict_term=predict_term, model_name=model_name,
                                          config=config, record_restrict=None,
                                          without_history_power=True, scene='ensemble_learn')

    sub_dir_path = "%s%s/%s/%s/" % (config['model_savepath'], str(station_id), predict_term, model_name)
    os.makedirs(sub_dir_path, exist_ok=True)
    save_model(fitted_model, sub_dir_path + predict_term + '_' + model_name + '.pkl')
    if predict_term == 'ultra_short':
        save_model(config["ultra_short_usecols"], sub_dir_path + 'ultra_short_usecols.pkl')
    else:
        save_model(config["short_usecols"], sub_dir_path + 'short_usecols.pkl')


# 程凯
@catch_exception("predict_short_power error: ", exc_info=True, default_return=None)
def predict_short_power(model_name_state='BPNN_without_history_power', sr_col=None, online_capacity=None,
                        predict_type=None, model_path=None, station_id=None, nwp_np_short=None):
    """
    为结果存储文件撰写表头，初始化时调用
    :param model_name_state: 模型名
    :param predict_type: 预测类型
    :param model_path: 模型存放路径
    :param station_id: 场站名
    :param sr_col: 辐照度所在列
    :param online_capacity: 开机容量
    :param nwp_np_short: 短期预测数据集
    :return:
    """
    model_name, model_state = model_name_state.split('_', 1)
    fitted_model = load_model(model_path + str(station_id) + '/short/' + model_name + '/'
                              + 'short' + '_' + model_name + '.pkl')
    fitted_model = eval('fitted_model' + '.predict_' + model_state)

    predict_result_cluster = numpy.zeros((0, 1))
    time_predict_cluster = nwp_np_short[:, 0].tolist()
    # 去掉了短期预测输入history_power，2023/3/18
    for i in range(int(len(nwp_np_short) / 288)):
        test_feature = nwp_np_short[288 * i: 288 * i + 288, 1:].astype("float64")
        # history_power = nwp_np_short[288 * i, -1]
        predict_result_raw = fitted_model(feature=test_feature, history_power=None,
                                          predict_type=predict_type, irradiance_col=sr_col)
        predict_result = data_postprocess.result_post_process(irradiance_threshold=10,
                                                              pre_processing_predict_result=predict_result_raw,
                                                              online_capacity=online_capacity,
                                                              capacity=online_capacity,
                                                              predict_type=predict_type,
                                                              irradiance_col=0, feature=test_feature)

        predict_result = predict_result.reshape(-1, 1)
        predict_result_cluster = numpy.vstack((predict_result_cluster, predict_result))
    return predict_result_cluster, time_predict_cluster


# 程凯
@catch_exception("predict_ultra_short_power error: ", exc_info=True, default_return=None)
def predict_ultra_short_power(model_name_state='BPNN_without_history_power', predict_type=None,
                              model_path=None, station_id=None, sr_col=None, online_capacity=None,
                              nwp_np_ultra_short=None):
    """
    为结果存储文件撰写表头，初始化时调用
    :param model_name_state: 模型名
    :param predict_type: 预测类型
    :param model_path: 模型存放路径
    :param station_id: 场站名
    :param sr_col: 辐照度所在列
    :param online_capacity: 开机容量
    :param nwp_np_ultra_short: 预测数据集
    :return:
    """
    model_name, model_state = model_name_state.split('_', 1)
    fitted_model = load_model(model_path + str(station_id) + '/ultra_short/' + model_name + '/'
                              + 'ultra_short' + '_' + model_name + '.pkl')
    fitted_model = eval('fitted_model' + '.predict_' + model_state)

    predict_result_cluster = numpy.zeros((0, 1))
    time_predict_cluster = list()
    for i in range(len(nwp_np_ultra_short)):
        test_feature = nwp_np_ultra_short[i, 1:-1].astype("float64").reshape(16, -1)
        history_power = nwp_np_ultra_short[i, -1]
        predict_result_raw = fitted_model(feature=test_feature, history_power=history_power,
                                          predict_type=predict_type, irradiance_col=sr_col)
        predict_result = data_postprocess.result_post_process(irradiance_threshold=10,
                                                              pre_processing_predict_result=predict_result_raw,
                                                              online_capacity=online_capacity,
                                                              capacity=online_capacity,
                                                              predict_type=predict_type,
                                                              irradiance_col=0, feature=test_feature)
        time_predict = [nwp_np_ultra_short[i, 0]]
        for j in range(15):
            time_predict.append(datetime.datetime.strptime(
                time.strftime("%Y/%m/%d %H:%M", time.localtime(time_predict[0].timestamp() + j * 900 - 31 * 900)),
                '%Y/%m/%d %H:%M'))
        time_predict_cluster = time_predict_cluster + time_predict
        predict_result = predict_result.reshape(-1, 1)
        predict_result_cluster = numpy.vstack((predict_result_cluster, predict_result))
    return predict_result_cluster, time_predict_cluster


@catch_exception("get_best_feature_parameter error: ", exc_info=True, default_return=None)
def get_best_feature_parameter(dataframe_best,
                               model_name_cluster_setting, station_id, config_cluster,
                               short_best_parameter_default, ultra_short_best_parameter_default,
                               ultra_short_usecols_default, short_usecols_default):
    """
    :param dataframe_best: 数据库读取的最优特征和参数
    :param station_id: 场站id
    :param config_cluster: 场站配置
    :param model_name_cluster_setting: 集成学习模型列表
    :param ultra_short_usecols_default: 超短期默认特征
    :param short_usecols_default: 短期默认特征
    :param ultra_short_best_parameter_default: 超短期默认参数
    :param short_best_parameter_default: 短期默认参数
    :return:
    """
    # 模型短期最优参数，如果不存在，则使用默认值
    best_parameters_exist = False
    best_parameter_short = short_best_parameter_default.copy()
    for j in range(len(dataframe_best)):
        if dataframe_best.loc[:, 'id'][j] == station_id and dataframe_best.loc[:, 'predict_term'][j] == 'short':
            best_parameters_exist = True
            for name in model_name_cluster_setting:
                if dataframe_best.loc[:, name][j] is not None and len(
                        dataframe_best.loc[:, name][j]) > 0:
                    best_parameter_short[name] = eval(dataframe_best.loc[:, name][j])
    if not best_parameters_exist:
        logs.debug("最优参数路径未找到！！！采用默认参数训练")
    config_cluster[station_id]["best_parameter_short"] = best_parameter_short
    # 模型超短期最优参数，如果不存在，则使用默认值
    best_parameters_exist = False
    best_parameter_ultra_short = ultra_short_best_parameter_default.copy()
    for j in range(len(dataframe_best)):
        if dataframe_best.loc[:, 'id'][j] == station_id and dataframe_best.loc[:, 'predict_term'][j] == 'ultra_short':
            best_parameters_exist = True
            for name in model_name_cluster_setting:
                if dataframe_best.loc[:, name][j] is not None and len(
                        dataframe_best.loc[:, name][j]) > 0:
                    best_parameter_ultra_short[name] = eval(dataframe_best.loc[:, name][j])
    if not best_parameters_exist:
        logs.debug("最优参数路径未找到！！！采用默认参数训练")
    config_cluster[station_id]["best_parameter_ultra_short"] = best_parameter_ultra_short

    # 超短期最优特征，如果不存在，则使用默认值
    ultra_short_usecols = ultra_short_usecols_default.copy()
    for j in range(len(dataframe_best)):
        if dataframe_best.loc[:, 'id'][j] == station_id and \
                dataframe_best.loc[:, 'predict_term'][j] == 'ultra_short' and \
                dataframe_best.loc[:, 'usecols'][j] is not None and \
                len(dataframe_best.loc[:, 'usecols'][j]) > 0:
            ultra_short_usecols = eval(dataframe_best.loc[:, 'usecols'][j])
    config_cluster[station_id]["ultra_short_usecols"] = ultra_short_usecols

    # 短期最优特征，如果不存在，则使用默认值
    short_usecols = short_usecols_default.copy()
    for j in range(len(dataframe_best)):
        if dataframe_best.loc[:, 'id'][j] == station_id and \
                dataframe_best.loc[:, 'predict_term'][j] == 'short' and \
                dataframe_best.loc[:, 'usecols'][j] is not None and \
                len(dataframe_best.loc[:, 'usecols'][j]) > 0:
            short_usecols = eval(dataframe_best.loc[:, 'usecols'][j])
    config_cluster[station_id]["short_usecols"] = short_usecols

    return config_cluster

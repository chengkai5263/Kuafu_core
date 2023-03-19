
from common import data_postprocess
from common.tools import load_model
import pandas
from common.logger import logs


def predict_ultra_short_power(station_name, model_path, station_id, sr_col, predict_type, online_capacity, capacity,
                              forecast_time, best_model, second_model, predict_result_without_nwp,
                              test_feature, history_power, rate_for_transfer):
    """
    预测超短期功率（单次）
    :param station_name: 场站名称:
    :param station_id: 场站id:
    :param model_path: 文件路径:
    :param sr_col: 数据天数:
    :param predict_type: 预测类型:
    :param online_capacity: 开机容量:
    :param capacity: 装机容量:
    :param forecast_time: 预报的时间
    :param best_model: 预测的方法
    :param second_model: 预测的方法
    :param predict_result_without_nwp:
    :param test_feature:
    :param history_power:
    :param rate_for_transfer: 迁移学习的模型需要乘以系数，非迁移学习下该系数为1
    :return:
    """
    if len(test_feature) != 16:
        # 启动备胎计划，修改配置文件的标志位
        predict_result = predict_result_without_nwp
        predict_result_dataframe = pandas.DataFrame(predict_result.reshape(1, -1), index=[forecast_time])
        model_used = 'no_NWP'
        logs.warning(station_name + ' | ' + forecast_time + ' | 超短期预测 | NWP缺失，转为备胎计划')
    else:
        # NWP完整
        try:
            if history_power is not None:
                try:
                    model_name, state_name = best_model.split('_', 1)
                    fitted_model = load_model(model_path + str(station_id) + '/ultra_short/' + model_name + '/'
                                              + 'ultra_short' + '_' + model_name + '.pkl')
                    fitted_model = eval('fitted_model.predict_' + state_name)
                    predict_result_raw = fitted_model(feature=test_feature, history_power=history_power,
                                                      predict_type=predict_type, irradiance_col=0)
                    model_used = best_model
                    logs.info(station_name + ' | ' + forecast_time + ' | 超短期预测 | ，采用最优模型：' + best_model)
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    model_name, state_name = second_model.split('_', 1)
                    fitted_model = load_model(model_path + str(station_id) + '/ultra_short/' + model_name + '/'
                                              + 'ultra_short' + '_' + model_name + '.pkl')
                    fitted_model = eval('fitted_model.predict_' + state_name)
                    predict_result_raw = fitted_model(feature=test_feature,
                                                      history_power=history_power,
                                                      predict_type=predict_type, irradiance_col=0)
                    model_used = second_model
                    logs.info(station_name + ' | ' + forecast_time + ' | 超短期预测 | ，采用次优模型：' + second_model)
            else:
                # 历史功率缺失时
                try:
                    model_name, state_name = best_model.split('_', 1)
                    fitted_model = load_model(model_path + str(station_id) + '/ultra_short/' + model_name +
                                              '/ultra_short' + '_' + model_name + '.pkl')
                    fitted_model = fitted_model.predict_without_history_power
                    predict_result_raw = fitted_model(feature=test_feature, predict_type=predict_type, irradiance_col=0)
                    model_used = model_name + '_without_history_power'
                    logs.info(station_name + ' | ' + forecast_time + ' | 超短期预测 | ，采用最优模型：' + model_name +
                              '，由于历史功率缺失，采用为无历史功率模式')
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.warning(station_name + '_ultra_short, ' + forecast_time + '，最优模型：' + best_model +
                                 '预测出错，改为次优模型：' + second_model)
                    model_name, state_name = second_model.split('_', 1)
                    fitted_model = load_model(model_path + str(station_id) + '/ultra_short/' + model_name +
                                              '/ultra_short' + '_' + model_name + '.pkl')
                    fitted_model = fitted_model.predict_without_history_power
                    predict_result_raw = fitted_model(feature=test_feature, predict_type=predict_type, irradiance_col=0)
                    model_used = model_name + '_without_history_power'
                    logs.info(station_name + ' | ' + forecast_time + ' | 超短期预测 | ，采用次优模型：' + model_name +
                              '，且由于历史功率缺失，采用无历史功率模式')
            try:
                predict_result = data_postprocess.result_post_process(irradiance_threshold=10,
                                                                      pre_processing_predict_result=predict_result_raw,
                                                                      online_capacity=capacity,
                                                                      capacity=online_capacity,
                                                                      predict_type=predict_type,
                                                                      irradiance_col=sr_col, feature=test_feature)
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning(station_name + ' | ' + forecast_time + ' | 超短期预测 | ，数据后处理出错！')
                predict_result = predict_result_raw
            # 如果是迁移学习，rate_for_transfer为实际容量除以源场站容量；如果不为迁移学习，rate_for_transfer为1
            predict_result_dataframe = pandas.DataFrame(predict_result.reshape(1, -1)*rate_for_transfer,
                                                        index=[forecast_time])
        except Exception as err:
            logs.error(str(err), exc_info=True)
            predict_result = predict_result_without_nwp
            predict_result_dataframe = pandas.DataFrame(predict_result.reshape(1, -1), index=[forecast_time])
            model_used = 'no_NWP'
            logs.warning(station_name + ' | ' + forecast_time + ' | 超短期预测 | 模型预测出错，转为备胎计划！')
    sr = None  # 返回辐照度区间预测后处理需要用到
    try:
        if predict_type == 'solar':
            sr = test_feature[:, sr_col]
    except Exception as err:
        logs.error(str(err), exc_info=True)
    return predict_result_dataframe, sr, model_used


# 程凯
def predict_short_power(station_name, model_path, station_id, sr_col, predict_type, online_capacity, capacity,
                        forecast_time, best_model, second_model, predict_result_without_nwp, test_feature,
                        rate_for_transfer):
    """
    预测超短期功率（单次）
    :param station_name: 场站名称:
    :param station_id: 场站id:
    :param model_path: 文件路径:
    :param sr_col: 数据天数:
    :param predict_type: 预测类型:
    :param online_capacity: 开机容量:
    :param capacity: 装机容量:
    :param forecast_time: 时间:
    :param best_model: 预测的方法
    :param second_model: 预测的方法
    :param predict_result_without_nwp:
    :param test_feature:
    :param rate_for_transfer: 迁移学习的模型需要乘以系数，非迁移学习下该系数为1
    :return:
    """
    if len(test_feature) != 288:
        # 启动备胎计划，修改配置文件的标志位
        predict_result = predict_result_without_nwp
        predict_result_dataframe = pandas.DataFrame(predict_result.reshape(1, -1), index=[forecast_time])
        model_used = 'no_NWP'
        logs.warning(station_name + ' | ' + forecast_time + ' | 短期预测 | NWP缺失，转为备胎计划')
    else:
        # NWP完整
        try:
            try:
                model_name, state_name = best_model.split('_', 1)
                fitted_model = load_model(model_path + str(station_id) + '/short/' + model_name + '/'
                                          + 'short' + '_' + model_name + '.pkl')
                fitted_model = eval('fitted_model.predict_' + state_name)
                predict_result_raw = fitted_model(feature=test_feature, predict_type=predict_type, irradiance_col=0)
                model_used = best_model
                logs.info(station_name + ' | ' + forecast_time + ' | 短期预测 | ，采用最优模型：' + best_model)
            except Exception as err:
                logs.error(str(err), exc_info=True)
                model_name, state_name = second_model.split('_', 1)
                fitted_model = load_model(model_path + str(station_id) + '/short/' + model_name + '/'
                                          + 'short' + '_' + model_name + '.pkl')
                fitted_model = eval('fitted_model.predict_' + state_name)
                predict_result_raw = fitted_model(feature=test_feature, predict_type=predict_type, irradiance_col=0)
                model_used = second_model
                logs.info(station_name + ' | ' + forecast_time + ' | 短期预测 | ，采用次优模型：' + second_model)
            try:
                predict_result = data_postprocess.result_post_process(irradiance_threshold=10,
                                                                      pre_processing_predict_result=predict_result_raw,
                                                                      online_capacity=online_capacity,
                                                                      capacity=capacity,
                                                                      predict_type=predict_type,
                                                                      irradiance_col=0, feature=test_feature)
            except Exception as err:
                logs.error(str(err), exc_info=True)
                logs.warning(station_name + ' | ' + forecast_time + ' | 短期预测 | ，数据后处理出错！')
                predict_result = predict_result_raw
            # 如果是迁移学习，rate_for_transfer为实际容量除以源场站容量；如果不为迁移学习，rate_for_transfer为1
            predict_result_dataframe = pandas.DataFrame(predict_result.reshape(1, -1)*rate_for_transfer,
                                                        index=[forecast_time])
        except Exception as err:
            logs.error(str(err), exc_info=True)
            predict_result = predict_result_without_nwp
            predict_result_dataframe = pandas.DataFrame(predict_result.reshape(1, -1), index=[forecast_time])
            model_used = 'no_NWP'
            logs.warning(station_name + ' | ' + forecast_time + ' | 短期预测 | 模型预测出错，转为备胎计划！')
    sr = None  # 返回辐照度给区间预测后处理需要用到
    try:
        if predict_type == 'solar':
            sr = test_feature[:, sr_col]
    except Exception as err:
        logs.error(str(err), exc_info=True)
    return predict_result_dataframe, sr, model_used

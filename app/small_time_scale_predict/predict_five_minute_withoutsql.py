import numpy

from common.logger import logs
from common.tools import load_model
import pandas
import time
from numpy import zeros
from app.small_time_scale_predict.method_five_minute.NWP_interpolation import predict_with_NWP_interpolation_model
from app.small_time_scale_predict.method_five_minute.NWP_multidimensional_mapping import \
    predict_with_NWP_multidimensional_mapping_model
from app.small_time_scale_predict.method_five_minute.cloud_predict import predict_with_cloud_predict_model
from app.small_time_scale_predict.method_five_minute.history_power_iterative_predict import \
    predict_with_history_power_iterative_prediction_model
from app.small_time_scale_predict.method_five_minute.power_cubic_spline_interpolation import \
    predict_with_power_cubic_spline_interpolation_model
from app.small_time_scale_predict.method_five_minute.power_machinelearning_interpolation import \
    predict_with_power_machinelearning_interpolation_model
from app.small_time_scale_predict.method_five_minute.power_nearest_interpolation import \
    predict_with_power_nearest_interpolation_model
from common import data_postprocess
from task.mysql.small_time_scale.load_predict_data_iterate_five_minute import LoadPredictdataiterateFiveminute


def predict_five_minute_power_withoutsql(station_id, model_path, file_path, day, sr_col, predict_type,
                                          online_capacity, capacity,
                                          label, usecols, predict_time, fifth_model, fifteen_minute_model,
                                          data_resource, host, user,
                                          password, database, charset, port, sql):
    """
    预测5分钟分辨率功率（单次）
    :param station_name: 场站名称:
    :param model_path: 文件路径:
    :param file_path: 文件路径:
    :param day: 数据天数:
    :param sr_col: 数据天数:
    :param predict_type: 预测类型:
    :param online_capacity: 开机容量:
    :param capacity: 装机容量:
    :param label: 计数器:
    :param model_set_path: 预测配置文件:
    :param predict_time: 预报的时间，已转换成数字
    :param usecols: 预报的输入特征
    :param data_resource: 数据来源是‘CSV’或‘SQL’
    :param model_name: 预测的方法
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口
    :return:
    """
    # ------------------------------------------------------------------------------------------------------------------
    # logs.info('one_station_predict_five_minute_power')
    history_power_4_hours_ago = zeros((48, 48)) + online_capacity / 2  # 用容量的一半生成48个点作为过去4小时的历史功率
    test_data_load_five_minute = LoadPredictdataiterateFiveminute()
    if data_resource == 'CSV':
        # 从csv读取测试数据
        test_feature, history_power, forecast_time = test_data_load_five_minute.load_test_data_forpredict(
            file_path=file_path,
            nrows=288 * day + 1,
            usecols=usecols,
            forecast_time=3,
            predict_term="ultra_short",
            label=label,
            predict_type=predict_type)
    else:
        # 从MySQL读取测试数据
        test_feature, history_power = test_data_load_five_minute.upload_predict_nwp_five_minute(start_time=predict_time,
                                                                                                file_path=file_path,
                                                                                                usecols=usecols,
                                                                                                predict_type=predict_type,
                                                                                                host=host,
                                                                                                user=user,
                                                                                                password=password,
                                                                                                database=database,
                                                                                                charset=charset,
                                                                                                port=port,
                                                                                                online_capacity=online_capacity,
                                                                                                sql=sql)

        history_power_4_hours_ago = test_data_load_five_minute.upload_4_hours_ago_power(start_time=predict_time,
                                                                                        file_path=file_path, host=host,
                                                                                        user=user, password=password,
                                                                                        database=database,
                                                                                        charset=charset,
                                                                                        port=port,
                                                                                        online_capacity=online_capacity,
                                                                                        sql=sql)

        nwp_15min_row, nwp_15min_tcc = test_data_load_five_minute.upload_predict_nwp_five_minute_for_cloud_predict(start_time=predict_time,
                                                                                                file_path=file_path,
                                                                                                usecols=usecols,
                                                                                                predict_type=predict_type,
                                                                                                host=host,
                                                                                                user=user,
                                                                                                password=password,
                                                                                                database=database,
                                                                                                charset=charset,
                                                                                                port=port,
                                                                                                online_capacity=online_capacity,
                                                                                                sql=sql)
        # logs.info('after upload_4_hours_ago_power')
        if len(history_power_4_hours_ago) == 48:
            # print(size(history_power_4_hours_ago))
            # logs.info('before history_power_4_hours_ago = zeros((48, 48)) + numpy.array')
            history_power_4_hours_ago = zeros((48, 48)) + numpy.array(history_power_4_hours_ago)  # 扩展为每行相同的矩阵
            # logs.info('after history_power_4_hours_ago = zeros((48, 48)) + numpy.array')
            # logs.info(time.strftime("%Y-%m-%d %H:%M", time.localtime(float(predict_time))))
        else:
            history_power_4_hours_ago = zeros((48, 48)) + online_capacity / 2
            # logs.info('after history_power_4_hours_ago = zeros((48, 48)) + online_capacity / 2')
        # 起报时间当前时间，转换成年月日时分格式。
        forecast_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(predict_time)))
        # logs.info('forecast_time = time.strftime')
        # ------------------------------------------------------------------------------------------------------------------
    if len(test_feature) != 16:  # 如果数值天气预报不全
        # 启动5分钟时序法的备胎计划
        # 写日志
        # logs.info('write nwp lost log')
        # 使用时间序列法，利用过去4小时的5分钟分辨率的历史功率开展功率预测
        try:
            HPIP_model = load_model(
                model_path + str(station_id) + "/five_minute/" + str(station_id) + "_five_minute_HPIP_BPNN.pkl")
            all_prediction_results_raw = HPIP_model(feature=history_power_4_hours_ago)
            all_prediction_results = data_postprocess.result_post_process(irradiance_threshold=10,
                                                                          pre_processing_predict_result=all_prediction_results_raw,
                                                                          online_capacity=capacity,
                                                                          capacity=online_capacity,
                                                                          predict_type=predict_type,
                                                                          irradiance_col=sr_col, feature=test_feature)
            predict_result = all_prediction_results.reshape(48, 1)
            model_name = "HPIP" + "_" + fifth_model[2] + fifth_model[3]
        except Exception as err:
            logs.error(str(err), exc_info=True)
            # 运算错误
            logs.warning('5分钟预测任务失败！可能是实时NWP不全且无5分钟历史功率')

    else:
        # NWP正常情况下，开展正常的5分钟功率预测
        predict_result_raw = predict_raw(fifth_model=fifth_model, fifteen_minute_model=fifteen_minute_model,
                                         feature=test_feature, history_power=history_power,
                                         history_power_4_hours_ago=history_power_4_hours_ago,
                                         forecast_time=predict_time, nwp_15min_row=nwp_15min_row, nwp_15min_tcc=nwp_15min_tcc)
        # 结果后处理
        predict_result = data_postprocess.result_post_process(irradiance_threshold=10,
                                                              pre_processing_predict_result=predict_result_raw,
                                                              online_capacity=capacity,
                                                              capacity=online_capacity,
                                                              predict_type=predict_type,
                                                              irradiance_col=sr_col, feature=test_feature)

        predict_result_dataframe = pandas.DataFrame(predict_result.reshape(1, -1), index=[forecast_time])
        predict_result = predict_result_dataframe.values.T
        model_name = fifth_model[1] + "_" + fifth_model[2] + fifth_model[3]
    return predict_result, model_name


def predict_raw(fifth_model, fifteen_minute_model, feature, history_power, history_power_4_hours_ago, forecast_time,
                nwp_15min_row, nwp_15min_tcc):
    """
    单个电场5分钟功率预测
    :param five_minute_model: 5分钟最优模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻历史功率
    :param label: 时刻标志，0代表整点，1代表整点多5分钟，2代表整点多10分钟，3代表整点。整点为正常15分钟一个时刻。
    :param fifteen_minute_model: 15分钟最优功率
    :param history_power_for_method5: 方法5的历史功率，默认为前48个时刻的历史功率
    :param online_capacity: 开机容量
    :return:
    """

    if fifth_model[1] == "NMM":
        predict_result_raw = predict_with_NWP_multidimensional_mapping_model(five_minute_model=fifth_model[0],
                                                                             feature=feature,
                                                                             history_power=history_power)
    elif fifth_model[1] == "PCSI":
        predict_result_raw = predict_with_power_cubic_spline_interpolation_model(five_minute_model=fifth_model[0],
                                                                                 feature=feature,
                                                                                 history_power=history_power,
                                                                                 forecast_time=forecast_time)
    elif fifth_model[1] == "PNI":
        predict_result_raw = predict_with_power_nearest_interpolation_model(five_minute_model=fifth_model[0],
                                                                            feature=feature,
                                                                            history_power=history_power,
                                                                            forecast_time=forecast_time)
    elif fifth_model[1] == "NI":
        predict_result_raw = predict_with_NWP_interpolation_model(five_minute_model=fifth_model[0],
                                                                  feature=feature,
                                                                  history_power=history_power)
    elif fifth_model[1] == "PMI":
        predict_result_raw = predict_with_power_machinelearning_interpolation_model(
            five_minute_model=fifth_model[0],
            feature=feature,
            history_power=history_power,
            fifteen_minute_model=fifteen_minute_model, forecast_time=forecast_time)
    elif fifth_model[1] == "HPIP":
        predict_result_raw = predict_with_history_power_iterative_prediction_model(
            five_minute_model=fifth_model[0],
            history_power_4_hours_ago=history_power_4_hours_ago)
    elif fifth_model[1] == "CP":
        predict_result_raw = predict_with_cloud_predict_model(
            five_minute_model=fifth_model[0], nwp_15min_row=nwp_15min_row, nwp_15min_tcc=nwp_15min_tcc,
            history_power=history_power)
    else:
        predict_result_raw = zeros((48))
    return predict_result_raw

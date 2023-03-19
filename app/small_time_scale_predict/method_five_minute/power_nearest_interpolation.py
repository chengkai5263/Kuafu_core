import numpy
from numpy import linspace, zeros
from scipy import interpolate
from app.small_time_scale_predict.model_evaluation_five_minute import result_evaluation_five_minute
from common import data_postprocess


def nearest_interpolation(history_power=None, fifteenminutes_result=None):
    """
    最邻近插值
    :param history_power: 历史功率
    :param fifteenminutes_result: 15分钟分辨率的预测功率，共16个功率值
    :return: 返回插值后的5分钟分辨率的48个功率值
    """
    fiveminute_result = []
    x = linspace(0, 240, 17)
    y = numpy.append(history_power, fifteenminutes_result)
    xnew = linspace(5, 240, 48)
    f = interpolate.interp1d(x, y, kind="nearest")
    result1 = f(xnew)
    fiveminute_result.append(result1)

    x = linspace(0, 240, 17)
    y = numpy.append(history_power, fifteenminutes_result)
    xnew = linspace(10, 245, 48)
    f = interpolate.interp1d(x, y, kind="nearest", fill_value="extrapolate")
    result2 = f(xnew)
    fiveminute_result.append(result2)

    x = linspace(0, 240, 17)
    y = numpy.append(history_power, fifteenminutes_result)
    xnew = linspace(15, 250, 48)
    f = interpolate.interp1d(x, y, kind="nearest", fill_value="extrapolate")
    result3 = f(xnew)
    fiveminute_result.append(result3)

    return fiveminute_result


def nearest_interpolation_for_predict(history_power=None, fifteenminutes_result=None, forecast_time=0):
    """
    最邻近插值
    :param history_power: 历史功率
    :param fifteenminutes_result: 15分钟分辨率的预测功率，共16个功率值
    :param label: 时刻标志，0代表整点，1代表整点多5分钟，2代表整点多10分钟，3代表整点。整点为正常15分钟一个时刻。
    :return: 返回插值后的5分钟分辨率的48个功率值
    """
    if forecast_time % 180 == 0:
        x = linspace(0, 240, 17)
        y = numpy.append(history_power, fifteenminutes_result)
        xnew = linspace(5, 240, 48)
        f = interpolate.interp1d(x, y, kind="nearest")
        result1 = f(xnew)
        fiveminute_result = result1
    elif forecast_time % 180 == 120:
        x = linspace(0, 240, 17)
        y = numpy.append(history_power, fifteenminutes_result)
        xnew = linspace(10, 245, 48)
        f = interpolate.interp1d(x, y, kind="nearest", fill_value="extrapolate")
        result2 = f(xnew)
        fiveminute_result = result2
    elif forecast_time % 180 == 60:
        x = linspace(0, 240, 17)
        y = numpy.append(history_power, fifteenminutes_result)
        xnew = linspace(15, 250, 48)
        f = interpolate.interp1d(x, y, kind="nearest", fill_value="extrapolate")
        result3 = f(xnew)
        fiveminute_result = result3
    else:
        fiveminute_result = zeros((48))
    return fiveminute_result


def predict_with_power_nearest_interpolation_model(five_minute_model, feature, history_power, forecast_time):
    """
    使用最邻近插值法的模型（先预测15分钟分辨率功率，再使用最邻近插值法插值）开展预测
    :param five_minute_model: 在该方法中，传入的模型实质为15分钟预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    fifteen_minute_model = five_minute_model
    # 15分钟功率预测
    fifteenminutes_result = fifteen_minute_model(feature=feature, history_power=history_power)
    # 插值
    predict_result_raw = nearest_interpolation_for_predict(
        history_power=history_power, fifteenminutes_result=fifteenminutes_result, forecast_time=forecast_time)
    return predict_result_raw


def test_PNI(config, test_feature, test_target, fifteen_model):
    """
    使用最邻近插值法的模型（先预测15分钟分辨率功率，再使用最邻近插值法插值）开展模型测试
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # 每次用于预测的特征集数据长度。默认为超短期功率预测时的特征集数据长度
    feature_len = 16
    # 前后两次预测的（特征集数据）间隔。默认为超短期功率预测时的特征集数据间隔
    interval = 1
    test_feature_data = test_feature

    # 获取测试集功率预测结果
    predict_result_group = []

    # 因为要输入历史功率，故需丢弃特征集的第一行数据
    start_index = 1
    end_index = start_index + feature_len
    max_index = len(test_feature_data)
    while end_index <= max_index:
        feature = test_feature_data[start_index:end_index]
        history_power = test_target[3 * (start_index - 1) + 1]
        predict_result = fifteen_model(feature=feature, history_power=history_power)
        predict_result = nearest_interpolation(history_power=history_power, fifteenminutes_result=predict_result)
        predict_result[0] = data_postprocess.result_post_process(irradiance_threshold=10,
                                                                 pre_processing_predict_result=predict_result[0],
                                                                 online_capacity=config['online_capacity'],
                                                                 capacity=config['online_capacity'],
                                                                 predict_type=config['type'],
                                                                 irradiance_col=config['sr_col'], feature=feature)
        predict_result[1] = data_postprocess.result_post_process(irradiance_threshold=10,
                                                                 pre_processing_predict_result=predict_result[1],
                                                                 online_capacity=config['online_capacity'],
                                                                 capacity=config['online_capacity'],
                                                                 predict_type=config['type'],
                                                                 irradiance_col=config['sr_col'], feature=feature)
        predict_result[2] = data_postprocess.result_post_process(irradiance_threshold=10,
                                                                 pre_processing_predict_result=predict_result[2],
                                                                 online_capacity=config['online_capacity'],
                                                                 capacity=config['online_capacity'],
                                                                 predict_type=config['type'],
                                                                 irradiance_col=config['sr_col'], feature=feature)
        if predict_result is not None:
            predict_result_group.append(predict_result[0])
            predict_result_group.append(predict_result[1])
            predict_result_group.append(predict_result[2])

        start_index += interval
        end_index = start_index + feature_len

    feature_len = 48
    end_index = interval * (len(predict_result_group) - 1) + feature_len + 1
    actual_result = test_target[1:end_index]

    test_target_data = []
    predict_step = len(predict_result_group[0])
    inteval = 1
    for i in range(len(predict_result_group)):
        test_target_data.append(actual_result[i * inteval: i * inteval + predict_step])

    five_minute_accuracy = result_evaluation_five_minute(predict_result_group, test_target_data, config['online_capacity'],
                                                         predict_term="five_minute_ultra_short")
    # 结果统计分析
    # generate_page_five_minute(predict_result_group, test_target_data, station_name, five_minute_accuracy, method="PNI",
    #                          save_page=True)

    return predict_result_group, five_minute_accuracy
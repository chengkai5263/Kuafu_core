import numpy
from numpy import zeros, vstack, array
from app.small_time_scale_predict.model_evaluation_five_minute import result_evaluation_five_minute
from common.tools import load_model
from common import data_postprocess


def extract_15minutes_power(train_target_data):
    i = 0
    power_interpolation_train_feature_data = numpy.zeros(shape=(int(len(train_target_data) / 3)))
    while i < (len(train_target_data) / 3):
        power_interpolation_train_feature_data[i] = train_target_data[3 * i]
        i = i + 1
    return power_interpolation_train_feature_data


def expansion_data(raw_data=None):
    """
    待复制扩展数据扩展为与5分钟功率一一对应的数据，一个15分钟的数据扩展为前后5分钟三个相同的数据
    :param raw_data:待复制扩展的数据
    :return:扩展后的数据，分辨率为5分钟
    """
    feature_data_for_5minute = zeros((len(raw_data) * 3, len(raw_data[0, :])))
    in_end = 0
    while in_end in range(len(raw_data)):
        feature_data_for_5minute[3 * in_end] = raw_data[in_end, :]
        feature_data_for_5minute[3 * in_end + 1] = raw_data[in_end, :]
        feature_data_for_5minute[3 * in_end + 2] = raw_data[in_end, :]
        in_end = in_end + 1
    return feature_data_for_5minute


def fit_power_machinelearning_interpolation_model(target_data, model_name, irradiance_col, predict_model, predict_type,
                                                  best_paprameter, predict_term="five_minute_ultra_short"):
    """
    获取功率数据机器学习插值法的模型（先预测15分钟分辨率功率，再用机器学习模型插值生成5分钟预测功率）及其预测精度
    :param target_data: 实际功率（分辨率5分钟）
    :param irradiance_col: 辐照度所在列的序号
    :param predict_model: 待训练模型
    :param predict_type: 预测类型，wind是风电，solar是光伏
    :param predict_term: 功率预测时期，short为短期预测，ultra_short为超短期预测,five_minute_ultra_short为5分钟超短期
    :return: predict_model: 预测模型（仅保留不用历史功率的模型.predict_without_history_power）
    """
    print("功率数据机器学习插值法-power_machinelearning_interpolation")
    # if model_name == "BLSTM":
    #     predict_model = BLSTM(input_size=2, params=best_paprameter[model_name])
    # if model_name == "BNN":
    #     predict_model = BNN(input_size=2, params=best_paprameter[model_name])
    # if model_name == "LSTM":
    #     predict_model = LSTM(n_features=1, params=best_paprameter[model_name])
    # 抽取15分钟功率
    fifteen_minute_train_target = extract_15minutes_power(target_data)
    # 对15分钟功率进行复制扩展为相邻两个5分钟功率
    power_interpolation_train_feature_data = expansion_data(
        fifteen_minute_train_target.reshape(len(fifteen_minute_train_target), 1))
    # 模型训练
    # if predict_type == 'solar':
    predict_model.fit(feature_data=power_interpolation_train_feature_data, target_data=target_data, predict_term=None,
                      epochs=20, batch_size=1440, predict_type=predict_type, irradiance_col=irradiance_col,
                      without_history_power=True)
    # else:
        # predict_model.fit(feature_data=power_interpolation_train_feature_data, target_data=target_data, predict_term=predict_term,
        #                   epochs=20, batch_size=1440, predict_type=predict_type, irradiance_col=irradiance_col,
        #                   without_history_power=False)
    return predict_model.predict_without_history_power


def predict_with_power_machinelearning_interpolation_model(five_minute_model, feature, history_power,
                                                           fifteen_minute_model, forecast_time):
    """
    使用功率数据机器学习插值法的模型（先预测15分钟分辨率功率，再用机器学习模型插值生成5分钟预测功率）开展预测
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :param fifteen_minute_model:15分钟功率预测模型
    :param label: 时刻标志，0代表整点，1代表整点多5分钟，2代表整点多10分钟，3代表整点。整点为正常15分钟一个时刻。
    :return:后处理前的预测功率
    """
    # 15分钟预测
    fifteen_minute_result = fifteen_minute_model(feature=feature, history_power=history_power)
    # 15分钟功率值复制扩展
    fifteenminute_result = expansion_data(
        fifteen_minute_result.reshape(len(fifteen_minute_result), 1))
    # 根据不同时刻5分钟插值
    if forecast_time % 180 == 0:
        predict_result_raw = five_minute_model(feature=vstack((array(history_power), fifteenminute_result))[:-1])
    elif forecast_time % 180 == 120:
        predict_result_raw = five_minute_model(feature=fifteenminute_result)
    elif forecast_time % 180 == 60:
        predict_result_raw = five_minute_model(feature=vstack((fifteenminute_result[1:], fifteenminute_result[47, 0])))
    else:
        predict_result_raw = zeros((48))
    return predict_result_raw


def test_PMI(config, test_feature, station_id, test_target, fifteen_model, model_name='BPNN_without_history_power'):
    """
    使用功率数据机器学习插值法的模型（先预测15分钟分辨率功率，再用机器学习模型插值生成5分钟预测功率）开展模型测试
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # 加载PMI模型
    PMI_model = load_model(
        config[
            'model_savepath'] + str(station_id) + "/five_minute/" + "five_minute_PMI_" + model_name
        + ".pkl")
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
        with_history = []
        feature = test_feature_data[start_index:end_index]
        history_power = test_target[3 * (start_index - 1) + 1]
        # 使用15分钟功率预测模型预测15分钟分辨率的功率
        fifteenminute_result = fifteen_model(feature=feature, history_power=history_power)
        # 5分钟功率预测
        # 有历史功率的15分钟功率预测值的拓展
        fifteenminute_result = expansion_data(
            fifteenminute_result.reshape(len(fifteenminute_result), 1))
        # 使用5分钟功率预测模型进行5分钟分辨率的功率预测
        with_history.append(PMI_model(
            feature=vstack((array(history_power), fifteenminute_result))[:-1]))
        with_history.append(PMI_model(feature=fifteenminute_result))
        with_history.append(PMI_model(
            feature=vstack((fifteenminute_result[1:], fifteenminute_result[47, 0]))))

        with_history[0] = data_postprocess.result_post_process(irradiance_threshold=10,
                                                               pre_processing_predict_result=with_history[0],  # 预测结果
                                                               online_capacity=config['online_capacity'],  # 开机容量
                                                               capacity=config['online_capacity'],
                                                               irradiance_col=config['sr_col'], feature=feature,
                                                               predict_type=config['type'])

        with_history[1] = data_postprocess.result_post_process(irradiance_threshold=10,
                                                               pre_processing_predict_result=with_history[1],  # 预测结果
                                                               online_capacity=config['online_capacity'],  # 开机容量
                                                               capacity=config['online_capacity'],
                                                               irradiance_col=config['sr_col'], feature=feature,
                                                               predict_type=config['type'])

        with_history[2] = data_postprocess.result_post_process(irradiance_threshold=10,
                                                               pre_processing_predict_result=with_history[2],  # 预测结果
                                                               online_capacity=config['online_capacity'],  # 开机容量
                                                               capacity=config['online_capacity'],
                                                               irradiance_col=config['sr_col'], feature=feature,
                                                               predict_type=config['type'])
        if with_history is not None:
            predict_result_group.append(with_history[0])
            predict_result_group.append(with_history[1])
            predict_result_group.append(with_history[2])

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
    # generate_page_five_minute(predict_result_group, test_target_data, station_name, five_minute_accuracy, method="PMI", model_name=model_name, save_page=True)
    return predict_result_group, five_minute_accuracy
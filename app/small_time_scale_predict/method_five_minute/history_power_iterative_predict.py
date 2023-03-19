import numpy
from numpy import zeros

from app.small_time_scale_predict.model_evaluation_five_minute import result_evaluation_five_minute
from common.tools import load_model
from common import data_postprocess


def generate_power_prediction_training_set(train_target_data, input_size):
    """
    生成两组1×4的历史功率作为输入，未来两个时刻的功率作为输出。
    :param train_target_data:
    :param input_size:
    :return:
    """
    iterative_prediction_train_feature_data = zeros((len(train_target_data) - input_size, input_size))
    i = 0
    while i < (len(train_target_data) - input_size - 48):
        iterative_prediction_train_feature_data[i, :] = train_target_data[i: input_size + i]  # 把要预测的48个功率值的前48个历史功率作为每一组的特征量
        i = i + 1
    iterative_prediction_train_target_data = train_target_data[input_size:]
    return iterative_prediction_train_feature_data, iterative_prediction_train_target_data


def fit_history_power_iterative_predict_model(target_data, model_name, irradiance_col, predict_model, predict_type,
                                              best_paprameter, predict_term="five_minute_ultra_short"):
    """
    获取历史功率时序预测法的模型（纯用历史4小时共48个时刻的功率预测未来4小时功率）及其预测精度
    :param target_data: 实际功率（分辨率5分钟）
    :param irradiance_col: 辐照度所在列的序号
    :param predict_model: 待训练模型
    :param predict_type: 预测类型，wind是风电，solar是光伏
    :param predict_term: 功率预测时期，short为短期预测，ultra_short为超短期预测,five_minute_ultra_short为5分钟超短期
    :return: predict_model: 预测模型（仅保留不用历史功率的模型.predict_without_history_power）
    """
    print("历史功率时序预测法-iterative_prediction_by_history_power")
    # if model_name == "BLSTM":
    #     predict_model = BLSTM(input_size=49, params=best_paprameter[model_name])
    # if model_name == "BNN":
    #     predict_model = BNN(input_size=49, params=best_paprameter[model_name])
    # if model_name == "LSTM":
    #     predict_model = LSTM(n_features=48, params=best_paprameter[model_name])
    # 生成功率预测训练集
    input_size = 48  # 输入的历史功率长度
    # 构造输入为48个历史功率（过去4小时），输出为未来48个时刻（未来4小时）的输入/输出对应训练集
    iterative_prediction_train_feature_data, iterative_prediction_train_target_data = \
        generate_power_prediction_training_set(target_data, input_size=input_size)
    # 模型训练
    # if predict_type == 'solar':
    predict_model.fit(feature_data=iterative_prediction_train_feature_data,
                      target_data=iterative_prediction_train_target_data.flatten(), predict_term=None,
                      epochs=20, batch_size=1440, predict_type=predict_type, irradiance_col=irradiance_col,
                      without_history_power=True)
    # else:
    #     predict_model.fit(feature_data=iterative_prediction_train_feature_data,
    #                       target_data=iterative_prediction_train_target_data.flatten(), predict_term=predict_term,
    #                       epochs=20, batch_size=1440, predict_type=predict_type, irradiance_col=irradiance_col,
    #                       without_history_power=False)

    return predict_model.predict_without_history_power


def predict_with_history_power_iterative_prediction_model(five_minute_model, history_power_4_hours_ago):
    """
    使用历史功率时序预测法的模型（纯用历史4小时共48个时刻的功率预测未来4小时功率）开展预测
    :param five_minute_model: 5分钟功率预测模型
    :param history_power_for_method5:前48个时刻的历史功率
    :return:后处理前的预测功率
    """
    predict_result_raw = five_minute_model(feature=history_power_4_hours_ago)
    return predict_result_raw


def test_HPIP(config, station_id, test_target, train_target, model_name='BPNN_without_history_power'):
    """
    使用历史功率时序预测法的模型（纯用历史4小时共48个时刻的功率预测未来4小时功率）开展模型测试
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # 加载HPIP模型
    HPIP_model = load_model(
        config[
            'model_savepath'] + str(station_id) + "/five_minute/" + "five_minute_HPIP_" + model_name
        + ".pkl")
    input_size = 48  # 输入的历史功率长度
    # 每次用于预测的特征集数据长度。默认为超短期功率预测时的特征集数据长度
    feature_len = 48
    # 前后两次预测的（特征集数据）间隔。默认为超短期功率预测时的特征集数据间隔
    interval = 1

    # 获取测试集功率预测结果
    predict_result_group = []

    # 因为要输入历史功率，故需丢弃特征集的第一行数据
    start_index = 1
    end_index = start_index + feature_len
    max_index = len(test_target)
    # all_history_power = numpy.vstack((train_target[-47:], test_target))
    all_history_power = numpy.append(train_target[-47:], test_target)
    while end_index <= max_index:
        without_history = []
        # i = 0
        # history_power = zeros((48, 48))
        # while i < 48:
            # history_power[i, :] = all_history_power[start_index - 1:start_index - 1 + input_size].flatten()
        history_power = all_history_power[start_index - 1:start_index - 1 + input_size].reshape(48, 1) * numpy.ones((48, 48))
            # i = i + 1
        without_history.append(HPIP_model(feature=history_power))
        without_history = data_postprocess.result_post_process(irradiance_threshold=10,
                                                               pre_processing_predict_result=without_history[0],
                                                               online_capacity=config['online_capacity'],  # 开机容量
                                                               capacity=config['online_capacity'],
                                                               irradiance_col=config['sr_col'],
                                                               feature="wind",
                                                               predict_type="wind"
                                                               )
        predict_result_group.append(without_history)
        start_index += interval
        end_index = start_index + feature_len


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
    # generate_page_five_minute(predict_result_group, test_target_data, station_name, five_minute_accuracy, method="HPIP", model_name=model_name, save_page=True)

    return predict_result_group, five_minute_accuracy
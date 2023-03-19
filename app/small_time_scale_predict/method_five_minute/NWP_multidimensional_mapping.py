from numpy import zeros
from app.small_time_scale_predict.model_evaluation_five_minute import result_evaluation_five_minute
from common.tools import load_model
from common import data_postprocess


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


def fit_NWP_multidimensional_mapping_model(feature_data, target_data, irradiance_col, predict_model, predict_type,
                                           predict_term="five_minute_ultra_short"):
    """
    获取NWP多点功率映射法的模型（1个15分钟NWP预测3个5分钟功率）及其预测精度
    :param feature_data: NWP(分辨率15分钟)
    :param target_data: 实际功率（分辨率5分钟）
    :param irradiance_col: 辐照度所在列的序号
    :param predict_model: 待训练模型
    :param predict_type: 预测类型，wind是风电，solar是光伏
    :param predict_term: 功率预测时期，short为短期预测，ultra_short为超短期预测,five_minute_ultra_short为5分钟超短期
    :return: predict_model: 预测模型
    """
    print("训练-NWP多点功率映射法-NWP_multidimensional_mapping")
    # NWP复制扩展，一条NWP数据复制为前后相邻5分钟的NWP
    train_feature_data = expansion_data(raw_data=feature_data)
    # 模型训练
    # if predict_type == 'solar':
    predict_model.fit(feature_data=train_feature_data, target_data=target_data, predict_term=None,
                      epochs=20, batch_size=1440, predict_type=predict_type, irradiance_col=irradiance_col,
                      without_history_power=True)
    # else:
    #     predict_model.fit(feature_data=train_feature_data, target_data=target_data, predict_term=predict_term,
    #                       epochs=20, batch_size=1440, predict_type=predict_type, irradiance_col=irradiance_col,
    #                       without_history_power=False)

    return predict_model


def predict_with_NWP_multidimensional_mapping_model(five_minute_model, feature, history_power):
    """
    使用NWP多点功率映射法（1个15分钟NWP预测3个5分钟功率）开展预测
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # NWP数据拓展
    feature = expansion_data(raw_data=feature)
    # 5分钟功率预测
    predict_result_raw = five_minute_model(feature=feature, history_power=history_power)
    return predict_result_raw


def test_NMM(config, test_feature, station_id, test_target, model_name='BPNN_without_history_power'):
    """
    使用NWP多点功率映射法（1个15分钟NWP预测3个5分钟功率）开展预测
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # 加载NMM模型
    NMM_model = load_model(
        config[
            'model_savepath'] + str(station_id) + "/five_minute/" + "five_minute_NMM_" + model_name
        + ".pkl")
    # NWP复制扩展
    test_feature = expansion_data(raw_data=test_feature)
    # 每次用于预测的特征集数据长度。默认为超短期功率预测时的特征集数据长度
    feature_len = 48
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
        feature = test_feature_data[start_index:end_index]  # 未来48个时刻的NWP
        history_power = test_target[start_index - 1]  # 前一个时刻的历史功率
        predict_result_raw = NMM_model(feature, history_power)  # 使用模型预测功率
        predict_result = data_postprocess.result_post_process(irradiance_threshold=10,  # 结果后处理
                                                              pre_processing_predict_result=predict_result_raw,
                                                              online_capacity=config['online_capacity'],
                                                              capacity=config['online_capacity'],
                                                              predict_type=config['type'],
                                                              irradiance_col=config['sr_col'], feature=feature)
        predict_result_group.append(predict_result)  # 添加结果至list
        start_index += interval
        end_index = start_index + feature_len

    end_index = interval * (len(predict_result_group) - 1) + feature_len + 1
    actual_result = test_target[1:end_index]

    test_target_data = []
    predict_step = len(predict_result_group[0])
    inteval = 1
    for i in range(len(predict_result_group)):  # 把真实功率拆成跟预测功率一样的48个值一组
        test_target_data.append(actual_result[i * inteval: i * inteval + predict_step])
    # 误差评价，48个数值算均方根误差
    five_minute_accuracy = result_evaluation_five_minute(predict_result_group, test_target_data, config['online_capacity'],
                                                         predict_term="five_minute_ultra_short")
    # 结果统计分析
    # generate_page_five_minute(predict_result_group, test_target_data, station_name, five_minute_accuracy, method="NMM", model_name=model_name, save_page=True)

    return predict_result_group, five_minute_accuracy

from numpy import zeros, linspace
from scipy import interpolate

from app.small_time_scale_predict.model_evaluation_five_minute import result_evaluation_five_minute
from common.tools import load_model
from common import data_postprocess


def interpolate_NWP(feature_data=None):
    """
    对15分钟的NWP插值，生成5分钟分辨率的NWP
    :param feature_data: NWP
    :return: 5分钟分辨率的NWP
    """
    # 构造5分钟级NWP
    feature_data_for_5minute = zeros((len(feature_data) * 3, len(feature_data[0, :])))
    in_end = 0
    while in_end in range(len(feature_data[0, :])):
        x = linspace(0, (len(feature_data) - 1) * 15, len(feature_data))
        y = feature_data[:, in_end]
        xnew = linspace(-5, (len(feature_data) - 1) * 15 + 5, len(feature_data) * 3)
        f = interpolate.interp1d(x, y, kind="cubic", fill_value="extrapolate")
        feature_data_for_5minute[:, in_end] = f(xnew)
        in_end = in_end + 1

    return feature_data_for_5minute


def fit_NWP_interpolation_model(feature_data, target_data, irradiance_col, predict_model, predict_type,
                                predict_term="five_minute_ultra_short"):
    """
    获取NWP插值法的模型（插值生成5分钟分辨率的NWP）及其预测精度
    :param feature_data: NWP(分辨率15分钟)
    :param target_data: 实际功率（分辨率5分钟）
    :param irradiance_col: 辐照度所在列的序号
    :param predict_model: 待训练模型
    :param predict_type: 预测类型，wind是风电，solar是光伏
    :param predict_term: 功率预测时期，short为短期预测，ultra_short为超短期预测,five_minute_ultra_short为5分钟超短期
    :return: predict_model: 预测模型
    """
    print("NWP插值法-NWP_interpolation")
    # NWP插值,使用三次条样插值法对15分钟的NWP插值生成5分钟分辨率的NWP
    train_feature_data = interpolate_NWP(feature_data=feature_data)
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


def predict_with_NWP_interpolation_model(five_minute_model, feature, history_power):
    """
    使用NWP插值法的模型（插值生成5分钟分辨率的NWP）开展预测
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # NWP插值
    train_feature_data = interpolate_NWP(feature_data=feature)
    # 5分钟功率预测
    predict_result_raw = five_minute_model(feature=train_feature_data, history_power=history_power)
    return predict_result_raw


def test_NI(config, test_feature, station_id, test_target, model_name='BPNN_without_history_power'):
    """
    使用NWP插值法的模型（插值生成5分钟分辨率的NWP）开展模型测试
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # 加载NMM模型
    NI_model = load_model(
        config[
            'model_savepath'] + str(station_id) + "/five_minute/" + "five_minute_NI_" + model_name
        + ".pkl")

    # 对15分钟的NWP插值，生成5分钟分辨率的NWP
    test_feature = interpolate_NWP(feature_data=test_feature)
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
        feature = test_feature_data[start_index:end_index]
        history_power = test_target[start_index - 1]
        predict_result_raw = NI_model(feature, history_power)
        predict_result = data_postprocess.result_post_process(irradiance_threshold=10,
                                                              pre_processing_predict_result=predict_result_raw,
                                                              online_capacity=config['online_capacity'],
                                                              capacity=config['online_capacity'],
                                                              predict_type=config['type'],
                                                              irradiance_col=config['sr_col'], feature=feature)
        predict_result_group.append(predict_result)
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
    # generate_page_five_minute(predict_result_group, test_target_data, station_name, five_minute_accuracy, method="NI", model_name=model_name, save_page=True)

    return predict_result_group, five_minute_accuracy
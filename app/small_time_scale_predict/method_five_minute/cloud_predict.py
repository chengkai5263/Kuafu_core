import numpy
import pandas

from app.small_time_scale_predict.model_evaluation_five_minute import result_evaluation_five_minute
from common.tools import load_model
from common import data_postprocess


def interpolate_nwp(nwp_15min_row, nwp_15min_tcc, n=7):
    """
    生成光伏5分钟的NWP数据
    :param nwp_15min_row: 光伏15分钟的NWP数据。
    :param n: NWP网格边长，一般为奇数，且大于5。
    :return: NWP_5min: 光伏5分钟的NWP数据。
    """
    nwp_15min = pandas.DataFrame(nwp_15min_row)
    nwp_15min_tcc = pandas.DataFrame(nwp_15min_tcc)
    # wg网格中的格子数
    wg = n * n
    # 用于匹配的网格边长
    m = n - 4
    num = int((wg - 1) / 2)
    nwp_5min = numpy.zeros((int(nwp_15min_tcc.shape[0] / wg) * 3, nwp_15min.shape[1]))
    for i in range(int(nwp_15min_tcc.shape[0] / wg) - 1):
        # 取方块
        nwp_15min_tcc_wg1 = nwp_15min_tcc.values[i * wg: i * wg + wg].reshape(n, n)
        nwp_15min_tcc_wg2 = nwp_15min_tcc.values[i * wg + wg: i * wg + 2 * wg].reshape(n, n)
        nwp_15min_tcc_wg1_central = nwp_15min_tcc_wg1[int((n - m) / 2):int((n + m) / 2), int((n-m)/2):int((n + m) / 2)]
        # 先用插值法填充
        nwp_5min[3 * i, :] = nwp_15min.values[wg * i + num, :]
        nwp_5min[3 * i + 1, :] = nwp_15min.values[wg * i + num, :] + (
                nwp_15min.values[wg * (i + 1) + num, :] - nwp_15min.values[wg * i + num, :]) / 3
        nwp_5min[3 * i + 2, :] = nwp_15min.values[wg * i + num, :] + 2 * (
                nwp_15min.values[wg * (i + 1) + num, :] - nwp_15min.values[wg * i + num, :]) / 3

        # 判断是否需要快匹配
        if nwp_15min_tcc_wg1_central.sum(axis=0).sum(axis=0) > -0.01:
            # 计算方向
            table = numpy.zeros((n - m + 1, n - m + 1))
            for j in range(n - m + 1):
                for k in range(n - m + 1):
                    table[j, k] = numpy.square(nwp_15min_tcc_wg2[j:j + m, k:k + m] - nwp_15min_tcc_wg1_central).sum(
                        axis=0).sum(axis=0)
            whe = table.argmin()
            wheh = whe // n - m + 1
            whel = whe % n - m + 1
            # x，y为方向
            whe5h = int((n - 1) / 2 - ((n - 1) / 2 - wheh) / 3)
            whe5l = int((n - 1) / 2 - ((n - 1) / 2 - whel) / 3)
            whe10h = int((n - 1) / 2 - 2 * ((n - 1) / 2 - wheh) / 3)
            whe10l = int((n - 1) / 2 - 2 * ((n - 1) / 2 - whel) / 3)
            nwp_5min[3 * i, :] = nwp_15min.values[wg * i + num, :]
            nwp_5min[3 * i + 1, :] = nwp_15min.values[wg * i + (n - m + 1) * (whe5h + 2) + 2 + whe5l, :]
            nwp_5min[3 * i + 2, :] = nwp_15min.values[wg * i + (n - m + 1) * (whe10h + 2) + 2 + whe10l, :]
    return nwp_5min


def fit_cloud_predict_model(nwp_15min_row, nwp_15min_tcc, target_data, irradiance_col, predict_model, predict_type,
                            predict_term="five_minute_ultra_short"):
    """
    获取云团追踪预测模型
    :param feature_data: NWP(分辨率15分钟)
    :param target_data: 实际功率（分辨率5分钟）
    :param irradiance_col: 辐照度所在列的序号
    :param predict_model: 待训练模型
    :param predict_type: 预测类型，wind是风电，solar是光伏
    :param predict_term: 功率预测时期，short为短期预测，ultra_short为超短期预测,five_minute_ultra_short为5分钟超短期
    :return: predict_model: 预测模型
    """
    print("云层迁移法-cloud_interpolation")
    # NWP预测
    train_feature_data = interpolate_nwp(nwp_15min_row=nwp_15min_row, nwp_15min_tcc=nwp_15min_tcc)
    # 模型训练
    predict_model.fit(feature_data=train_feature_data, target_data=target_data, predict_term=None,
                      epochs=20, batch_size=1440, predict_type=predict_type, irradiance_col=irradiance_col,
                      without_history_power=True)
    return predict_model


def predict_with_cloud_predict_model(five_minute_model, nwp_15min_row, nwp_15min_tcc, history_power):
    """
    使用云团追踪的模型（云团追踪插值生成5分钟分辨率的NWP）开展预测
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # NWP插值
    train_feature_data = interpolate_nwp(nwp_15min_row=nwp_15min_row, nwp_15min_tcc=nwp_15min_tcc)
    # 5分钟功率预测
    predict_result_raw = five_minute_model(feature=train_feature_data, history_power=history_power)
    return predict_result_raw


def test_CP(config, test_feature, station_name, test_target, model_name='BPNN_without_history_power'):
    """
    使用云团追踪的模型（云团追踪插值生成5分钟分辨率的NWP）开展模型测试
    :param five_minute_model: 5分钟功率预测模型
    :param feature: NWP特征量
    :param history_power: 前一个时刻的历史功率
    :return:后处理前的预测功率
    """
    # 加载NMM模型
    pass

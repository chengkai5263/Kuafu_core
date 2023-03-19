
import pandas
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from common.tools import load_model
from common import data_postprocess
import datetime
import time


class FeatureSelection:
    """
    特征选取模型封装类
    """
    def __init__(self):
        # 初始化类
        self.features_selected = None

    def feature_select_by_rf(self, feature_data=None, target_data=None, threshold=0.05, usecols=None):
        """
        基于随机森林筛选特征量
        :param feature_data: 特征集（NWP天气数据, 参数类型为ndarray, 与预测模型训练时所用数据集相同
        :param target_data: 目标集（功率数据）, 用于随机森林模型的训练, 参数类型为ndarray, 与预测模型训练时所用数据集相同
        :param threshold: 特征重要性评分阈值, 对每个特征进行重要性评分, 高于阈值的特征将会被选取用于下一步的预测模型训练,
                重要性评分基于基尼指数（Gini index）错误率，用于衡量单个特征对模型预测结果影响的大小
        :param usecols: 加载特征集数据时, 所要筛选的特征, 默认全选
        :return: self.features_selected 根据随机森林模型的特征重要性评分和阈值, 筛选出的特征值, 参数类型为list,
                list中的单个元素为String
        """
        rf_random = RandomForestRegressor(bootstrap=True, criterion='squared_error', max_depth=10,
                                          max_features='sqrt', min_samples_leaf=10,
                                          min_samples_split=4, n_estimators=200, n_jobs=-1)
        rf_random.fit(feature_data, target_data)
        best_feature = []
        feat_imps = pandas.DataFrame(rf_random.feature_importances_, index=usecols,
                                     columns=['Importance score']).sort_values('Importance score', ascending=False)
        feat_imps = feat_imps.reset_index()
        feat_imps.columns = ["Feature", "Importance Score"]
        scores = rf_random.feature_importances_
        index = numpy.array(usecols)
        for i in range(len(scores)):
            if scores[i] >= threshold:
                best_feature.append(index[i])
        self.features_selected = best_feature

    def feature_select_by_correlation_withinitialdata(self, usecols, feature_data=None, target_data=None):
        """
        利用原始NWP数据并基于相关性分析筛选特征量
        :param feature_data: 特征集（NWP天气数据, 参数类型为ndarray, 与预测模型训练时所用数据集相同
        :param target_data: 目标集（功率数据）, 用于随机森林模型的训练, 参数类型为ndarray, 与预测模型训练时所用数据集相同
        :param usecols: 加载特征集数据时, 所要筛选的特征, 默认全选
        :return: None
        """
        target_data = numpy.reshape(target_data, (target_data.shape[0], 1))
        train_data = numpy.hstack((feature_data, target_data))
        usecols.append('power')
        train_data = pandas.DataFrame(train_data, columns=usecols)
        corrtarget = train_data.corr()
        diccorrelations = dict(corrtarget['power'])
        for k, v in diccorrelations.items():
            if k == 'power':
                pass
            elif v < 0:
                diccorrelations[k] = abs(v)
        import heapq
        # 选取相关性最大的前6个（包含功率power）特征量
        self.features_selected = heapq.nlargest(6, diccorrelations, key=diccorrelations.get)
        self.features_selected.remove('power')
        usecols.remove('power')

    def feature_select_by_rfe(self, feature_type, usecols, feature_data=None, target_data=None, feature_num=5):
        """
        基于递归消除筛选特征量
        :param feature_data: 特征集（NWP天气数据, 参数类型为ndarray, 与预测模型训练时所用数据集相同
        :param target_data: 目标集（功率数据）, 用于随机森林模型的训练, 参数类型为ndarray, 与预测模型训练时所用数据集相同
        :param feature_type: 特征重类别
        :param usecols: 加载特征集数据时, 所要筛选的特征, 默认全选
        :param feature_num: 特征个数
        :return: self.features_selected 根据随机森林模型的特征重要性评分和阈值, 筛选出的特征值, 参数类型为list,
                list中的单个元素为String
        """
        clf_rf = RandomForestRegressor()
        rfe = RFE(estimator=clf_rf, n_features_to_select=feature_num)
        feature_data = pandas.DataFrame(feature_data, columns=usecols)
        target_data = pandas.DataFrame(target_data)
        target_data = target_data.values.ravel()
        rfe.fit(feature_data, target_data)
        col_list = feature_data.columns[rfe.support_].tolist()
        if feature_type == 'solar':
            if 'SR' not in col_list:
                col_list.append('SR')
            if 'TCC' not in col_list:
                col_list.append('TCC')
            if 'HCC' not in col_list:
                col_list.append('HCC')
        if col_list == 'wind':
            if col_list[0] is None:
                col_list.append('WS')
        for i in range(len(col_list)):
            if col_list[i] == 'SR':
                col_list[0], col_list[i] = col_list[i], col_list[0]

        self.features_selected = col_list


# 随机森林特征选取
def select_feature_by_rf(usecols, threshold=0.05, feature_type="wind", train_feature=None, train_target=None):
    """
    :param train_feature: 特征集（NWP天气数据, 参数类型为ndarray, 与预测模型训练时所用数据集相同
    :param train_target: 目标集（功率数据）, 用于随机森林模型的训练, 参数类型为ndarray, 与预测模型训练时所用数据集相同
    :param usecols: 加载特征集数据时, 所要筛选的特征, 默认全选
    :param feature_type: 特征类别
    :param threshold: 随机森林模型的门槛值，低于门槛值的特征将会被舍弃，默认为0.05
    :return: self.features_selected 根据递归特征消除筛选出的特征值, 类型为list, 例如:['UU_hpa_700','VV_hpa_700','WS']
    """
    feature_selection = FeatureSelection()

    if feature_type == 'solar':
        train_feature = train_feature[:, :-1]
    feature_selection.feature_select_by_rf(feature_data=train_feature, target_data=train_target, threshold=threshold,
                                           usecols=usecols)
    return feature_selection.features_selected


# 使用未预处理的数据的相关性分析特征选取
def select_feature_by_correlation_withinitialdata(usecols, feature_type='wind', train_feature=None, train_target=None):
    """
    使用未预处理的数据的相关性分析特征选取
    :param train_feature: 特征集（NWP天气数据, 参数类型为ndarray, 与预测模型训练时所用数据集相同
    :param train_target: 目标集（功率数据）, 用于随机森林模型的训练, 参数类型为ndarray, 与预测模型训练时所用数据集相同
    :param usecols: 加载特征集数据时, 所要筛选的特征, 默认全选
    :param feature_type: 特征类别
    :return: self.features_selected 根据递归特征消除筛选出的特征值, 类型为list, 例如:['UU_hpa_700','VV_hpa_700','WS']
    """
    feature_selection = FeatureSelection()
    if feature_type == 'solar':
        train_feature = train_feature[:, :-1]
    feature_selection.feature_select_by_correlation_withinitialdata(feature_data=train_feature,
                                                                    target_data=train_target, usecols=usecols)
    # 光伏添加SR，并且放在第一位
    if feature_type == 'solar':
        if 'SR' not in feature_selection.features_selected:
            feature_selection.features_selected.append('SR')
        for i in range(len(feature_selection.features_selected)):
            if feature_selection.features_selected[i] == 'SR':
                feature_selection.features_selected[0], feature_selection.features_selected[i] = \
                    feature_selection.features_selected[i], feature_selection.features_selected[0]
    return feature_selection.features_selected


# 递归消除特征选取
def select_feature_by_rfe(all_cols, feature_type, feature_num,
                          train_feature_ultra_short=None,
                          train_target=None):
    """
    :param feature_type: 预测类型
    :param all_cols: 加载特征集数据时，待筛选的特征集，类型为list
    :param train_feature_ultra_short:
    :param train_target:
    :param feature_type: 加载天气数据类型，'wind'为风电，'solar'为光伏
    :param feature_num: 特征工程筛选得出的最少的特征数量
    :return: self.features_selected 根据递归特征消除筛选出的特征值, 类型为list, 例如:['UU_hpa_700','VV_hpa_700','WS']
    """
    feature_selection = FeatureSelection()
    if feature_type == 'solar':
        train_feature = train_feature_ultra_short.loc[:, all_cols + ['time']].values
    else:
        train_feature = train_feature_ultra_short.loc[:, all_cols].values
    if feature_type == 'solar':
        train_feature = train_feature[:, :-1]
    feature_selection.feature_select_by_rfe(feature_type, usecols=all_cols, feature_data=train_feature,
                                            target_data=train_target, feature_num=feature_num)
    return feature_selection.features_selected


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
    for i in range(int(len(nwp_np_short) / 288)):
        test_feature = nwp_np_short[288 * i: 288 * i + 288, 1:-1].astype("float64")
        history_power = nwp_np_short[288 * i, -1]
        predict_result_raw = fitted_model(feature=test_feature, history_power=history_power,
                                          predict_type=predict_type, irradiance_col=sr_col)
        predict_result = data_postprocess.result_post_process(irradiance_threshold=10,
                                                              pre_processing_predict_result=predict_result_raw,
                                                              online_capacity=online_capacity,
                                                              capacity=online_capacity,
                                                              predict_type=predict_type,
                                                              irradiance_col=0, feature=test_feature)

        predict_result = predict_result.reshape(-1, 1)
        predict_result_cluster = numpy.vstack((predict_result_cluster, predict_result))
        # --------------------------------------------------------------------------------------------------------------
    return predict_result_cluster, time_predict_cluster


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
        # --------------------------------------------------------------------------------------------------------------
    return predict_result_cluster, time_predict_cluster


def get_all_features(feature_type='wind'):
    if feature_type == 'solar':
        all_cols = ["HCC", "LCC", "MCC", "MSL", "RHU_hpa_700", "RHU_hpa_500", "RHU_hpa_400", "RHU_hpa_300",
                    "RHU_hpa_250", "RHU_hpa_200", "RHU_meter", "SR", "SWDDIF", "SWDDIR", "TCC", "TEM_hpa_700",
                    "TEM_hpa_500", "TEM_hpa_400", "TEM_hpa_300", "TEM_hpa_250", "TEM_hpa_200", "TEM_meter"]
    else:
        all_cols = ['Density', 'PRS', 'RHU_hpa_700', 'RHU_hpa_500', 'RHU_hpa_400', 'RHU_hpa_300', 'RHU_hpa_250',
                    'RHU_hpa_200', 'RHU_meter', 'TEM_hpa_700', 'TEM_hpa_500', 'TEM_hpa_400', 'TEM_hpa_300',
                    'TEM_hpa_250', 'TEM_hpa_200', 'TEM_meter', 'UU_hpa_700', 'UU_hpa_500', 'UU_hpa_400',
                    'UU_hpa_300', 'UU_hpa_250', 'UU_hpa_200', 'VIS', 'VV_hpa_700', 'VV_hpa_500', 'VV_hpa_400',
                    'VV_hpa_300', 'VV_hpa_250', 'VV_hpa_200', 'WD', 'WS']
    return all_cols


from model import BPNN, MFFS, BLSTM, BNN, GRU, LSTM, MLP, GBRT, DecisionTreeRegressor, RandomForest, SVM, XGBoost
from common import data_preprocess
from common.tools import save_model
from common.logger import logs
import os
from common.tools import load_model


def fit_model(predict_term, model_name, config, train_feature=None, train_target=None,
              record_restrict=None, without_history_power=None, scene='opration'):
    """
    用阿普风电站的数据来检验模型预测效果
    :param config: 配置信息
    :param predict_term: 功率预测时期，short为短期预测，其他为超短期预测
    :param model_name: 预测可惜“wind”，“solar”
    :param predict_term: 功率预测类型 "wind" or "solar"
    :param train_feature: 训练特征
    :param train_target: 训练历史功率
    :param record_restrict: 限电记录
    :param without_history_power: 是否训练带历史功率的模型
    :param scene: 应用场景
    :return:
        result: 预测结果,
        predict_model: 预测模型
    """
    predict_type = config["type"]
    online_capacity = config["online_capacity"]
    irradiance_col = config["sr_col"]
    if predict_term == 'short':
        best_parameter = config["best_parameter_short"]
        usecols = config["short_usecols"]
    else:
        best_parameter = config["best_parameter_ultra_short"]
        usecols = config["ultra_short_usecols"]
    n_features = len(usecols)
    if predict_type == 'solar':
        n_features = len(usecols) + 1

    predict_model = None
    if model_name == "BPNN":
        predict_model = BPNN(params=best_parameter[model_name])
    if model_name == "MFFS":
        predict_model = MFFS(params=best_parameter[model_name])
    if model_name == "BLSTM":
        predict_model = BLSTM(input_size=n_features + 1, params=best_parameter[model_name])
    if model_name == "BNN":
        predict_model = BNN(input_size=n_features + 1, params=best_parameter[model_name])
    if model_name == "GRU":
        predict_model = GRU(params=best_parameter[model_name])
    if model_name == "LSTM":
        predict_model = LSTM(n_features=n_features, params=best_parameter[model_name])
    if model_name == "MLP":
        predict_model = MLP()
    if model_name == "GBRT":
        predict_model = GBRT(params=best_parameter[model_name])
    if model_name == "DecisionTreeRegressor":
        predict_model = DecisionTreeRegressor(params=best_parameter[model_name])
    if model_name == "RandomForest":
        predict_model = RandomForest(params=best_parameter[model_name])
    if model_name == "SVM":
        predict_model = SVM(params=best_parameter[model_name])
    if model_name == "XGBoost":
        predict_model = XGBoost(params=best_parameter[model_name])

    # ------------------------------------------------------------------------------------------------------------------
    if scene == 'ensemble_learn':
        sub_dir_path = config['model_savepath'] + str(config['id']) + '/' + predict_term + '/'
        train_feature = load_model(sub_dir_path + predict_term + '_train_feature.pkl')
        train_target = load_model(sub_dir_path + predict_term + '_train_target.pkl')
    else:
        # ----------------------------------------------------------------------------------------------------------
        # 数据预处理
        train_feature, train_target = data_preprocess.DataPreprocess.data_preprocess(train_feature, train_target,
                                                                                     online_capacity,
                                                                                     record_restrict)

    predict_model.fit(feature_data=train_feature, target_data=train_target, predict_term=predict_term,
                      epochs=20, batch_size=1440, predict_type="wind", irradiance_col=irradiance_col,
                      without_history_power=without_history_power)

    return predict_model


def save_a_fitted_model(station_id, config, model_name_cluster, train_feature=None, train_target=None,
                        record_restrict=None, without_history_power=None):
    """
    针对特定的电场和预测类型，通过测试选择最优的模型并保存
    :param station_id: 场站名称
    :param config: 配置信息
    :param model_name_cluster: 模型名称集合
    :param train_feature: 训练特征
    :param train_target: 训练历史功率
    :param record_restrict: 限电记录
    :param without_history_power: 是否训练带历史功率的模型
    :return:
    """
    term_cluster = ["short", "ultra_short"]
    for term in term_cluster:
        for model_name in model_name_cluster:
            fitted_model = fit_model(predict_term=term, model_name=model_name, config=config,
                                     train_feature=train_feature, train_target=train_target,
                                     record_restrict=record_restrict,
                                     without_history_power=without_history_power)

            sub_dir_path = config['model_savepath'] + str(station_id) + "/" + term + "/" + model_name + "/"
            os.makedirs(sub_dir_path, exist_ok=True)
            save_model(fitted_model, sub_dir_path + term + "_" + model_name + ".pkl")
            if term == 'ultra_short':
                save_model(config["ultra_short_usecols"], sub_dir_path + 'ultra_short_usecols.pkl')
            else:
                save_model(config["short_usecols"], sub_dir_path + 'short_usecols.pkl')


def save_all_fitted_model(station_id_cluster, config_cluster, model_name_cluster, train_feature=None, train_target=None,
                          record_restrict=None, without_history_power=None):
    """
    针对特定的电场和预测类型，通过测试选择最优的模型并保存
    :param config_cluster: 配置信息集合
    :param station_id_cluster: 场站名称集合
    :param model_name_cluster: 模型名称集合
    :param train_feature: 训练特征
    :param train_target: 训练历史功率
    :param record_restrict: 限电记录
    :param without_history_power: 是否训练带历史功率的模型
    :return:
    """
    logs.info('模型训练开始')
    for station_id in station_id_cluster:
        save_a_fitted_model(station_id=station_id, config=config_cluster[station_id],
                            model_name_cluster=model_name_cluster,
                            train_feature=train_feature, train_target=train_target,
                            record_restrict=record_restrict,
                            without_history_power=without_history_power)
    logs.info('模型训练结束')

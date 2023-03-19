import os

from app.small_time_scale_predict.method_five_minute.NWP_interpolation import fit_NWP_interpolation_model
from app.small_time_scale_predict.method_five_minute.cloud_predict import fit_cloud_predict_model
from app.small_time_scale_predict.method_five_minute.NWP_multidimensional_mapping import \
    fit_NWP_multidimensional_mapping_model
from app.small_time_scale_predict.method_five_minute.history_power_iterative_predict import \
    fit_history_power_iterative_predict_model
from app.small_time_scale_predict.method_five_minute.power_machinelearning_interpolation import \
    fit_power_machinelearning_interpolation_model
from model import BPNN, GBRT, XGBoost, MFFS, BLSTM, BNN, GRU, LSTM, MLP, DecisionTreeRegressor, RandomForest, SVM
from common.tools import save_model, load_model
from task.mysql.small_time_scale.load_train_data_five_minute import LoadFiveMinuteEnsembledata


def fit_five_minute_model(predict_term, model_name, config, train_feature, train_target, station_id,
                          nwp_15min_row, nwp_15min_tcc):
    """
    用阿普风电站的数据来检验模型预测效果
    :param config: 配置信息
    :param predict_term: 功率预测时期，short为短期预测，其他为超短期预测
    :param model_name: 预测可惜“wind”，“solar”
    :param predict_term: 功率预测类型 "wind" or "solar"
    :param data_resource: 数据来源是‘CSV’或‘SQL’
    :return:
        result: 预测结果,
        predict_model: 预测模型
    """
    usecols = config["usecols"]
    predict_type = config["type"]
    irradiance_col = config["sr_col"]
    best_parameter = config["best_paprameter_ultra_short"]
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
    # 训练NWP多点功率映射法模型NMM
    five_minute_model1 = fit_NWP_multidimensional_mapping_model(feature_data=train_feature, target_data=train_target,
                                                                irradiance_col=irradiance_col,
                                                                predict_model=predict_model, predict_type=predict_type,
                                                                predict_term=predict_term
                                                                )
    sub_dir_path = "%s%s%s" % (config['model_savepath'], str(station_id), '/five_minute/')
    os.makedirs(sub_dir_path, exist_ok=True)

    save_model(five_minute_model1.predict_without_history_power,
               sub_dir_path + "five_minute_NMM_" + model_name + "_without_history_power.pkl")
    save_model(five_minute_model1.predict_with_history_power, sub_dir_path + "five_minute_NMM_" + model_name + "_with_history_power.pkl")
    save_model(five_minute_model1.predict_mix_history_power, sub_dir_path + "five_minute_NMM_" + model_name + "_mix_history_power.pkl")

    # 训练NWP插值法模型NI
    five_minute_model2 = fit_NWP_interpolation_model(feature_data=train_feature, target_data=train_target,
                                                     irradiance_col=irradiance_col,
                                                     predict_model=predict_model, predict_type=predict_type,
                                                     predict_term=predict_term
                                                     )
    save_model(five_minute_model2.predict_without_history_power,
               sub_dir_path + "five_minute_NI_" + model_name + "_without_history_power.pkl")
    save_model(five_minute_model2.predict_with_history_power, sub_dir_path + "five_minute_NI_" + model_name + "_with_history_power.pkl")
    save_model(five_minute_model2.predict_mix_history_power, sub_dir_path + "five_minute_NI_" + model_name + "_mix_history_power.pkl")

    # 训练功率数据机器学习插值法模型PMI
    five_minute_model3 = fit_power_machinelearning_interpolation_model(target_data=train_target, model_name=model_name,
                                                                       irradiance_col=irradiance_col,
                                                                       predict_model=predict_model,
                                                                       predict_type=predict_type,
                                                                       best_paprameter=best_parameter,
                                                                       predict_term=predict_term)
    save_model(five_minute_model3, sub_dir_path + "five_minute_PMI_" + model_name + ".pkl")

    # 训练时序法模型HPIP
    five_minute_model4 = fit_history_power_iterative_predict_model(target_data=train_target, model_name=model_name,
                                                                   irradiance_col=irradiance_col,
                                                                   predict_model=predict_model,
                                                                   predict_type=predict_type,
                                                                   best_paprameter=best_parameter,
                                                                   predict_term=predict_term)
    save_model(five_minute_model4, sub_dir_path + "five_minute_HPIP_" + model_name + ".pkl")

    # 训练云团追踪预测模型
    if predict_type == "solar":
        cloud_predict_model = fit_cloud_predict_model(nwp_15min_row=nwp_15min_row, nwp_15min_tcc=nwp_15min_tcc,
                                                      target_data=train_target,
                                                      irradiance_col=irradiance_col,
                                                      predict_model=predict_model, predict_type=predict_type,
                                                      predict_term=predict_term)
        save_model(cloud_predict_model.predict_without_history_power,
                   sub_dir_path + "five_minute_NI_" + model_name + "_without_history_power.pkl")
        save_model(cloud_predict_model.predict_with_history_power,
                   sub_dir_path + "five_minute_NI_" + model_name + "_with_history_power.pkl")
        save_model(cloud_predict_model.predict_mix_history_power,
                   sub_dir_path + "five_minute_NI_" + model_name + "_mix_history_power.pkl")
    else:
        cloud_predict_model = None

    return five_minute_model1, five_minute_model2, five_minute_model3, five_minute_model4, cloud_predict_model


def fit_power_interpolation_model(best_fitted_model, config, station_id):
    """
    训练插值法模型，即三次条样插值法的模型， 最邻近插值法的模型
    :param best_fitted_model: 最优的15分钟预测模型
    :param config: 配置信息
    :param station_name: 场站名称
    """
    sub_dir_path = config['model_savepath'] + str(station_id) + "/five_minute/"
    os.makedirs(sub_dir_path, exist_ok=True)
    save_model(best_fitted_model, sub_dir_path + "fifteen_minute_best_model.pkl")


def save_a_fitted_five_minute_model(station_id, config, model_name_cluster, train_feature, train_target, nwp_15min_row, nwp_15min_tcc):
    """
    针对特定的电场和预测类型，通过测试选择最优的模型并保存
    :param station_name: 场站名称
    :param config: 配置信息
    :param model_name_cluster: 模型名称集合
    :param data_resource: 数据来源“CSV”，“SQL”
    :return:
    """
    # 最优的15分钟预测模型
    model_name, model_state = config['best_model_ultra_short'].split('_', 1)
    best_fitted_model = load_model(config['model_savepath'] + str(station_id) + '/ultra_short/' + model_name + '/'
                              + 'ultra_short' + '_' + model_name + '.pkl')
    best_fitted_model = eval('best_fitted_model' + '.predict_' + model_state)
    # 保存直接功率插值法的模型
    fit_power_interpolation_model(best_fitted_model=best_fitted_model, config=config, station_id=station_id)

    # # --------------------------------------------加载数据---------------------------------------------------------------
    # 训练其他5分钟预测模型
    for model_name in model_name_cluster:
        fit_five_minute_model(predict_term="five_minute_ultra_short", model_name=model_name,
                              config=config,
                              train_feature=train_feature, train_target=train_target, station_id=station_id,
                              nwp_15min_row=nwp_15min_row, nwp_15min_tcc=nwp_15min_tcc)


def save_all_fitted_model(station_name_cluster, config_cluster, model_name_cluster, data_resource='CSV',
                          host='localhost', user='root', password='123456', database='kuafu', charset='utf8',
                          port=3306, rate=0.75):
    """
    针对特定的电场和预测类型，通过测试选择最优的模型并保存
    :param config_cluster: 配置信息集合
    :param station_name_cluster: 场站名称集合
    :param model_name_cluster: 模型名称集合
    :param data_resource: 数据来源“CSV”，“SQL”
    :return:
    """
    for station_name in station_name_cluster:
        # --------------------------------------------加载数据---------------------------------------------------------------
        train_data_load_five_minute = LoadFiveMinuteEnsembledata()
        # 从SQL读取训练数据
        train_feature, train_target = train_data_load_five_minute.load_train_data_for_five_minute_sql(host=host,
                                                                                                      user=user,
                                                                                                      password=password,
                                                                                                      database=database,
                                                                                                      charset=charset,
                                                                                                      port=port,
                                                                                                      file_path=config_cluster[station_name],
                                                                                                      usecols=config_cluster[station_name]['usecols'],
                                                                                                      rate=rate,
                                                                                                      predict_type=config_cluster[station_name]["type"])
        save_a_fitted_five_minute_model(station_name=station_name, config=config_cluster[station_name],
                                        model_name_cluster=model_name_cluster, data_resource=data_resource, host=host, user=user,
                                        password=password, database=database, charset=charset, port=port, rate=rate,
                                        train_feature=train_feature, train_target=train_target)

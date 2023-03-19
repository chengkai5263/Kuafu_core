import numpy

from app.small_time_scale_predict_model.method_of_NWP_interpolation import method_of_NWP_interpolation
from app.small_time_scale_predict_model.method_of_NWP_multidimensional_mapping import \
    method_of_NWP_multidimensional_mapping
from app.small_time_scale_predict_model.method_of_iterative_prediction_by_history_power import \
    method_of_iterative_prediction_by_history_power
from app.small_time_scale_predict_model.method_of_power_interpolation import method_of_power_interpolation
from app.small_time_scale_predict_model.method_of_power_machinelearning_interpolation import \
    method_of_power_machinelearning_interpolation
from common.load_data_from_csv import load_csv_data
from model._blstm import BLSTM


def load_three_day_forecast_data(file_path, nrows=None, skiprows=None, usecols=None, rate=0.75):
    """
    加载训练集及测试集数据。此处加载的数据，均是预处理后的数据
    :param file_path: 特征集/目标集文件路径。参数类型为字典，key分别为feature、target，value为对应文件路径所组成的列表。
    :param nrows: 加载特征集/目标集文件时所要选取的记录条数，默认全加载
    :param skiprows: 加载特征集/目标集文件时的忽略记录条数（从文件头开始算起，表头也算一条数据），默认不忽略。
                     该值必须是288的倍数+1
    :param usecols: 加载特征集CSV文件时，所要筛选的列，默认全选。当要筛选所有列时，该参数取值None；当要筛选某些列时，若skiprows参数
                    为None，该参数的值可以是由所要筛选的列名所组成的一个列表（如['Tmp', '170FengSu']）或列序号（从0开始）所组成的
                    一个列表（如[0,1])，若skiprows参数不为None或者原始CSV文件没有表头，则该参数的值只能是列序号所组成的列表
    :param rate: 训练集数据量与（训练集+测试集数据量）的比值
    :return:
    """
    # 加载特征集文件
    data_set = None
    for file in file_path["feature"]:
        if data_set is None:
            data_set = load_csv_data(file, nrows=nrows, skiprows=skiprows, usecols=usecols)
        else:
            data_set = numpy.concatenate((data_set,
                                          load_csv_data(file, nrows=nrows, skiprows=skiprows, usecols=usecols)
                                          ))
    # 起报时间总天数。预报未来3天的天气预报数据时，起报时间为同一天的记录共有288个点
    start_days = int(len(data_set) / 288)
    # 用于训练的起报时间天数
    train_start_days = int(start_days * rate)
    # 划分训练/测试特征集数据
    train_data = data_set[0:train_start_days * 288]
    test_feature_data = data_set[train_start_days * 288:start_days * 288]

    # 处理训练特征集，仅取预报未来一天的预报数据
    train_feature_data = numpy.reshape(train_data, (train_start_days, -1))[:, :96 * train_data.shape[1]]
    train_feature_data = numpy.reshape(train_feature_data, (train_start_days * 96, -1))

    data_set = None
    for file in file_path["target"]:
        if data_set is None:
            data_set = load_csv_data(file, nrows=nrows, skiprows=skiprows)
        else:
            data_set = numpy.concatenate((data_set,
                                          load_csv_data(file, nrows=nrows, skiprows=skiprows)
                                          ))
    test_data_end_index = start_days * 288
    train_data_len = train_start_days * 288
    train_target_data = data_set[:train_data_len, 1]
    test_target_data = data_set[train_data_len:test_data_end_index, 1]

    # 将目标集的数据类型转换成float64
    train_target_data = train_target_data.astype("float64")
    test_target_data = test_target_data.astype("float64")
    return train_feature_data, train_target_data, test_feature_data, test_target_data



def load_data(power_and_NWP_data_path, nrows=None, skiprows=None, usecols=None, rate=0.75, forecast_time=3):
    if forecast_time == 1:
        # load_one_day_forecast_data(power_and_NWP_data_path, nrows, skiprows, usecols, rate)
        print(1)
    else:
        train_feature_data, train_target_data, test_feature_data, test_target_data = \
            load_three_day_forecast_data(power_and_NWP_data_path, nrows, skiprows, usecols, rate)
        print(2)
    return train_feature_data, train_target_data, test_feature_data, test_target_data
if __name__ == '__main__':
    # 功率预测时期，short为短期预测，其他为超短期预测
    predict_term = "5minute_ultra_short"
    # 加载X天的数据。当预测短期发电功率时，该值不小于16，且测试集中的第一天数据会被弃用
    days = 4
    # 天气预报csv文件所要筛选的列
    wind_usecols = ['WS', 'Density', 'PRS', 'WD', 'RHU_meter', 'TEM_meter']
    wind_file_path = {
        "feature": [
            r"./resource/阿普风电气象数据.csv",
        ],
        "target": [
            r"./resource/阿普风电功率数据.csv",
        ],
    }

    # 开机容量
    # 装机容量188MW
    wind_online_capacity = 188
    # 模型初始化
    wind_blstm = BLSTM(input_size=len(wind_file_path) + 1, predict_type="wind")

    # 加载5分钟功率数据和15分钟NWP
    train_feature_data, train_target_data, test_feature_data, test_target_data = \
        load_data(power_and_NWP_data_path=wind_file_path, nrows=288 * days + 1, usecols=wind_usecols, forecast_time=3)

    # NWP多点功率映射法
    result1 = method_of_NWP_multidimensional_mapping(fiveminte_model=wind_blstm, train_feature_data=train_feature_data,
                                                     train_target_data=train_target_data,
                                                     test_feature_data=test_feature_data,
                                                     test_target_data=test_target_data,
                                                     online_capacity=wind_online_capacity,
                                                     predict_term=predict_term)
    # 功率插值法
    result2 = method_of_power_interpolation(fiveminte_model=wind_blstm, train_feature_data=train_feature_data,
                                            train_target_data=train_target_data,
                                            test_feature_data=test_feature_data,
                                            test_target_data=test_target_data,
                                            online_capacity=wind_online_capacity,
                                            predict_term=predict_term)

    # NWP插值法
    result3 = method_of_NWP_interpolation(fiveminte_model=wind_blstm, train_feature_data=train_feature_data,
                                          train_target_data=train_target_data,
                                          test_feature_data=test_feature_data,
                                          test_target_data=test_target_data,
                                          online_capacity=wind_online_capacity,
                                          predict_term=predict_term)

    # 功率数据机器学习插值法
    result4 = method_of_power_machinelearning_interpolation(fifteenminute_model=wind_blstm,
                                                            train_feature_data=train_feature_data,
                                                            train_target_data=train_target_data,
                                                            test_feature_data=test_feature_data,
                                                            test_target_data=test_target_data,
                                                            online_capacity=wind_online_capacity,
                                                            predict_term=predict_term)

    # 利用历史功率迭代预测
    result5 = method_of_iterative_prediction_by_history_power(fifteenminute_model=wind_blstm, train_feature_data=train_feature_data,
                                                              train_target_data=train_target_data, test_feature_data=test_feature_data,
                                                              test_target_data=test_target_data, online_capacity=wind_online_capacity,
                                                              predict_term=predict_term)

    # 打印结果
    print(f"按照国标，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result1["evaluation_without_history"],
          result1["evaluation_with_history"],
          result1["evaluation_mix_history"]
          )
    print(f"按照南网总调标准，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result1["evaluation_csg_without_history"],
          result1["evaluation_csg_with_history"],
          result1["evaluation_csg_mix_history"]
          )

    # 打印结果
    print(f"按照国标，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result2["evaluation_without_history"],
          result2["evaluation_with_history"],
          result2["evaluation_mix_history"]
          )
    print(f"按照南网总调标准，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result2["evaluation_csg_without_history"],
          result2["evaluation_csg_with_history"],
          result2["evaluation_csg_mix_history"]
          )

    # 打印结果
    print(f"按照国标，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result3["evaluation_without_history"],
          result3["evaluation_with_history"],
          result3["evaluation_mix_history"]
          )
    print(f"按照南网总调标准，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result3["evaluation_csg_without_history"],
          result3["evaluation_csg_with_history"],
          result3["evaluation_csg_mix_history"]
          )

    # 打印结果
    print(f"按照国标，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result4["evaluation_without_history"],
          result4["evaluation_with_history"],
          result4["evaluation_mix_history"]
          )
    print(f"按照南网总调标准，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result4["evaluation_csg_without_history"],
          result4["evaluation_csg_with_history"],
          result4["evaluation_csg_mix_history"]
          )

    # 打印结果
    print(f"按照国标，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result5["evaluation_without_history"],
          result5["evaluation_with_history"],
          result5["evaluation_mix_history"]
          )
    print(f"按照南网总调标准，本模型不使用/使用/混合历史功率数据进行风电{predict_term}功率预测时的准确率分别为：",
          result5["evaluation_csg_without_history"],
          result5["evaluation_csg_with_history"],
          result5["evaluation_csg_mix_history"]
          )


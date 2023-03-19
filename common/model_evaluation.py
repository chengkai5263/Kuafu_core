
from math import sqrt
import numpy
from sklearn.metrics import mean_squared_error
import pandas
from common.logger import logs


def evaluate_GB_T_40607_2021_withtimetag(predict_power, actual_power, online_capacity, predict_term,
                                         error_data_label=None):
    """
    依据国标GB/T 40607-2021标准, 计算全网风电/光伏功率预测性能指标要求：
        1. 超短期功率预测：第4小时>=95%
        2. 短期功率预测：日前>=90%
        3. 中期功率预测：第10日（第217-240小时）>=80%
    :param predict_power: 一段时间内每次预测结果集
    :param actual_power: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测类型:
        1. "ultra_short"：超短期功率预测
        2. "short"：短期功率预测
        3. “medium”：中期功率预测
    :param error_data_label: 限电记录
    """
    # 单次预测长度，单次预测需要考核的数据长度---------------------------------------------------------------------------------
    interval = 288
    evaluation_term = 96
    if predict_term == "medium":
        interval = 960
        evaluation_term = 96
    if predict_term == "ultra_short":
        interval = 16
        evaluation_term = 4

    predict_power = pandas.DataFrame(predict_power, columns=['start_time', 'forecast_time', 'predict_power'])
    actual_power = pandas.DataFrame(actual_power, columns=['time', 'power'])
    # ------------------------------------------------------------------------------------------------------------------
    # 删除真实功率重复的点
    actual_power = actual_power.drop_duplicates(subset=['time'])
    # 判断每轮预测是否有完整的16、288、960个点
    start_time = pandas.DataFrame(predict_power.loc[:, 'start_time'], columns=['start_time'])
    start_time = start_time.drop_duplicates(subset=['start_time'])
    for i in range(len(start_time)):
        number_predict = len(predict_power.loc[predict_power['start_time'] == start_time.iloc[i, 0]])
        if number_predict % interval != 0:
            predict_power = predict_power.drop(
                index=predict_power[(predict_power.start_time == start_time.iloc[i, 0])].index.tolist())

    number_actual_power_raw = len(actual_power)

    if number_actual_power_raw == 0:
        logs.warning('被选中时段没有实际功率和预测功率重合的时刻！')
        return 0
    # 如果限电记录不为空时的处理---------------------------------------------------------------------------------------------
    if error_data_label is not None:
        error_data_label = pandas.DataFrame(error_data_label, columns=['time', 'record'])
        actual_power_restrict = pandas.merge(actual_power, error_data_label, left_on='time', right_on='time',
                                             how='left').values

        for i in range(len(actual_power_restrict) - 1, -1, -1):
            if actual_power_restrict[i, 2] == 1:
                actual_power_restrict = numpy.vstack((actual_power_restrict[:i, :], actual_power_restrict[i + 1:, :]))

        actual_power = pandas.DataFrame(actual_power_restrict[:, 0:2], columns=['time', 'power'])
    number_actual_power = len(actual_power)

    restrict_rate = (number_actual_power_raw - number_actual_power) / number_actual_power_raw
    if (number_actual_power_raw - number_actual_power) != 0:
        logs.debug('限电免考核数据比例：' + str(restrict_rate))

    # 实际功率与预测功率时间对齐---------------------------------------------------------------------------------------------
    train_data_merge = pandas.merge(predict_power, actual_power, left_on='forecast_time', right_on='time', how='left')
    day = int(len(train_data_merge) / interval)
    predict_power = train_data_merge.loc[:, 'predict_power'].values.reshape(day, interval)
    actual_power = train_data_merge.loc[:, 'power'].values.reshape(day, interval)

    # 按照标准要求取考核时间数据---------------------------------------------------------------------------------------------
    if predict_term == "medium":
        predict_power = predict_power[:, -96:]
        actual_power = actual_power[:, -96:]
    if predict_term == "short":
        predict_power = predict_power[:, :96]
        actual_power = actual_power[:, :96]
    if predict_term == "ultra_short":
        predict_power = predict_power[:, -4:]
        actual_power = actual_power[:, -4:]

    loss_number = 0

    # 预测功率为空时的异常处理
    if predict_power is None:
        return None

    # 去掉没有实际功率对应的预测时段------------------------------------------------------------------------------------------
    actual_power_list = []
    predict_power_list = []
    for i in range(len(predict_power)):
        power_cluster = numpy.hstack(
            (actual_power[i][-evaluation_term:].reshape(-1, 1), predict_power[i][-evaluation_term:].reshape(-1, 1)))
        power_cluster = pandas.DataFrame(power_cluster)
        power_cluster = power_cluster.dropna(axis=0, how='any').values  # 去除没有实际功率对应的预测时段
        loss_number = loss_number + (evaluation_term - len(power_cluster))
        actual_power_list.append(power_cluster[:, 0].tolist())
        predict_power_list.append(power_cluster[:, 1].tolist())

    loss_rate = loss_number / evaluation_term / len(predict_power)
    restrict_rate = (number_actual_power_raw - number_actual_power) / number_actual_power_raw
    logs.debug('没有实际功率对应的预测时段比例：' + str(min(loss_rate, abs(loss_rate - restrict_rate))))

    # 计算准确率----------------------------------------------------------------------------------------------------------
    accrucy = result_evaluation_continue_point(predict_power_list, actual_power_list, online_capacity,
                                               evaluation="capacity")
    return accrucy


# 光伏夜间时段不计入考核
def evaluate_NB_T_32011_2013_withtimetag_solar_without_night(predict_power, actual_power, online_capacity, predict_term,
                                                             error_data_label, sunrise_time, sunset_time):
    """
    依据行标NB_T 32011_2013标准, 计算全网光伏功率预测性能指标要求：
        1. 超短期功率预测：第4小时>90%
        2. 短期功率预测：日前>85%
    :param predict_power: 一段时间内每次预测结果集
    :param actual_power: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测时长:
        1. "ultra_short"：超短期功率预测
        2. "short"：短期功率预测
        3. "medium"：中期功率预测
    :param error_data_label: 限电记录
    :param sunrise_time:光伏发电站所在地理位置的日出时间
    :param sunset_time: 光伏发电站所在地理位置的日落时间
    """
    # 单次预测长度，单次预测需要考核的数据长度---------------------------------------------------------------------------------
    interval = 288
    evaluation_term = 96
    if predict_term == "medium":
        interval = 960
        evaluation_term = 96
    if predict_term == "ultra_short":
        interval = 16
        evaluation_term = 4

    predict_power = pandas.DataFrame(predict_power, columns=['start_time', 'forecast_time', 'predict_power'])
    actual_power = pandas.DataFrame(actual_power, columns=['time', 'power'])

    # 如果限电记录不为空时的处理---------------------------------------------------------------------------------------------
    if error_data_label is not None:
        error_data_label = pandas.DataFrame(error_data_label, columns=['time', 'record'])
        actual_power_restrict = pandas.merge(actual_power, error_data_label, left_on='time', right_on='time',
                                             how='left').values

        for i in range(len(actual_power_restrict) - 1, -1, -1):
            if actual_power_restrict[i, 2] == 1:
                actual_power_restrict = numpy.vstack((actual_power_restrict[1:i, :], actual_power_restrict[i + 1:, :]))

        actual_power = pandas.DataFrame(actual_power_restrict[:, 0:2], columns=['time', 'power'])

    # 实际功率与预测功率时间对齐---------------------------------------------------------------------------------------------
    train_data_merge = pandas.merge(predict_power, actual_power, left_on='forecast_time', right_on='time', how='left')
    # 剔除日落时间至日出时间之间的夜间时段
    df_temp = train_data_merge.iloc[:, 1:2]
    min_array = df_temp.applymap(lambda x: x.strftime('%H:%M:%S'))
    bool_array = min_array.applymap(lambda x: sunrise_time <= x <= sunset_time)
    train_data_merge.loc[train_data_merge[bool_array.forecast_time == False].index.tolist(), 'power'] = numpy.nan
    day = int(len(train_data_merge) / interval)
    predict_power = train_data_merge.loc[:, 'predict_power'].values.reshape(day, interval)
    actual_power = train_data_merge.loc[:, 'power'].values.reshape(day, interval)

    if len(actual_power) == 0:
        logs.warning('被选中时段没有实际功率和预测功率重合的时刻！')
        return 0
    # 按照标准要求取考核时间数据---------------------------------------------------------------------------------------------
    if predict_term == "medium":
        predict_power = predict_power[:, -96:]
        actual_power = actual_power[:, -96:]
    if predict_term == "short":
        predict_power = predict_power[:, :96]
        actual_power = actual_power[:, :96]
    if predict_term == "ultra_short":
        predict_power = predict_power[:, -4:]
        actual_power = actual_power[:, -4:]

    loss_number = 0

    # 预测功率为空时的异常处理
    if predict_power is None:
        return None

    # 去掉没有实际功率对应的预测时段--------------------------------------------------------------------------- --------------
    actual_power_list = []
    predict_power_list = []
    for i in range(len(predict_power)):
        power_cluster = numpy.hstack(
            (actual_power[i][-evaluation_term:].reshape(-1, 1), predict_power[i][-evaluation_term:].reshape(-1, 1)))
        power_cluster = pandas.DataFrame(power_cluster)
        power_cluster = power_cluster.dropna(axis=0, how='any').values  # 去除免考核时段，包括限电、实际功率缺失
        loss_number = loss_number + (evaluation_term - len(power_cluster))
        actual_power_list.append(power_cluster[:, 0].tolist())
        predict_power_list.append(power_cluster[:, 1].tolist())

    # 计算准确率----------------------------------------------------------------------------------------------------------
    actual_result = actual_power_list
    predict_result = predict_power_list
    if actual_result is None or len(actual_result) == 0 or predict_result is None or len(predict_result) == 0:
        return None

    evaluation_set = []
    for i in range(0, len(predict_result), 1):
        if actual_result[i] is None or len(actual_result[i]) == 0:
            continue
        rmse = sqrt(mean_squared_error(
            numpy.array(actual_result[i]), numpy.array(predict_result[i]))
        )
        score = (1 - rmse / online_capacity) * 100
        evaluation_set.append(score)

    logs.debug('没有实际功率对应的预测时段比例：' + str(loss_number / evaluation_term / len(predict_power)))
    return numpy.mean(evaluation_set)


# Q_CSG1211017_2018，超短期计算前4小时；短期计算前24小时
def result_evaluation_Q_CSG1211017_2018_without_time_tag(predict_power, actual_power, online_capacity, predict_term,
                                                         error_data_label=None, evaluation=None):
    """
    评估预测精度-基于Q/CSG1211017-2018标准，对于短期风电功率预测，计算单次预测的前96个点；对于超短期风电功率预测，计算单次预测的前16个点
    :param predict_power: 一段时间内每次预测结果集
    :param actual_power: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测类型，超短期"ultra_short"或短期"short"
    :param error_data_label: 真实功率错误而不考核的时间段记录
    :param evaluation: 分母按开始容量还是真实功率
    :return np.mean(evaluation_set): 日平均准确率为当日内全部超短期预测准确率的算术平均值
    """
    # 单次预测长度，单次预测需要考核的数据长度---------------------------------------------------------------------------------
    interval = 288
    evaluation_term = 96
    if predict_term == "medium":
        interval = 960
        evaluation_term = 96
    if predict_term == "ultra_short":
        interval = 16
        evaluation_term = 16

    predict_power = pandas.DataFrame(predict_power, columns=['start_time', 'forecast_time', 'predict_power'])
    actual_power = pandas.DataFrame(actual_power, columns=['time', 'power'])
    # ------------------------------------------------------------------------------------------------------------------
    # 删除真实功率重复的点
    actual_power = actual_power.drop_duplicates(subset=['time'])
    # 判断每轮预测是否有完整的16、288、960个点
    start_time = pandas.DataFrame(predict_power.loc[:, 'start_time'], columns=['start_time'])
    start_time = start_time.drop_duplicates(subset=['start_time'])
    for i in range(len(start_time)):
        number_predict = len(predict_power.loc[predict_power['start_time'] == start_time.iloc[i, 0]])
        if number_predict % interval != 0:
            predict_power = predict_power.drop(
                index=predict_power[(predict_power.start_time == start_time.iloc[i, 0])].index.tolist())

    # ------------------------------------------------------------------------------------------------------------------

    number_actual_power_raw = len(actual_power)
    if number_actual_power_raw == 0:
        logs.warning('被选中时段没有实际功率和预测功率重合的时刻！')
        return 0
    # 如果限电记录不为空时的处理---------------------------------------------------------------------------------------------
    if error_data_label is not None:
        error_data_label = pandas.DataFrame(error_data_label, columns=['time', 'record'])
        actual_power_restrict = pandas.merge(actual_power, error_data_label, left_on='time', right_on='time',
                                             how='left').values

        for i in range(len(actual_power_restrict) - 1, -1, -1):
            if actual_power_restrict[i, 2] == 1:
                actual_power_restrict = numpy.vstack((actual_power_restrict[:i, :], actual_power_restrict[i + 1:, :]))

        actual_power = pandas.DataFrame(actual_power_restrict[:, 0:2], columns=['time', 'power'])
    number_actual_power = len(actual_power)

    restrict_rate = (number_actual_power_raw - number_actual_power) / number_actual_power_raw
    if (number_actual_power_raw - number_actual_power) != 0:
        logs.debug('限电免考核数据比例：' + str(restrict_rate))

    # 实际功率与预测功率时间对齐---------------------------------------------------------------------------------------------
    train_data_merge = pandas.merge(predict_power, actual_power, left_on='forecast_time', right_on='time', how='left')
    day = int(len(train_data_merge) / interval)
    predict_power = train_data_merge.loc[:, 'predict_power'].values.reshape(day, interval)
    actual_power = train_data_merge.loc[:, 'power'].values.reshape(day, interval)

    # 按照标准要求取考核时间数据---------------------------------------------------------------------------------------------
    if predict_term == "medium":
        predict_power = predict_power[:, -96:]
        actual_power = actual_power[:, -96:]
    if predict_term == "short":
        predict_power = predict_power[:, :96]
        actual_power = actual_power[:, :96]
    if predict_term == "ultra_short":
        predict_power = predict_power[:, :]
        actual_power = actual_power[:, :]

    loss_number = 0

    # 预测功率为空时的异常处理
    if predict_power is None:
        return None

    # 去掉没有实际功率对应的预测时段------------------------------------------------------------------------------------------
    actual_power_list = []
    predict_power_list = []
    for i in range(len(predict_power)):
        power_cluster = numpy.hstack(
            (actual_power[i][-evaluation_term:].reshape(-1, 1), predict_power[i][-evaluation_term:].reshape(-1, 1)))
        power_cluster = pandas.DataFrame(power_cluster)
        power_cluster = power_cluster.dropna(axis=0, how='any').values  # 去除没有实际功率对应的预测时段
        loss_number = loss_number + (evaluation_term - len(power_cluster))
        actual_power_list.append(power_cluster[:, 0].tolist())
        predict_power_list.append(power_cluster[:, 1].tolist())

    loss_rate = loss_number / evaluation_term / len(predict_power)
    restrict_rate = (number_actual_power_raw - number_actual_power) / number_actual_power_raw
    logs.debug('没有实际功率对应的预测时段比例：' + str(min(loss_rate, abs(loss_rate - restrict_rate))))

    # 计算准确率----------------------------------------------------------------------------------------------------------
    accrucy = result_evaluation_continue_point(predict_power_list, actual_power_list, online_capacity,
                                               evaluation=evaluation)
    return accrucy


# 南网AI大赛
def result_evaluation_CSG_AI_without_time_tag(predict_result, actual_result, online_capacity,
                                              predict_term="ultra_short"):
    """
    评估预测精度-基于南网第三届电力调度AI应用大赛标准，对于短期风电功率预测，计算单次预测的前96个点；对于超短期风电功率预测，
                计算单次预测的前16个点，计算RMSE分母时使用0.2*实际功率与发电场开机容量之间的最大值
    :param predict_result: 一段时间内每次预测结果集
    :param actual_result: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测类型，超短期"ultra_short"或短期"short"
    :return np.mean(evaluation_set): 日平均准确率为当日内全部超短期预测准确率的算术平均值
    """
    if predict_term == "ultra_short":
        interval = 16
    else:
        interval = 96
    evaluation_set = []
    y_cap = 0.2 * online_capacity
    for j in range(len(actual_result)):
        s = 0
        for k in range(interval):
            y_true = actual_result[j][k]
            y_pred = predict_result[j][k]
            denominator = max(y_true, y_cap)
            s += ((y_true - y_pred) / denominator) ** 2
        score = (1 - sqrt(s / len(actual_result[j]))) * 100
        evaluation_set.append(score)

    return numpy.mean(evaluation_set)


# 两个细则，超短期考核第16个点，Two_Detailed_Rules
def result_evaluation_Two_Detailed_Rules_without_time_tag(predict_power, actual_power, online_capacity, predict_term,
                                                          error_data_label=None):
    """
    依据国标GB/T 40607-2021标准, 计算全网风电/光伏功率预测性能指标要求：
        1. 超短期功率预测：第4小时>=95%
        2. 短期功率预测：日前>=90%
        3. 中期功率预测：第10日（第217-240小时）>=80%
    :param predict_power: 一段时间内每次预测结果集
    :param actual_power: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测类型: 须输入符合格式要求的值
        1. "ultra_short"：超短期功率预测
        2. "short"：短期功率预测
        3. “medium”：中期功率预测
    :param error_data_label: 限电记录
    """
    # 单次预测长度，单次预测需要考核的数据长度---------------------------------------------------------------------------------
    interval = 288
    evaluation_term = 96
    if predict_term == "medium":
        interval = 960
        evaluation_term = 96
    if predict_term == "ultra_short":
        interval = 16
        evaluation_term = 1

    predict_power = pandas.DataFrame(predict_power, columns=['start_time', 'forecast_time', 'predict_power'])
    actual_power = pandas.DataFrame(actual_power, columns=['time', 'power'])
    # ------------------------------------------------------------------------------------------------------------------
    # 删除真实功率重复的点
    actual_power = actual_power.drop_duplicates(subset=['time'])
    # 判断每轮预测是否有完整的16、288、960个点
    start_time = pandas.DataFrame(predict_power.loc[:, 'start_time'], columns=['start_time'])
    start_time = start_time.drop_duplicates(subset=['start_time'])
    for i in range(len(start_time)):
        number_predict = len(predict_power.loc[predict_power['start_time'] == start_time.iloc[i, 0]])
        if number_predict % interval != 0:
            predict_power = predict_power.drop(
                index=predict_power[(predict_power.start_time == start_time.iloc[i, 0])].index.tolist())

    number_actual_power_raw = len(actual_power)
    if number_actual_power_raw == 0:
        logs.warning('被选中时段没有实际功率和预测功率重合的时刻！')
        return 0
    # 如果限电记录不为空时的处理---------------------------------------------------------------------------------------------
    if error_data_label is not None:
        error_data_label = pandas.DataFrame(error_data_label, columns=['time', 'record'])
        actual_power_restrict = pandas.merge(actual_power, error_data_label, left_on='time', right_on='time',
                                             how='left').values

        for i in range(len(actual_power_restrict) - 1, -1, -1):
            if actual_power_restrict[i, 2] == 1:
                actual_power_restrict = numpy.vstack((actual_power_restrict[:i, :], actual_power_restrict[i + 1:, :]))

        actual_power = pandas.DataFrame(actual_power_restrict[:, 0:2], columns=['time', 'power'])
    number_actual_power = len(actual_power)

    restrict_rate = (number_actual_power_raw - number_actual_power) / number_actual_power_raw
    if (number_actual_power_raw - number_actual_power) != 0:
        logs.debug('限电免考核数据比例：' + str(restrict_rate))

    # 实际功率与预测功率时间对齐---------------------------------------------------------------------------------------------
    train_data_merge = pandas.merge(predict_power, actual_power, left_on='forecast_time', right_on='time', how='left')
    day = int(len(train_data_merge) / interval)
    predict_power = train_data_merge.loc[:, 'predict_power'].values.reshape(day, interval)
    actual_power = train_data_merge.loc[:, 'power'].values.reshape(day, interval)

    # 按照标准要求取考核时间数据---------------------------------------------------------------------------------------------
    if predict_term == "medium":
        predict_power = predict_power[:, 3*96:4*96]
        actual_power = actual_power[:, 3*96:4*96]
    elif predict_term == "short":
        predict_power = predict_power[:, :96]
        actual_power = actual_power[:, :96]
    elif predict_term == "ultra_short":
        predict_power = predict_power[:, -1:]
        actual_power = actual_power[:, -1:]
    else:
        logs.warning('未输入正确格式的预测类型！')
        return

    loss_number = 0

    # 预测功率为空时的异常处理
    if predict_power is None:
        return None

    # 去掉没有实际功率对应的预测时段------------------------------------------------------------------------------------------
    actual_power_list = []
    predict_power_list = []
    for i in range(len(predict_power)):
        power_cluster = numpy.hstack(
            (actual_power[i][-evaluation_term:].reshape(-1, 1), predict_power[i][-evaluation_term:].reshape(-1, 1)))
        power_cluster = pandas.DataFrame(power_cluster)
        power_cluster = power_cluster.dropna(axis=0, how='any').values  # 去除没有实际功率对应的预测时段
        loss_number = loss_number + (evaluation_term - len(power_cluster))
        actual_power_list.append(power_cluster[:, 0].tolist())
        predict_power_list.append(power_cluster[:, 1].tolist())

    loss_rate = loss_number / evaluation_term / len(predict_power)
    restrict_rate = (number_actual_power_raw - number_actual_power) / number_actual_power_raw
    logs.debug('没有实际功率对应的预测时段比例：' + str(min(loss_rate, abs(loss_rate - restrict_rate))))

    # 计算准确率----------------------------------------------------------------------------------------------------------
    predict_result = predict_power_list
    actual_result = actual_power_list
    if actual_result is None or len(actual_result) == 0 or predict_result is None or len(predict_result) == 0:
        return None

    evaluation_set = []
    if predict_term == 'medium':
        y_cap = 0.2 * online_capacity
        for j in range(len(actual_result)):
            s = 0
            if len(actual_result[j]) == 0:
                continue
            for k in range(len(actual_result[j])):
                y_true = actual_result[j][k]
                y_pred = predict_result[j][k]
                denominator = max(y_true, y_cap)
                s += ((y_true - y_pred) / denominator) ** 2
            score = (1 - sqrt(s / len(actual_result[j]))) * 100
            evaluation_set.append(score)
    elif predict_term == 'short':
        y_cap = 0.2 * online_capacity
        for j in range(len(actual_result)):
            s = 0
            if len(actual_result[j]) == 0:
                continue
            for k in range(len(actual_result[j])):
                y_true = actual_result[j][k]
                y_pred = predict_result[j][k]
                denominator = max(y_true, y_cap)
                s += ((y_true - y_pred) / denominator) ** 2
            score = (1 - sqrt(s / len(actual_result[j]))) * 100
            evaluation_set.append(score)
    else:
        y_cap = 0.2 * online_capacity
        for j in range(int(len(actual_result)/96)):
            s = 0
            # 连续96次预测的第16个点组成一组，进行考核，多组求算数平均
            for k in range(96):
                y_true = actual_result[96*j+k]
                if len(y_true) == 0:
                    continue
                y_true = y_true[-1]
                y_pred = predict_result[96*j+k][-1]
                denominator = max(y_true, y_cap)
                s += ((y_true - y_pred) / denominator) ** 2
            score = (1 - sqrt(s / 96)) * 100
            evaluation_set.append(score)
    return numpy.mean(evaluation_set)


# GB_T_40607_2021，超短期第4小时；短期前24小时；中期第10天
def result_evaluation_GB_T_40607_2021_without_time_tag(predict_result, actual_result, online_capacity,
                                                       predict_term="ultra_short"):
    """
    依据国标GB/T 40607-2021标准, 计算全网风电/光伏功率预测性能指标要求：
        1. 超短期功率预测：第4小时>=95%
        2. 短期功率预测：日前>=90%
        3. 中期功率预测：第10日（第217-240小时）>=80%
    :param predict_result: 一段时间内每次预测结果集
    :param actual_result: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param predict_term: 预测类型:
        1. "ultra_short"：超短期功率预测
        2. "short"：短期功率预测
        3. “medium”：中期功率预测
    :return np.mean(evaluation_set): 日平均准确率为当日内全部超短期预测准确率的算术平均值
    """
    if predict_term == "short":
        return result_evaluation_Q_CSG1211017_2018_without_time_tag(predict_result, actual_result, online_capacity,
                                                                    predict_term="short")
    elif predict_term == "medium":
        evaluation_term = 96
    elif predict_term == "ultra_short":
        evaluation_term = 4
    else:
        logs.warning("预测时间尺度predict_term不是规定的格式！！！，应从'ultra_short', 'short', 'medium'选择一个，当前输入为："
                     + str(predict_term))
        return None

    evaluation_set = []
    for i in range(0, len(predict_result), 1):
        rmse = sqrt(mean_squared_error(
            numpy.array(actual_result[i][-evaluation_term:]), numpy.array(predict_result[i][-evaluation_term:]))
        )
        score = (1 - rmse / online_capacity) * 100
        evaluation_set.append(score)

    return numpy.mean(evaluation_set)


# 计算
def result_evaluation_continue_point(predict_result, actual_result, online_capacity, evaluation="capacity"):
    """
    评估预测精度-计算未来三天预测结果的准确率
    :param predict_result: 一段时间内每次预测结果集
    :param actual_result: 一段时间内每次预测功率时所对应的实际功率集
    :param online_capacity: 发电场开机容量
    :param evaluation: 评估预测标准，"capacity"为基于Q/CSG1211017-2018标准，计算RMSE分母时使用发电场开机容量，
                        "power"为基于南网总调AI比赛新算法，计算RMSE分母时使用0.2*实际功率与发电场开机容量之间的最大值
    :return np.mean(evaluation_set): 未来三天288个点预测结果取平均值
    """
    if actual_result is None or len(actual_result) == 0 or predict_result is None or len(predict_result) == 0:
        return None

    evaluation_set = []
    if evaluation == "capacity":
        for i in range(0, len(predict_result), 1):
            if actual_result[i] is None or len(actual_result[i]) == 0:
                continue
            rmse = sqrt(mean_squared_error(
                numpy.array(actual_result[i]), numpy.array(predict_result[i]))
            )
            score = (1 - rmse / online_capacity) * 100
            evaluation_set.append(score)
    else:
        y_cap = 0.2 * online_capacity
        for j in range(len(actual_result)):
            s = 0
            number_point = 0
            for k in range(len(actual_result[j])):
                y_true = actual_result[j][k]
                y_pred = predict_result[j][k]
                if y_true is None or y_pred is None:
                    continue
                number_point += 1
                denominator = max(y_true, y_cap)
                s += ((y_true - y_pred) / denominator) ** 2
            score = (1 - sqrt(s / number_point)) * 100
            evaluation_set.append(score)

    return numpy.mean(evaluation_set)

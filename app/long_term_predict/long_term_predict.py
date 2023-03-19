from datetime import datetime
import calendar
import pandas
import numpy
from dateutil.relativedelta import relativedelta


class LongTermPred:
    def __init__(self):
        self.db_monthly_power_generation = None
        self.db_historical_nwp_solar = None
        self.db_historical_nwp_wind = None
        self.db_matched_nwp_power_solar = None
        self.db_matched_nwp_power_wind = None

    def load_database(self, monthly_power_generation: pandas.DataFrame, historical_nwp_solar: pandas.DataFrame,
                      historical_nwp_wind: pandas.DataFrame, matched_nwp_power_solar: pandas.DataFrame,
                      matched_nwp_power_wind: pandas.DataFrame):
        self.db_monthly_power_generation = monthly_power_generation
        self.db_historical_nwp_solar = historical_nwp_solar
        self.db_historical_nwp_wind = historical_nwp_wind
        self.db_matched_nwp_power_solar = matched_nwp_power_solar
        self.db_matched_nwp_power_wind = matched_nwp_power_wind

    def long_term_predict(self, station_name: str, station_type: str, station_capacity: float or list, area: str,
                          predict_start_time: str, feature_weight: dict = None, predict_length: int = 12, ):
        # 初始化预测结果，包含预测日期（date）、发电量（pred_power_generation）、当月容量（capacity）、预测方法（method）、匹配目标（match_target）
        # 用上年度的平均功率数据预测，method=1
        # 用平均的nwp特征预测，method=2
        # 以上两种方法都无法执行，既没有历史功率数据、又没有历史气象数据，用pandas自带的填充，method=3
        predict_result = pandas.DataFrame(
            columns=['area', 'name', 'type', 'year', 'month', 'capacity',
                     'pred_1', 'pred_2', 'pred_mix', 'match_target', 'filled',
                     'pred_hours_of_utilization'])
        predict_start_time = datetime.strptime(predict_start_time, "%Y-%m-%d %H:%M:%S")
        start_date = predict_start_time.date() + relativedelta(months=1)
        date_range = pandas.date_range(start=start_date, periods=predict_length, freq="M")
        predict_result['year'] = date_range.year
        predict_result['month'] = date_range.month
        predict_result['area'] = area
        predict_result['name'] = station_name
        predict_result['type'] = station_type

        # 获取预测开始的年、月，用于后续加载长期电量预测所需的数据池
        start_year = predict_start_time.year
        start_month = predict_start_time.month

        # 根据场站类型判断在哪个数据池查找
        if station_type == "光伏":
            tn1 = self.db_historical_nwp_solar  # 查历史NWP数据的表名称
            tn2 = self.db_matched_nwp_power_solar  # 匹配最佳数据的表名称
        else:
            tn1 = self.db_historical_nwp_wind
            tn2 = self.db_matched_nwp_power_wind

        # 判断所输入的场站类型是float还是list, 并根据预测的时长进行取舍
        if isinstance(station_capacity, list):
            if len(station_capacity) > predict_length:
                station_capacity = station_capacity[0: predict_length]
            elif len(station_capacity) < predict_length:
                station_capacity += [station_capacity[-1]] * (predict_length - len(station_capacity))
        # 将逐月容量填入预测结果中
        predict_result['capacity'] = station_capacity

        # 判断特征-权重是否为None，如果未输入该参数，用以下的默认权重
        if feature_weight is None:
            if station_type == '风电':
                feature_weight = {'WS': 0.9, "WD": 0.1}
            else:
                feature_weight = {'SR': 0.9, "TEM_hpa_700": 0.1}

        for i in range(predict_result.shape[0]):
            year = predict_result.loc[i, 'year']
            month = predict_result.loc[i, 'month']
            # 读取月均功率、电量数据
            df_monthly_power_database = self.db_monthly_power_generation.loc[
                (self.db_monthly_power_generation['year'] < start_year) &
                (self.db_monthly_power_generation['month'] == month) &
                (self.db_monthly_power_generation['name'] == station_name)
                ]
            # 若在数据池中找到了该场站往年的数据，那直接用往年同期数据来推测
            if not df_monthly_power_database.empty:
                df = df_monthly_power_database.iloc[[-1]].reset_index(drop=True)
                # 考虑容量的变化，因为去年的容量和预测月份的容量可能不一样，所以要有一个系数ap_ratio，即去年同期平均功率/当时的装机容量
                ap_ratio = df.loc[0, 'avg_power'] / df.loc[0, 'capacity']
                # 考虑不同月份天数变化，用calendar包自动根据预测的年和月算出对应的天数，乘以平均功率即为预测的电量
                predict_result.loc[i, 'pred_1'] = ap_ratio * predict_result.loc[
                    i, 'capacity'] * calendar.monthrange(year, month)[1] * 24

            # 读取月均nwp数据
            df_monthly_nwp_database = tn1.loc[
                (tn1['year'] < start_year) &
                (tn1['month'] == month) &
                (tn1['name'] == station_name)
                ]

            # 查看数据池中有无往年同期的历史NWP数据，如果有的话就用它按权重进行匹配
            if not df_monthly_nwp_database.empty:
                df_monthly_nwp_power_database = tn2.loc[
                    ((tn2['year'] < start_year) |
                     ((tn2['year'] == start_year) & (tn2['month'] < month))) &
                    ((tn2['name'] != station_name) | (tn2['month'] != month))
                    ]
                df = df_monthly_nwp_database.iloc[[-1]]
                year = df['year'].values[0]
                cols = list(feature_weight.keys())
                df_nwp = df[cols]
                score = numpy.sum(df_nwp.values * numpy.array(list(feature_weight.values())))
                df_target_database = df_monthly_nwp_power_database[cols + ['area', 'name', 'year', 'month',
                                                                           'capacity', 'avg_power']]
                score_list = numpy.sum(df_target_database.loc[:, cols].values * numpy.array(
                    list(feature_weight.values())), axis=1)
                df_target_database['score'] = numpy.abs(score_list - score)
                nearest_index = df_target_database['score'].idxmin()

                # 考虑容量的变化，因为去年的容量和预测月份的容量可能不一样，所以要有一个系数ap_ratio，即去年同期平均功率/当时的装机容量
                ap_ratio = df_target_database.loc[
                               nearest_index, 'avg_power'] / df_target_database.loc[nearest_index, 'capacity']
                # 考虑不同月份天数变化，用calendar包自动根据预测的年和月算出对应的天数，乘以平均功率即为预测的电量
                predict_result.loc[i, 'pred_2'] = ap_ratio * predict_result.loc[
                    i, 'capacity'] * calendar.monthrange(year, month)[1] * 24
                predict_result.loc[i, 'match_target'] = ",".join(
                    str(i) for i in df_target_database.loc[nearest_index,
                                                           ['area', 'name', 'year', 'month']].values.tolist())
        predict_result = long_term_predict_postprocess(predict_result)
        return predict_result


def long_term_predict_postprocess(predict_result):
    # 插值法补全预测结果
    priority_list = ['pred_1', 'pred_2']
    index_nan = pandas.isna(predict_result[priority_list[0]])
    predict_result['pred_mix'] = predict_result[priority_list[0]]
    predict_result.loc[index_nan, 'pred_mix'] = predict_result.loc[index_nan, priority_list[1]]
    index_still_nan = pandas.isna(predict_result['pred_mix'])
    predict_result['pred_mix'] = predict_result['pred_mix'].fillna(method='pad')
    predict_result['pred_mix'] = predict_result['pred_mix'].fillna(method='ffill')
    predict_result['pred_mix'] = predict_result['pred_mix'].fillna(method='bfill').round(4)
    predict_result.loc[index_still_nan, 'filled'] = 1
    predict_result['pred_hours_of_utilization'] = (predict_result['pred_mix'] / predict_result['capacity']).round(4)
    return predict_result

from datetime import datetime, timedelta
import pandas
import numpy as np
from common.logger import logs
from common.tools import load_model
from fbprophet import Prophet
from common.data_postprocess import result_post_process


class MidTermPred:
    def __init__(self):
        self.generation_type = None
        self.id = None
        self.name = None
        self.capacity = None
        self.type = None
        self.sr_col = None
        self.use_cols = None
        self.model_save_path = None
        self.fitted_nwp_based_model_name = None
        self.fitted_nwp_based_model_state = None
        self.fitted_nwp_based_model = None
        self.available_model_list = ['BPNN', 'GBRT', 'DecisionTreeRegressor', 'RandomForest', 'XGBoost', 'SVM']
        self.use_nwp_based_model = False
        self.total_predict_length = None
        self.nwp_for_predict = None
        self.h_power_for_prediction = None
        self.max_nwp_length = None
        self.max_historical_power_length_for_prediction = None
        self.predict_start_time = None
        self.prediction_result: pandas.DataFrame = pandas.DataFrame()
        self.pred_real_power_for_test: pandas.DataFrame = pandas.DataFrame(columns=['start_time', 'forecast_time',
                                                                                    'pred_power', 'real_power'])
        self.day_time = None
        self.bin_num = None
        self.test_evaluation = None

    def set_configure(self,
                      id: int,
                      name: str,
                      capacity: float,
                      type: str,
                      sr_col: int,
                      usecols: str,
                      model_savepath: str,
                      fitted_nwp_based_model_name: str,
                      predict_start_time=None,
                      total_predict_length=None,
                      max_nwp_length=656,
                      max_historical_power_length_for_prediction=10000,
                      bin_num=50,
                      day_time=None,
                      ):
        if day_time is None:
            day_time = [6, 18]
        if total_predict_length is None:
            total_predict_length = 960
        self.id = id
        self.name = name
        self.capacity = capacity
        self.type = type
        self.sr_col = sr_col
        self.use_cols = eval(usecols)
        self.predict_start_time = predict_start_time
        self.total_predict_length = total_predict_length
        self.model_save_path = model_savepath
        self.max_nwp_length = max_nwp_length
        self.max_historical_power_length_for_prediction = max_historical_power_length_for_prediction
        self.bin_num = bin_num
        self.day_time = day_time
        if fitted_nwp_based_model_name:
            self.fitted_nwp_based_model_name, self.fitted_nwp_based_model_state = fitted_nwp_based_model_name.split('_',
                                                                                                                    1)
        else:
            self.fitted_nwp_based_model_name = None
            self.fitted_nwp_based_model_state = None
        self.check_nwp_based_model_available()

    def _init_predict_result(self):
        columns = ['id', 'predict_term', 'model_name', 'start_time', 'forecast_time', 'predict_power', 'upper_bound_90',
                   'lower_bound_90', 'upper_bound_80', 'lower_bound_80', 'upper_bound_70', 'lower_bound_70',
                   'upper_bound_60', 'lower_bound_60', 'upper_bound_50', 'lower_bound_50']
        self.prediction_result = pandas.DataFrame(columns=columns)
        forecast_time = generate_start_predict_time(self.predict_start_time)
        forecast_time_range = pandas.date_range(start=forecast_time, periods=self.total_predict_length, freq='15min')
        self.prediction_result['forecast_time'] = forecast_time_range
        self.prediction_result['predict_term'] = 'medium'

    def check_nwp_based_model_available(self):
        """
        通过检测最优模型路径是否存在，判断是否要用基于nwp的模型开展预测
        step1：判断所输入的模型是否在系统默认的可用模型库范围内
        step2：若输入的模型不可用，则自动遍历本地是否有已经训练好的可用模型，并选取检测到的第一个可用模型作为nwp预测模型
        step3：如果没有可以用的nwp模型，那么将self.use_nwp_based_model置位False，后续完全用历史功率开展预测
        """
        if self.fitted_nwp_based_model_name in self.available_model_list:
            try:
                self.__load_fitted_nwp_based_model()
                self.use_nwp_based_model = True
                msg = f"已成功加载{self.name}的短期预测模型！"
                logs.info(msg)
            except Exception as err:
                logs.error(err, exc_info=True)
        else:
            msg = f"{self.name}的短期预测模型不在可用模型范围内!"
            logs.info(msg)

    def __load_fitted_nwp_based_model(self):
        """
        加载基于NWP的短期预测模型，如果加载失败，将完全基于历史功率进行中期预测
        """
        predict_term = 'short'
        sub_dir_path = "%s%s/%s/%s/" % (
            self.model_save_path, str(self.id), predict_term, self.fitted_nwp_based_model_name)
        model_path = sub_dir_path + predict_term + "_" + self.fitted_nwp_based_model_name + ".pkl"
        usecols_path = sub_dir_path + predict_term + "_usecols.pkl"
        self.fitted_nwp_based_model = load_model(model_path)
        self.use_cols = load_model(usecols_path)

    def predict(self, nwp_data: np.ndarray = None, historical_power_data: pandas.DataFrame = None):
        self._init_predict_result()
        self.prediction_result['start_time'] = self.predict_start_time
        if nwp_data is None:
            nwp_data = self.nwp_for_predict
        if historical_power_data is None:
            historical_power_data = self.h_power_for_prediction
        if self.use_nwp_based_model:
            if nwp_data is None:
                msg = f"{self.name}未成功加载nwp数据，将完全基于历史数据进行中期预测！"
                predict_result = self.predict_without_nwp(historical_power_data)
                self.prediction_result['model_name'] = "TSD"
                logs.info(msg)
            else:
                msg = f"{self.name}成功加载nwp数据，将根据nwp可用长度调用AI模型+TSD进行组合预测！"
                predict_result = self.predict_with_nwp(nwp_data, historical_power_data)
                self.prediction_result['model_name'] = self.fitted_nwp_based_model_name + "+TSD"
                logs.info(msg)
        else:
            msg = f"未成功加载{self.name}的短期预测模型，将基于历史数据进行中期预测！"
            predict_result = self.predict_without_nwp(historical_power_data)
            self.prediction_result['model_name'] = "TSD"
            logs.info(msg)
        self.prediction_result['predict_power'] = predict_result
        return predict_result

    def predict_with_nwp(self, nwp_data, historical_power_data):
        predict_length_with_nwp = len(nwp_data)
        predict_length_remaining = self.total_predict_length - predict_length_with_nwp
        pred_func = "self.fitted_nwp_based_model." + "predict_" + self.fitted_nwp_based_model_state
        nearest_historical_power = historical_power_data.loc[historical_power_data.index[-1], 'power']
        if not pandas.isna(nearest_historical_power):
            predict_result_with_nwp = eval(pred_func)(feature=nwp_data,
                                                      history_power=nearest_historical_power,
                                                      irradiance_col=self.sr_col,
                                                      )
        else:
            predict_result_with_nwp = self.fitted_nwp_based_model.predict_without_history_power(
                feature=nwp_data,
                history_power=nearest_historical_power,
                irradiance_col=self.sr_col)
        predict_result_with_nwp = result_post_process(irradiance_threshold=10,
                                                      pre_processing_predict_result=predict_result_with_nwp,
                                                      capacity=self.capacity,
                                                      predict_type=self.type,
                                                      feature=nwp_data,
                                                      irradiance_col=0)

        if predict_length_remaining <= 0:
            msg = f"本次输入的NWP数据长度大于等于中期预测所需长度，系统自动取前{self.total_predict_length}数据作为中期预测结果！"
            logs.info(msg)
            return predict_result_with_nwp[:self.total_predict_length]
        else:
            df_predict_result_with_nwp = pandas.DataFrame(columns=['time', 'power'])
            df_predict_result_with_nwp['power'] = predict_result_with_nwp
            df_predict_result_with_nwp['time'] = pandas.date_range(start=self.predict_start_time,
                                                                   periods=predict_length_with_nwp,
                                                                   freq="15min")
            power_data = pandas.concat([historical_power_data, df_predict_result_with_nwp])
            predict_result_remaining = self.predict_without_nwp(power_data, predict_length_remaining)
            return np.append(predict_result_with_nwp, predict_result_remaining)

    def predict_without_nwp(self, power_data, predict_length=None):
        power_data.rename(columns={'power': 'y', 'time': 'ds'}, inplace=True)
        power_data['cap'] = 1
        power_data['y'] = power_data['y'] / self.capacity
        power_data['floor'] = 0
        if predict_length is None:
            predict_length = self.total_predict_length
        model = Prophet(changepoint_prior_scale=0.01)
        model.fit(power_data)
        future = model.make_future_dataframe(periods=predict_length, freq='15min', include_history=False)
        future['cap'] = 1
        future['floor'] = 0
        fcst = model.predict(future)
        fcst['ds'] = pandas.to_datetime(fcst['ds'])
        if self.type == 'solar':
            time_list = fcst['ds']
            night_time_list = time_list.apply(
                lambda x: True if (x.hour <= self.day_time[0]) or (x.hour >= self.day_time[1]) else False)
            fcst.loc[night_time_list, 'yhat'] = 0
        predict_result = fcst.loc[:, 'yhat'].values.flatten() * self.capacity
        return predict_result


def get_start_predict_time():
    predict_start_date = datetime.now().date() + timedelta(days=1)
    predict_start_time = datetime(
        year=predict_start_date.year,
        month=predict_start_date.month,
        day=predict_start_date.day,
        hour=0,
        minute=0,
        second=0
    )
    return predict_start_time


def generate_start_predict_time(input_time=None, to_str=False):
    if input_time is None:
        predict_start_date = datetime.now().date() + timedelta(days=1)
    else:
        if isinstance(input_time, str):
            predict_start_date = datetime.strptime(input_time, "%Y-%m-%d %H:%M:%S").date() + timedelta(days=1)
        elif isinstance(input_time, datetime):
            predict_start_date = input_time.date() + timedelta(days=1)
        else:
            logs.info('数据参数input_time类型错误！')
            return input_time

    predict_start_time = datetime(
        year=predict_start_date.year,
        month=predict_start_date.month,
        day=predict_start_date.day,
        hour=0,
        minute=0,
        second=0
    )
    if to_str:
        predict_start_time = predict_start_time.strftime("%Y-%m-%d %H:%M:%S")
    return predict_start_time


def prepare_nwp_data(df_data: pandas.DataFrame, predict_start_time, time_col=None, generation_type='solar'):
    if time_col is None:
        time_col = ['forecast_time', 'start_time']
    for col in time_col:
        try:
            df_data[col] = pandas.to_datetime(df_data[col])
        except Exception as err:
            logs.debug(err)
            return df_data
    df_data.sort_values(by=time_col, inplace=True)
    df_data.bfill(inplace=True)
    df_data.drop_duplicates(subset=['forecast_time'], inplace=True, keep='last')
    df_data.drop(columns=['start_time'], inplace=True)

    time_sta = predict_start_time
    time_end = df_data['forecast_time'].max()
    time_ser = pandas.date_range(start=time_sta, end=time_end, freq="15min")

    df_empty = pandas.DataFrame(columns=['forecast_time'])
    df_empty['forecast_time'] = time_ser.values

    df_new = pandas.merge(left=df_data, right=df_empty, left_on='forecast_time', right_on="forecast_time", how='outer')
    df_new.set_index('forecast_time', inplace=True)
    df_new = df_new.astype(float)
    df_new.interpolate(inplace=True)
    df_new.reset_index(inplace=True)
    df_new.rename(columns={'index': 'forecast_time'})
    if generation_type == 'solar':
        df_new['timelabel'] = df_new['forecast_time'].apply(dt2float)
    return df_new


def dt2float(dt: datetime):
    this_date = dt.date()
    base_dt = datetime(year=this_date.year, month=this_date.month, day=this_date.day, hour=0, minute=0, second=0)
    time_delta = (dt - base_dt) / timedelta(minutes=15)
    return time_delta / 96


def prepare_power_data(df_data: pandas.DataFrame, predict_start_time):
    df_data.drop_duplicates(subset=['time'], inplace=True)
    df_data['time'] = pandas.to_datetime(df_data['time'])

    if isinstance(predict_start_time, str):
        time_end = datetime.strptime(predict_start_time, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=15)
    else:
        time_end = predict_start_time - timedelta(minutes=15)
    time_sta = df_data['time'].min()
    time_ser = pandas.date_range(start=time_sta, end=time_end, freq="15min")

    df_empty = pandas.DataFrame(columns=['time'])
    df_empty['time'] = time_ser.values

    df_new = pandas.merge(left=df_data, right=df_empty, left_on='time', right_on="time", how='outer')
    df_new.sort_values(by=['time'], inplace=True, ignore_index=True)
    return df_new
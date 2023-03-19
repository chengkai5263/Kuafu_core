
from fbprophet import Prophet
from datetime import datetime, timedelta
import pandas


class PredictWithoutNwp:
    def __init__(self,
                 station_name: str,
                 station_id: int,
                 station_type: str,
                 station_capacity: float,
                 model_save_path: str,
                 bin_num: int,
                 predict_start_time: str,
                 predict_pattern: str,
                 day_time="08:00",
                 night_time='18:00',
                 ):
        self.station_name = station_name
        self.station_id = station_id
        self.station_type = station_type
        self.station_capacity = station_capacity
        self.model_save_path = model_save_path + station_name + '/short/'
        self.bin_num = bin_num
        self.predict_start_time = datetime.strptime(predict_start_time, "%Y-%m-%d %H:%M")
        self.predict_pattern = predict_pattern
        self.real_power = None
        self.day_time = datetime.strptime(day_time, "%H:%M")
        self.night_time = datetime.strptime(night_time, "%H:%M")
        self.predict_result = None
        self.__init_predict_result()

    def clean_power_data(self, cursor_fetchall):
        self.real_power = pandas.DataFrame(cursor_fetchall).rename(columns={0: 'ds', 1: 'y'})
        self.real_power['ds'] = pandas.to_datetime(self.real_power['ds'])
        self.real_power.sort_values(by='ds', inplace=True, ignore_index=True)
        self.real_power['y'] = self.real_power['y'].astype(float)
        self.real_power.loc[self.real_power['y'] < 0, 'y'] = 0
        self.real_power.loc[self.real_power['y'] > self.station_capacity, 'y'] = self.station_capacity

    def predict(self, predict_length: int):
        predict_start_time = self.predict_start_time
        if predict_length == 288:
            predict_term = "short"
        else:
            predict_term = "ultra_short"
        if self.predict_pattern == 'fast':
            self._tsd_predict(predict_length, predict_start_time, predict_term)
        else:
            self._persistence_predict(predict_length, predict_start_time, predict_term)
        self.__basically_correct_result()

    def _persistence_predict(self, predict_length, predict_start_time, predict_term):
        newest_historical_datetime = self.real_power['ds'].max()
        real_length = predict_length + int((predict_start_time - newest_historical_datetime) / timedelta(minutes=15))
        result = pandas.DataFrame(columns=['ds'])
        result['ds'] = pandas.date_range(start=newest_historical_datetime, periods=real_length, freq='15min')
        result = pandas.merge(left=result, right=self.real_power, on='ds', how='outer')
        result['date'] = result['ds'].apply(lambda x: x.date())
        result['time'] = result['ds'].apply(lambda x: x.time())
        result.sort_values(by=['time', 'date'], inplace=True)
        result.ffill(inplace=True)
        result.sort_values(by='ds', inplace=True)
        self.predict_result['forecast_time'] = result['ds'].values[-predict_length:]
        self.predict_result['predict_power'] = result['y'].values[-predict_length:]
        self.predict_result['model_name'] = "Persistence"
        self.predict_result['predict_term'] = predict_term
        self.predict_result['start_time'] = predict_start_time
        self.predict_result['id'] = self.station_id

    def _tsd_predict(self, predict_length, predict_start_time, predict_term):
        newest_historical_datetime = self.real_power['ds'].max()
        real_length = predict_length + int((predict_start_time - newest_historical_datetime) / timedelta(minutes=15))
        model = Prophet(changepoint_prior_scale=0.01)
        power_data = self.real_power
        power_data['cap'] = self.station_capacity
        power_data['floor'] = 0
        model.fit(power_data)
        future = model.make_future_dataframe(periods=real_length, freq='15min', include_history=False)
        future['cap'] = self.station_capacity
        future['floor'] = 0
        fst = model.predict(future)
        if self.station_type == 'solar':
            time_list = fst['ds']
            night_time_list = time_list.apply(
                lambda x: True if (x.time() <= self.day_time.time()) or (x.time() >= self.night_time.time()) else False)
            fst.loc[night_time_list, 'yhat'] = 0
        self.predict_result['forecast_time'] = fst['ds'].values[-predict_length:]
        self.predict_result['predict_power'] = fst['yhat'].values[-predict_length:]
        self.predict_result['model_name'] = "TSD"
        self.predict_result['predict_term'] = predict_term
        self.predict_result['start_time'] = predict_start_time
        self.predict_result['id'] = self.station_id

    def __init_predict_result(self):
        columns = ['id',
                   'predict_term',
                   'model_name',
                   'start_time',
                   'forecast_time',
                   'predict_power',
                   ]
        self.predict_result = pandas.DataFrame(columns=columns)

    def __basically_correct_result(self):
        self.predict_result.loc[self.predict_result['predict_power'] < 0, 'predict_power'] = 0
        self.predict_result.loc[
            self.predict_result['predict_power'] > self.station_capacity, 'predict_power'] = self.station_capacity

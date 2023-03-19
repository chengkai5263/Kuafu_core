# _*_ coding: utf-8 _*_

import os
import joblib
import tensorflow
from model import BaseModel
from keras import Sequential, layers
from keras.callbacks import ModelCheckpoint
import numpy as np


class LSTM(BaseModel):
    """
    LSTM模型，预测发电功率
    """
    def __init__(self, n_features=6, params={'optimizer': 'adam', 'activation': 'relu'}, predict_type="wind",
                 irradiance=None, without_history_power=None):
        """
        初始化。待完善
        """
        # 初始化父类
        super(LSTM, self).__init__(predict_type=predict_type, irradiance=irradiance,
                                   without_history_power=without_history_power)

        # 以下为本派生类的属性。若本派生类有自身的个性化属性时，请在下方添加相应的代码
        # 根据属性的访问权限，做相应的访问权限限制（如为本类私有的属性，命名时应以两个下划线(__)开头；
        # 如为本类保护的属性即允许其派生类访问，命名时应以一个下划线(_)开头；
        # 如为公有的属性，命名时应不以下划线(_)开头
        self.n_features = n_features
        optimizer = params['optimizer']
        activation = params['activation']

        # LSTM模型
        # 不使用历史功率数据的模型
        self.__model_lstm = Sequential([
            layers.LSTM(units=256, activation=activation, input_shape=(1, n_features), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=256, activation=activation, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=128, return_sequences=True),
            layers.LSTM(units=32),
            layers.Dense(1)
        ])
        self.__model_lstm.compile(optimizer=optimizer, loss='mse')

        # 使用历史功率数据的模型
        self.__model_lstm_with_power = Sequential([
            layers.LSTM(units=256, activation=activation, input_shape=(1, n_features + 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=256, activation=activation, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=128, return_sequences=True),
            layers.LSTM(units=32),
            layers.Dense(1)
        ])
        self.__model_lstm_with_power.compile(optimizer=optimizer, loss='mse')

    def create_dataset(self, feature_data, target_data, seq_len):
        features = []
        targets = []
        interval = 1
        if seq_len == 288:
            interval = 96
        for i in range(0, len(feature_data) - seq_len, interval):
            data = feature_data[i:i + seq_len]
            label = target_data[i:i + seq_len]
            features.append(data)
            targets.append(label)
        features = np.array(features)
        targets = np.array(targets)
        return features, targets

    def build_lstmmodel(self, optimizer='adam', activation='relu'):
        model = Sequential([
            layers.LSTM(units=256, activation=activation, input_shape=(1, self.n_features), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=256, activation=activation, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units=128, return_sequences=True),
            layers.LSTM(units=32),
            layers.Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def save_model(self, filename):
        """
        保存模型。
        tensorflow模型有一套自己的模型加载/保存方法，故需对本类的tensorflow实例成员属性进行单独处理
        """
        model = self.__model_lstm
        modelwith_power = self.__model_lstm_with_power
        # import tensorflow
        dir_path = os.path.dirname(filename)
        sub_dir_path = os.path.join(dir_path, "without_power")
        os.makedirs(sub_dir_path, exist_ok=True)
        model.save(sub_dir_path)

        sub_dir_path = os.path.join(dir_path, "with_power")
        os.makedirs(sub_dir_path, exist_ok=True)
        modelwith_power.save(sub_dir_path)

        # tensorflow模型、训练/测试相关的属性先临时存到临时变量，然后全部置None或删除，再保存以减小模型文件大小
        self.__model_lstm = None
        self.__model_lstm_with_power = None
        joblib.dump(self, filename=filename)
        # 恢复原状
        self.__model_lstm = model
        self.__model_lstm_with_power = modelwith_power

    def load_model(self, dir_path):
        """
        加载模型。
        """
        self.__model_lstm = tensorflow.keras.models.load_model(os.path.join(dir_path, "without_power"))
        self.__model_lstm_with_power = tensorflow.keras.models.load_model(
            os.path.join(dir_path, "with_power"))

    def fit(self, feature_data, target_data, *args, **kwargs):
        """
        模型训练。当有传特征集、目标集数据进来时，则使用这两个数据对模型进行训练，否则使用类自身的特征集、目标集数据进行训练
        :param feature_data: 特征集（即天气预报数据）
                            数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64
        :param target_data: 目标集（即功率数据）
                            数据类型为ndarray，大小为m*1，即m行1列。列的含义是代表功率值。每个元素的类型是float64
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                    （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
                       当厂家不提供辐照度阈值时，若要从天气预报数据中估算该值，则需以irradiance_col=XXX的形式指明
                       辐照度特征在feature中列序号（从0开始计算），否则将取默认值10
                       说明：本模型可以输入预测周期predict_term，会被调用。当预测周期predict_term为超短期ultra_short时，
                       模型单次预测结果输出长度为seq_len = 16；
                       当预测周期predict_term为短期short时，模型单次预测结果输出长度为seq_len = 288；
                       当预测周期predict_term为5分钟超短期five_minute_ultra_short时，模型单次预测结果输出长度为seq_len = 48.
        :return:
        """
        epochs = kwargs["epochs"]
        batch_size = kwargs["batch_size"]
        predict_term = kwargs["predict_term"]
        if predict_term == "ultra_short":
            seq_len = 16
        elif predict_term == "short":
            seq_len = 288
        elif predict_term == "five_minute_ultra_short":
            seq_len = 48
        else:
            seq_len = 16

        features, targets = self.create_dataset(feature_data, target_data, seq_len)

        features = np.reshape(features, (features.shape[0] * features.shape[1], 1, features.shape[2]))
        targets = np.reshape(targets, (targets.shape[0] * targets.shape[1],))

        # checkpoint 保存最优一次运行结果
        checkpoint_file = "./work_dir/data/best_model_hdf5"
        checkpoint_callback = ModelCheckpoint(checkpoint_file,
                                              monitor='loss',
                                              mode='min',
                                              save_best_only=True,
                                              save_weights_only=True)
        print(str(checkpoint_callback)+'-----------------------------------------------------------------------------')

        # 不使用历史功率数据的模型
        if "without_history_power" not in kwargs or kwargs["without_history_power"] is None or kwargs["without_history_power"] is True:
            self.__model_lstm.fit(features, targets, epochs=epochs, batch_size=batch_size,
                                  callbacks=[checkpoint_callback])  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbe in position 121: invalid start byte

        if "predict_term" not in kwargs or kwargs["predict_term"] is None or kwargs["predict_term"] == 'ultra_short':
            # 使用历史功率数据的模型
            features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
            features_for_power_model = []
            targets_for_power_model = []
            start_index = 16
            step = 16
            end_index = start_index + step
            while end_index < features.shape[0]:
                feature = features[start_index:end_index]
                target = targets[start_index:end_index]
                history_target = targets[start_index - step:start_index]
                history_target = np.reshape(history_target, (history_target.shape[0], 1))
                new_feature = np.hstack((feature, history_target))
                features_for_power_model.append(new_feature)
                targets_for_power_model.append(target)
                start_index += step
                end_index += step

            features_for_power_model = np.array(features_for_power_model)
            targets_for_power_model = np.array(targets_for_power_model)
            features_for_power_model = np.reshape(features_for_power_model, (
            features_for_power_model.shape[0] * features_for_power_model.shape[1], 1, features_for_power_model.shape[2]))
            targets_for_power_model = np.reshape(targets_for_power_model,
                                                 (targets_for_power_model.shape[0] * targets_for_power_model.shape[1],))

            self.__model_lstm_with_power.fit(features_for_power_model,
                                             targets_for_power_model,
                                             epochs=epochs,
                                             batch_size=batch_size,
                                             callbacks=[checkpoint_callback]
                                             )

        # 对于光伏发电，若厂家没有提供光伏电场不发电时的辐照度阈值，需要自行计算
        if self._predict_type == "solar" and self._irradiance is None:
            if "irradiance_col" in kwargs:
                # 有提供辐照度特征在天气预报数据（特征集）中所在列的序号（序号从0开始算），按照一定算法计算辐照度阈值
                self._irradiance = self.compute_irradiance_gate(feature_data, target_data, kwargs["irradiance_col"])
            else:
                # 没有提供辐照度特征在天气预报数据（特征集）中所在列的序号，则按照经验给一个参考值（或做其他处理）
                self._irradiance = 10

    def predict_without_history_power(self, feature, *args, **kwargs):
        """
        不使用历史功率数据的功率预测。
        1、特征集（feature，即天气预报数据），不区分时段（白天黑夜）。光伏发电在处理时可能需要在方法实现的时候，注意将晚上的发电功率置为0。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
        3、此处不区分短期/超短期功率预测。派生类在实现时，若有需要，可通过len(feature) == 288、len(feature) == 16来
          区分短期/超短期预测
        4、此处不区分风电/光伏功率预测。若有需要，可在派生类实现。
        :param feature: （预测）特征集
                        数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64，
                        当是短期预测时，m=288；当是超短期预测时，m=16
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
                       当预测光伏发电功率时，需要以irradiance_col=XXX的形式指明辐照度特征在feature中列序号（从0开始计算）
        :return:
        """
        feature_reshape = np.reshape(feature, (feature.shape[0], 1, feature.shape[1]))
        result = self.__model_lstm.predict(feature_reshape, verbose=1)
        result = np.reshape(result, (result.shape[0],))

        # 光伏发电时，需要对光照较弱的情况（比如夜晚）进行特殊处理（发电功率置零）
        if self._predict_type == "solar" and "irradiance_col" in kwargs:
            for index in range(len(feature)):
                if feature[index][kwargs["irradiance_col"]] <= self._irradiance:
                    result[index] = 0

        return result

    def predict_with_history_power(self, feature, history_power, *args, **kwargs):
        """
        使用历史功率数据的（单次）功率预测。本模型没有使用历史功率的功率预测实现方法，返回None。
        1、特征集（feature，即天气预报数据），不区分时段（白天黑夜）。光伏发电在处理时可能需要在方法实现的时候，注意将晚上的发电功率置为0。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
        3、此处不区分短期/超短期功率预测。派生类在实现时，若有需要，可通过len(feature) == 288、len(feature) == 16来
          区分短期/超短期预测
        4、此处不区分风电/光伏功率预测。若有需要，可在派生类实现。
        :param feature: （预测）特征集
                        数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64，
                        当是短期预测时，m=288；当是超短期预测时，m=16
        :param history_power: 上一时刻历史功率数据。其数据类型为float64
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return:
        """
        # 本模型使用了历史功率的功率预测实现方法
        history = np.full((len(feature), 1), history_power)
        new_feature = np.hstack((feature, history))
        feature_reshape = np.reshape(new_feature, (new_feature.shape[0], 1, new_feature.shape[1]))
        result = self.__model_lstm_with_power.predict(feature_reshape, verbose=1)
        result = np.reshape(result, (result.shape[0],))

        # 光伏发电时，需要对光照较弱的情况（比如夜晚）进行特殊处理（发电功率置零）
        if self._predict_type == "solar" and "irradiance_col" in kwargs:
            for index in range(len(feature)):
                if feature[index][kwargs["irradiance_col"]] <= self._irradiance:
                    result[index] = 0

        return result

    def predict_mix_history_power(self, feature, history_power, num_with_history_power=7, *args, **kwargs):
        """
        融合不使用与使用历史功率数据的功率预测。一般不建议派生类重新实现该方法。
        取不使用历史功率数据时的预测数据的前N个数据，以及使用历史功率数据时的预测数据排除了前N个数据的数据，组成最终结果数据并返回。
        1、特征集（feature，即天气预报数据），不区分时段（白天黑夜）。光伏发电在处理时可能需要在方法实现的时候，注意将晚上的发电功率置为0。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
        3、此处不区分短期/超短期功率预测。派生类在实现时，若有需要，可通过len(feature) == 288、len(feature) == 16来
          区分短期/超短期预测
        4、此处不区分风电/光伏功率预测。若有需要，可在派生类实现。
        :param feature: （预测）特征集
        :param history_power: 上一时刻历史功率数据
        :param num_with_history_power: 融合不使用历史功率的预测结果和使用历史功率的预测结果时，后者数据长度所占最终结果数据长度的比例
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return:
        """
        without_history_result = self.predict_without_history_power(feature)
        if num_with_history_power <= 0:
            return without_history_result

        with_history_result = self.predict_with_history_power(feature, history_power)
        if with_history_result is None:
            return without_history_result

        return np.hstack((with_history_result[:num_with_history_power],
                          without_history_result[num_with_history_power:]))

# _*_ coding: utf-8 _*_

import os
import joblib
import tensorflow
from tensorflow import keras
from keras.layers import GRU as KERAS_GRU
import numpy
from model import BaseModel
from numpy import array


class GRU(BaseModel):
    """
    GUR模型，预测发电功率
    """
    def __init__(self, params={'gru_alpha': 0.35, 'epochs': 20}, predict_type="wind", irradiance=None):
        """
        初始化。待完善
        """
        # 初始化父类
        super(GRU, self).__init__(predict_type=predict_type, irradiance=irradiance)

        # 以下为本派生类的属性。若本派生类有自身的个性化属性时，请在下方添加相应的代码
        # 根据属性的访问权限，做相应的访问权限限制（如为本类私有的属性，命名时应以两个下划线(__)开头；
        # 如为本类保护的属性即允许其派生类访问，命名时应以一个下划线(_)开头；
        # 如为公有的属性，命名时应不以下划线(_)开头
        gru_alpha = params['gru_alpha']
        epochs = params['epochs']
        self.epochs = epochs

        # GRU模型
        self.__model_grua = keras.Sequential()
        self.__model_grua.add(KERAS_GRU(256))
        self.__model_grua.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua.add(keras.layers.Dense(256))
        self.__model_grua.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua.add(keras.layers.Dense(128))
        self.__model_grua.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua.add(keras.layers.Dense(64))
        self.__model_grua.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua.add(keras.layers.Dense(32))
        self.__model_grua.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua.add(keras.layers.Dropout(0.75))
        # 输出层
        self.__model_grua.add(keras.layers.Dense(1))
        self.__model_grua.add(keras.layers.LeakyReLU(alpha=0.3))
        # 定义优化器
        self.__model_grua.compile(loss='mse', metrics=['accuracy'])

        # GRU模型
        # 不带历史功率
        self.__model_grua_withoutpower = keras.Sequential()
        self.__model_grua_withoutpower.add(KERAS_GRU(256))
        self.__model_grua_withoutpower.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua_withoutpower.add(keras.layers.Dense(256))
        self.__model_grua_withoutpower.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua_withoutpower.add(keras.layers.Dense(128))
        self.__model_grua_withoutpower.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua_withoutpower.add(keras.layers.Dense(64))
        self.__model_grua_withoutpower.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua_withoutpower.add(keras.layers.Dense(32))
        self.__model_grua_withoutpower.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        self.__model_grua_withoutpower.add(keras.layers.Dropout(0.75))
        # 输出层
        self.__model_grua_withoutpower.add(keras.layers.Dense(1))
        self.__model_grua_withoutpower.add(keras.layers.LeakyReLU(alpha=0.3))
        # 定义优化器
        self.__model_grua_withoutpower.compile(loss='mse', metrics=['accuracy'])

    def build_grumodel(self, gru_alpha=0.35):
        model = keras.Sequential()
        model.add(KERAS_GRU(256))
        model.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.LeakyReLU(alpha=gru_alpha))
        model.add(keras.layers.Dropout(0.75))
        # 输出层
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        # 定义优化器
        model.compile(loss='mse', metrics=['accuracy'])
        return model

    def save_model(self, filename):
        """
        保存模型。
        tensorflow模型有一套自己的模型加载/保存方法，故需对本类的tensorflow实例成员属性进行单独处理
        """
        model = self.__model_grua
        modelwithout_power = self.__model_grua_withoutpower
        dir_path = os.path.dirname(filename)
        sub_dir_path = os.path.join(dir_path, "without_power")
        os.makedirs(sub_dir_path, exist_ok=True)
        modelwithout_power.save(sub_dir_path)

        sub_dir_path = os.path.join(dir_path, "with_power")
        os.makedirs(sub_dir_path, exist_ok=True)
        model.save(sub_dir_path)

        # tensorflow模型、训练/测试相关的属性先临时存到临时变量，然后全部置None或删除，再保存以减小模型文件大小
        self.__model_grua = None
        self.__model_grua_withoutpower = None
        joblib.dump(self, filename=filename)
        # 恢复原状
        self.__model_grua = model
        self.__model_grua_withoutpower = modelwithout_power

    def load_model(self, dir_path):
        """
        加载模型。
        """
        self.__model_grua_withoutpower = tensorflow.keras.models.load_model(os.path.join(dir_path, "without_power"))
        self.__model_grua = tensorflow.keras.models.load_model(
            os.path.join(dir_path, "with_power"))

    def fit(self, feature_data, target_data, *args, **kwargs):
        """
        模型训练。当有传特征集、目标集数据进来时，则使用这两个数据对模型进行训练，否则使用类自身的特征集、目标集数据进行训练
        :param feature_data: 特征集（即天气预报数据）
        :param target_data: 目标集（即功率数据）
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                    （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
                       当厂家不提供辐照度阈值时，若要从天气预报数据中估算该值，则需以irradiance_col=XXX的形式指明
                       辐照度特征在feature中列序号（从0开始计算），否则将取默认值10
                       说明：本模型需要输入predict_term，会调用。
                       当预测周期predict_term为超短期ultra_short时，
                       模型单次预测结果输出长度为predict_step = 16，预测间隔predict_inter = 1；
                       当预测周期predict_term为短期short时，
                       模型单次预测结果输出长度为predict_step = 288，预测间隔predict_inter = 96；
                       当预测周期predict_term为5分钟超短期five_minute_ultra_short时，
                       模型单次预测结果输出长度为predict_step = 48，预测间隔predict_inter = 1；
        :return:
        """
        if "predict_term" in kwargs:
            predict_term = kwargs["predict_term"]
        else:
            predict_term = "ultra_short"

        if predict_term == "ultra_short":
            predict_step = 16
            predict_inter = 1
        elif predict_term == "short":
            predict_step = 288
            predict_inter = 96
        elif predict_term == "five_minute_ultra_short":
            predict_step = 48
            predict_inter = 1
        else:
            # print("预测周期输入错误")
            predict_step = 16
            predict_inter = 1

        # 不使用历史功率数据的模型
        X, y = list(), list()
        in_start = 0
        in_start = in_start + predict_inter
        in_end = in_start + predict_step
        while in_end in range(len(feature_data)):
            Power_the_step = feature_data[in_start:in_end, :]
            X.append(Power_the_step)
            y.append(target_data[in_start:in_end])
            in_start = in_start + predict_inter
            in_end = in_start + predict_step
        X_fit = array(X)
        Y_fit = array(y)
        X_fit = X_fit.reshape((X_fit.shape[0] * X_fit.shape[1], 1, X_fit.shape[2]))
        Y_fit = Y_fit.reshape(Y_fit.shape[0] * Y_fit.shape[1], 1)
        Y_fitdata = Y_fit.flatten()
        # 模型适配
        self.__model_grua_withoutpower.fit(X_fit, Y_fitdata, epochs=self.epochs, batch_size=96)

        # 使用历史功率数据的模型
        X, y = list(), list()
        in_start = 0
        in_start = in_start + predict_inter
        in_end = in_start + predict_step
        while in_end in range(len(feature_data)):
            Power_the_step = numpy.c_[feature_data[in_start:in_end, :], target_data[in_start-1]*numpy.ones(predict_step)]
            X.append(Power_the_step)
            y.append(target_data[in_start:in_end])
            in_start = in_start + predict_inter
            in_end = in_start + predict_step
        X_fit = array(X)
        Y_fit = array(y)
        X_fit = X_fit.reshape((X_fit.shape[0] * X_fit.shape[1], 1, X_fit.shape[2]))
        Y_fit = Y_fit.reshape(Y_fit.shape[0] * Y_fit.shape[1], 1)
        Y_fitdata = Y_fit.flatten()
        # 模型适配
        self.__model_grua.fit(X_fit, Y_fitdata, epochs=self.epochs, batch_size=96)

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
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
                       当预测光伏发电功率时，需要以irradiance_col=XXX的形式指明辐照度特征在feature中列序号（从0开始计算）
        :return:
        """
        result = (self.__model_grua_withoutpower.predict(
            feature.reshape(len(feature), 1, feature.shape[1]))).flatten()

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
        :param history_power: 上一时刻历史功率数据
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return:
        """
        # result = []
        # 超短期每15min做16个点的预测 tic1 = time.perf_counter()
        input_x = (numpy.c_[feature, history_power * numpy.ones(len(feature))]).reshape(len(feature), 1,
                                                                                        feature.shape[1] + 1)
        result = (self.__model_grua.predict(input_x)).flatten()
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

        return numpy.hstack((with_history_result[:num_with_history_power],
                             without_history_result[num_with_history_power:]))

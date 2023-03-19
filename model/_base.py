# _*_ coding: utf-8 _*_

import numpy
from abc import ABCMeta
from abc import abstractmethod


class BaseModel(metaclass=ABCMeta):
    def __init__(self, predict_type="wind", irradiance=None, without_history_power=None):
        # 预测类型，取值范围为：wind,solar。分别代表风电预测和光伏发电预测
        self._predict_type = predict_type
        # 辐照度阈值。该变量值一般由厂家（光伏发电设备生厂商、光伏发电场站）提供，表示当太阳光辐照度低于该数值时，光伏设备不再发电
        # 取值为None时，代表厂家不提供该数值，后续需根据一定的算法计算出该值
        self._irradiance = irradiance
        self._with_history_power = without_history_power

    @abstractmethod
    def fit(self, feature_data, target_data, *args, **kwargs):
        """
        模型训练。派生类必须实现模型训练功能。
        1、特征集（feature_data，即天气预报数据），不区分时段（白天黑夜）。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
           如选取总辐射量作为标准，该值原则上由厂家（光伏设备生产商、光伏场站等）提供，并作为派生类的一个属性在初始化时保存起来；
           若厂家无法提供该值，则可以在本方法实现时，分析天气特征数据（总辐射量）与发电功率的关系，即当辐射量小于某个数值时，
           对应的发电功率为0（须剔除限光的情况），从而确定总辐射量临界值并保存起来供后续预测时使用
        3、预测功率时，有使用历史功率数据来预测未来功率、不使用历史功率数据来预测未来功率两种接口。在模型训练时，
          若需要对这两种情况进行区分训练，则请在实现本方法时自行区分实现（必要时可使用不同的属性进行区分）。
        :param feature_data: 特征集（即天气预报数据，数据为原始数据，未作标准化处理）
                            数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64
        :param target_data: 目标集（即功率数据）
                            数据类型为ndarray，大小为m*1，即m行1列。列的含义是代表功率值。每个元素的类型是float64
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                    （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return:
        """
        pass

    def predict(self, feature, history_power=None, history_rate=0.5, *args, **kwargs):
        """
        （单次）功率预测。一般不建议派生类重新实现该方法。
        1、特征集（feature，即天气预报数据），不区分时段（白天黑夜）。光伏发电在处理时可能需要在方法实现的时候，注意将晚上的发电功率置为0。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
        3、此处不区分短期/超短期功率预测。派生类在实现时，若有需要，可通过len(feature) == 288、len(feature) == 16来
          区分短期/超短期预测
        4、此处不区分风电/光伏功率预测。若有需要，可在派生类实现。
        :param feature: 特征集（即预测的天气预报数据，数据为原始数据，未作标准化处理）
                        数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64，
                        当是短期预测时，m=288；当是超短期预测时，m=16
        :param history_power: 上一时刻历史功率数据。为None时，返回不使用历史功率的预测结果
                        不为None时，其数据类型为float64
        :param history_rate: 融合不使用历史功率的预测结果和使用历史功率的预测结果时，后者数据长度所占最终结果数据长度的比例
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                    （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return: 返回功率预测结果，数据类型为ndarray，大小为m的一维数组
        """
        if history_power is None:
            return self.predict_without_history_power(feature, *args, **kwargs)
        else:
            return self.predict_mix_history_power(feature, history_power, history_rate, *args, **kwargs)

    @abstractmethod
    def predict_without_history_power(self, feature, *args, **kwargs):
        """
        不使用历史功率数据的（单次）功率预测。基类抽象方法，派生类必须实现该方法。
        1、特征集（feature，即天气预报数据），不区分时段（白天黑夜）。光伏发电在处理时可能需要在方法实现的时候，注意将晚上的发电功率置为0。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
        3、此处不区分短期/超短期功率预测。派生类在实现时，若有需要，可通过len(feature) == 288、len(feature) == 16来
          区分短期/超短期预测
        4、此处不区分风电/光伏功率预测。若有需要，可在派生类实现。
        :param feature: （预测）特征集（数据为原始数据，未作标准化处理）
                        数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64，
                        当是短期预测时，m=288；当是超短期预测时，m=16
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return: 返回功率预测结果，数据类型为ndarray，大小为m的一维数组
        """
        pass

    @abstractmethod
    def predict_with_history_power(self, feature, history_power, *args, **kwargs):
        """
        使用历史功率数据的（单次）功率预测。需要时，派生类可重新实现该方法。
        1、特征集（feature，即天气预报数据），不区分时段（白天黑夜）。光伏发电在处理时可能需要在方法实现的时候，注意将晚上的发电功率置为0。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
        3、此处不区分短期/超短期功率预测。派生类在实现时，若有需要，可通过len(feature) == 288、len(feature) == 16来
          区分短期/超短期预测
        4、此处不区分风电/光伏功率预测。若有需要，可在派生类实现。
        :param feature: （预测）特征集（数据为原始数据，未作标准化处理）
                        数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64，
                        当是短期预测时，m=288；当是超短期预测时，m=16
        :param history_power: 上一时刻历史功率数据。其数据类型为float64
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return: 返回功率预测结果，数据类型为ndarray，大小为m的一维数组
        """
        pass

    def predict_mix_history_power(self, feature, history_power, history_rate=0.5, *args, **kwargs):
        """
        融合不使用与使用历史功率数据的功率预测。一般不建议派生类重新实现该方法。
        取不使用历史功率数据时的预测数据的前N个数据，以及使用历史功率数据时的预测数据排除了前N个数据的数据，组成最终结果数据并返回。
        1、特征集（feature，即天气预报数据），不区分时段（白天黑夜）。光伏发电在处理时可能需要在方法实现的时候，注意将晚上的发电功率置为0。
        2、对于光伏发电，建议选取若干天气特征作为标准，当实际的天气特征低于标准时，则在预测的时候将相应的预测结果置0。
        3、此处不区分短期/超短期功率预测。派生类在实现时，若有需要，可通过len(feature) == 288、len(feature) == 16来
          区分短期/超短期预测
        4、此处不区分风电/光伏功率预测。若有需要，可在派生类实现。
        :param feature: （预测）特征集（数据为原始数据，未作标准化处理）
                         数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64，
                        当是短期预测时，m=288；当是超短期预测时，m=16
        :param history_power: 上一时刻历史功率数据。其数据类型为float64
        :param history_rate: 融合不使用历史功率的预测结果和使用历史功率的预测结果时，后者数据长度所占最终结果数据长度的比例
        :param args: 个数可变的参数集，类型为元组。
                     实现该方法时，若要使用元组中的参数值，可以通过下标（如args[0]）的方式或者遍历元组的方式
                     （for item in args）来使用。建议实现时，先对元组进行判断是否大小为0，非0时才进行相应的操作
        :param kwargs: 个数可变的参数集，类型为字典。
                       实现该方法时，若要使用字典中的参数值，可以通过key的方式直接获取值（如kwargs[ke])，或者遍历字典
                       （for key, value in kwargs.items()）的方式来获取。
        :return: 返回功率预测结果，数据类型为ndarray，大小为m的一维数组
        """
        without_history_result = self.predict_without_history_power(feature, *args, **kwargs)
        if history_rate <= 0:
            return without_history_result

        with_history_result = self.predict_with_history_power(feature, history_power, *args, **kwargs)
        if with_history_result is None:
            return without_history_result

        result_len = len(feature)
        history_len = int(result_len * history_rate)
        return numpy.concatenate((with_history_result[history_len:], without_history_result[:result_len - history_len]))

    @staticmethod
    def compute_irradiance_gate(feature, target, irradiance_col):
        """
        根据一定的规则，估算出光伏发电的辐照度阈值（即当辐照度低于阈值时，光伏发电设备的发电功率为0）。
        若功率数据中，没有为0的数值，则默认返回10。
        本方法实现的规则是：对照目标集，把发电功率为0的对应特征集中的辐照度数据找出来，从中选取最大值作为辐照度阈值。
        若有必要，派生类可另行重写该方法。
        :param feature: 特征集（二维数组），即天气预报数据，必须包含辐照度特征
                       数据类型为ndarray，大小为m*n，即m行n列。每一列都是天气特征（如100米风速）。每个元素的类型是float64，
        :param target: 目标集，即功率数据。
                       数据类型为ndarray，大小为m*1，即m行1列。列的含义是代表功率值。每个元素的类型是float64
        :param irradiance_col: 辐照度在特征集（二维数组）所在的列的序号（从0开始计算序号）
        :return: 辐照度阈值
        """
        flag_filter = {}
        for index in range(len(target)):
            key = feature[index][irradiance_col]
            if target[index] <= 0:
                if key not in flag_filter:
                    flag_filter[key] = 0
            else:
                flag_filter[key] = 1

        result = 10
        order_keys = list(flag_filter.keys())
        order_keys.sort()
        for key in order_keys:
            if flag_filter[key] <= 0:
                result = flag_filter[key]
            else:
                break
        return result

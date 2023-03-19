# _*_ coding: utf-8 _*_

from model import BaseModel
import numpy


class MFFS(BaseModel):
    """
    MFFS模型，预测发电功率
    gam：距离系数
    aph：惩罚系数
    """
    def __init__(self, params={'gam': 0.1, 'aph': 2.7}, predict_type="wind", irradiance=None):
        """
        初始化。待完善
        """
        # 初始化父类
        super(MFFS, self).__init__(predict_type=predict_type, irradiance=irradiance)

        # 以下为本派生类的属性。若本派生类有自身的个性化属性时，请在下方添加相应的代码
        # 根据属性的访问权限，做相应的访问权限限制（如为本类私有的属性，命名时应以两个下划线(__)开头；
        # 如为本类保护的属性即允许其派生类访问，命名时应以一个下划线(_)开头；
        # 如为公有的属性，命名时应不以下划线(_)开头
        gam = params['gam']
        aph = params['aph']
        self.__model_mffs = (gam, aph)
        self.__model_mffs_fit = None

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
                       说明：本模型可以输入predict_term，但不会被调用
        :return:
        """
        gam = self.__model_mffs[0]
        aph = self.__model_mffs[1]
        ma = numpy.max(feature_data, axis=0)
        mi = numpy.min(feature_data, axis=0)
        dpower = numpy.max(target_data, axis=0)

        input_train = feature_data.copy()
        n = len(feature_data)
        for i in range(n):
            input_train[i, :] = (input_train[i, :] - mi) / (ma - mi)
        self.__model_mffs_fit = (input_train, target_data.copy(), ma, mi, n, gam, aph, dpower)

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
        input_train = self.__model_mffs_fit[0]
        output_train = self.__model_mffs_fit[1]
        ma = self.__model_mffs_fit[2]
        mi = self.__model_mffs_fit[3]
        n = self.__model_mffs_fit[4]
        gam = self.__model_mffs_fit[5]
        aph = self.__model_mffs_fit[6]

        input_test = feature.copy()
        m = len(feature)
        for i in range(m):
            input_test[i, :] = (input_test[i, :] - mi) / (ma - mi)

        result = numpy.zeros(m)
        for i in range(m):
            a = numpy.ones((n, 1))
            t = input_test[i, :] * a
            d = ((input_train - t) ** 2).mean(axis=1) ** 0.5

            threshold = d.min() + gam * (d.max() - d.min())
            p1 = numpy.zeros(0)
            for j in range(n):
                if d[j] < threshold:
                    p1 = numpy.append(p1, numpy.array([aph ** (-d[j]), output_train[j]]), axis=0)
            p1 = p1.reshape(int(p1.size / 2), 2)
            result[i] = (p1[:, 0] * p1[:, 1]).sum(axis=0) / p1[:, 0].sum(axis=0)

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
        input_train = self.__model_mffs_fit[0]
        output_train = self.__model_mffs_fit[1]
        dpower = self.__model_mffs_fit[7]
        output_train = output_train / dpower
        input_train = numpy.hstack((input_train[1:, :], output_train[:-1].reshape(-1, 1)))
        output_train = output_train[1:]
        ma = self.__model_mffs_fit[2]
        mi = self.__model_mffs_fit[3]
        n = self.__model_mffs_fit[4]
        gam = self.__model_mffs_fit[5]
        aph = self.__model_mffs_fit[6]
        history_power = history_power / dpower

        input_test = feature.copy()
        m = len(feature)
        for i in range(m):
            input_test[i, :] = (input_test[i, :] - mi) / (ma - mi)

        result = numpy.zeros(m)
        for i in range(m):
            a = numpy.ones((n - 1, 1))
            t = numpy.hstack((input_test[i, :], history_power)).reshape(1, -1) * a
            d = ((input_train - t) ** 2).mean(axis=1) ** 0.5

            threshold = d.min() + gam * (d.max() - d.min())
            p1 = numpy.zeros(0)
            for j in range(n - 1):
                if d[j] < threshold:
                    p1 = numpy.append(p1, numpy.array([aph ** (-d[j]), output_train[j]]), axis=0)
            p1 = p1.reshape(int(p1.size / 2), 2)
            result[i] = (p1[:, 0] * p1[:, 1]).sum(axis=0) / p1[:, 0].sum(axis=0)
            history_power = 1 * result[i]
        result = result * dpower

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

        return numpy.hstack((with_history_result[:num_with_history_power],
                             without_history_result[num_with_history_power:]))
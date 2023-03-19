# _*_ coding: utf-8 _*_
import numpy
from numpy import array
import torch
from torch import nn, optim
from model import BaseModel


class BLSTMPytorch(BaseModel):
    """
    BLSTM模型，预测发电功率
    """

    def __init__(self, input_size, params={'n_hidden': 5}, predict_type="wind", irradiance=None):
        """
        初始化。待完善
        """
        # 初始化父类
        super(BLSTMPytorch, self).__init__(predict_type=predict_type, irradiance=irradiance)

        n_hidden = params['n_hidden']
        self.__model_blstm = blstmModel(input_size=input_size, n_hidden=n_hidden)

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
                       说明：本模型需要输入predict_term，会调用.
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
            print("预测周期输入错误")
            predict_step = 16
            predict_inter = 1

        X, y = list(), list()
        in_start = 0
        in_start = in_start + predict_inter
        in_end = in_start + predict_step
        while in_end in range(len(feature_data)):
            Power_the_step = numpy.c_[
                feature_data[in_start:in_end, :], target_data[in_start - 1] * numpy.ones(predict_step)]
            X.append(Power_the_step)
            y.append(target_data[in_start:in_end])
            in_start = in_start + predict_inter
            in_end = in_start + predict_step
        X_fit = array(X).reshape(len(X) * predict_step, 1, feature_data.shape[1] + 1)
        Y_fit = array(y)
        Y_fitdata = Y_fit.flatten().reshape(len(X) * predict_step, 1, 1)
        aaa = torch.from_numpy(X_fit)  # 变量名没有物理意义
        bbb = torch.from_numpy(Y_fitdata)
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.__model_blstm.parameters(), lr=1e-2)
        # 模型适配
        for epoch in range(1500):
            inputs = aaa.to(torch.float32)
            target = bbb.to(torch.float32)
            pred = self.__model_blstm(inputs)
            loss = criterion(pred.reshape(-1), target.reshape(-1))
            # if (epoch + 1) % 300 == 0:
            #     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        input_x = torch.from_numpy((numpy.c_[feature, feature[:, 0]]).reshape(len(feature), 1, feature.shape[1] + 1))
        input_x = input_x.to(torch.float32)
        result = array(self.__model_blstm(input_x).data).reshape(-1)

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
        input_x = (numpy.c_[feature, history_power * numpy.ones(len(feature))]).reshape(len(feature), 1,
                                                                                        feature.shape[1] + 1)
        input_x = torch.from_numpy(input_x)
        input_x = input_x.to(torch.float32)
        result = array(self.__model_blstm(input_x).data).reshape(-1)
        return result


class blstmModel(nn.Module):
    def __init__(self, input_size, n_hidden):
        super(blstmModel, self).__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=n_hidden, num_layers=2, bidirectional=True)
        # fc
        self.fc = nn.Linear(n_hidden * 2, 1)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, X):
        out = self.dropout(X)
        batch_size = out.shape[0]
        input = out.transpose(0, 1)

        hidden_state = torch.randn(1 * 4, batch_size,
                                   self.n_hidden)
        cell_state = torch.randn(1 * 4, batch_size,
                                 self.n_hidden)

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model

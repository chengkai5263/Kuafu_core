# _*_ coding: utf-8 _*_

from model import BaseModel
from itertools import chain
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)


class LSTMPytorch(BaseModel):
    """
    LSTM模型，预测发电功率
    """
    def __init__(self, input_size, predict_type="wind", predict_term="ultra_short",
                 params={'hidden_size': 256, 'num_layers':4, 'num_directions':1}, irradiance=None):
        """
        初始化。待完善
        """
        # 初始化父类
        super(LSTMPytorch, self).__init__(predict_type=predict_type, irradiance=irradiance)

        # 以下为本派生类的属性。若本派生类有自身的个性化属性时，请在下方添加相应的代码
        # 根据属性的访问权限，做相应的访问权限限制（如为本类私有的属性，命名时应以两个下划线(__)开头；
        # 如为本类保护的属性即允许其派生类访问，命名时应以一个下划线(_)开头；
        # 如为公有的属性，命名时应不以下划线(_)开头

        # LSTM模型
        output_size = 16
        if predict_term == 'short':
            output_size = 288
        elif predict_term == "five_minute_ultra_short":
            output_size = 48
        self.__model_lstm = LSTM(input_size, output_size, params).to(device)

    def create_dataset(self, feature_data, target_data, predict_term):
        print('---dataset process---')
        before = 32
        seq_len = 16
        if predict_term == 'short':
            before = 96
            seq_len = 288
        elif predict_term == "five_minute_ultra_short":
            before = 48
            seq_len = 48
        feature_data = feature_data.tolist()
        target_data = target_data.tolist()
        seq = []
        for i in range(0, len(feature_data) - before - seq_len, seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + before):
                x = [target_data[j]]
                for c in range(0, len(feature_data[0])-1):
                    x.append(feature_data[j][c])
                train_seq.append(x)
            for j in range(i + before, i + before + seq_len):
                train_label.append(target_data[j])
            train_seq = torch.DoubleTensor(train_seq)
            train_label = torch.DoubleTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        dataloader = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0)

        return dataloader

    def fit(self, feature_data, target_data, predict_type="wind", *args, **kwargs):
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
        predict_term = kwargs["predict_term"]
        dataloader_train = self.create_dataset(feature_data, target_data, predict_term)
        print("---fit---")
        loss_function = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(self.__model_lstm.parameters(), lr=0.05)
        # 训练
        epochs = 8
        if predict_type == "solar":
            epochs = 6
        for i in range(epochs):
            cnt = 0
            print('epoch:', i)
            for (seq, label) in dataloader_train:
                cnt += 1
                seq = seq.to(device)
                label = label.to(device)
                y_pred = self.__model_lstm(seq)
                loss = loss_function(y_pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cnt % 100 == 0:
                    print('epoch', i, ':', cnt - 100, '~', cnt, loss.item())
        # 对于光伏发电，若厂家没有提供光伏电场不发电时的辐照度阈值，需要自行计算
        if self._predict_type == "solar" and self._irradiance is None:
            if "irradiance_col" in kwargs:
                # 有提供辐照度特征在天气预报数据（特征集）中所在列的序号（序号从0开始算），按照一定算法计算辐照度阈值
                self._irradiance = self.compute_irradiance_gate(feature_data, target_data, kwargs["irradiance_col"])
            else:
                # 没有提供辐照度特征在天气预报数据（特征集）中所在列的序号，则按照经验给一个参考值（或做其他处理）
                self._irradiance = 10

    def predict_without_history_power(self, feature, predict_type="wind", *args, **kwargs):
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
        self.__model_lstm.eval()
        features_seq = torch.from_numpy(feature)
        seq = []
        pred = []
        seq.append(features_seq)
        dataloader_test = DataLoader(dataset=seq, batch_size=16, shuffle=False, num_workers=0)
        for (seq) in dataloader_test:
            y_pred = self.__model_lstm(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
        result = np.array(pred)
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
        self.__model_lstm.eval()
        features_seq = torch.from_numpy(feature)
        seq = []
        pred = []
        seq.append(features_seq)
        dataloader_test = DataLoader(dataset=seq, batch_size=16, shuffle=False, num_workers=0)
        for (seq) in dataloader_test:
            y_pred = self.__model_lstm(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
        result = np.array(pred)
        # 光伏发电时，需要对光照较弱的情况（比如夜晚）进行特殊处理（发电功率置零）
        if self._predict_type == "solar" and "irradiance_col" in kwargs:
            for index in range(len(feature)):
                if feature[index][kwargs["irradiance_col"]] <= self._irradiance:
                    result[index] = 0

        return result


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, params):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = params['hidden_size']
        self.num_layers = params['num_layers']
        self.num_directions = params['num_directions']
        self.batch_size = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print("input_seq.size()", input_seq.size())
        seq_len = input_seq.shape[1]
        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        # print('output.size=', output.size())
        # print("self.batch_size * seq_len, self.hidden_size", self.batch_size * seq_len, self.hidden_size)
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)
        pred = self.linear(output)
        # print('pred=', pred.shape)
        pred = pred.view(self.batch_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred

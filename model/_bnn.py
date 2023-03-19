# _*_ coding: utf-8 _*_
import torch
import torch.nn as nn
import numpy
from numpy import array
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from model import BaseModel


class BNN(BaseModel):
    """
    GUR模型，预测发电功率
    """

    def __init__(self, input_size, params={'hidden_units': 32, 'BNN_prior_var': 10}, predict_type="wind",
                 irradiance=None):
        """
        初始化。待完善
        """
        # 初始化父类
        super(BNN, self).__init__(predict_type=predict_type, irradiance=irradiance)

        # 以下为本派生类的属性。若本派生类有自身的个性化属性时，请在下方添加相应的代码
        # 根据属性的访问权限，做相应的访问权限限制（如为本类私有的属性，命名时应以两个下划线(__)开头；
        # 如为本类保护的属性即允许其派生类访问，命名时应以一个下划线(_)开头；
        # 如为公有的属性，命名时应不以下划线(_)开头
        # BNN模型
        hidden_units = params['hidden_units']
        BNN_prior_var = params['BNN_prior_var']
        # initialize the network like you would with a standard multilayer perceptron,
        # but using the BNN layer像使用标准多层感知器一样初始化网络，但使用BNN层
        self.__model_bnn = MLP_BNN(input_size=input_size, hidden_units=hidden_units, prior_var=BNN_prior_var)
        # 不带历史功率
        self.__model_bnn_withoutpower = MLP_BNN(input_size=input_size - 1, hidden_units=hidden_units, prior_var=BNN_prior_var)

    def get_model(self):
        return self.__model_bnn

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

        # 不带历史功率
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
        X_fit = array(X).reshape(len(X) * predict_step, feature_data.shape[1])
        Y_fit = array(y)
        Y_fitdata = Y_fit.flatten().reshape(len(X) * predict_step, 1)
        aaa = torch.tensor(X_fit)  # 变量名没有物理意义
        bbb = torch.tensor(Y_fitdata)
        # 模型适配
        optimizer = optim.Adam(self.__model_bnn_withoutpower.parameters(), lr=.1)
        epochs = 2000
        self.__model_bnn_withoutpower.double()
        for epoch in range(epochs):  # loop over the dataset multiple times
            optimizer.zero_grad()
            loss = self.__model_bnn_withoutpower.sample_elbo(aaa, bbb, 1)
            loss.backward()
            optimizer.step()

        # 带历史功率
        X, y = list(), list()
        in_start = 0
        in_start = in_start + predict_inter
        in_end = in_start + predict_step
        while in_end in range(len(feature_data)):
            Power_the_step = numpy.c_[
                feature_data[in_start:in_end, :],
                target_data[in_start-1]*numpy.ones(predict_step)
            ]
            X.append(Power_the_step)
            y.append(target_data[in_start:in_end])
            in_start = in_start + predict_inter
            in_end = in_start + predict_step
        X_fit = array(X).reshape(len(X)*predict_step, feature_data.shape[1]+1)
        Y_fit = array(y)
        Y_fitdata = Y_fit.flatten().reshape(len(X)*predict_step, 1)
        aaa = torch.tensor(X_fit)  # 变量名没有物理意义
        bbb = torch.tensor(Y_fitdata)
        # 模型适配
        optimizer = optim.Adam(self.__model_bnn.parameters(), lr=.1)
        epochs = 2000
        self.__model_bnn.double()
        for epoch in range(epochs):  # loop over the dataset multiple times
            optimizer.zero_grad()
            loss = self.__model_bnn.sample_elbo(aaa, bbb, 1)
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
        input_x = torch.tensor(feature.reshape(len(feature), 1, feature.shape[1]))
        result = (self.__model_bnn_withoutpower(input_x).detach().numpy()).flatten()

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
        input_x = (numpy.c_[feature, history_power*numpy.ones(len(feature))]).reshape(len(feature), 1, feature.shape[1]+1)
        input_x = torch.tensor(input_x)
        result = (self.__model_bnn(input_x).detach().numpy()).flatten()
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


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class MLP_BNN(nn.Module):
    def __init__(self, input_size, hidden_units=50, noise_tol=.1, prior_var=1.):
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.hidden = Linear_BBB(input_size, hidden_units, prior_var=prior_var)
        self.out = Linear_BBB(hidden_units, 1, prior_var=prior_var)
        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        # initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                target.reshape(-1)).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss

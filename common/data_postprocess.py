
import numpy


def result_post_process(capacity, predict_type, pre_processing_predict_result=None, feature=None, online_capacity=None,
                        irradiance_threshold=10, irradiance_col=None, restrict_record=None):
    """
    预测结果后处理
    :param irradiance_threshold: 辐照度阈值
    :param pre_processing_predict_result: 待处理的预测结果（一维的numpy数组）
    :param online_capacity: 开机容量
    :param capacity: 装机容量
    :param irradiance_col: 辐照度在feature的所在列的索引（序号从0开始）
    :param feature: 特征集（即天气预报数据）或添加历史功率的特征集
    :param predict_type: 预测类型，风电或光伏
    :param restrict_record: 限电记录
    :return: 输出处理后的预测结果pre_processing_predict_result（一维的numpy数组）
    """
    # 判断预测功率是否为空
    if pre_processing_predict_result is not None and feature is not None:
        if online_capacity is None:
            online_capacity = capacity

        pre_processing_predict_result = pre_processing_predict_result / capacity * online_capacity
        # 新能源场站发电功率不存在负数，因此预测的负功率须置零
        pre_processing_predict_result = numpy.maximum(pre_processing_predict_result, 0)
        # 新能源场站发电功率不能大于给定的开机容量，因此预测出大于开机容量的预测功率，置为开机容量
        pre_processing_predict_result = numpy.minimum(pre_processing_predict_result, online_capacity)

        # 判断预测的场站类型是否为光伏电站
        if predict_type == "solar" and irradiance_col is not None:
            # 正常情况下，在晚上或辐照度极小时，光伏电站的发电功率为0。因此，若在数值天气预报中辐照度低于阈值，则将预测功率置零
            pre_processing_predict_result[numpy.where(feature[:, irradiance_col] < irradiance_threshold)] = 0
        # 如果有限电记录，在预测结果和限电记录之间取较小值
        if restrict_record is not None and restrict_record.shape == pre_processing_predict_result.shape:
            pre_processing_predict_result = numpy.minimum(pre_processing_predict_result, restrict_record)

    return pre_processing_predict_result

import math
import shutil
import numpy
import os
from common.logger import logs


def copy_file(srcpath, targetpath):
    """
    输入：
    :param srcpath: 源文件路径
    :param targetpath: 目标文件路径
    功能：将文件以固定命名格式复制到目标路径下
    输出：无
    """
    # 原有的过渡模型如果存在，先删掉
    try:
        if os.path.exists(targetpath + '_temp'):
            shutil.rmtree(targetpath + '_temp')
    except Exception as err:
        logs.error(str(err), exc_info=True)
        logs.error("原有的过渡模型无法删除：" + targetpath + '_temp')
        return
    # 将源文件夹拷贝到目标地址，但文件夹名称带temp
    try:
        shutil.copytree(srcpath, targetpath + '_temp')
    except Exception as err:
        logs.error(str(err), exc_info=True)
        logs.error("新的模型模型无法拷贝：" + srcpath)
        return
    # 如果新的模型拷贝成功了，将旧模型删掉，同时将拷贝的新模型的名称改成标准模式
    if os.path.exists(targetpath + '_temp'):
        if os.path.exists(targetpath):
            shutil.rmtree(targetpath)
        shutil.move(targetpath + '_temp', targetpath)


def select_similar_farm(farm_database, select_mode, num_of_similar_farm, geography_base, capacity_base, weight, target_farm):
    """
    输入：
    farm_database——数据库中的dataframe,v1.0版拥有key：'longitude'，'latitude'，'online_capacity'，'farm_name'，'model_status'
    select_mode——模式选择，若值=1，则选择相似电场的最佳预测模型；否则，选择相似电场的指定模型
    num_of_similar_farm——相似电场数目，自定义输出的相似距离排在前num_of_similar_farm的电场的预测模型
    weight——计算综合相似距离时，赋予的各类距离的权重值，如经纬度距离权重、容量距离权重
    target_farm——目标电场的信息，v1.0版拥有key：'longitude'，'latitude'，'online_capacity'
    功能：计算数据库中每一个电场与目标电场的相对综合距离，取综合距离最小的一个或相似距离排在前num_of_similar_farm的电场名
    输出：综合距离最小的一个或相似距离排在前num_of_similar_farm的电场名,将最后选取的电场名字用similar_farm_list(list)存储并输出
    """
    farm_similarity = {}
    # distance_dic字典对数据库中所有电场相对目标电场的综合距离进行存储，键值对格式：综合距离(float) - 电场名(list)
    distance_dic = {}
    # distance_array数组对数据库中所有电场相对目标电场的综合距离进行存储，保存格式：综合距离数组(list)
    distance_array = []
    # 将最后选取的电场名字用similar_farm_list(list)存储并输出
    similar_farm_list = []

    # select_mode == 1，表示选择相似电场的最佳预测模型
    if select_mode == 1:
        # 兼容性判断v1.0，判断farm_database是否拥有key：'longitude'，'latitude'，'online_capacity'
        if farm_database.__contains__('longitude') and \
                farm_database.__contains__('latitude') and \
                farm_database.__contains__('capacity'):

            farm_similarity['geography_weight'] = weight[0]
            farm_similarity['capacity_weight'] = weight[1]

            farm_similarity['target_farm_longitude'] = target_farm['longitude']
            farm_similarity['target_farm_latitude'] = target_farm['latitude']
            farm_similarity['target_farm_capacity'] = target_farm['capacity']

            for indexs in farm_database.index:
                farm_similarity['similar_farm_longitude'] = farm_database.loc[indexs]['longitude']
                farm_similarity['similar_farm_latitude'] = farm_database.loc[indexs]['latitude']
                farm_similarity['similar_farm_capacity'] = farm_database.loc[indexs]['capacity']
                integrated_distance = cal_similarity_distance(geography_base, capacity_base, **farm_similarity)
                distance_dic.setdefault(integrated_distance, []).append(str(farm_database.loc[indexs]['id']))
                distance_array.append(integrated_distance)

            sorted_distance_array = sorted(distance_array)
            if num_of_similar_farm == 1:  # 选一个相似场站
                min_distance = numpy.min(distance_array)
                simialr_farm = distance_dic[min_distance]
                for val in simialr_farm:
                    similar_farm_list = numpy.append(similar_farm_list, val)
            else:
                for i in range(num_of_similar_farm):  # 选综合相似距离排前num_of_similar_farm的相似场站
                    simialr_farm = distance_dic[sorted_distance_array[i]]
                    for val in simialr_farm:
                        similar_farm_list = numpy.append(similar_farm_list, val)

        # 兼容性判断v2.0，判断farm_database是否拥有key：'similar_key1'
        elif farm_database.__contains__('similar_key1'):
            pass
        else:
            pass
    else:  # 选择相似电场的指定模型
        pass

    return similar_farm_list


def cal_similarity_distance(geography_base, capacity_base, **farm_similarity):
    """
    输入：farm_similarity字典
        farm_similarity字典结构v1.0拥有key：'target_farm_longitude'，'target_farm_latitude'，'target_farm_capacity'，
                                         'similar_farm_longitude'，'similar_farm_latitude'，'similar_farm_capacity'
    功能：计算目标电场与farm_similarity电场的相似距离
    输出：综合相似距离
    """
    # 兼容性判断v1.0，判断farm_similarity是否拥有key：'target_farm_longitude'，'target_farm_latitude'，'target_farm_capacity'
    if farm_similarity.__contains__('target_farm_longitude') and \
            farm_similarity.__contains__('target_farm_latitude') and \
            farm_similarity.__contains__('target_farm_capacity'):
        geography_distance = math.sqrt(  # 地理距离
            (farm_similarity['target_farm_longitude'] - farm_similarity['similar_farm_longitude']) ** 2 +
            (farm_similarity['target_farm_latitude'] - farm_similarity['similar_farm_latitude']) ** 2)

        online_capacity_distance = math.fabs(  # 开机容量距离
            farm_similarity['target_farm_capacity'] - farm_similarity['similar_farm_capacity'])

        # 计算综合距离
        integrated_distance = farm_similarity['geography_weight'] * geography_distance / geography_base + \
                              farm_similarity['capacity_weight'] * online_capacity_distance / capacity_base

    # 兼容性判断v2.0，判断farm_similarity是否拥有key：'similar_key1'
    elif farm_similarity.__contains__('similar_key1'):
        integrated_distance = []
    else:
        integrated_distance = []

    return integrated_distance

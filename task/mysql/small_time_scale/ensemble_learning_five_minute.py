# import ksycopg2
import os

import numpy
import pandas

from app.small_time_scale_predict.method_five_minute.NWP_interpolation import test_NI
from app.small_time_scale_predict.method_five_minute.NWP_multidimensional_mapping import test_NMM
from app.small_time_scale_predict.method_five_minute.history_power_iterative_predict import test_HPIP
from app.small_time_scale_predict.method_five_minute.power_cubic_spline_interpolation import test_PCSI
from app.small_time_scale_predict.method_five_minute.power_machinelearning_interpolation import test_PMI
from app.small_time_scale_predict.method_five_minute.power_nearest_interpolation import test_PNI
from app.small_time_scale_predict.training_five_minute import save_a_fitted_five_minute_model
from common.logger import logs
from common.tools import load_model
import pymysql
from common.tools import save_model
from task.mysql.small_time_scale.load_train_data_five_minute import LoadFiveMinuteEnsembledata


def fiveminute_modeltrain_based_15power(station_id, config_cluster, host='localhost', user='root',
                                        password='123456', database='kuafu', charset='utf8', port=3306,sql='kingbas'):
    """
    针对只有15分钟历史功率的模型，将训练好的15分钟模型转到插值法的模型
    :param station_name: 场站名称
    :param config: 配置信息
    :param model_name_cluster: 模型名称集合
    :param data_resource: 数据来源“CSV”，“SQL”
    :return:
    """
    # 最优的15分钟预测模型
    model_name, model_state = config_cluster[station_id]['best_model_ultra_short'].split('_', 1)
    best_fitted_model = load_model(config_cluster[station_id]['model_savepath'] + str(station_id) + '/ultra_short/' + model_name + '/'
                              + 'ultra_short' + '_' + model_name + '.pkl')
    best_fitted_model = eval('best_fitted_model' + '.predict_' + model_state)
    sub_dir_path = "%s%s%s" % (config_cluster[station_id]['model_savepath'], str(station_id), '/five_minute/')
    os.makedirs(sub_dir_path, exist_ok=True)
    # 保存直接功率插值法的模型
    save_model(best_fitted_model, sub_dir_path + "fifteen_minute_best_model.pkl")
    fifteen_model = load_model(sub_dir_path + "fifteen_minute_best_model.pkl")
    import time
    ta = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    time_label = ["updata_time", ta]
    time_label_dataframe = pandas.DataFrame(time_label)
    sub_dir_path = "%s%s%s" % ('./work_dir/log/', str(station_id), '/five_minute/')
    os.makedirs(sub_dir_path, exist_ok=True)
    time_label_dataframe.to_csv(sub_dir_path + "最优模型记录.csv", index=False,
                                header=False, mode='a+', encoding="utf_8_sig")
    # 最邻近插值法
    best_method = "PNI"
    best_model_name = "no_model_name"
    best_model_state = "no_model_state"
    best_accuracy = -100

    model_accuracy = [best_method + best_model_name + best_model_state, best_accuracy]
    model_accuracy_dataframe = pandas.DataFrame(model_accuracy,
                                                index=['best method and model for' + str(station_id) + 'five_minute',
                                                       'accuracy']).T
    model_accuracy_dataframe.to_csv('./work_dir/log/' + str(station_id) + "/five_minute/" + "最优模型记录.csv", index=False,
                                    header=True, mode='a+', encoding="utf_8_sig")
    logs.info(str(station_id) + 'five_minute' + "最优方法是" + best_method + "，最优模型是" + best_model_name + "，准确率是" + str(best_accuracy))

    fifth_model = fifteen_model
    best_model_and_method_list = [fifth_model, best_method, best_model_name, best_model_state]
    save_model(best_model_and_method_list,
               './work_dir/data/' + str(station_id) + "/five_minute/" + "five_minute_best_model.pkl")
    # 获取参与集成学习的模型名称

    if sql == "kingbas":
        db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
        c = db.cursor()
        c.execute("update configure set best_model_five_minute = %s where id = %s;",
                  (best_method + '_' + best_model_name + best_model_state, str(config_cluster[str(station_id)]['id'])))
        c.execute("update syy_ensemble_learning set accuracy = %s where station_id = %s;",
                  (best_accuracy, str(config_cluster[str(station_id)]['id'])))
        c.execute("update syy_ensemble_learning set best_model = %s where station_id = %s;",
                  (best_method + '_' + best_model_name, str(config_cluster[str(station_id)]['id'])))
        # c.close()
    else:
        db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
        c = db.cursor()
        c.execute("update syy_5_minute_ensemble_learning set best_model_five_minute = %s where id = %s;",
                  (best_method + '_' + best_model_name + best_model_state, config_cluster[station_id]['id']))
        c.execute("update syy_5_minute_ensemble_learning set accuracy = %s where station_id = %s;",
                  (best_accuracy, config_cluster[station_id]['id']))
        # c.execute("update syy_5_minute_ensemble_learning set best_model = %s where station_id = %s;",
        #           (best_method + '_' + best_model_name, config_cluster[station_id]['id']))
    c.execute("update syy_5_minute_ensemble_learning set status = %s where station_id = %s;",
              ('集成完成', str(config_cluster[station_id]['id'])))  # 状态列变2表示已完成
    c.execute("update syy_5_minute_ensemble_learning set ready_to_predict = 1 where station_id = %s;",
              (config_cluster[station_id]['id'],))  # 配置项中1表示模型已经训练好可以开始循环预测
    # logs.info(station_name + '小时间尺度集成学习已完成')
    logs.info(str(station_id) + ':small scale prediction finished')
    db.commit()
    db.close()


def one_station_ensemble_learn_five_minute(host='localhost', user='root', password='123456', database='kuafu',
                                           charset='utf8', port=3306, start_time_str='2021/8/1 00:00',
                                           end_time_str='2021/12/12 00:00', rate=0.75, sql="kingbas", station_id=None,
                                           config_cluster=None, model_name_cluster=None, model_state_cluster=None):
    """

    """
    # --------------------------------------------加载数据---------------------------------------------------------------
    train_data_load_five_minute = LoadFiveMinuteEnsembledata()
    # 从SQL读取训练数据
    ensemble_feature, ensemble_target = train_data_load_five_minute.load_train_data_for_five_minute_sql(host=host,
                                          user=user, password=password, database=database, charset=charset, port=port,
                                          file_path=config_cluster[station_id], usecols=config_cluster[station_id][
                                          'usecols'], rate=rate, predict_type=config_cluster[station_id]['type'],
                                          sql=sql, start_time_str=start_time_str, end_time_str=end_time_str)
    nwp_15min_row, nwp_15min_tcc = train_data_load_five_minute.upload_predict_nwp_five_minute_for_cloud_predict(host=host,
                                                                                                        user=user,
                                                                                                        password=password,
                                                                                                        database=database,
                                                                                                        charset=charset,
                                                                                                        port=port,
                                                                                                        file_path=
                                                                                                        config_cluster[
                                                                                                            station_id],
                                                                                                        usecols=
                                                                                                        config_cluster[
                                                                                                            station_id][
                                                                                                            'usecols'],
                                                                                                        rate=rate,
                                                                                                        predict_type=
                                                                                                        config_cluster[
                                                                                                            station_id][
                                                                                                            'type'],
                                                                                                        sql=sql,
                                                                                                        start_time_str=start_time_str,
                                                                                                        end_time_str=end_time_str)

    if numpy.size(ensemble_target) == 1:
        fiveminute_modeltrain_based_15power(station_id=station_id, config_cluster=config_cluster, host=host,
                            user=user, password=password, database=database, charset=charset, port=port, sql=sql)
    else:
        ensemble_target = ensemble_target.flatten()
        # 训练集和测试集样本拆分
        train_feature = ensemble_feature[0:int(len(ensemble_feature) * rate), :]
        test_feature = ensemble_feature[int(len(ensemble_feature) * rate):, :]
        train_target = ensemble_target[0:int(len(ensemble_feature) * rate) * 3:]
        test_target = ensemble_target[int(len(ensemble_feature) * rate) * 3:]
        # 模型训练
        save_a_fitted_five_minute_model(station_id=station_id, config=config_cluster[station_id],
                                        model_name_cluster=model_name_cluster,
                                        train_feature=train_feature, train_target=train_target,
                                        nwp_15min_row=nwp_15min_row, nwp_15min_tcc=nwp_15min_tcc)
        sub_dir_path = "%s%s%s" % (config_cluster[station_id]['model_savepath'], str(station_id), '/five_minute/')
        os.makedirs(sub_dir_path, exist_ok=True)
        fifteen_model = load_model(
            sub_dir_path + "fifteen_minute_best_model.pkl")

        best_accuracy = 0
        best_method = "CSI"
        best_model_name = "_"
        best_model_state = "_"
        NMM_result = []
        NI_result = []
        CP_result = []
        PMI_result = []
        HPIP_result = []
        PCSI_result = []
        PNI_result = []
        # 加载跟不同算法模型相关的方法并测试起精确度
        for model_name in model_name_cluster:
            for model_state in model_state_cluster:
                # 获取模型NMM的预测功率值（组）,用NWP多点功率映射法
                predict_result1, accuracy1 = test_NMM(config=config_cluster[station_id], test_feature=test_feature,
                                                      station_id=station_id, test_target=test_target,
                                                      model_name=model_name + model_state)
                if accuracy1 > best_accuracy:
                    best_accuracy = accuracy1
                    best_method = "NMM"
                    best_model_name = model_name
                    best_model_state = model_state
                    NMM_result = predict_result1

                # 获取模型NI的预测功率值（组）,NWP插值法
                predict_result2, accuracy2 = test_NI(config=config_cluster[station_id], test_feature=test_feature,
                                                     station_id=station_id, test_target=test_target,
                                                     model_name=model_name + model_state)
                if accuracy2 > best_accuracy:
                    best_accuracy = accuracy2
                    best_method = "NI"
                    best_model_name = model_name
                    best_model_state = model_state
                    NI_result = predict_result2

                # if config_cluster[station_name]["type"] == "solar":  # 云团追踪法
                # predict_result5, accuracy5 = test_CP(config=config_cluster[station_name], test_feature=test_feature,
                #                                      station_name=station_name, test_target=test_target,
                #                                      model_name=model_name + model_state)
                # if accuracy5 > best_accuracy:
                #     best_accuracy = accuracy5
                #     best_method = "CP"
                #     best_model_name = model_name
                #     best_model_state = model_state
                #     CP_result = predict_result5
            # 功率数据机器学习插值法
            predict_result3, accuracy3 = test_PMI(config=config_cluster[station_id], test_feature=test_feature,
                                                  station_id=station_id, test_target=test_target,
                                                  fifteen_model=fifteen_model,
                                                  model_name=model_name)
            if accuracy3 > best_accuracy:
                best_accuracy = accuracy3
                best_method = "PMI"
                best_model_name = model_name
                best_model_state = "no_model_state"
                PMI_result = predict_result3
            # 历史功率时序预测法
            predict_result4, accuracy4 = test_HPIP(config=config_cluster[station_id],
                                                   station_id=station_id, test_target=test_target,
                                                   train_target=train_target, model_name=model_name)
            if accuracy4 > best_accuracy:
                best_accuracy = accuracy4
                best_method = "HPIP"
                best_model_name = model_name
                best_model_state = "no_model_state"
                HPIP_result = predict_result4

        # 加载跟直接功率插值法的模型方法并测试起精确度。三次条样插值法
        predict_result6, accuracy6 = test_PCSI(config=config_cluster[station_id], test_feature=test_feature,
                                               test_target=test_target, fifteen_model=fifteen_model)
        if accuracy6 > best_accuracy:
            best_accuracy = accuracy6
            best_method = "PCSI"
            best_model_name = "no_model_name"
            best_model_state = "no_model_state"
            PCSI_result = predict_result6
        # 最邻近插值法
        predict_result7, accuracy7 = test_PNI(config=config_cluster[station_id], test_feature=test_feature,
                                              test_target=test_target, fifteen_model=fifteen_model)
        if accuracy7 > best_accuracy:
            best_accuracy = accuracy7
            best_method = "PNI"
            best_model_name = "no_model_name"
            best_model_state = "no_model_state"
            PNI_result = predict_result7
        best_accuracy = round(best_accuracy, 2)
        # ------------------------------------------------------------------------------------------------------------------
        # 保存最优模型
        import time
        logs.info(str(station_id) + 'five_minute' + "最优方法是" + best_method + "，最优模型是" + best_model_name + "，准确率是" + str(best_accuracy))

        if best_model_name == "no_model_name":  # 没有模型名称，说明是两种插值的一种，最优模型是15分钟的模型
            fifth_model = fifteen_model
        elif best_model_state == "no_model_state":  # 有模型名称，但没有模型状态，说明是历史功率时序预测法HPIP，最优模型加载带模型名称即可
            fifth_model = load_model(sub_dir_path + "five_minute_" + best_method + "_" + best_model_name + ".pkl")
        else:  # 有模型名称，有模型状态，最优模型加载带模型名称+模型状态
            fifth_model = load_model(sub_dir_path + "five_minute_" + best_method + "_" + best_model_name + best_model_state + ".pkl")

        best_model_and_method_list = [fifth_model, best_method, best_model_name, best_model_state]
        save_model(best_model_and_method_list,
                   sub_dir_path + "five_minute_best_model.pkl")
        # 获取参与集成学习的模型名称

        if sql == "kingbas":
            db = ksycopg2.connect(database=database, user=user, password=password, host=host, port=port)
            c = db.cursor()
            c.execute("update configure set best_model_five_minute = %s where id = %s;",
                      (best_method + '_' + best_model_name + best_model_state, str(config_cluster[station_id]['id'])))
            c.execute("update syy_5_minute_ensemble_learning set accuracy = %s where station_id = %s;",
                      (best_accuracy, str(config_cluster[station_id]['id'])))
            c.execute("update syy_5_minute_ensemble_learning set best_model = %s where station_id = %s;",
                      (best_method + '_' + best_model_name, str(config_cluster[station_id]['id'])))
            # c.close()
        else:
            db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
            c = db.cursor()
            c.execute("update syy_5_minute_ensemble_learning set best_model_five_minute = %s where station_id = %s;",
                      (best_method + '_' + best_model_name + best_model_state, config_cluster[station_id]['id']))
            c.execute("update syy_5_minute_ensemble_learning set accuracy = %s where station_id = %s;",
                      (best_accuracy, config_cluster[station_id]['id']))
            c.execute("update syy_5_minute_ensemble_learning set best_method_five_minute = %s where station_id = %s;",
                      (best_method, str(config_cluster[station_id]['id'])))
            # c.execute("update syy_5_minute_ensemble_learning set best_model = %s where station_id = %s;",
            #           (best_method + '_' + best_model_name, config_cluster[station_id]['id']))
        c.execute("update syy_5_minute_ensemble_learning set status = %s where station_id = %s;",
                  ('集成完成', str(config_cluster[station_id]['id'])))  # 状态列变2表示已完成
        c.execute("update syy_5_minute_ensemble_learning set ready_to_predict = 1 where station_id = %s;",
                  (str(config_cluster[station_id]['id']),))  # 配置项中1表示模型已经训练好可以开始循环预测
        # logs.info(station_name + '小时间尺度集成学习已完成')
        logs.info(str(station_id) + ':small scale prediction finished')
        db.commit()
        db.close()


def satic_vars(**kwargs):
    # 模拟测试数据迭代产生
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


if __name__ == '__main__':
    # trainning_five_minute_model(host='localhost', user='root', password='111111', database='kuafu', charset='utf8',
    #                             port=3306, rate=0.75,
    #                             set_path="./configure/5分钟训练配置文件.csv",
    #                             model_set_path="./configure/训练模型及初始参数配置文件.csv",
    #                             data_resource='CSV')
    # ensemble_learning_five_minute(days=1, start_time_str='2021/08/17 08:00')

    # ensemble_learning_five_minute(host='localhost', user='root', password='111111', database='kuafu', charset='utf8',
    #                               port=3306, rate=0.75)
    print('')

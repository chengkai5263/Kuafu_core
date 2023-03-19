
import pandas
import pymysql
import numpy
import time
from common.logger import logs
from common.tools import catch_exception
from app.transfer_learning.transfer_learning import copy_file
from app.transfer_learning.transfer_learning import select_similar_farm


@catch_exception("transfer_learning error: ")
def run_transfer_learning(host, user, password, database, charset, port):
    """
    功能：迁移学习
    输出：迁移学习后,将相似场站模型迁移复制到目标场站文件夹下
    """
    # 迁移学习的配置信息
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()

    c.execute("select id from configure where station_status = 3;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe_config = pandas.DataFrame(record, columns=coul)

    station_id_clusters = []
    for i in range(len(dataframe_config)):
        station_id_clusters.append(dataframe_config.loc[:, "id"][i])

    for station_id in station_id_clusters:
        c.execute("DELETE FROM parallel_tasks_station where station_id=%s and task_name='transfer';", station_id)
        db.commit()
        c.execute("INSERT INTO parallel_tasks_station (station_id, task_name, task_status)"
                  " VALUES (%s, %s, %s)", tuple([station_id, 'transfer', str(station_id) + '迁移学习开始']))
        db.commit()
        transfer_learning(host, user, password, database, charset, port, station_id=station_id)
    c.close()
    db.close()
    return station_id_clusters


@catch_exception("transfer_learning error: ")
def transfer_learning(host, user, password, database, charset, port, station_id=None):
    """
    功能：迁移学习
    输出：迁移学习后,将相似场站模型迁移复制到目标场站文件夹下
    """
    # 迁移学习的配置信息
    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    c = db.cursor()
    result = c.execute(
        "update parallel_tasks_station set task_status = '" + str(
            station_id) + "正在进行迁移学习' where task_name = 'transfer' and task_status = '" + str(
                            station_id) + "迁移学习开始' and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        c.execute("select * from configure where id = %s;", station_id)
        db.commit()
        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        transfer_config = pandas.DataFrame(record, columns=coul)

        if len(transfer_config) == 0:
            c.close()
            db.close()
            return

        if transfer_config['type'][0] == 'wind':
            c.execute("select * from configure where type = 'wind' and trained = 1;")
            db.commit()
            des = c.description
            record = c.fetchall()
            if len(record) == 0:  # 读数据库记录的时候，判断是否为空，如果是空要提前退出，2023/3/19
                c.close()
                db.close()
                return
            coul = list(iterm[0] for iterm in des)
            farm_database = pandas.DataFrame(record, columns=coul)
        else:
            c.execute("select * from configure where type = 'solar' and trained = 1;")
            db.commit()
            des = c.description
            record = c.fetchall()
            if len(record) == 0:  # 读数据库记录的时候，判断是否为空，如果是空要提前退出，2023/3/19
                c.close()
                db.close()
                return
            coul = list(iterm[0] for iterm in des)
            farm_database = pandas.DataFrame(record, columns=coul)

        # 模式选择，若值=1，则选择相似电场的最佳预测模型；否则，选择相似电场的指定模型
        try:
            select_mode = transfer_config['transfer_mode'][0]
            if select_mode is None:
                select_mode = 1
        except Exception as err:
            logs.error(str(err), exc_info=True)
            select_mode = 1
        # 相似电场数目，自定义输出的相似距离排在前num_of_similar_farm的电场的预测模型
        try:
            num_of_similar_farm = transfer_config['num_of_similar_farm'][0]
            if num_of_similar_farm is None:
                num_of_similar_farm = 1
            else:
                num_of_similar_farm = int(num_of_similar_farm)
        except Exception as err:
            logs.error(str(err), exc_info=True)
            num_of_similar_farm = 1
        # 计算综合相似距离时，赋予的各类距离的权重值，如经纬度距离权重、容量距离权重
        try:
            weight_temp = transfer_config['weight'][0]
            if weight_temp is None or len(weight_temp) == 0:
                weight = [0.5, 0.5]
            else:
                temp = weight_temp[0].strip()
                temp = temp.strip('[]')
                temp_arr = temp.split(",")
                weight = numpy.array(temp_arr)
                weight = weight.astype(float)
            if len(weight) < 2:
                weight = [0.5, 0.5]
        except Exception as err:
            logs.error(str(err), exc_info=True)
            weight = [0.5, 0.5]

        # 约定的训练模型保存名称格式
        target_farm = transfer_config

        similar_farm_list = select_similar_farm(farm_database, int(select_mode), num_of_similar_farm, 50, 50, weight,
                                                target_farm.iloc[0, :])

        if farm_database.__contains__('station_status'):
            for val in similar_farm_list:
                try:
                    copy_file(transfer_config['model_savepath'][0] + val + "/short",
                              transfer_config['model_savepath'][0] + str(target_farm['id'][0]) + "/short")
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.error("模型库中无" + val + "短期预测模型")

                try:
                    copy_file(transfer_config['model_savepath'][0] + val + "/ultra_short",
                              transfer_config['model_savepath'][0] + str(target_farm['id'][0]) + "/ultra_short")
                except Exception as err:
                    logs.error(str(err), exc_info=True)
                    logs.error("模型库中无" + val + "超短期预测模型")

                c.execute("update configure set station_status = '1' where id = %s;", target_farm['id'][0])
                db.commit()
            try:
                # 在结果表里添加记录
                result = c.execute("select id, predict_term from best_feature_parameters_and_model"
                                   " where id = %s and predict_term = %s;", (target_farm['id'][0], 'short'))
                db.commit()
                if result == 0:
                    c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                              " VALUES (%s, %s, %s)", tuple([target_farm['id'][0], 'short', target_farm['name'][0]]))
                    db.commit()

                result = c.execute("select id, predict_term from best_feature_parameters_and_model"
                                   " where id = %s and predict_term = %s;", (target_farm['id'][0], 'ultra_short'))
                db.commit()
                if result == 0:
                    c.execute("INSERT INTO best_feature_parameters_and_model (id, predict_term, station_name)"
                              " VALUES (%s, %s, %s)", tuple([target_farm['id'][0], 'ultra_short', target_farm['name'][0]]))
                    db.commit()

                # 将源场站的最优模型写到数据库
                c.execute("select best_model from best_feature_parameters_and_model"
                          " where id=%s and predict_term=%s;", (eval(similar_farm_list[0]), 'short'))
                db.commit()
                record = c.fetchall()
                best_model_name = record[0][0]

                c.execute("select second_model from best_feature_parameters_and_model"
                          " where id=%s and predict_term=%s;", (eval(similar_farm_list[0]), 'short'))
                db.commit()
                record = c.fetchall()
                second_model_name = record[0][0]

                c.execute("update best_feature_parameters_and_model set best_model = %s"
                          " where id = %s and predict_term=%s;",
                          (best_model_name, target_farm['id'][0], 'short'))
                db.commit()
                c.execute("update best_feature_parameters_and_model set second_model=%s"
                          " where id=%s and predict_term=%s;",
                          (second_model_name, target_farm['id'][0], 'short'))
                db.commit()

                c.execute("select best_model from best_feature_parameters_and_model"
                          " where id=%s and predict_term=%s;", (eval(similar_farm_list[0]), 'ultra_short'))
                db.commit()
                record = c.fetchall()
                best_model_name = record[0][0]

                c.execute("select second_model from best_feature_parameters_and_model"
                          " where id=%s and predict_term=%s;", (eval(similar_farm_list[0]), 'ultra_short'))
                db.commit()
                record = c.fetchall()
                second_model_name = record[0][0]

                c.execute("update best_feature_parameters_and_model set best_model = %s"
                          " where id = %s and predict_term=%s;",
                          (best_model_name, target_farm['id'][0], 'ultra_short'))
                db.commit()
                c.execute("update best_feature_parameters_and_model set second_model=%s"
                          " where id=%s and predict_term=%s;",
                          (second_model_name, target_farm['id'][0], 'ultra_short'))
                db.commit()

                # 将源场站的容量写到数据库
                c.execute("select capacity from configure where id=%s;", eval(similar_farm_list[0]))
                db.commit()
                record = c.fetchall()
                source_capacity_for_transfer = record[0][0]

                c.execute("update configure set source_capacity_for_transfer=%s where id=%s;",
                          (source_capacity_for_transfer, target_farm['id'][0]))
                db.commit()
            except Exception as err:
                logs.error(str(err), exc_info=True)
                db.rollback()
                logs.error("拷贝模型失败")
        else:
            logs.info("数据库中无电场模型状态！")

        c.execute(
            "update parallel_tasks_station set task_status = '" + str(
                station_id) + "迁移学习完成' where task_name = 'transfer' and task_status = '" + str(
                station_id) + "正在进行迁移学习' and station_id = %s;", station_id)
        db.commit()
        time.sleep(3)

    result = c.execute(
        "select task_status from parallel_tasks_station where task_name = 'transfer' and task_status = '" + str(
            station_id) + "迁移学习完成' and station_id = %s;", station_id)
    db.commit()

    while result == 0:
        task_stop_sign = task_stop(db, c, station_id)  # 如果数据库已将任务设为停止将终止运行
        if task_stop_sign == 1:
            return
        time.sleep(1)
        result = c.execute(
            "select task_status from parallel_tasks_station where task_name = 'transfer' and task_status = '" + str(
                station_id) + "迁移学习完成' and station_id = %s;", station_id)
        db.commit()

    c.close()
    db.close()


def task_stop(db, c, station_id):
    result = c.execute(
        "select id from parallel_tasks_station where task_name = 'transfer' and task_status = 'task_stopped'"
        " and station_id = %s;", station_id)
    db.commit()
    if result == 1:
        c.close()
        db.close()
    return result

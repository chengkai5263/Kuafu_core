
import pymysql
import pandas
import numpy
import datetime
from common.model_evaluation import evaluate_GB_T_40607_2021_withtimetag, \
    result_evaluation_Two_Detailed_Rules_without_time_tag
import math
from common.logger import logs
import time
import os

try:
    from pyecharts import options as opts
    from pyecharts.charts import Line, Tab, Page, Bar
except Exception as err:
    logs.error(str(err), exc_info=True)
    logs.info('加载pyecharts失败，不影响预测，但无法自动出图。')


def line_markpoint(predict_power=None, actual_power=None, title="page") -> Line:
    c = (
        Line()
            .add_xaxis(list(range(len(predict_power))))
            .add_yaxis(
            "预测功率/MW",
            [float('{:.2f}'.format(i)) for i in predict_power],
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]),
            label_opts=opts.LabelOpts(is_show=False),
        )
            .add_yaxis(
            "实际功率/MW",
            [float('{:.2f}'.format(i)) for i in actual_power],
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]),
            label_opts=opts.LabelOpts(is_show=False),
        )
            .set_global_opts(title_opts=opts.TitleOpts(title=title),
                             datazoom_opts=[opts.DataZoomOpts()],
                             legend_opts=opts.LegendOpts(pos_right=True))
    )
    return c


def bar_datazoom_slider(station_name, evaluations) -> Bar:
    bar = Bar()
    result_list = []
    result_list = numpy.append(result_list, evaluations)
    result_list.tolist()
    bar.add_xaxis(" ")
    bar.add_yaxis(" ", list(result_list))
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title=station_name, pos_top='bottom', pos_left='center'),
        legend_opts=opts.LegendOpts(pos_right=True),
    )
    return bar


def generate_page(predict_power_list, actual_power_list, station_name,
                  evaluation, save_file_path=None, save_page=False, id=None, titlename='超短期第4小时时刻全天曲线'):
    if save_file_path is None:
        save_file_path = f'./work_dir/{station_name}_{datetime.datetime.now().strftime("%Y_%m%d_%H%M")}.html'
    else:
        save_file_path = save_file_path + '_' +str(id) + '.html'
    page = Page(layout=Page.DraggablePageLayout)
    test_times = len(predict_power_list)
    page.add(bar_datazoom_slider(station_name + "预测模型的精度", evaluation))
    for i in range(test_times):
        if titlename == '超短期第4小时时刻全天曲线':
            title = f"测试环节的第{i + 1}天 超短期第4小时时刻全天曲线"
        elif titlename == '超短期预测结果':
            title = f"测试环节的第{i + 1}次超短期预测结果"
        else:
            title = f"测试环节的第{i + 1}次短期预测结果"
        predict_power = list(predict_power_list[i])
        actual_power = list(actual_power_list[i])
        page.add(line_markpoint(predict_power, actual_power, title))
    if save_page:
        page.render(save_file_path)
    return page


def evaluate(host, user, password, database, charset, port,
             model_name='GBRT', model_state='_without_history_power', term='short', save_file_path=None, station_id=None,
             scene='train'):
    """
    为结果存储文件撰写表头，初始化时调用
    :param host: 主机
    :param user: 用户名
    :param password: 密码
    :param database: 数据库名
    :param charset: 解码方式
    :param port: 端口号
    :param model_name: 模型
    :param model_state: 方法
    :param term: 预测时间尺度
    :return:
    """
    interval = 288
    if term == "medium":
        interval = 960
    if term == "ultra_short":
        interval = 16

    db = pymysql.connect(host=host, user=user, password=password, database=database, charset=charset, port=port)
    # 定义游标
    c = db.cursor()
    if station_id is not None:
        c.execute("select * from configure where station_status = 2 and id = %s;", station_id)
    else:
        c.execute("select * from configure where station_status = 2;")
    db.commit()
    des = c.description
    record = c.fetchall()
    coul = list(iterm[0] for iterm in des)
    dataframe = pandas.DataFrame(record, columns=coul)

    # 场站名称
    station_name_cluster = []
    for i in range(len(dataframe)):
        station_name_cluster.append(dataframe.loc[:, "name"][i])

    # 配置信息
    config_cluster = {}
    for i in range(len(dataframe)):
        config_cluster[dataframe.loc[:, "name"][i]] = {"id": dataframe.loc[:, 'id'][i],
                                                       "type": dataframe.loc[:, "type"][i],
                                                       "sr_col": 0,
                                                       "online_capacity": dataframe.loc[:, "capacity"][i],
                                                       "model_savepath": dataframe.loc[:, "model_savepath"][i]}

    accuracy_GB = {}
    accuracy_Two = {}
    for name in station_name_cluster:
        if scene == 'train':
            c.execute('select start_time, forecast_time, predict_power, predict_term, model_name from predict_power_' +
                      str(config_cluster[name]['id']) + '_train where model_name = %s and predict_term = %s'
                                                        ' ORDER BY start_time asc;',
                      (model_name + model_state, term))
        else:
            c.execute(
                'select start_time, forecast_time, predict_power, predict_term, model_name from predict_power_' + str(
                    config_cluster[name][
                        'id']) + ' where model_name = %s and predict_term = %s ORDER BY start_time asc;',
                (model_name + model_state, term))
        db.commit()

        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        predict_power = pandas.DataFrame(record, columns=coul)

        predict_power = predict_power.loc[predict_power['model_name'] == model_name + model_state]
        predict_power = predict_power.loc[predict_power['predict_term'] == term]
        predict_power = predict_power.loc[:, ['start_time', 'forecast_time', 'predict_power']]

        c.execute('select distinct time, power from real_power_' + str(config_cluster[name]['id']) + ';')
        db.commit()
        des = c.description
        record = c.fetchall()
        coul = list(iterm[0] for iterm in des)
        real_power_row = pandas.DataFrame(record, columns=coul)

        # 删除真实功率重复的点
        real_power_row = real_power_row.drop_duplicates(subset=['time'])
        # 判断每轮预测是否有完整的16、288、960个点
        start_time = pandas.DataFrame(predict_power.loc[:, 'start_time'], columns=['start_time'])
        start_time = start_time.drop_duplicates(subset=['start_time'])
        for i in range(len(start_time)):
            number_predict = len(predict_power.loc[predict_power['start_time'] == start_time.iloc[i, 0]])
            if number_predict % interval != 0:
                predict_power = predict_power.drop(
                    index=predict_power[(predict_power.start_time == start_time.iloc[i, 0])].index.tolist())

        train_data_merge_row = pandas.merge(predict_power, real_power_row, left_on='forecast_time', right_on='time', how='left')
        train_data_merge = train_data_merge_row.dropna(axis=0, how='any')
        real_power = train_data_merge.loc[:, ['time', 'power']]
        real_power = real_power.drop_duplicates(subset=['time'])

        restrict_label =None

        accuracy_GB[name] = evaluate_GB_T_40607_2021_withtimetag(predict_power.values, real_power.values,
                                                                 config_cluster[name]['online_capacity'],
                                                                 term, restrict_label)
        accuracy_Two[name] = result_evaluation_Two_Detailed_Rules_without_time_tag(predict_power.values,
                                                                                   real_power.values,
                                                                                   config_cluster[name][
                                                                                       'online_capacity'],
                                                                                   term, restrict_label)

        restrict_label = pandas.DataFrame(restrict_label, columns=['time', 'record'])

        actual_power_restrict = pandas.merge(real_power, restrict_label, left_on='time', right_on='time',
                                             how='left').values

        for i in range(len(actual_power_restrict) - 1, -1, -1):
            if actual_power_restrict[i, 2] == 1:
                actual_power_restrict = numpy.vstack((actual_power_restrict[0:i, :], actual_power_restrict[i + 1:, :]))

        real_power = pandas.DataFrame(actual_power_restrict[:, 0:2], columns=['time', 'power'])

        train_data_merge = pandas.merge(predict_power, real_power, left_on='forecast_time', right_on='time', how='left')
        if term == 'short':
            n = 288
        elif term == 'medium':
            n = 960
        else:
            n = 16
        day = int(len(train_data_merge)/n)
        predict_power = train_data_merge.loc[:, 'predict_power'].values.reshape(day, n)
        real_power = train_data_merge.loc[:, 'power'].values.reshape(day, n)
        if term == 'ultra_short':
            if scene == 'train':
                c.execute(
                    'select start_time, forecast_time, predict_power from predict_power_' + str(
                        config_cluster[name][
                            'id']) + '_train where model_name = %s and predict_term = %s and'
                                     ' UNIX_TIMESTAMP(forecast_time) - UNIX_TIMESTAMP(start_time) = 14400'
                                     ' ORDER BY forecast_time asc;',
                    (model_name + model_state, term))
            else:
                c.execute(
                    'select start_time, forecast_time, predict_power from predict_power_' + str(
                        config_cluster[name][
                            'id']) + ' where model_name = %s and predict_term = %s and'
                                     ' UNIX_TIMESTAMP(forecast_time) - UNIX_TIMESTAMP(start_time) = 14400'
                                     ' ORDER BY forecast_time asc;',
                    (model_name + model_state, term))
            db.commit()

            des = c.description
            record = c.fetchall()
            coul = list(iterm[0] for iterm in des)
            predict_power_last = pandas.DataFrame(record, columns=coul)

            predict_power_last_df = predict_power_last.loc[:, ['forecast_time', 'predict_power']]

            middle_time_float = predict_power_last_df.iloc[0, 0].timestamp()
            end_time_float = predict_power_last_df.iloc[-1, 0].timestamp()
            if middle_time_float % 86400 > 57600:
                middle_time_float = middle_time_float - (middle_time_float % 86400 - 57600) + 86400
            else:
                middle_time_float = middle_time_float - (middle_time_float % 86400 - 57600)

            if end_time_float % 86400 > 57600:
                end_time_float = end_time_float - (end_time_float % 86400 - 57600)
            else:
                end_time_float = end_time_float - (end_time_float % 86400 - 57600) - 86400

            middle_time_float = middle_time_float - 900
            end_time_float = end_time_float - 900

            time_list_power = []
            for j in range(int(middle_time_float), int(end_time_float), 900):
                time_list_power.append(
                    datetime.datetime.strptime(time.strftime("%Y/%m/%d %H:%M", time.localtime(j)), '%Y/%m/%d %H:%M'))
            time_dic = {'time_list': time_list_power}
            time_dataframe = pandas.DataFrame(time_dic)
            data_base = pandas.merge(time_dataframe, predict_power_last_df, left_on='time_list',
                                     right_on='forecast_time', how='left')
            data_base_power = data_base[['time_list', 'predict_power']]
            data_base = pandas.merge(data_base_power, real_power_row, left_on='time_list', right_on='time', how='left')
            predict_power_last_list = data_base.loc[:, 'predict_power'].values.reshape(int(len(data_base)/96), 96)
            real_power_last_list = data_base.loc[:, 'power'].values.reshape(int(len(data_base) / 96), 96)

            page = generate_page(predict_power_last_list, real_power_last_list, name, accuracy_Two[name],
                                 save_file_path=save_file_path, save_page=True, id=config_cluster[name]['id'],
                                 titlename='超短期第4小时时刻全天曲线')

            n = math.ceil(len(predict_power)/96)
            for i in range(n):
                if i == n-1:
                    predict_power_i = predict_power[i * 96:, :]
                    real_power_i = real_power[i * 96:, :]
                else:
                    predict_power_i = predict_power[i*96: i*96+96, :]
                    real_power_i = real_power[i*96: i*96+96, :]
                os.makedirs(save_file_path+'/', exist_ok=True)
                page = generate_page(predict_power_i, real_power_i, name, accuracy_GB[name],
                                     save_file_path=save_file_path+'/'+model_name + model_state+ str(i+1)+'-'+str(n),
                                     save_page=True,
                                     id=config_cluster[name]['id'], titlename='超短期预测结果')
        elif term == 'medium':
            page = generate_page(predict_power, real_power, name, accuracy_GB[name], save_file_path=save_file_path,
                                 save_page=True, id=config_cluster[name]['id'], titlename='中预测结果')
        else:
            page = generate_page(predict_power, real_power, name, accuracy_GB[name], save_file_path=save_file_path,
                                 save_page=True, id=config_cluster[name]['id'], titlename='短期预测结果')
    c.close()
    db.close()
    return accuracy_GB

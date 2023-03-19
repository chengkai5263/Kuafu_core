# _*_ coding: utf-8 _*_

import logging
import configparser

settings = {}


class DefaultOption(dict):
    """
    获取默认配置项value类。当配置文件中没有对应项目（key）时，结合该类来设置默认值
    """

    def __init__(self, config, section, **kv):
        self._config = config
        self._section = section
        dict.__init__(self, **kv)

    def items(self):
        _items = []
        for option in self:
            if not self._config.has_option(self._section, option):
                _items.append((option, self[option]))
            else:
                value_in_config = self._config.get(self._section, option)
                _items.append((option, value_in_config))
        return _items


class Settings(object):
    def __init__(self):
        self._conf = configparser.ConfigParser()

    # def get_settings(self):
    #     return self._conf

    def init_settings(self, conf_path, encoding="utf-8"):
        if settings:
            return

        self._conf.read(conf_path, encoding=encoding)
        for section in self._conf.sections():
            section_value = {}
            for (key, value) in self._conf.items(section):
                section_value[key] = value
            settings[section] = section_value
        self._process_log_settings()
        self._process_scheduler_settings()
        self._process_database_settings()
        self._process_job_test_serial_tasks_feature_parameter_ensemble_settings()
        self._process_job_feature_select_settings()
        self._process_job_best_parameter_search_settings()
        self._process_job_ensemble_learn_settings()
        self._process_job_interval_predict_ensemble_learning_settings()
        self._process_job_interval_learning_settings()
        self._process_job_transfer_learning_settings()
        self._process_job_trigger_of_cloud_version_settings()
        self._process_job_predict_short_power_settings()
        self._process_job_predict_ultra_short_power_settings()
        self._process_execute_once_at_startup_settings()
        self._process_job_trigger_of_small_time_scale_ensemble_learning_settings()
        self._process_job_trigger_of_small_time_scale_predict_settings()
        self._process_job_trigger_of_small_time_scale_transfer_learning_settings()

    def _process_log_settings(self):
        if "log" not in settings:
            settings["log"] = {
                "level": logging.INFO,
            }
            return

        # 对日志级别进行数据类型转换
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = self._conf.get("log", "level", vars=DefaultOption(self._conf, "log", level="INFO"))
        settings["log"]["level"] = levels.get(level, logging.INFO)

        # 设置控制台输出标志
        console = self._conf.get("log", "console", vars=DefaultOption(self._conf, "log", console="FALSE"))
        if console.lower() == "true":
            settings["log"]["console"] = True
        else:
            settings["log"]["console"] = False

        # 对日志轮换（午夜12点）时，最多保留的历史日志文件个数进行数据类型转换
        backup_count = self._conf.get("log", "backup_count",
                                      vars=DefaultOption(self._conf, "log", backup_count="0"))
        settings["log"]["backup_count"] = int(backup_count)

    def _process_scheduler_settings(self):
        if "scheduler" not in settings:
            settings["scheduler"] = {
                "process_pool_size": 5,
                "thread_pool_size": 20,
                "job_max_instances": 5,
            }
            return

        process_pool_size = self._conf.get("scheduler", "process_pool_size",
                                           vars=DefaultOption(self._conf, "scheduler", process_pool_size="5"))
        thread_pool_size = self._conf.get("scheduler", "thread_pool_size",
                                          vars=DefaultOption(self._conf, "scheduler", thread_pool_size="20"))
        job_max_instances = self._conf.get("scheduler", "job_max_instances",
                                           vars=DefaultOption(self._conf, "scheduler", job_max_instances="5"))
        settings["scheduler"]["process_pool_size"] = int(process_pool_size)
        settings["scheduler"]["thread_pool_size"] = int(thread_pool_size)
        settings["scheduler"]["job_max_instances"] = int(job_max_instances)

    def _process_database_settings(self):
        if "database" not in settings:
            return

        if "port" in settings["database"]:
            settings["database"]["port"] = int(settings["database"]["port"])

    def _process_execute_once_at_startup_settings(self):
        session_name = "execute_once_at_startup"

        enable_test_serial_tasks_feature_parameter_ensemble = \
            self._conf.get(session_name, "enable_test_serial_tasks_feature_parameter_ensemble",
                           vars=DefaultOption(self._conf, session_name,
                                              enable_test_serial_tasks_feature_parameter_ensemble="True"))
        if enable_test_serial_tasks_feature_parameter_ensemble.lower() == "false":
            settings[session_name]["enable_test_serial_tasks_feature_parameter_ensemble"] = False
        else:
            settings[session_name]["enable_test_serial_tasks_feature_parameter_ensemble"] = True

        enable_feature_select = self._conf.get(session_name, "enable_feature_select",
                                               vars=DefaultOption(self._conf, session_name,
                                                                  enable_feature_select="True"))
        if enable_feature_select.lower() == "false":
            settings[session_name]["enable_feature_select"] = False
        else:
            settings[session_name]["enable_feature_select"] = True

        enable_best_parameter_search = self._conf.get(session_name, "enable_best_parameter_search",
                                                      vars=DefaultOption(self._conf, session_name,
                                                                         enable_best_parameter_search="True"))
        if enable_best_parameter_search.lower() == "false":
            settings[session_name]["enable_best_parameter_search"] = False
        else:
            settings[session_name]["enable_best_parameter_search"] = True

        enable_ensemble_learn = self._conf.get(session_name, "enable_ensemble_learn",
                                               vars=DefaultOption(self._conf, session_name,
                                                                  enable_ensemble_learn="True"))
        if enable_ensemble_learn.lower() == "false":
            settings[session_name]["enable_ensemble_learn"] = False
        else:
            settings[session_name]["enable_ensemble_learn"] = True

    def _process_job_test_serial_tasks_feature_parameter_ensemble_settings(self):
        self._process_job_trigger_settings("job_test_serial_tasks_feature_parameter_ensemble")

    def _process_job_feature_select_settings(self):
        self._process_job_trigger_settings("job_feature_select")

    def _process_job_best_parameter_search_settings(self):
        self._process_job_trigger_settings("job_best_parameter_search")

    def _process_job_ensemble_learn_settings(self):
        session_name = "job_ensemble_learn"
        self._process_job_trigger_settings(session_name)
        enable_interval_predict = self._conf.get(session_name, "enable_interval_predict",
                                                 vars=DefaultOption(self._conf, session_name,
                                                                    enable_interval_predict="True"))
        if enable_interval_predict.lower() == "false":
            settings[session_name]["enable_interval_predict"] = False
        else:
            settings[session_name]["enable_interval_predict"] = True

    def _process_job_interval_predict_ensemble_learning_settings(self):
        self._process_job_trigger_settings("job_interval_predict_ensemble_learning")

    def _process_job_interval_learning_settings(self):
        self._process_job_trigger_settings("job_interval_learning")

    def _process_job_transfer_learning_settings(self):
        self._process_job_trigger_settings("job_transfer_learning")

    def _process_job_trigger_of_cloud_version_settings(self):
        self._process_job_trigger_settings("job_trigger_of_cloud_version")

    def _process_job_predict_short_power_settings(self):
        session_name = "job_predict_short_power"
        self._process_job_trigger_settings(session_name)
        enable_interval_predict = self._conf.get(session_name, "enable_interval_predict",
                                                 vars=DefaultOption(self._conf, session_name,
                                                                    enable_interval_predict="True"))
        if enable_interval_predict.lower() == "false":
            settings[session_name]["enable_interval_predict"] = False
        else:
            settings[session_name]["enable_interval_predict"] = True

    def _process_job_predict_ultra_short_power_settings(self):
        session_name = "job_predict_ultra_short_power"
        self._process_job_trigger_settings(session_name)
        enable_interval_predict = self._conf.get(session_name, "enable_interval_predict",
                                                 vars=DefaultOption(self._conf, session_name,
                                                                    enable_interval_predict="True"))
        if enable_interval_predict.lower() == "false":
            settings[session_name]["enable_interval_predict"] = False
        else:
            settings[session_name]["enable_interval_predict"] = True

    def _process_job_trigger_of_small_time_scale_ensemble_learning_settings(self):
        self._process_job_trigger_settings("job_trigger_of_small_time_scale_ensemble_learning")

    def _process_job_trigger_of_small_time_scale_predict_settings(self):
        self._process_job_trigger_settings("job_trigger_of_small_time_scale_predict")

    def _process_job_trigger_of_small_time_scale_transfer_learning_settings(self):
        self._process_job_trigger_settings("job_trigger_of_small_time_scale_transfer_learning")

    def _process_job_trigger_settings(self, session_name):
        """
        设置通用任务配置（是否启用任务、触发器及其参数）
        """
        if session_name not in settings:
            settings[session_name] = {
                "enable": False,
                "trigger_args": {
                    "trigger": "cron",
                }
            }
            return
        result = {}
        enable = self._conf.get(session_name, "enable",
                                vars=DefaultOption(self._conf, session_name, enable="True"))
        if enable.lower() == "false":
            result["enable"] = False
        else:
            result["enable"] = True

        # 调度器执行任务函数/方法时，需要传的参数
        job_func_kwargs = {}
        if "start_time" in settings[session_name]:
            job_func_kwargs["start_time"] = settings[session_name]["start_time"]
        if "end_time" in settings[session_name]:
            job_func_kwargs["end_time"] = settings[session_name]["end_time"]
        result["job_func_kwargs"] = job_func_kwargs

        trigger = self._conf.get(session_name, "trigger",
                                 vars=DefaultOption(self._conf, session_name, trigger="None"))
        if trigger not in ["cron", "interval", "date"]:
            result["trigger_args"] = dict(trigger=None)
            settings[session_name] = result
            return

        result["trigger_args"] = {
            "trigger": trigger,
        }
        if trigger == "date":
            if "run_date" in settings[session_name]:
                result["trigger_args"]["run_date"] = settings[session_name]["run_date"]

        if trigger == "interval":
            if "weeks" in settings[session_name]:
                result["trigger_args"]["weeks"] = int(settings[session_name]["weeks"])
            if "days" in settings[session_name]:
                result["trigger_args"]["days"] = int(settings[session_name]["days"])
            if "hours" in settings[session_name]:
                result["trigger_args"]["hours"] = int(settings[session_name]["hours"])
            if "minutes" in settings[session_name]:
                result["trigger_args"]["minutes"] = int(settings[session_name]["minutes"])
            if "seconds" in settings[session_name]:
                result["trigger_args"]["seconds"] = int(settings[session_name]["seconds"])
            if "start_date" in settings[session_name]:
                result["trigger_args"]["start_date"] = settings[session_name]["start_date"]
            if "end_date" in settings[session_name]:
                result["trigger_args"]["end_date"] = settings[session_name]["end_date"]

        if trigger == "cron":
            if "year" in settings[session_name]:
                result["trigger_args"]["year"] = settings[session_name]["year"]
            if "month" in settings[session_name]:
                result["trigger_args"]["month"] = settings[session_name]["month"]
            if "day" in settings[session_name]:
                result["trigger_args"]["day"] = settings[session_name]["day"]
            if "week" in settings[session_name]:
                result["trigger_args"]["week"] = settings[session_name]["week"]
            if "day_of_week" in settings[session_name]:
                result["trigger_args"]["day_of_week"] = settings[session_name]["day_of_week"]
            if "hour" in settings[session_name]:
                result["trigger_args"]["hour"] = settings[session_name]["hour"]
            if "minute" in settings[session_name]:
                result["trigger_args"]["minute"] = settings[session_name]["minute"]
            if "second" in settings[session_name]:
                result["trigger_args"]["second"] = settings[session_name]["second"]
            if "start_date" in settings[session_name]:
                result["trigger_args"]["start_date"] = settings[session_name]["start_date"]
            if "end_date" in settings[session_name]:
                result["trigger_args"]["end_date"] = settings[session_name]["end_date"]

        settings[session_name] = result


def init_settings(conf_path, encoding="utf-8", update=None):
    Settings().init_settings(conf_path, encoding)
    if isinstance(update, dict):
        for key, value in update.items():
            if key in settings:
                settings[key].update(value)


# _*_ coding: utf-8 _*_
# 线程安全日志类，单例模式，可配置将日志输出到标准输出（屏幕）、常规日志文件、错误日志文件中的一种或多种
# 其中输出到日志文件时，日志文件将在午夜00:00:00进行日志轮转切换，开启新日志（即每天一个日志文件）
# 仅需在程序启动时调用init方法初始化一次。
# 多次调用时，仅第一次调用生效；不调用时，日志配置不生效，但不影响info等日志记录方法的调用

import logging
import os
from logging import handlers


class Logger(logging.Logger):
    def __init__(self, name="green_energy", level=logging.INFO):
        self._log_queue_listener = None
        super().__init__(name, level)

    def init(self, **kwargs):
        if self.hasHandlers():
            # 或者使用如下语句判断
            # if self.handlers:
            return
        if self._log_queue_listener and kwargs.get("main_process", True):
            return

        level = kwargs.get('level', logging.INFO)
        datefmt = kwargs.get('datefmt', None)
        default_fmt = "%(asctime)s - %(module)s - %(levelname)s - %(lineno)d - %(process)d - %(thread)d: %(message)s"
        fmt = kwargs.get('format', default_fmt)

        self.setLevel(level)
        format_str = logging.Formatter(fmt, datefmt)

        # 创建一个StreamHandler，用于输出到控制台
        if kwargs.get('console', None) and kwargs.get("main_process", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(format_str)
            console_handler.setLevel(kwargs.get('console_level', level))
            self.addHandler(console_handler)

        # 创建一个TimedRotatingFileHandler，按时间自动切分日志，用于输出到常规日志
        normal_logfile = kwargs.get('normal_logfile', None)
        if normal_logfile and kwargs.get("main_process", True):
            dir_path = os.path.dirname(normal_logfile)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # backupCount 保存日志的数量，轮转时过期的日志文件自动删除
            # when 按什么日期格式切分
            normal_handler = handlers.TimedRotatingFileHandler(filename=normal_logfile, when='midnight',
                                                               interval=1, encoding='utf-8',
                                                               backupCount=kwargs.get('backup_count', 0))
            normal_handler.setFormatter(format_str)
            normal_handler.setLevel(kwargs.get('normal_level', level))
            self.addHandler(normal_handler)

        # 创建一个TimedRotatingFileHandler，按时间自动切分日志，用于输出到错误日志
        error_logfile = kwargs.get('error_logfile', None)
        if error_logfile and kwargs.get("main_process", True):
            dir_path = os.path.dirname(error_logfile)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # backupCount 保存日志的数量，轮转时过期的日志文件自动删除
            # when 按什么日期格式切分
            error_handler = handlers.TimedRotatingFileHandler(filename=error_logfile, when='midnight',
                                                              interval=1, encoding='utf-8',
                                                              backupCount=kwargs.get('backup_count', 0))
            error_handler.setFormatter(format_str)
            error_handler.setLevel(logging.ERROR)
            self.addHandler(error_handler)

        logs_queue = kwargs.get("logs_queue", None)
        if logs_queue:
            if self._log_queue_listener:
                self._log_queue_listener.stop()
                del self._log_queue_listener.handlers[:]
            elif kwargs.get("main_process", True):
                queue_handler = logging.handlers.QueueHandler(logs_queue)
                queue_handler.setLevel(kwargs.get('normal_level', level))
                self._log_queue_listener = logging.handlers.QueueListener(logs_queue, *self.handlers,
                                                                          respect_handler_level=True)
                del self.handlers[:]
                self.addHandler(queue_handler)
                self._log_queue_listener.start()
            else:
                queue_handler = logging.handlers.QueueHandler(logs_queue)
                queue_handler.setLevel(kwargs.get('normal_level', level))
                del self.handlers[:]
                self.addHandler(queue_handler)

    def close(self):
        for handler in self.handlers:
            handler.close()
            del self.handlers[:]
        if self._log_queue_listener:
            self._log_queue_listener.stop()
            self._log_queue_listener = None


logs = Logger()


def init_logger(**kwargs):
    logs.init(**kwargs)

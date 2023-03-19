
import os
import joblib
import shutil
from model import BaseModel
from common.logger import logs
from functools import wraps


def save_model(model, filename):
    """
    保存模型到文件
    对于类BaseModel及其派生类，若有save_model方法，则额外执行该方法
    """
    if isinstance(model, BaseModel) and hasattr(model, "save_model") \
            and callable(getattr(model, "save_model")):
        model.save_model(filename)
    else:
        # 先保存成临时文件，后move成目标文件，防止在生成临时文件的过程中，有其他进程或线程来读取该文件而读到脏数据
        # linux跨文件系统的情况下，应该用软链接ln -s
        tmp_file = filename + ".temp"
        joblib.dump(model, filename=tmp_file)
        shutil.move(tmp_file, filename)


def load_model(filename):
    """
    从文件将模型加载到程序
    对于类BaseModel及其派生类，若有load_model方法，则额外执行该方法
    """
    model = joblib.load(filename)
    if isinstance(model, BaseModel) and hasattr(model, "load_model") \
            and callable(getattr(model, "save_model")):
        model.load_model(os.path.dirname(filename))
    return model


def catch_exception(err_msg="", exc_info=True, exc_str=False, default_return=None):
    """
    带参数的异常捕获装饰器，捕获方法/函数异常并将异常信息送至日志系统中处理
    :param err_msg: 异常发生时，所要打印的附加出错信息。当为None时，表示不打印出错信息，此时参数exc_info、exc_str均无效
    :param exc_info: 异常发生时，是否打印堆栈调用信息
    :param exc_str: 异常发生时，是否打印简略出错信息（即Exception e中，str(e)的值）
    :param default_return: 被装饰的方法/函数发生异常时所返回的值
    :return: 发生异常时，返回参数default_return所指定的值；否则返回被装饰的方法/函数正常运行时所返回的值
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if err_msg is not None:
                    if exc_str:
                        logs.error("%s%s" % (err_msg, str(e)), exc_info=exc_info)
                    else:
                        logs.error(err_msg, exc_info=exc_info)
                return default_return
        return wrapper
    return decorator

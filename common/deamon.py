# _*_ coding: utf-8 _*_
# description: 一个守护进程的简单包装类, 具备常用的start|stop|restart|status功能, 使用方便
#              需要改造为守护进程的程序只需要重写基类的run函数就可以了
# usage: 启动: python3 daemon.py start
#        关闭: python3 daemon.py stop
#        状态: python3 daemon.py status
#        重启: python3 daemon.py restart
#        查看: ps -axj | grep daemon

import atexit
import os
import sys
import time
import signal
import platform


class Daemon:
    """
    a generic daemon class.
    usage: 继承Daemon类并重写run()方法
    pid_file 表示守护进程pid文件的绝对路径
    """

    def __init__(self, name, pid_file="./daemon_class.pid", home_dir='.', umask=0o22,
                 stdout_file="/dev/null", stderr_file="/dev/null"):
        self.name = name  # 派生守护进程类的名称
        self.pidfile = pid_file  # pid文件绝对路径
        self.home_dir = home_dir
        self.umask = umask
        self.alive = True
        self.stdout = stdout_file
        self.stderr = stderr_file

    def daemonize(self):
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            print('fork #1 failed: %d (%s)' % (e.errno, e.strerror))
            sys.exit(1)

        if self.home_dir and self.home_dir != ".":
            os.makedirs(self.home_dir, exist_ok=True)
        os.chdir(self.home_dir)
        os.setsid()
        os.umask(self.umask)

        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            print('fork #1 failed: %d (%s)' % (e.errno, e.strerror))
            sys.exit(1)

        sys.stdout.flush()
        sys.stderr.flush()

        # 重定向标准文件描述符
        stdin = '/dev/null'
        # stdout = '/dev/null'
        # stderr = '/dev/null'
        if self.stdout and self.stdout != '/dev/null':
            os.makedirs(os.path.dirname(self.stdout), exist_ok=True)
        if self.stderr and self.stderr != '/dev/null':
            os.makedirs(os.path.dirname(self.stderr), exist_ok=True)
        si = open(stdin, 'r')
        so = open(self.stdout, 'a+')
        se = open(self.stderr, 'a+')
        # dup2函数原子化关闭和复制文件描述符
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        def sig_handler(signum, frame):
            print(signum, frame)
            self.alive = False
        signal.signal(signal.SIGTERM, sig_handler)
        signal.signal(signal.SIGINT, sig_handler)

        print('daemon process started ...')
        self.write_pidfile()

    def write_pidfile(self):
        dir_path = os.path.dirname(self.pidfile)
        if os.path.exists(dir_path):
            if not os.path.isdir(dir_path):
                os.remove(dir_path)
                os.makedirs(dir_path, exist_ok=True)
        else:
            os.makedirs(dir_path, exist_ok=True)
        pid = str(os.getpid())
        with open(self.pidfile, 'w+') as fd:
            fd.write('%s\n' % pid)
        atexit.register(self.del_pidfile)

    def get_pid(self):
        if not os.path.exists(self.pidfile):
            return None
        try:
            pf = open(self.pidfile, 'r')
            pid = int(pf.read().strip())
            pf.close()
        except IOError:
            pid = None
        except SystemExit:
            pid = None
        return pid

    def del_pidfile(self):
        if os.path.exists(self.pidfile):
            os.remove(self.pidfile)

    def start(self, *args, **kwargs):
        print('ready to starting ......')
        if self.is_running():
            msg = 'pid file %s already exists, is it already running?\n'
            sys.stderr.write(msg % self.pidfile)
            sys.exit(1)
        if platform.system() == 'Linux':
            # start the daemon
            self.daemonize()
        else:
            self.write_pidfile()

            def sig_handler(signum, frame):
                print(signum, frame)
                self.alive = False

            signal.signal(signal.SIGTERM, sig_handler)
            signal.signal(signal.SIGINT, sig_handler)
        self.run(*args, **kwargs)

    def stop(self):
        print('stopping ...')
        pid = self.get_pid()
        if not pid:
            msg = 'pid file [%s] does not exist. Not running?\n' % self.pidfile
            sys.stderr.write(msg)
            if os.path.exists(self.pidfile):
                os.remove(self.pidfile)
            return
        # try to kill the daemon process
        try:
            i = 0
            while 1:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.3)
                i = i + 1
                if i % 10 == 0:
                    # os.kill(pid, signal.SIGHUP)
                    stop_cmd = "ps -ef | grep -w %d | grep -v grep | awk '{print $2}' | xargs kill -9" % pid
                    os.system(stop_cmd)
        except OSError as err:
            err = str(err)
            if err.find('No such process') > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            elif err.find("Operation not permitted") > 0:
                print("kill process which pid is %s err: %s" % (pid, err))
                print("remove pidfile ", self.pidfile)
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print(str(err))
                sys.exit(1)
            print('Stopped!')

    def restart(self, *args, **kwargs):
        self.stop()
        self.start(*args, **kwargs)

    def is_running(self):
        pid = self.get_pid()
        return pid and os.path.exists('/proc/%d' % pid) and pid in self.get_process_pid_list()

    def get_process_pid_list(self):
        """获取进程号列表"""
        # 定义一个空列表，放置按空格切割split(' ')后的进程信息
        process_info_list = []
        if platform.system() != "Linux":
            return process_info_list
        # 命令行输入，获取进程信息
        # process = os.popen('ps -ef | grep -w $LOGNAME | grep -w %s | grep -v grep' % self.name)
        process = os.popen('ps -ef | grep python | grep -w $LOGNAME | grep -v grep')
        # 读取进程信息，获取字符串
        process_info = process.read()
        if not process_info:
            return process_info_list
        cur_self_pid = os.getpid()
        # print("process_info [%s]" %process_info)
        # 按空格切割split(" ")，
        for line in process_info.split('\n'):
            # 判断不为空，添加到process_info_list中
            if line and line.strip():
                # noinspection PyBroadException
                try:
                    pid = int(line.split(' ')[5])
                # except Exception as err:
                except Exception:
                    continue
                if pid != cur_self_pid:
                    process_info_list.append(int(pid))
        return process_info_list

    def run(self, *args, **kwargs):
        # 'NOTE: override the method in subclass'
        print('base class run()')


if __name__ == '__main__':
    help_msg = 'Usage: python %s <start|stop|restart|status> [processname]' % sys.argv[0]
    pid_fn = './work_dir/data/daemon_class.pid'  # 守护进程pid文件的绝对路径

    opt = "start"
    if len(sys.argv) > 1:
        opt = sys.argv[1]
    elif platform.system() != "Windows":
        print(help_msg)
        sys.exit(0)

    p_name = None
    if len(sys.argv) > 2:
        p_name = sys.argv[2]  # 守护进程名称

    class ClientDaemon(Daemon):
        def __init__(self, name, pid_file, home_dir='.', umask=0o22):
            Daemon.__init__(self, name, pid_file, home_dir, umask)

        def run(self, *args, **kwargs):
            while self.alive:
                line = time.ctime()
                print(line)
                time.sleep(1)

    cD = ClientDaemon(p_name, pid_fn)
    if opt == 'start':
        cD.start()
    elif opt == 'stop':
        cD.stop()
    elif opt == 'restart':
        cD.restart()
    elif opt == 'status':
        alive = cD.is_running()
        if alive:
            print('process [%s] is running ......' % cD.get_pid())
        else:
            print('daemon process [%s] stopped' % cD.name)
    else:
        print('invalid argument!')
        print(help_msg)

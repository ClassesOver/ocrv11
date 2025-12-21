# Gunicorn 配置文件
import multiprocessing
import os
import sys

# 服务器绑定
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8078")

# 工作进程数：默认 CPU 核心数 * 2 + 1（适用于 I/O 密集型任务）
cpu_count = multiprocessing.cpu_count()
default_workers = max(1, cpu_count * 2 + 1)
workers = int(os.getenv("GUNICORN_WORKERS", 1))

# 工作模式：sync/gevent/eventlet/uvicorn
worker_class_env = os.getenv("GUNICORN_WORKER_CLASS", "sync")
if worker_class_env == "uvicorn":
    worker_class = "uvicorn.workers.UvicornWorker"
elif worker_class_env in ["gevent", "eventlet"]:
    worker_class = worker_class_env
else:
    worker_class = "sync"

# 每个工作进程的线程数（仅 sync 模式有效）
threads = int(os.getenv("GUNICORN_THREADS", 2))

# 超时时间（秒）- OCR 处理较慢，设置较大值
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))

# 优雅重启超时
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", 30))

# 最大请求数（防止内存泄漏）
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", 50))

# 连接保持时间（秒）
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", 5))

# 监听队列大小
backlog = int(os.getenv("GUNICORN_BACKLOG", 2048))

# 日志配置
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info").lower()

# 访问日志格式（包含响应时间）
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" '
    '%(D)s %(p)s %(L)s'
)

# 进程名称
proc_name = os.getenv("GUNICORN_PROC_NAME", "ocr-service")

# 预加载应用（减少内存占用，加快启动）
preload_app = os.getenv("GUNICORN_PRELOAD", "true").lower() == "true"

# 使用内存文件系统提高性能（仅 Linux 系统）
if sys.platform == "linux" and os.path.exists("/dev/shm"):
    worker_tmp_dir = "/dev/shm"
else:
    # Windows 或其他系统使用默认临时目录
    worker_tmp_dir = None

# 限制请求行大小（防止过大请求）
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# 用户和组（生产环境建议设置）
# user = "www-data"
# group = "www-data"

# 钩子函数
def when_ready(server):
    """服务器就绪时调用"""
    server.log.info(
        f"Gunicorn 就绪 - 监听地址: {bind}, "
        f"工作进程数: {workers}, "
        f"工作模式: {worker_class}, "
        f"线程数: {threads if worker_class == 'sync' else 'N/A'}, "
        f"CPU 核心数: {cpu_count}"
    )

def on_starting(server):
    """服务器启动时调用"""
    server.log.info("正在启动 Gunicorn 服务器...")

def post_fork(server, worker):
    """工作进程 fork 后调用"""
    server.log.info(f"工作进程 {worker.pid} 已启动")

def worker_int(worker):
    """工作进程收到 INT 信号时调用"""
    worker.log.info(f"工作进程 {worker.pid} 收到中断信号")

def pre_fork(server, worker):
    """工作进程 fork 前调用"""
    pass

def post_worker_init(worker):
    """工作进程初始化后调用"""
    worker.log.info(f"工作进程 {worker.pid} 初始化完成")

def worker_abort(worker):
    """工作进程异常退出时调用"""
    worker.log.warning(f"工作进程 {worker.pid} 异常退出")


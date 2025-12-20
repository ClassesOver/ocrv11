# Gunicorn 配置文件
import multiprocessing
import os
import sys

# 服务器绑定
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8078")

# 工作进程数：默认 CPU 核心数 * 2 + 1（适用于 I/O 密集型任务）
cpu_count = multiprocessing.cpu_count()
default_workers = max(1, cpu_count * 2 + 1)
workers = int(os.getenv("GUNICORN_WORKERS", default_workers))

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
# 日志文件夹（可通过环境变量指定，默认 /app/logs）
log_dir = os.getenv("GUNICORN_LOG_DIR", "/app/logs")

# 确保日志目录存在
os.makedirs(log_dir, exist_ok=True)

# 访问日志和错误日志文件路径
accesslog = os.getenv("GUNICORN_ACCESSLOG", os.path.join(log_dir, "gunicorn_access.log"))
errorlog = os.getenv("GUNICORN_ERRORLOG", os.path.join(log_dir, "gunicorn_error.log"))
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info").lower()

# 日志轮转配置（使用 loguru）
from loguru import logger
import logging

# 日志文件最大大小，可通过环境变量配置，默认 100MB
log_max_size = os.getenv("GUNICORN_LOG_MAX_SIZE", "100 MB")

# 保留时间，可通过环境变量配置，默认 7 天
log_retention = os.getenv("GUNICORN_LOG_RETENTION", "7 days")

# 配置日志轮转（使用 loguru）
def setup_logging():
    """设置日志轮转（使用 loguru）"""
    # 移除 loguru 的默认 handler
    logger.remove()
    
    # 将 Python logging 重定向到 loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # 获取对应的 loguru 级别
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                # 映射 logging 级别到 loguru 级别
                level_map = {
                    50: "CRITICAL",
                    40: "ERROR",
                    30: "WARNING",
                    20: "INFO",
                    10: "DEBUG",
                    0: "NOTSET"
                }
                level = level_map.get(record.levelno, "INFO")
            
            # 根据 logger 名称决定输出格式和文件
            if record.name == "gunicorn.access":
                # 访问日志：使用 bind 添加标识，然后通过 filter 过滤
                logger.bind(name="gunicorn.access").opt(
                    depth=6, 
                    exception=record.exc_info
                ).log(level, record.getMessage())
            else:
                # 错误日志：包含完整信息
                logger.opt(
                    depth=6, 
                    exception=record.exc_info
                ).log(level, record.getMessage())
    
    # 配置错误日志轮转
    if errorlog and errorlog != "-":
        # 使用 loguru 添加错误日志文件，支持轮转
        # 过滤掉访问日志
        logger.add(
            errorlog,
            rotation=log_max_size,
            retention=log_retention,
            level=loglevel.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
            encoding="utf-8",
            enqueue=True,  # 异步写入，提高性能
            backtrace=True,  # 记录堆栈跟踪
            diagnose=True,  # 显示变量值
            filter=lambda record: record["extra"].get("name") != "gunicorn.access"
        )
    
    # 配置访问日志轮转
    if accesslog and accesslog != "-":
        # 访问日志使用简单的格式，直接记录消息
        logger.add(
            accesslog,
            rotation=log_max_size,
            retention=log_retention,
            level="INFO",
            format="{message}",
            encoding="utf-8",
            enqueue=True,  # 异步写入，提高性能
            filter=lambda record: record["extra"].get("name") == "gunicorn.access"
        )
    
    # 同时输出到控制台（如果日志文件未设置或设置为 "-"）
    if errorlog == "-" or not errorlog:
        logger.add(
            sys.stderr,
            level=loglevel.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
            colorize=True
        )
    
    # 拦截 Gunicorn 的日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for logger_name in ["gunicorn", "gunicorn.error", "gunicorn.access", "uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

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
    setup_logging()

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


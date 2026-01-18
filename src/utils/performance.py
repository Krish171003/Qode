import time
import psutil
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.process = psutil.Process()
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        logger.debug(f"Performance monitoring started (Memory: {self.start_memory:.2f} MB)")
        
    def update_peak(self):
        current = self.process.memory_info().rss / 1024 / 1024
        if current > self.peak_memory:
            self.peak_memory = current
    
    def stop(self):
        self.end_time = time.time()
        self.update_peak()
        
    def get_stats(self):
        elapsed = (self.end_time or time.time()) - (self.start_time or time.time())
        
        return {
            'elapsed_time': elapsed,
            'start_memory_mb': self.start_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': self.peak_memory - self.start_memory
        }


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def log_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / 1024 / 1024
        logger.debug(f"{func.__name__} memory: {mem_before:.2f} -> {mem_after:.2f} MB "
                    f"(delta {mem_after - mem_before:.2f} MB)")
        return result
    return wrapper


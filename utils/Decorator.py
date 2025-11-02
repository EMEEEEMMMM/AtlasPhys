import time
from functools import wraps
from typing import Callable, Any

def time_counter(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A time counter for functions

    Args:
        func : function
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        StartTime = time.perf_counter()
        Reuslt = func(*args, **kwargs)
        EndTime = time.perf_counter()
        Elapsed = EndTime - StartTime

        print(f"函数{func.__name__}已被执行，用时{Elapsed:.6f}秒")

        return Reuslt
    
    return wrapper
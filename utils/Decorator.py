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
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        StartTime: float = time.perf_counter()
        Reuslt: Any = func(*args, **kwargs)
        EndTime: float = time.perf_counter()
        Elapsed: float = EndTime - StartTime

        print(f"函数{func.__name__}已被执行，用时{Elapsed:.6f}秒")

        return Reuslt
    
    return wrapper

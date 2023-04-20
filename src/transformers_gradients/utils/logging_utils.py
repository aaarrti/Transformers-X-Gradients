import inspect
import logging
from functools import wraps

log = logging.getLogger(__name__)


def log_before(func, logger=log.debug):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        logger(
            f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )"
        )
        return func(*args, **kwargs)

    return wrapper


def log_after(func, logger=log.debug):
    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        retval = func(*func_args, **func_kwargs)
        logger("Exited " + func.__name__ + "() with value: " + repr(retval))
        return retval

    return wrapper

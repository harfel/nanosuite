from functools import wraps
import warnings


def deprecated_feature(func):
    @wraps(func)
    def wrapper(*args, **opts):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            func(*args, **opts)
    return wrapper

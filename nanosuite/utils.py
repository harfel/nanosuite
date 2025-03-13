from typing import Any, Callable
from functools import wraps


class Cache:
    """LRU Cache for function calls

    This is similar to functools.lru_cache with two major differents.
    Firstly, Cache does not require function arguments to be immutable.
    Secondly, the Cache needs to be created explicitly and independently
    from the function one wants to cache.
    """
    def __init__(self, maxsize: int = -1):
        self.maxsize = maxsize
        self.dict: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def key(self, args: tuple[Any], opts: dict[str, Any]) -> str:
        """Generate cache key for function arguments

        The key value is based on the string representations of
        the arguments.

        Parameters
        ----------
        args, opts:
            packed function arguments and keyword arguments

        Returns
        -------
        A string representing the provided parameters
        """
        return f'{args}{opts}'

    def compute(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorate function for caching

        Parameters
        ----------
        func: any python callable

        Returns
        -------
        a caching version of the provided function
        """
        @wraps(func)
        def wrapper(*args, **opts) -> Any:
            key = self.key(args, opts)
            if key not in self.dict:
                self.dict[key] = func(*args, **opts)
                self.misses += 1
                while len(self.dict) > self.maxsize > -1:
                    self.dict.pop(next(iter(self.dict)))
            else:
                self.hits += 1
            return self.dict[key]
        return wrapper

import functools


def substitutive(func):
    func.__args_list__ = []
    @functools.wraps(func)
    def wrapper(*args):
        try:
            func(*[*func.__args_list__, *args])
            func.__args_list__ = []
        except:
            for arg in args:
                func.__args_list__.append(arg)
            return wrapper
    return wrapper

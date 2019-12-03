import functools


def substitutive(func):
    
    @functools.wraps(func)
    def wrapper(*args):
        tmp_args = []
        args_for_save = []
        tmp_wrapper = None
        arguments_amount_real = len(args) + len(wrapper.__args_list__)
        if arguments_amount_real > func.__code__.co_argcount:
            raise TypeError
        try:
            for arg in wrapper.__args_list__:
                tmp_args.append(arg)
            for arg in args:
                tmp_args.append(arg)
            return func(*tmp_args)
        except:
            wrapper.__args_list__ = tmp_args[:]
            return wrapper
    wrapper.__args_list__ = []                
    return wrapper

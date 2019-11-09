from collections.abc import Iterable


def linearize(iterable_object):
    for elem in iterable_object:
        if isinstance(elem, Iterable):
            if isinstance(elem, str) & (len(elem) == 1):
                yield elem
            else:
                for elem2 in linearize(elem):
                    yield elem2
            continue       
        yield elem

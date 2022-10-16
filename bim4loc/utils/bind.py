#from:
#https://stackoverflow.com/questions/7811247/how-to-fill-specific-positional-arguments-with-partial-in-python

from functools import partial

class bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """
    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)

if __name__ == "__main__":
    def foo(a, b, c, /, *, d):
        print(f"A({a}) B({b}) C({c}) D({d})")

    f1 = bind(foo, 1, 2, 3, d=4)
    f1()

    f2 = bind(foo, 1, 2, d=4)
    f2(3)

    f3 = bind(foo, 1, ..., 3, d=4)
    f3(2)

    f4 = bind(foo, ..., 2, ..., d=4)
    f4(1, 3)

    f5 = bind(foo, ..., d=5)
    f5(1, 2, 3, d=4)


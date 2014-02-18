"""
Combine functionality from ordered and default dict.
"""

# ATTRIBUTION
# The implementation is taken from:
# http://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
# I consider Apache 2.0 shared alike to http://creativecommons.org/licenses/by-sa/2.5/
# the license of stackoverflow questions
#
# However,
# I did not change the code, so It should not effect the license of the project!
from __future__ import unicode_literals

from collections import Callable
try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict


class DefaultOrderedDict(OrderedDict):

    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory, OrderedDict.__repr__(self))

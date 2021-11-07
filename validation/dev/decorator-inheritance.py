from flowdyn.modelphy.base import methoddict

import logging

class a():
    _dico = methoddict('f_')
    def __init__(self):
        self._dico = a._dico.copy()
    @_dico.register()
    def f_ma(self):
        return
    @_dico.register('')
    def m(self):
        return

class b(a):
    _dico = methoddict()
    def __init__(self):
        a.__init__(self)
        self._dico.merge(b._dico)
    @_dico.register('f_')
    def f_mb(self):
        return
    @_dico.register()
    def m(self):
        return

class c(a):
    _dico = methoddict('f_')
    def __init__(self):
        a.__init__(self)
        self._dico.merge(c._dico)
    @_dico.register(name='zmc')
    def f_mc(self):
        return
    try:
        @_dico.register()
        def m(self):
            return
    except LookupError as error:
        logging.traceback.print_exc()
        print('Error is ignored')
        @_dico.register(name='zm')
        def m(self):
            return

xa = a()
xb = b()
xc = c()
for z in a, b, c:
    print("Class", z._dico.dict)
for x in xa, xb, xc:
    print("Instance", x._dico.dict)

print("end")

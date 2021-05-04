from flowdyn.modelphy.base import methoddict
class a():
    _dico = methoddict()
    prop = 0
    def __init__(self):
        self._dico = a._dico.copy()
    @_dico.register('ma')
    def ma(self):
        return

class b(a):
    _dico = methoddict()
    def __init__(self):
        a.__init__(self)
        self._dico.merge(b._dico)
    @_dico.register('mb') 
    def mb(self):
        return
    @_dico.register('m') 
    def m(self):
        return

class c(a):
    _dico = methoddict()
    def __init__(self):
        a.__init__(self)
        self._dico.merge(c._dico)
    @_dico.register('mc') 
    def mc(self):
        return
    @_dico.register('m') 
    def m(self):
        return

xa = a()
xb = b()
xc = c()
for m in a, b, c, xa, xb, xc:
    print(m._dico.dict)
for m in xa, xb, xc:
    print(m._dico.dict)

print("end")
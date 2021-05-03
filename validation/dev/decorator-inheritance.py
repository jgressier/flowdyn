class methoddict():
    """decorator to register decorated method as specific and tagged in the class model
    """
    def __init__(self):
        self.dict = {}

    def register(self, name):
        def decorator(classmeth):
            self.dict[name] = classmeth
            return classmeth
        return decorator

class a():
    _dico = methoddict()
    prop = 0
    def __init__(self):
        self.aa = ''
    @_dico.register('ma')
    def ma(self):
        return

class b(a):
    def __init__(self):
        self.bb = ''
    @a._dico.register('mb') # this function will be registered in class a too
    def mb(self):
        return

xa = a()
xb = b()
print(a.prop)
xa.prop = 1
print(a.prop, xa.prop)
a.prop = 2
print(a.prop, xa.prop)
print(b.prop, xb.prop)

class c():
    def __init__(self):
        self.cc = ''
        self._dico = methoddict()
    #@self._dico.register('mc')
    def mc(self):
        return

class d(c):
    def __init__(self):
        a.__init__(self)
        self.dd = ''
    #@self._dico.register('md')
    def md(self):
        return

xc = c()
xd = d()

print("end")
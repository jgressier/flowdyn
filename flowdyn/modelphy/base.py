# -*- coding: utf-8 -*-
"""
    The ``base`` module of modelphy library
    =========================

    Provides virtual class for all other model

    :Example:

    >>> import flowdyn.modelphy.base as modelbase
    >>> model = modelbase.model(name='test', neq=1)
    >>> print(model.neq, model.equation)
    1 test

    Available functions
    -------------------

 """

class methoddict():
    """decorator to register decorated method as specific and tagged in the class model
    """
    def __init__(self, items={}, pref=""): # pref = prefix to be stripped off the method's name
        if type(items) == type(""): # if only the prefix is given as argument
            pref = items
            items = {}
        self.dict = dict(items)
        self.pref = pref

    def register(self, pref=None, name=None): # name = alternate name for the method in the dict
        def decorator(classmeth):
            rpref = self.pref if pref is None else pref
            if name is None:
                rname = classmeth.__name__
                if not rname[:len(rpref)] == rpref:
                    raise(LookupError("Prefix "+repr(rpref)+" not found in name "+repr(rname)))
                rname = rname[len(rpref):]
            else:
                rname = name
            self.dict[rname] = classmeth
            return classmeth
        return decorator

    def merge(self, mdict):
        self.dict.update(mdict.dict)

    def update(self, ddict: dict):
        self.dict.update(ddict)

    def copy(self):
        return methoddict(self.dict)

# ===============================================================
# implementation of MODEL class

class model():
    """
    Class model (as virtual class)

    attributes:
        neq
        islinear
        has_firstorder_terms
        has_secondorder_terms
        has_source_terms

    """
    _bcdict = methoddict('bc_')   # dict and associated decorator method to register BC
    _vardict = methoddict()
    _numfluxdict = methoddict('numflux_')

    def __init__(self, name='not defined', neq=0):
        self.equation = name
        self.neq      = neq
        self.source   = None
        self.islinear = 0
        self.has_firstorder_terms  = 0
        self.has_secondorder_terms = 0
        self.has_source_terms      = 0
        self._bcdict  = model._bcdict.copy()
        self._vardict  = model._vardict.copy()
        self._numfluxdict  = model._numfluxdict.copy()

    def __repr__(self):
        print("model: ", self.equation)
        print("nb eq: ", self.neq)

    def list_bc(self):
        return ['per']+list(self._bcdict.dict.keys())

    def list_var(self):
        return self._vardict.dict.keys()

    def cons2prim(self, qdata):  # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")

    def prim2cons(self, pdata):  # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")

    def initdisc(self, mesh):
        return

    def numflux(self, name, pL, pR): # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")

    def timestep(self, data, dx, condition):  # NEEDS definition by derived model
        raise NameError("must be implemented in derived class")

    def nameddata(self, name, data):
        return (self._vardict.dict[name])(self, data)

    def namedBC(self, name, dir, data, param):
        return (self._bcdict.dict[name])(self, dir, data, param)

    #------------------------------------
    # definition of boundary conditions with name bc_*

    @_bcdict.register()
    def bc_dirichlet(self, dir, data, param):
        return param['prim']

# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest
    doctest.testmod()


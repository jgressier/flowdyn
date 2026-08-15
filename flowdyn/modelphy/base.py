# -*- coding: utf-8 -*-
"""Provide base classes and registration helpers for physical models.

Example:

    >>> import flowdyn.modelphy.base as modelbase
    >>> model = modelbase.model(name='test', neq=1)
    >>> print(model.neq, model.equation)
    1 test
"""


class methoddict:
    """decorator to register decorated method as specific and tagged in the class model"""

    def __init__(self, items=None, pref=""):  # pref = prefix to be stripped off the method's name
        if isinstance(items, str):  # if only the prefix is given as argument
            pref = items
            items = {}
        self.dict = dict(items or {})
        self.pref = pref

    def register(self, pref=None, name=None):  # name = alternate name for the method in the dict
        def decorator(classmeth):
            rpref = self.pref if pref is None else pref
            if name is None:
                rname = classmeth.__name__
                if not rname[: len(rpref)] == rpref:
                    raise (LookupError("Prefix " + repr(rpref) + " not found in name " + repr(rname)))
                rname = rname[len(rpref) :]
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


class model:
    """
    Class model (as virtual class)

    attributes:
        neq
        islinear
        has_firstorder_terms
        has_secondorder_terms
        has_source_terms

    """

    _bcdict = methoddict('bc_')  # dict and associated decorator method to register BC
    _vardict = methoddict()
    _numfluxdict = methoddict('numflux_')

    def __init__(self, name='not defined', neq=0):
        if not isinstance(neq, int) or neq < 0:
            raise ValueError("neq must be a non-negative integer")
        self.equation = name
        self.neq = neq
        self.source = None
        self.islinear = 0
        self.has_firstorder_terms = 0
        self.has_secondorder_terms = 0
        self.has_source_terms = 0
        self._bcdict = model._bcdict.copy()
        self._vardict = model._vardict.copy()
        self._numfluxdict = model._numfluxdict.copy()

    def __repr__(self):
        return f"model: {self.equation}\nnb eq: {self.neq}"

    def list_bc(self):
        return ['per'] + list(self._bcdict.dict.keys())

    def list_var(self):
        return self._vardict.dict.keys()

    def cons2prim(self, qdata):  # NEEDS definition by derived model
        raise NotImplementedError("cons2prim must be implemented in a derived model")

    def prim2cons(self, pdata):  # NEEDS definition by derived model
        raise NotImplementedError("prim2cons must be implemented in a derived model")

    def initdisc(self, mesh):
        return

    def numflux(self, name, pL, pR):  # NEEDS definition by derived model
        raise NotImplementedError("numflux must be implemented in a derived model")

    def timestep(self, data, dx, condition):  # NEEDS definition by derived model
        raise NotImplementedError("timestep must be implemented in a derived model")

    def nameddata(self, name, data):
        if name not in self._vardict.dict:
            available = ", ".join(sorted(self._vardict.dict))
            raise ValueError(f"unknown variable {name!r}; available variables: {available}")
        return self._vardict.dict[name](self, data)

    def namedBC(self, name, direction, data, param):
        if name not in self._bcdict.dict:
            available = ", ".join(self.list_bc())
            raise ValueError(f"unknown boundary condition {name!r}; available conditions: {available}")
        return self._bcdict.dict[name](self, direction, data, param)

    # ------------------------------------
    # definition of boundary conditions with name bc_*

    @_bcdict.register()
    def bc_dirichlet(self, direction, data, param):
        return param['prim']


# ===============================================================
# automatic testing

if __name__ == "__main__":
    import doctest

    doctest.testmod()

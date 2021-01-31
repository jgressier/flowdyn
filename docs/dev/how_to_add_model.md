# How to add a new model

The numerical methods in `integration` and `modeldisc` need a definition of a physical model with at least some methods which are listed in `modelphybase`. Obviously, existing models are examples but all definitions are not necessary.

## compulsory methods

* `__init__` should be adapted the new model ; at least define model name and number of equations
* `prim2cons` and `cons2prim` methods must be defined even if they are the same (see Burgers)
* `numflux` *is* the definition of the physics of the model ; there is currently no different definition of physical flux (of one only state) and numerical flux (from two states)
* `timestep` must be defined as a kind of definition of spectral radius of the dynamical system, with dependency to mesh size and a user _time_ condition


## variables

* primitive (user) variables must be defined by the `prim2cons` and `cons2prim` methods
* a set of variables can be defined through a dictionnary `model._vardict` with a name and associated fonction, computed from conservative data. For example :
```python
def __init__(self, ...):
    ...
    self._vardict = { 'pressure': self.pressure, 'velocity': self.velocity }

def pressure(self, qdata): # returns (gam-1)*( rho.et) - .5 * (rho.u)**2 / rho )
        return (self.gamma-1.0)*(qdata[2]-0.5*qdata[1]**2/qdata[0])

def velocity(self, qdata): # returns (rho u)/rho
    return qdata[1]/qdata[0]
```

## boundary conditions

Available boundary conditions are also listed in a dictionnary which is updated with
```python
self._bcdict.update({'sym': self.bc_sym, 'insub': self.bc_insub, ...)
```
and then defined with the above defined name
```python
def bc_outsub(self, dir, data, param):
        return [ data[0], data[1], param['p'] ] 
```
where
* `dir` is a variable used to specify either the direction or side of the boundary condition
* `data` is the local _interior_ *primitive* data
* `param` is a dictionnary which has been defined by the user and is associated to this boundary condition

Note that `per` for periodic and `dirichlet` conditions are already defined in `modelphy.base`.
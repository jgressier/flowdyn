# Models

## linear convection model

## Burgers model
## 1D Euler model

A very classical 1D inviscid compressible model is defined in `modelphy.euler`. It is only dependend on $`\gamma`$ coefficient (specific heats ratio) which is 1.4 by default. Additional source terms can be added to mass, momentum, energy equations. It allows the definition of derived models. The model uses
* the Euler model for inviscid fluid
* the ideal gas laws
Note that the temperature is never directly used, so the $`r`$ constant of the gas is not needed.

### data model

* Dedicated to the Finite Volume method, the genuine variables are the conservative variables
```math
 Q = ( \rho, \rho\, u, \rho e_t)
```
* the primitive variables are $`\rho, u, p`$.

### boundary conditions

Available boundary conditions are
* `per` periodic
* `dirichlet` apply/force a set a primitive condition (may be overdetermined)
* `sym` symmetry or slipping wall (inviscid)
* `insub` subsonic inlet: needs `ptot` and `rttot` parameters
* `insup` supersonic inlet: needs 'ptot', 'rttot' and either 'p' or 'mach' parameters
* `outsub` subsonic outlet: needs `p` parameter 
* `outsup` supersonic outlet: no additional parameter

### available variables

* conservative variables are $`\rho, \rho\, u, \rho e_t`$
* primitive variables are $`\rho, u, p`$
* post-processed variables are `pressure`, `density`, `velocity`, `mach`, `enthalpy`, `entropy`, `ptot`, `htot`

### specific numerical methods

* the only specific numerical method is HLLC numerical flux

### derived models

Derived models of the base `euler` model are available
* `modelphy.euler.nozzle(sectionlaw, gamma)` : source terms are computed to model varying section effect

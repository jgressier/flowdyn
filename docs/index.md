# flowdyn documentation

`flowdyn` is a finite volume based toy project to test methods for cfd courses and some research preliminary and exploratory work.

[![PyPi Version](https://img.shields.io/pypi/v/flowdyn.svg?style=flat)](https://pypi.org/project/flowdyn)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/flowdyn.svg?style=flat)](https://pypi.org/pypi/flowdyn/)
[![Doc](https://readthedocs.org/projects/flowdyn/badge/?version=latest)](https://readthedocs.org/projects/flowdyn/)
[![GitHub stars](https://img.shields.io/github/stars/jgressier/flowdyn.svg?style=flat&logo=github&label=Stars&logoColor=white)](https://github.com/jgressier/flowdyn)
[![PyPi downloads](https://img.shields.io/pypi/dm/flowdyn.svg?style=flat)](https://pypistats.org/packages/flowdyn)
[![codecov](https://img.shields.io/codecov/c/github/jgressier/flowdyn.svg?style=flat)](https://codecov.io/gh/jgressier/flowdyn)

## to start

## models

* [convection](models/convection): physical model for 1D scalar equation for convection (and maybe diffusion)
* [burgers](models/burgers): physical model for 1D scalar non linear convection equation
* [euler](models/euler): physical model for compressible 1D inviscid equation, derived models with source terms to model nozzle are proposed.

## numerical methods

* [Finite Volume RHS discretization](userguide/num#finite-volume-method) `modeldisc`
* [time integrators](num/time_integrators)

## development

* [structure of python libraries](dev/flowdyn_structure)
* [how to define a new model](dev/how_to_add_model)
* [automatic testing](dev/automatic_testing)
* future developments
    - 2D euler model
    - more implicit integration methods
    - non reflecting conditions
    - impedance wall conditions
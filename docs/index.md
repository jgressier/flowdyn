# flowdyn documentation

`flowdyn` is a finite volume based toy project to test methods for cfd courses and some research preliminary and exploratory work.

[![PyPi Version](https://img.shields.io/pypi/v/flowdyn.svg?style=flat)](https://pypi.org/project/flowdyn)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/flowdyn.svg?style=flat)](https://pypi.org/pypi/flowdyn/)
[![Build Status](https://travis-ci.com/jgressier/flowdyn.svg?branch=master)](https://travis-ci.com/jgressier/flowdyn)
[![GitHub stars](https://img.shields.io/github/stars/jgressier/flowdyn.svg?style=flat&logo=github&label=Stars&logoColor=white)](https://github.com/jgressier/flowdyn)

[![Doc](https://readthedocs.org/projects/flowdyn/badge/?version=latest)](https://flowdyn.readthedocs.io/en/latest/)
[![Slack](https://img.shields.io/static/v1?logo=slack&label=slack&message=contact&style=flat)](https://join.slack.com/t/isae-opendev/shared_invite/zt-obqywf6r-UUuHR4_hc5iTzyL5bFCwpw
)
[![PyPi downloads](https://img.shields.io/pypi/dm/flowdyn.svg?style=flat)](https://pypistats.org/packages/flowdyn)
[![codecov](https://img.shields.io/codecov/c/github/jgressier/flowdyn.svg?style=flat)](https://codecov.io/gh/jgressier/flowdyn)
[![lgtm](https://img.shields.io/lgtm/grade/python/github/jgressier/flowdyn.svg?style=flat)](https://lgtm.com/projects/g/jgressier/flowdyn/)

## to start

## models

* [convection](userguide/models/convection): physical model for 1D scalar equation for convection (and maybe diffusion)
* [burgers](userguide/models/burgers): physical model for 1D scalar non linear convection equation
* [shallow water](userguide/models/shallowwater): physical model for 1D shallow water equations
* [euler](userguide/models/euler): physical model for compressible 1D inviscid equation, derived models with source terms to model nozzle are proposed.

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
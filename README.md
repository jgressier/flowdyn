flowdyn
-----

[![PyPi Version](https://img.shields.io/pypi/v/flowdyn.svg?style=flat)](https://pypi.org/project/flowdyn)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/flowdyn.svg?style=flat)](https://pypi.org/pypi/flowdyn/)
[![Build Status](https://travis-ci.com/jgressier/flowdyn.svg?branch=master)](https://travis-ci.com/jgressier/flowdyn)
[![Doc](https://readthedocs.org/projects/flowdyn/badge/?version=latest)](https://flowdyn.readthedocs.io/en/latest/)
[![Slack](https://img.shields.io/static/v1?logo=slack&label=slack&message=contact&style=flat)](https://join.slack.com/t/isae-opendev/shared_invite/zt-obqywf6r-UUuHR4_hc5iTzyL5bFCwpw
)

[![GitHub stars](https://img.shields.io/github/stars/jgressier/flowdyn.svg?style=flat&logo=github&label=Stars&logoColor=white)](https://github.com/jgressier/flowdyn)
[![PyPi downloads](https://img.shields.io/pypi/dm/flowdyn.svg?style=flat)](https://pypistats.org/packages/flowdyn)
[![codecov](https://img.shields.io/codecov/c/github/jgressier/flowdyn.svg?style=flat)](https://codecov.io/gh/jgressier/flowdyn)
[![lgtm](https://img.shields.io/lgtm/grade/python/github/jgressier/flowdyn.svg?style=flat)](https://lgtm.com/projects/g/jgressier/flowdyn/)

### Documentation and examples

* documentation of the [official release](https://flowdyn.readthedocs.io/en/latest/) or [development branch](https://flowdyn.readthedocs.io/en/develop/)
* some [examples in the documentation](https://flowdyn.readthedocs.io/en/latest/examples) pages
* some examples in the [github repository](https://github.com/jgressier/flowdyn/tree/master/validation)

### Features

Current version includes
* 1D scalar models: linear convection, Burgers
* 1D model: inviscid compressible flow (Euler), section law effects with source terms
* 1st to 3rd order linear extrapolation, 2nd order MUSCL extrapolation
* various centered or upwind/Riemann fluxes
* explicit, runge-kutta and implicit integrators

### Installation and usage

    pip install flowdyn

### Requirements

* `numpy`
* examples are plotted using [matplotlib](http://matplotlib.org)


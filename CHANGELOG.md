# Changelog & Release Notes

## Upgrading

To upgrade to the latest version of `flowdyn` use `pip`:

```bash
pip install flowdyn --upgrade
```

You can determine your currently installed version using this command:

```bash
pip show flowdyn
```

## Versions

### [1.x.x](https://pypi.org/project/flowdyn/) (2022-xx-xx)

#### new features

- Euler model: new outlet CBC (characteristic based) condition `outsub_cbc`
- Euler model: NRCBC (non-reflecting CBC) with relaxation

### [1.3.0](https://pypi.org/project/flowdyn/) (2021-11-05)

#### new features

- Finite Volume method on 2D cartesian grids
  - 'extrapol2d1' and 'extrapol2dk' linear extrapolation (1st to 3rd order)
  - geometric symmetry or periodic conditions, or model based
- euler 2d model on cartesian grids
  - 'hlle' and 'centered' flux
  - set of 2d variables
  - 'insup', 'insub', 'ousub', 'outsub' conditions
- new subsonic inlet/oulet boundary condition for `euler1d` model
  - `insub_cbc` with stagnation parameters but more stable
  - `outsub_rh` able to trigger shock wave even in supersonic flow
- `field.semilogy('name')` for semi-logarithmic plots
- specific `fieldlist` object output from integration
- monitoring features:
  - new monitor `data_average`
  - monitor type handled via monitor name or type
  - monitor class provides plotting functions

### [1.2.0](https://pypi.org/project/flowdyn/) (2021-06-05)

#### new features

- shallow water model `modelphy.shallowwater`
- new subsonic outlet boundary condition for `euler1d` model
  - `outsub_prim` as a legacy primitive variables set, same as `outsub`
  - `outsub_qtot` computed with `p` parameter and `ptot` and `rttot` extrapolation
  - `outsub_nrcbc` for non-reflective characteristics conditions
- new option to `timemodel.solve`: stop parameter with dictionary `tottime` or `maxit`
- new monitoring feature to `timemodel.solve`

#### fixed

- bad initialization of cpu time computation in `show_perf()` for successive integration
- fix `solution.nozzle` for fully supersonic cases

### [1.1.2](https://pypi.org/project/flowdyn/) (2021-04-17)

#### changed

- add `asound` variable to `euler` model

#### fixed

- fix MUSCL vanleer limiter (defective since 1.1.0)

### [1.1.1](https://pypi.org/project/flowdyn/) (2021-04-06)

#### fixed

- bug fix in 1d euler supersonic inlet condition

### [1.1.0](https://pypi.org/project/flowdyn/) (2021-03-10)

#### new features

- Runge-Kutta methods `rk2_heun`and `'rk3_heun` from Heun
- low storage Runge-Kutta (LSRK) methods from Hu and Hussaini
- LSRK implementation of Bogey and Bailly `lsrk25bb`and `lsrk26bb``

#### changed

- analytical 1D solution for nozzle flows in `solution.euler_nozzle`
- improve test coverage
- optimize some mesh computation
- allow additional sources in (source based) `nozzle` euler model
- `euler1d` and `nozzle` models have new `massflow` output variable

#### fixed

- avoid warnings with `vanalbada` and `vanleer` limiters when uniform flows

### [1.0.1](https://pypi.org/project/flowdyn/) (2021-01-29)

#### fixed

- fix computation of `modelphy.euler` supersonic inlet condition

### [1.0.0](https://pypi.org/project/flowdyn/) (2021-01-27)

- models: convection, Burgers, Euler and derived (nozzle)
- Finite Volume method for 1d mesh
- numerical methods: linear 1st to 3rd order extrapolation ; MUSCL method and associated limiters
- integrators: explicit, Runge-Kutta, implicit

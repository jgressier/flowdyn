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

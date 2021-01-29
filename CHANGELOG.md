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

### [1.0.1](https://pypi.org/project/flowdyn/) (2021-01-27)

#### Bug fix

- fix computation of `modelphy.euler` supersonic inlet condition 
### [1.0.0](https://pypi.org/project/flowdyn/) (2021-01-27)

- models: convection, Burgers, Euler and derived (nozzle)
- Finite Volume method for 1d mesh
- numerical methods: linear 1st to 3rd order extrapolation ; MUSCL method and associated limiters
- integrators: explicit, Runge-Kutta, implicit

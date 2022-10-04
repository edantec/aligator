# proxddp

A primal-dual augmented Lagrangian-type trajectory optimization solver.

## Features

This package provides

* a modelling interface for optimal control problems, node-per-node
* an efficient solver algorithm
* (optional) an interface to [Crocoddyl](https://github.com/loco-3d/crocoddyl)

## Installation

### Dependencies

* [proxnlp](https://github.com/Simple-Robotics/proxnlp.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [Boost](https://www.boost.org) >= 1.71.0
* [eigenpy](https://github.com/stack-of-tasks/eigenpy) >= 2.7.2
* (optional) [Crocoddyl](https://github.com/loco-3d/crocoddyl)

Python:

* [typed-argument-parser](https://github.com/swansonk14/typed-argument-parser)
* [pin-meshcat-utils](https://github.com/Simple-Robotics/pin-meshcat-utils)

### Build from source

```bash
git clone repo_link --recursive
# define envars here
cmake -DCMAKE_INSTALL_PREFIX=your_install_folder -S . -B build/
cd build/
cmake --build . -jNCPUS
```

Options:

* For developers, add the `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` when working with language servers e.g. clangd.
* To use the Crocoddyl interface, add `-DBUILD_CROCODDYL_COMPAT=ON`

**Building against conda:** define

```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
```

and use `$CONDA_PREFIX` as your install folder.

## Contributors

* [Antoine Bambade](https://bambade.github.io/)
* [Justin Carpentier](https://jcarpent.github.io/)
* [Wilson Jallet](https://manifoldfr.github.io/)
* Sarah Kazdadi
* [Quentin Le Lidec](https://quentinll.github.io/)
* Nicolas Mansard
* Guilhem Saurel
* Fabian Schramm

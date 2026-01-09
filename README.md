# TODOs
- [x] implement using B_pinv as solver
- [x] implement assembling B_T, B_T dense on device
- [x] eta updates for B_pinv
- [x] storage efficient implementation of solvers
- [x] investigate better ways to store B, Bt and to compute BtB
- [ ] implement customized sparse matrix and vector representations
- [x] implement assembling B from A on device
- [ ] figure out the cost of creating and destroying handles and matrix representations in cuSPARSE
- [ ] investigate "nearly constant" size per row in the sparse matrix representation of B and Bt (fast reconstruction without reallocating memory every time)
- [ ] implement dense matrix - sparse vector multiplication kernel
- [ ] test the new idea of parallelizing the pivoting strategy with MPI --> in parallel-pivot-strategies
- [ ] investigate the tradeoff of keeping basic_list on device vs copying it every time
- [ ] move more data structures to device to reduce data transfer overhead (more kernels for various steps)
- [ ] implement more analysis tools for optimal kernel launching

# Prototype Folder

This is a Python playground to test dual simplex ideas quickly.

# C++ Modules

This directory contains the C++ modules for the cuOpt project with custom GPU acceleration. 
Use the build.sh script to create an executable. Note that many flags need to be changed depending on your environment.

Please refer to the [CMakeLists.txt](CMakeLists.txt) file for details on how to add new modules and tests.

Most of the dependencies are defined in the [dependencies.yaml](../dependencies.yaml) file. Please refer to different sections in the [dependencies.yaml](../dependencies.yaml) file for more details. However, some of the dependencies are defined in [thirdparty modules](cmake/thirdparty/) in case where source code is needed to build, for example, `cccl` and `rmm`.


## Include Structure

Add any new modules in the `include` directory under `include/cuopt/<module_name>` directory.

```bash
cpp/
├── include/
│   ├── cuopt/
│   │   └── linear_programming/
│   │       └── ...
│   │   └── routing/
│   │       └── ...
│   └── ...
└── ...
```

## Source Structure

Add any new modules in the `src` directory under `src/cuopt/<module_name>` directory.

```bash
cpp/
├── src/
│   ├── cuopt/
│   │   └── linear_programming/
│   │       └── ...
│   │   └── routing/
│   │       └── ...
└── ...
```

## MPS parser

The MPS parser is a standalone module that parses MPS files and converts them into a format that can be used by the cuOpt library.

It is located in the `libmps_parser` directory. This also contains the `CMakeLists.txt` file to build the module.


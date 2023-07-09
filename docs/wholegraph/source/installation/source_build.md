# Build from Source

WholeGraph has two packages, one is C library `libwholegraph`, the other is Python library `pylibwholegraph`.
If you want to build these WholeGraph, you need to build these two packages.

Before build, you may first set up the environment.
```text
CUDA Toolkit>=11.5
cmake>=3.26.4
ninja
nccl
cython
setuputils3
scikit-learn
scikit-build
nanobind>=0.2.0
```

To build these two packages, you can simply run the following script from root of WholeGraph repository:
```shell
./build.sh
```
Then the script will build both `libwholegraph` and `pylibwholegraph`.
And if `-n` flag is not used, it will also install `libwholegraph` and `pylibwholegraph`.
More details of the `build.sh` script can be found by running
```shell
./build.sh --help
```
Which may output:
```text
./build.sh [<target> ...] [<flag> ...]
 where <target> is:
   clean                    - remove all existing build artifacts and configuration (start over).
   uninstall                - uninstall libwholegraph and pylibwholegraph from a prior build/install (see also -n)
   libwholegraph            - build the libwholegraph C++ library.
   pylibwholegraph          - build the pylibwholegraph Python package.
   tests                    - build the C++ (OPG) tests.
   benchmarks               - build benchmarks.
   docs                     - build the docs
 and <flag> is:
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   --allgpuarch               - build for all supported GPU architectures
   --cmake-args=\"<args>\" - add arbitrary CMake arguments to any cmake call
   --compile-cmd               - only output compile commands (invoke CMake without build)
   --clean                    - clean an individual target (note: to do a complete rebuild, use the clean target described above)
   -h | --h[elp]               - print this text

 default action (no args) is to build and install 'libwholegraph' and then 'pylibwholegraph'

 libwholegraph build dir is:

 Set env var LIBWHOLEGRAPH_BUILD_DIR to override libwholegraph build dir.
```

## Manually build libwholegraph from source
If you don't want to use the `build.sh` script, but want to manually build `libwholegraph`. You can follow these steps:
```shell
cd cpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install ../
make -j
make install
```
Then the built library `libwholegraph` will be installed to `cpp/build/install`. If you want to install it to other
location, you may need to specify that path in `-DCMAKE_INSTALL_PREFIX`.

## Manually build pylibwholegraph from source
`pylibwholegraph` is a python package, most part of this package is python code that don't need build phase.
However, in `python/pylibwholegraph/pylibwholegraph/binding` directory, there is Cython binding for python.
This part need to be compiled to so files to be called from Python code.
To build `pylibwholegraph`, you can follow these steps:
```shell
export LIBWHOLEGRAPH_DIR=`pwd`/cpp/build/install
cd python/pylibwholegraph
mkdir build
cd build
cmake ../
make -j
```
Then `python/pylibwholegraph/build/pylibwholegraph/binding/wholememory_binding.cpython*` will be generated.
If you want to run the code locally, this Cython file may be linked into the Python source directory.
From `python/pylibwholegraph/build` directory:
```shell
ln -s `pwd`/pylibwholegraph/binding/wholememory_binding.cpython* ../pylibwholegraph/binding/
```

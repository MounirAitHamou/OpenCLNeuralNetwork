# Installation

Requirements:

- A C++20 compiler (MSVC 2022, recent GCC, or recent Clang)
- CMake 3.20 or newer
- Ninja when using the supplied presets
- A vendor OpenCL runtime/graphics driver for GPU execution

Configure, build, and test:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Or use `./build.sh debug` / `build.bat debug`. No OpenCL SDK, OpenCL headers, HDF5, CLBlast, or package manager is required. CLNN dynamically loads `OpenCL.dll`, `libOpenCL.so`, or the macOS OpenCL framework at runtime.

The test suite skips OpenCL-only cases when no GPU runtime is installed. The `clnn_examples` example requires a GPU and prints the selected device.

To install the library:

```sh
cmake --install out/build/release --prefix /your/prefix
```
(Example prefixes: `/usr/local` on Linux, `C:/Libraries/CLNN` on Windows, or `~/clnn` for a user-local install.)

Downstream CMake projects can then use `find_package(CLNN CONFIG REQUIRED)` and link `CLNN::clnn`.

## Python package

Install the Python package with pip from PyPI:

```sh
python -m pip install .
```

Python 3.9 or newer and NumPy are required. Build an installable wheel with:

```sh
python -m pip install build
python -m build --wheel
python -m pip install dist/clnn_autograd-*.whl
```

For an in-tree CMake build, enable `CLNN_BUILD_PYTHON`:

```sh
cmake -S . -B out/build/python -DCLNN_BUILD_PYTHON=ON -DCLNN_BUILD_TESTS=ON
cmake --build out/build/python --config Release
ctest --test-dir out/build/python -C Release --output-on-failure
```

CMake finds an installed pybind11 3.x package first and otherwise downloads pybind11 3.1.0
when `CLNN_FETCH_PYBIND11=ON` (the default).

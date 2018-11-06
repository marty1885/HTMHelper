# HTMHelper
Header only NuPIC.core wrapper for low level HTM algorithm abstraction

## Fratures
 * Python like interface for C++ NuPIC.core users
 * Header only
   * No need to deal with yet another linker issue
 * Easy to use
   * No more dealing with std::vector when it should be a N-D array
   * No more switching between dense/indexed SDR
   * Fimilar patterns for Deep Learning developers

## Dependencies
 * NuPIC.core - https://github.com/numenta/nupic.core
 * xtensor - https://github.com/QuantStack/xtensor
 * CMake (To build the exmaples) - https://cmake.org/
 * A C++14 capable compiler

## Supported Layers and Encoders
 * Layers
   * SpatialPooler
   * TemporalMemory
   * TemporalPooler (Cells4)
 * Encoders
   * ScalarEncoder
   * CategoryEncoder
     * Including a decocder!
 * Classifers
   * SDRClassifer
 * Other
   * Raw anomoly

## Build and install
You don't need ot build! It is header only. Still, CMake build examples and installs header foor you.
``` sh
mkdir build
cd build
cmake ..
make
sudo make install
```

### NOTE: Building NuPIC.core
``` sh
NuPIC.core is a bit annoning to build. Here is how
git clone https://github.com/numenta/nupic.core
cd nupic.core
export NUPIC_CORE=`pwd`
cd $NUPIC_CORE/build/scripts
cmake $NUPIC_CORE -DCMAKE_BUILD_TYPE=Release -NUPIC_TOGGLE_INSTALL=ON -DPY_EXTENSIONS_DIR=$NUPIC_CORE/bindings/py/src/nupic/bindings .
make -j4
sudo make install
```

And since NuPIC has problems installing all the headers. You’ll need to copy them manually.

``` sh
cd $NUPIC_CORE
sudo cp -r src/nupic /usr/local/include
sudo cp -r build/scripts/src/nupic /usr/local/include
```


## Todo
 - [ ] Saving and loading models
 - [ ] More encoders
 - [ ] More classifers
 - [ ] Python binding via cookiecutter and pybind11 (?)


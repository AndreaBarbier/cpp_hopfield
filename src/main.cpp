/**
@mainpage Hopfield Network

@brief Interactive C++20 Hopfield simulator: train a network on images, corrupt
them with noise, and watch it recall the stored patterns.

@section intro Overview

This project implements a discrete **Hopfield network** that stores binary
images as attractors of an energy function. Given a noisy or partial image,
the network evolves until it settles on the closest stored pattern.

The state of the network is a vector of neurons @f$ s_i \in \{-1, +1\} @f$,
coupled by a symmetric weight matrix built with the Hebb rule:

@f[
  W_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^\mu \, \xi_j^\mu ,
  \qquad W_{ii} = 0
@f]

and the dynamics lowers the energy

@f[
  E(s) = -\frac{1}{2} \sum_{i \neq j} W_{ij}\, s_i\, s_j .
@f]

@section modules Code structure

| Module | Files | Role |
| :--- | :--- | :--- |
| Network | `hopfield.hpp`, `hopfield.cpp` | Weights, training, recall dynamics, energy, Hamming distance |
| Images | `image_processor.hpp`, `image_processor.cpp` | Image to bipolar vector conversion (SFML), noise and corruption |
| Application | `main.cpp` | Command-line interface and recall viewer (SFML window) |
| Tests | `src/tests/` | Unit tests with doctest |

See the @ref classes "class list" and the @ref files "file list" for the full API.

@section usage Quick start

@code{.sh}
cmake -S . -B build
cmake --build build

./build/hopfield init  <dir> <name>          # create an empty network
./build/hopfield train <net.hop> <dataset>   # compute the weight matrix
./build/hopfield recall <net.hop> <image>    # open the viewer
@endcode

@note Adjust the executable name and the subcommands to match your actual CLI.

@section files_format File formats

- **Dataset**: a folder of images. The network size is derived automatically
  from the image dimensions.
- **`.hop`**: binary file holding the weight matrix. The network name is the
  file name; the extension is fixed.

@section tests Running the tests

@code{.sh}
ctest --test-dir build --output-on-failure
@endcode

@section refs References

- J. J. Hopfield, *Neural networks and physical systems with emergent collective
  computational abilities*, PNAS 79 (1982).
*/


#include "repl.hpp"
#include "session.hpp"
#include <iostream>

int main(){
    hopfield::Session session;
    hopfield::runRepl(session, std::cin, std::cout);
}
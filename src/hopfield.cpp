#include <SFML/Graphics.hpp>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

namespace hopfield {
// ============================================================
// WEIGHT MATRIX
// ============================================================

/// @brief Matrix struct -- it represents the network's memory
struct Matrix {
  size_t rows;
  size_t cols;
  std::vector<double> data;

  Matrix(size_t rows, size_t cols)
      : rows{rows}, cols{cols}, data{std::vector<double>(rows * cols, 0.0)} {}

  double& operator()(size_t i, size_t j) { return data[j * cols + i]; }
  double const& operator()(size_t i, size_t j) const {
    return data[j * cols + i];
  }
};
}  // namespace hopfield

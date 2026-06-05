#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "image_processor.hpp"

namespace hopfield {
// ============================================================
// SIGN FUNCTION
// ============================================================

/// @brief Returns the sign of a value
/// @tparam T Numeric type
/// @param[in] val The value to evaluate
/// @return Returns 1 if val > 0, -1 if val < 0, 0 if val == 0
template <class T>
int sgn(T val) {
  return ((T{0} < val) - (T{0} > val));
}

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

// ============================================================
// NEURLA NETWORK
// ============================================================

/// @brief Hopfield neural network class
class Network {
  std::vector<sf::Image> trainImgs{};
  sf::Vector2u validSize{0u, 0u};
  std::string wMatrixFilePath;

 public:
  Network(std::string const& wMatrixFilePath)
      : wMatrixFilePath(wMatrixFilePath) {}

  /// @brief Adds an image into Networks's train images vector
  /// @param[in] img Image to add
  void addImage(sf::Image const& img) {
    auto size{img.getSize()};
    assert(size.x > 0u && size.y > 0u);
    if (size.x == 0u || size.y == 0u) {
      throw std::runtime_error("Error: invalid image.\n");
    }
    trainImgs.push_back(img);
  }

  /// @brief Adds multiple images into Networks's train images vector
  /// @param[in] imgs Vector of images to add
  void addImages(std::vector<sf::Image> const& imgs) {
    for (auto img : imgs) {
      addImage(img);
    }
  }

  /// @brief Adds multiple images into Networks's train images vector
  /// @throws std::runtime_error if it is not possible to open the file
  /// @param[in] file File in which are written the paths of the images to add
  void addImages(std::ifstream& file) {
    if (!file.is_open()) {
      std::runtime_error("Error: impossible to open the file.");
    }
    std::string absPath;
    while (std::getline(file, absPath)) {
      sf::Image img;
      if (!img.loadFromFile(absPath)) {
        throw std::runtime_error("Error: impossible to load " + absPath + '\n');
      }
      addImage(img);
    }
  }

  /// @brief Trains the Hopfield network on the loaded images using the Hebb
  /// rule.
  /// @details Computes the symmetric weight matrix from the binary patterns
  ///          extracted from the training images. The diagonal is set to zero
  ///          to prevent self-reinforcement. The resulting matrix is written
  ///          to the file specified at construction and cached for later use.
  /// @return The computed weight matrix.
  /// @throws std::runtime_error If no images have been loaded via addImage().
  /// @throws std::runtime_error If all loaded images have invalid size.
  /// @throws std::runtime_error If the weight matrix file cannot be opened.
  /// @note Call addImage() at least once before invoking this function.
  Matrix train() {
    if (trainImgs.empty()) {
      throw std::runtime_error(
          "Error: empty training dataset. Please call addImage() or "
          "addImages() before train().\n");
    }
    auto resizedImgs{resizeImages(validateImages(trainImgs))};
    if (trainImgs.empty()) {
      throw std::runtime_error(
          "Error: invalid training dataset. Every image had an invalid "
          "size.\n");
    }
    validSize = {resizedImgs[0].getSize()};
    for (auto const& img : resizedImgs) {
      assert(img.getSize().x != 0u);
      assert(img.getSize().y != 0u);
      assert(img.getSize() == validSize);
    }
    size_t const N{validSize.x * validSize.y};
    std::vector<PatternInt> binaryPatterns{imageToBinaries(resizedImgs)};

    Matrix w{N, N};
    std::ofstream file{wMatrixFilePath};
    if (!file.is_open()) {
      std::runtime_error("Error: impossible to open the file " +
                         wMatrixFilePath);
    }
    for (size_t j{0}; j != N; ++j) {
      for (size_t i{j}; i != N; ++i) {
        if (i == j) {
          w(j, i) = 0;
        } else {
          for (auto pattern : binaryPatterns) {
            w(j, i) += static_cast<double>(pattern.data[i]) *
                       static_cast<double>(pattern.data[j]) / N;
          }
        }
        w(i, j) = w(j, i);
      }
    }

    for (size_t j{0}; j != N; ++j) {
      for (size_t i{0}; i != N; ++i) {
        file << w(j, i);
      }
    }
    file.close();
    return w;
  }

  /// @brief Reconstructs a stored pattern from a partial or noisy input.
  /// @details Iteratively updates the pattern using the trained weight matrix
  ///          until convergence, following the Hopfield network dynamics.
  /// @param inPattern The input pattern to reconstruct. Must match the size
  ///                  of the patterns used during training.
  /// @return The reconstructed pattern after convergence.
  /// @throws std::runtime_error If the network has not been trained yet.
  /// @throws std::runtime_error If the weight matrix file cannot be opened.
  PatternInt recall(PatternInt const& inPattern) {
    if (validSize == sf::Vector2u{0u, 0u}) {
      throw std::runtime_error(
          "Error: the network has not been trained yet. "
          "Please call train() before recall().\n");
    }
    assert(inPattern.size == validSize);

    size_t const N{validSize.x * validSize.y};
    Matrix w{N, N};
    std::ifstream file{wMatrixFilePath};
    if (!file.is_open()) {
      std::runtime_error("Error: impossible to open the file " +
                         wMatrixFilePath);
    }
    for (size_t j{0}; j != N; ++j) {
      for (size_t i{0}; i != N; ++i) {
        file >> w(j, i);
      }
    }

    PatternInt oldPattern{};
    PatternInt newPattern{inPattern};
    oldPattern.size = inPattern.size;

    while (newPattern != oldPattern) {
      oldPattern.data = newPattern.data;
      for (size_t j{0}; j != N; ++j) {
        newPattern.data[j] = 0;
        for (size_t i{0}; i != N; ++i) {
          newPattern.data[j] += sgn(w(j, i) * oldPattern.data[i]);
        }
      }
    }

    file.close();
    return newPattern;
  }
};
}  // namespace hopfield

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
  return ((T{0} < val) - (T{0} > val))
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
  std::vector<sf::Image> trainImgs;
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

  /// @brief Trains the neural network using the Hebb rule
  /// @throws std::runtime_error if it is not possible to open the file
  /// @return Returns the weights matrix and writes it on a file
  Matrix train() const {
    auto resizedImgs{resizeImages(validateImages(trainImgs))};
    size_t newWidth{resizedImgs[0].getSize().x};
    size_t newHeight{resizedImgs[0].getSize().y};
    for (auto const& img : resizedImgs) {
      assert(img.getSize().x == newWidth);
      assert(img.getSize().y == newHeight);
    }
    size_t const N{newWidth * newHeight};
    std::vector<PatternInt> binaryPatterns{imageToBinaries(resizedImgs)};

    Matrix w{N, N};
    std::ofstream file{wMatrixFilePath};
    if (!file.is_open()) {
      std::runtime_error("Error: impossible to open the file " +
                         wMatrixFilePath);
    }
    for (size_t j{0}; j != newHeight; ++j) {
      for (size_t i{0}; i != newWidth; ++i) {
        for (auto pattern : binaryPatterns) {
          w(j, i) = static_cast<double>(pattern.data[i]) *
                    static_cast<double>(pattern.data[j]) / N;
          file << w(i, j);
        }
      }
    }
    file.close();
    return w;
  }
};
}  // namespace hopfield

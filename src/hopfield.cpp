#include <SFML/Graphics.hpp>
#include <algorithm>
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

/// @brief Hopfield neural network class
class Network {
  std::vector<sf::Image> trainImgs;

 public:
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
  /// @return Returns the weights matrix and writes it on a file
  Matrix train() const {
    size_t minWidth;
    size_t minHeight;
    auto N{minWidth * minHeight};
    // Resize all the images
    std::vector<std::vector<int>> binaryPatterns;
    // Converts every image into PatternInt

    Matrix w{N, N};
    std::ofstream file{};
    if (!file.is_open()) {
      std::runtime_error("Error: impossible to open the file.");
    }
    for (size_t j{0}; j != minHeight; ++j) {
      for (size_t i{0}; i != minWidth; ++i) {
        for (auto pattern : binaryPatterns) {
          w(j, i) = pattern[i] * pattern[j] / N;
          file << w(i, j);
        }
      }
    }
    file.close();
    return w;
  }
};
}  // namespace hopfield

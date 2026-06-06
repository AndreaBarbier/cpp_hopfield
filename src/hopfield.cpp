#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "image_processor.hpp"
#include "tests/doctest.h"

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

/// @brief 2D matrix with row-major storage, used to represent the network's
/// weight matrix.
/// @details Stores elements as a flat vector of doubles, accessible via (i, j)
/// indexing. The element at row i and column j is stored at data[i * cols + j].
struct Matrix {
  size_t rows;
  size_t cols;
  std::vector<double> matrix;

  /// @brief Constructs a zero-initialized matrix of the given dimensions.
  /// @param r Number of rows. Must be different from 0.
  /// @param c Number of columns. Must be different from 0.
  Matrix(size_t r, size_t c) : rows{r}, cols{c}, matrix(r * c, 0.0) {
    if (r == 0 || c == 0) {
      throw std::runtime_error("Error: invalid matrix size inserted.");
    }
  }
  /// @brief Constructs a matrix of the given dimensions from an existing data
  /// vector.
  /// @param r Number of rows. Must be different from 0.
  /// @param c Number of columns. Must be different from 0.
  /// @param data Flat vector of doubles in row-major order. Must have size r *
  /// c.
  /// @throws std::runtime_error If r or c are 0.
  /// @throws std::runtime_error If data.size() != r * c.
  Matrix(size_t r, size_t c, std::vector<double> const& data)
      : rows{r}, cols{c}, matrix{data} {
    if (rows == 0 || cols == 0) {
      throw std::runtime_error("Error: invalid matrix size inserted.");
    }
    if (data.size() != rows * cols) {
      throw std::runtime_error("Error: invalid data lenght inserted");
    }
  }

  /// @brief Returns a reference to the element at row i, column j.
  /// @param i Row index (0-based).
  /// @param j Column index (0-based).
  /// @throws std::runtime_error If i or j are out of range.
  double& operator()(size_t i, size_t j) {
    assert(rows > 0 && cols > 0);
    if (i >= rows || j >= cols) {
      throw std::runtime_error("Error: index out of range.");
    }
    return matrix[i * cols + j];
  }
  /// @brief Returns a const reference to the element at row i, column j.
  /// @param i Row index (0-based).
  /// @param j Column index (0-based).
  /// @throws std::runtime_error If i or j are out of range.
  double const& operator()(size_t i, size_t j) const {
    assert(rows > 0 && cols > 0);
    if (i >= rows || j >= cols) {
      throw std::runtime_error("Error: index out of range.");
    }
    return matrix[i * cols + j];
  }
};

// ============================================================
// NEURLA NETWORK
// ============================================================

/// @brief Hopfield neural network class
class Network {
  std::vector<sf::Image> trainImgs_{};
  sf::Vector2u validSize_{0u, 0u};
  std::string wMatrixFilePath_;

 public:
  /// @brief Contructs an empty hopfield neural network, only with the path of
  /// the matrix storege file.
  /// @param[in] wMatrixFilePath Matrix storege file.
  Network(std::string const& wMatrixFilePath)
      : wMatrixFilePath_(wMatrixFilePath) {}

  /// @brief Adds an image into Networks's train images vector
  /// @param[in] img Image to add
  void addImage(sf::Image const& img) {
    auto size{img.getSize()};
    assert(size.x > 0u && size.y > 0u);
    if (size.x == 0u || size.y == 0u) {
      throw std::runtime_error("Error: invalid image.\n");
    }
    trainImgs_.push_back(img);
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
    if (trainImgs_.empty()) {
      throw std::runtime_error(
          "Error: empty training dataset. Please call addImage() or "
          "addImages() before train().\n");
    }
    auto resizedImgs{resizeImages(validateImages(trainImgs_))};
    if (trainImgs_.empty()) {
      throw std::runtime_error(
          "Error: invalid training dataset. Every image had an invalid "
          "size.\n");
    }
    validSize_ = {resizedImgs[0].getSize()};
    for (auto const& img : resizedImgs) {
      assert(img.getSize().x != 0u);
      assert(img.getSize().y != 0u);
      assert(img.getSize() == validSize_);
    }
    size_t const N{validSize_.x * validSize_.y};
    std::vector<PatternInt> binaryPatterns{imageToBinaries(resizedImgs)};

    Matrix w{N, N};
    std::ofstream file{wMatrixFilePath_};
    if (!file.is_open()) {
      std::runtime_error("Error: impossible to open the file " +
                         wMatrixFilePath_);
    }
    for (size_t j{0}; j != N; ++j) {
      for (size_t i{j}; i != N; ++i) {
        if (i == j) {
          w(j, i) = 0;
        } else {
          for (auto pattern : binaryPatterns) {
            w(j, i) += static_cast<double>(pattern.data[i]) *
                       static_cast<double>(pattern.data[j]) /
                       static_cast<double>(N);
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
  /// @param[in] inPattern The input pattern to reconstruct. Must match the size
  ///                  of the patterns used during training.
  /// @param[in] callback An optional callable invoked at each iteration with
  /// the current intermediate pattern and the iteration index. Defaults to a
  /// no-op.
  /// @return The reconstructed pattern after convergence.
  /// @throws std::runtime_error If the network has not been trained yet.
  /// @throws std::runtime_error If the weight matrix file cannot be opened.
  PatternInt recall(
      PatternInt const& inPattern,
      std::function<void(PatternInt const& newPattern, size_t id)> const&
          callback = [](PatternInt const&, size_t) {}) const {
    if (validSize_ == sf::Vector2u{0u, 0u}) {
      throw std::runtime_error(
          "Error: the network has not been trained yet. "
          "Please call train() before recall().\n");
    }
    assert(inPattern.size == validSize_);

    size_t const N{validSize_.x * validSize_.y};
    Matrix w{N, N};
    std::ifstream file{wMatrixFilePath_};
    if (!file.is_open()) {
      std::runtime_error("Error: impossible to open the file " +
                         wMatrixFilePath_);
    }
    for (size_t j{0}; j != N; ++j) {
      for (size_t i{0}; i != N; ++i) {
        file >> w(j, i);
      }
    }

    PatternInt oldPattern{};
    PatternInt newPattern{inPattern};
    oldPattern.size = inPattern.size;

    size_t id{0};
    while (newPattern != oldPattern) {
      oldPattern.data = newPattern.data;
      for (size_t j{0}; j != N; ++j) {
        newPattern.data[j] = 0;
        for (size_t i{0}; i != N; ++i) {
          newPattern.data[j] += sgn(w(j, i) * oldPattern.data[i]);
        }
      }
      callback(newPattern, id);
      ++id;
    }

    file.close();
    return newPattern;
  }
};
}  // namespace hopfield

TEST_CASE("MATRIX STRUCT") {
  SUBCASE("Invalid Matrix") {
    CHECK_THROWS(hopfield::Matrix{0, 3});
    CHECK_THROWS(hopfield::Matrix{3, 0});
    CHECK_THROWS(hopfield::Matrix{0, 0});
  }
  SUBCASE("Matrix operator()") {
    std::vector<double> data{0.0, 1.0, 2.0, 3.0, 4.0,  5.0,
                             6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
    hopfield::Matrix m{3, 4, data};

    CHECK(m(0, 0) == 0.0);
    CHECK(m(0, 3) == 3.0);
    CHECK(m(2, 3) == 11.0);
    CHECK_THROWS(m(3, 0));
    CHECK_THROWS(m(0, 4));
    CHECK_THROWS(m(3, 4));
  }
}
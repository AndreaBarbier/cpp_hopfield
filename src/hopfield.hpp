#ifndef HOPFIELD_HPP
#define HOPFIELD_HPP

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <functional>
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
int sgn(T val);

// ============================================================
// GET THE DATA DIR
// ============================================================

/// @brief Returns the path to the project's data directory, creating it if it
/// does not exist.
/// @return A std::filesystem::path pointing to the "data" directory inside
/// PROJECT_ROOT.
/// @throws std::runtime_error If the path exists but is not a directory.
std::filesystem::path getDataDir();

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
  Matrix(size_t r, size_t c);

  /// @brief Constructs a matrix of the given dimensions from an existing data
  /// vector.
  /// @param r Number of rows. Must be different from 0.
  /// @param c Number of columns. Must be different from 0.
  /// @param data Flat vector of doubles in row-major order. Must have size r *
  /// c.
  /// @throws std::runtime_error If r or c are 0.
  /// @throws std::runtime_error If data.size() != r * c.
  Matrix(size_t r, size_t c, std::vector<double> const& data);

  /// @brief Returns a reference to the element at row i, column j.
  /// @param i Row index (0-based).
  /// @param j Column index (0-based).
  /// @throws std::runtime_error If i or j are out of range.
  double& operator()(size_t i, size_t j);

  /// @brief Returns a const reference to the element at row i, column j.
  /// @param i Row index (0-based).
  /// @param j Column index (0-based).
  /// @throws std::runtime_error If i or j are out of range.
  double const& operator()(size_t i, size_t j) const;
};

// ============================================================
// NEURAL NETWORK
// ============================================================

/// @brief Hopfield neural network class
class Network {
  std::vector<sf::Image> trainImgs_{};
  sf::Vector2u validSize_{0u, 0u};
  std::filesystem::path wMatrixFilePath_{getDataDir() / "matrix.txt"};

 public:
  Network() = default;

  /// @brief Contructs an empty hopfield neural network, only with the path of
  /// the matrix storege file.
  /// @param[in] wMatrixFilePath Matrix storege file.
  Network(std::string const& wMatrixFilePath);

  /// @brief Adds an image into Networks's train images vector
  /// @param[in] img Image to add
  void addImage(sf::Image const& img);

  /// @brief Adds multiple images into Networks's train images vector
  /// @param[in] imgs Vector of images to add
  void addImages(std::vector<sf::Image> const& imgs);

  /// @brief Adds multiple images into Networks's train images vector
  /// @throws std::runtime_error if it is not possible to open the file
  /// @param[in] file File in which are written the paths of the images to add
  void addImages(std::filesystem::path const& imgsPath = getDataDir() /
                                                         "train");

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
  Matrix train();

  /// @brief Calculates the total energy of the neuronal network.
  ///
  /// This function evaluates the energy state of a network (e.g., Hopfield
  /// network) using the state vector of the neurons and the weight matrix.
  ///
  /// @param neuron A constant reference to a vector of integers representing
  /// the neuron states.
  /// @param w A constant reference to the weight matrix of the network.
  /// @return The calculated double-precision energy value of the network.
  /// @note Uses assertions to ensure the network is not empty and that the
  /// weight matrix
  ///       dimensions match the size of the neuron vector.
  double energy(std::vector<int> const& neuron, Matrix const& w);

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
          callback = [](PatternInt const&, size_t) {}) const;
};
}  // namespace hopfield

#endif
#ifndef HOPFIELD_HPP
#define HOPFIELD_HPP

#include <SFML/Graphics.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "image_processor.hpp"

/// @file hopfield.hpp
/// @brief Core classes and utilities for managing datasets and Hopfield networks

namespace hopfield {

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// @brief Checks whether a filename starts with a timestamp-like prefix
/// @param[in] name Filename to inspect
/// @param[in] prefixLen Length of the prefix to be skipped by nameOf()
/// @return True if the expected timestamp separators and digits are present, false otherwise
bool hasPrefix(std::string const& name, std::size_t prefixLen);

/// @brief Checks whether a path has a supported image extension
/// @param[in] file Path to inspect
/// @return True for PNG, JPG, or JPEG files (case-insensitive), false otherwise
bool hasImageExtension(std::filesystem::path const& file);

/// @brief Checks whether a path identifies a regular image file
/// @param[in] file Path to inspect
/// @return True if the path is a regular file with a supported extension, false otherwise
bool isImageFile(std::filesystem::path const& file);

/// @brief Generates the timestamp prefix used for dataset filenames
/// @return A UTC timestamp formatted as YYYYMMDD-HHMMSS_
std::string timestamp();

/// @brief Removes a leading filename prefix
/// @param[in] file Path whose filename is used
/// @param[in] prefixLen Number of characters to remove from the filename
/// @return The filename without its leading prefix
std::string nameOf(std::filesystem::path const& file, std::size_t prefixLen);

/// @brief Computes the number of neurons represented by an image size
/// @param[in] size Width and height of the image/network
/// @throws std::runtime_error if either dimension is zero
/// @return The product of width and height
std::size_t neuronCount(sf::Vector2u const& size);

// ============================================================
// WEIGHT MATRIX
// ============================================================

/// @brief 2D matrix with row-major storage, used to represent the network's weight matrix
/// @details Stores elements as a flat vector of doubles, accessible via (i, j) indexing.
///          The element at row i and column j is stored at matrix[i * cols + j].
struct Matrix {
  std::size_t rows;
  std::size_t cols;
  std::vector<double> matrix;

  /// @brief Constructs a zero-initialized matrix of the given dimensions
  /// @param[in] r Number of rows; must be different from 0
  /// @param[in] c Number of columns; must be different from 0
  /// @throws std::runtime_error if r or c are 0
  Matrix(std::size_t r, std::size_t c);

  /// @brief Constructs a matrix of the given dimensions from an existing data vector
  /// @param[in] r Number of rows; must be different from 0
  /// @param[in] c Number of columns; must be different from 0
  /// @param[in] data Flat vector of doubles in row-major order; must have size r * c
  /// @throws std::runtime_error if r or c are 0, or if data.size() != r * c
  Matrix(std::size_t r, std::size_t c, std::vector<double> data);

  /// @brief Returns a reference to the element at row i, column j
  /// @param[in] i Row index (0-based)
  /// @param[in] j Column index (0-based)
  /// @throws std::runtime_error if i or j are out of range
  /// @return Mutable reference to the requested element
  double& operator()(std::size_t i, std::size_t j);

  /// @brief Returns a const reference to the element at row i, column j
  /// @param[in] i Row index (0-based)
  /// @param[in] j Column index (0-based)
  /// @throws std::runtime_error if i or j are out of range
  /// @return Constant reference to the requested element
  double const& operator()(std::size_t i, std::size_t j) const;
};

/// @brief Computes the Hopfield network energy of a binary pattern
/// @pre neuron.data must not be empty and its size must match w.rows and w.cols
/// @param[in] neuron Pattern whose energy is evaluated
/// @param[in] w Square network weight matrix
/// @return The energy, calculated as -0.5 * sum(w(i, j) * neuron[i] * neuron[j])
double energy(PatternInt const& neuron, Matrix const& w);

// ============================================================
// DATASET CLASS
// ============================================================

/// @brief Manages the image patterns stored in a dataset directory
/// @details Dataset images are identified by a timestamp prefix followed by their original filename.
///          Images are resized to a common size when patterns() or size() is called.
class Dataset {
  std::filesystem::path dir_;
  static constexpr std::size_t kPrefixLen_{16};

  /// @brief Returns the valid, timestamp-prefixed image files in the dataset
  /// @return Vector of valid image file paths
  std::vector<std::filesystem::path> getFiles() const;

 public:
  /// @brief Opens an existing dataset directory
  /// @param[in] dir Directory containing the image patterns
  /// @throws std::runtime_error if the path does not exist or is not a directory
  explicit Dataset(std::filesystem::path const& dir);

  /// @brief Gets the sorted names of all patterns in the dataset
  /// @return Vector of pattern filenames without their timestamp prefixes
  std::vector<std::string> getNames() const;

  /// @brief Adds an image to the dataset
  /// @param[in] img Image file to validate, copy, and convert to a binary pattern
  /// @throws std::runtime_error if the file, format, image validity, or duplicate check fails
  /// @return The image converted to a PatternInt
  PatternInt add(std::filesystem::path const& img);

  /// @brief Removes a pattern from the dataset
  /// @param[in] name Pattern filename without its timestamp prefix
  /// @throws std::runtime_error if the pattern is not found, cannot be loaded/removed, or is invalid
  /// @return The removed image converted to a PatternInt
  PatternInt remove(std::string const& name);

  /// @brief Gets the common resized dimensions of the dataset images
  /// @return Image dimensions, or {0, 0} if the dataset is empty
  sf::Vector2u size() const;

  /// @brief Loads and converts all dataset images to binary patterns
  /// @return Vector of resized images represented as PatternInt
  std::vector<PatternInt> patterns() const;
};

// ============================================================
// NETWORK CLASS
// ============================================================

/// @brief Hopfield network with a symmetric weight matrix
/// @details Stores the dimensions of its patterns and their associated weights.
///          Supports Hebbian learning, binary persistence, and iterative pattern recall.
class Network {
 public:
  /// @brief Callback invoked during recall with the current pattern, sweep ID, and its energy
  using RecallCallback =
      std::function<void(PatternInt const& newPattern, std::size_t id,
                         double energy)>;

 private:
  sf::Vector2u size_;
  Matrix w_;
  static constexpr char kMagic[4]{'H', 'O', 'P', 'F'};
  static constexpr std::uintmax_t kHeaderBytes{
      4 + 2 * sizeof(std::uint32_t)};

 public:
  /// @brief Constructs a network with an existing weight matrix
  /// @param[in] size Width and height of each pattern
  /// @param[in] w Square weight matrix with one row and column per neuron
  /// @throws std::runtime_error if the matrix is not square or its dimensions do not match size
  explicit Network(sf::Vector2u const& size, Matrix const& w);

  /// @brief Constructs an untrained network for patterns of the given size
  /// @param[in] size Width and height of each pattern; neither dimension may be zero
  /// @throws std::runtime_error if either dimension is zero
  explicit Network(sf::Vector2u const& size);

  /// @brief Gets the dimensions of patterns handled by the network
  /// @return Network width and height
  sf::Vector2u size() const;

  /// @brief Loads a network and its weights from a binary file
  /// @param[in] file Binary network file to read
  /// @throws std::runtime_error if the file cannot be opened, has an invalid header/size, or cannot be read
  /// @return The Network reconstructed from the file
  static Network load(std::filesystem::path const& file);

  /// @brief Saves the network and its weights to a binary file
  /// @param[in] file Destination file path
  /// @throws std::runtime_error if the network is not trained or writing fails
  void save(std::filesystem::path const& file) const;

  /// @brief Applies the Hebbian learning rule to one pattern
  /// @pre Network weight matrix must be initialized and match pattern dimensions
  /// @param[in] p Pattern to learn
  /// @param[in] sign Scale/sign applied to the Hebbian contribution (default +1.0)
  void learn(PatternInt const& p);

  /// @brief Trains the network using a collection of patterns
  /// @param[in] patterns Vector of patterns used to initialize and train the weight matrix
  /// @throws std::runtime_error if the pattern collection is empty
  void train(std::vector<PatternInt> const& patterns);

  /// @brief Checks whether the network has non-zero learned weights
  /// @return True if at least one weight is non-zero, false otherwise
  bool isTrained() const;

  /// @brief Recalls a stored pattern from an input pattern
  /// @details Neurons are updated sequentially until the state stops changing or maxSweep is reached.
  ///          The callback is invoked initially and after each sweep that changes the pattern.
  /// @param[in] inPattern Initial pattern to restore
  /// @param[in] maxSweep Maximum number of sweeps (default 1000)
  /// @param[in] callback Function called with the current pattern, sweep index, and energy
  /// @throws std::runtime_error if the recall does not converge within maxSweep sweeps
  /// @return The recalled PatternInt
  PatternInt recall(
      PatternInt const& inPattern, std::size_t maxSweep = 1000,
      RecallCallback const& callback =
          [](PatternInt const&, std::size_t, double) {}) const;
};

}  // namespace hopfield

#endif
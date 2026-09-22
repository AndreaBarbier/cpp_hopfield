#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <filesystem>
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
// GET THE DATA DIR
// ============================================================

/// @brief Returns the path to the project's data directory, creating it if it
/// does not exist.
/// @return A std::filesystem::path pointing to the "data" directory inside
/// PROJECT_ROOT.
/// @throws std::runtime_error If the path exists but is not a directory.
std::filesystem::path getDataDir() {
  auto data_dir{std::filesystem::path(PROJECT_ROOT) / "data"};
  if (!std::filesystem::exists(data_dir)) {
    std::filesystem::create_directories(data_dir);
  } else if (!std::filesystem::is_directory(data_dir)) {
    throw std::runtime_error("Error: " + data_dir.string() +
                             "exists, but it is not a directory.");
  }
  return data_dir;
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
// NEURAL NETWORK
// ============================================================

/// @brief Hopfield neural network class
class Network {
  std::vector<sf::Image> trainImgs_{};
  sf::Vector2u validSize_{0u, 0u};
  std::filesystem::path wMatrixFilePath_{getDataDir() / "matrix.txt"};

 public:
 Network() {}

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
  void addImages(std::filesystem::path const& imgsPath = "data/train/") {
    assert(std::filesystem::exists(imgsPath));
    for (auto const& file : std::filesystem::directory_iterator{imgsPath}) {
      sf::Image img{};
      img.loadFromFile(file.path());
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
      throw std::runtime_error("Error: impossible to open the file " +
                               wMatrixFilePath_.string());
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
  double energy(std::vector<int> const& neuron, Matrix const& w) {
    assert(!neuron.empty());
    assert(neuron.size() == w.cols && neuron.size() == w.rows);

    double energy{0.0};
    const size_t size = neuron.size();

    for (size_t i{0}; i != size; ++i) {
      for (size_t j{0}; j != size; ++j) {
        energy += -0.5 * w(i, j) * neuron[i] * neuron[j];
      }
    }

    return energy;
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
                         wMatrixFilePath_.string());
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
      double sum{0.0};
      oldPattern.data = newPattern.data;
      for (size_t j{0}; j != N; ++j) {
        newPattern.data[j] = 0;
        for (size_t i{0}; i != N; ++i) {
          sum += w(j, i) * oldPattern.data[i];
        }
        newPattern.data[j] = sgn(sum);
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

TEST_CASE("HOPFIELD NETWORK") {
  SUBCASE("Network::train") {
    hopfield::Network net{"data/test_matrix.txt"};

    SUBCASE("Empty dataset throws") {
      CHECK_THROWS_AS(net.train(), std::runtime_error);
    }

    SUBCASE("Weight matrix is symmetric with zero diagonal") {
      sf::Image img1{};
      img1.create(2, 2, sf::Color::White);
      sf::Image img2{};
      img2.create(2, 2, sf::Color::Black);
      net.addImage(img1);
      net.addImage(img2);
      auto w = net.train();
      for (size_t i = 0; i < w.rows; ++i) {
        CHECK(w(i, i) == doctest::Approx(0.0));
        for (size_t j = 0; j < w.cols; ++j) {
          CHECK(w(i, j) == doctest::Approx(w(j, i)));
        }
      }
    }
  }

  SUBCASE("Network::energy") {
    hopfield::Network net{"data/test_matrix.txt"};
    hopfield::Matrix w{2, 2, {0.0, 1.0, 1.0, 0.0}};
    std::vector<int> pattern{1, -1};

    SUBCASE("Known value") {
      // E = -0.5*(w01*x0*x1 + w10*x1*x0) = -0.5*(1*1*-1 + 1*-1*1) = 1.0
      CHECK(net.energy(pattern, w) == doctest::Approx(1.0));
    }

    SUBCASE("Energy is invariant under global sign flip") {
      std::vector<int> flipped{-1, 1};
      CHECK(net.energy(pattern, w) == doctest::Approx(net.energy(flipped, w)));
    }
  }

  // ============================================================
  // NEW ADVANCED TESTS & EDGE CASES
  // ============================================================

  SUBCASE("Network::train with multiple small images and matrix inspection") {
    hopfield::Network net{"data/multi_small_matrix.txt"};

    // Creiamo 3 piccole immagini 2x2 con configurazioni diverse
    sf::Image img1{};
    img1.create(2, 2, sf::Color::White);

    sf::Image img2{};
    img2.create(2, 2, sf::Color::Black);

    sf::Image img3{};
    img3.create(
        2, 2,
        sf::Color::White);  // Duplicato o simile per testare il cumulo dei pesi

    net.addImage(img1);
    net.addImage(img2);
    net.addImage(img3);

    auto w = net.train();

    // La dimensione della matrice deve essere N x N dove N = 2 * 2 = 4
    CHECK(w.rows == 4);
    CHECK(w.cols == 4);

    // Proprietà fondamentali della matrice di Hopfield:
    // 1. Diagonale principale rigorosamente a zero
    // 2. Simmetria (w(i,j) == w(j,i))
    for (size_t i{0}; i < w.rows; ++i) {
      CHECK(w(i, i) == doctest::Approx(0.0));
      for (size_t j{0}; j < w.cols; ++j) {
        CHECK(w(i, j) == doctest::Approx(w(j, i)));
        // Controllo che i pesi non contengano valori non validi (NaN o Inf)
        CHECK_FALSE(std::isnan(w(i, j)));
        CHECK_FALSE(std::isinf(w(i, j)));
      }
    }
  }

  SUBCASE("Network::train with mixed image sizes (automatic resizing)") {
    hopfield::Network net{"data/mixed_sizes_matrix.txt"};

    sf::Image imgSmall{};
    imgSmall.create(2, 2, sf::Color::White);

    sf::Image imgLarge{};
    imgLarge.create(4, 4, sf::Color::Black);  // Dimensione diversa

    net.addImage(imgSmall);
    net.addImage(imgLarge);

    // Il training DEVE completare con successo (grazie a resizeImages)
    CHECK_NOTHROW(net.train());
  }
}
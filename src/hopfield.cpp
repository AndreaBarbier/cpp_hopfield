#include "hopfield.hpp"

#include <cmath>

namespace hopfield {

std::filesystem::path getDataDir() {
  auto data_dir{std::filesystem::path(PROJECT_ROOT) / "data"};
  if (!std::filesystem::exists(data_dir)) {
    std::filesystem::create_directories(data_dir);
  } else if (!std::filesystem::is_directory(data_dir)) {
    throw std::runtime_error("Error: " + data_dir.string() +
                             " exists, but it is not a directory.");
  }
  return data_dir;
}

Matrix::Matrix(size_t r, size_t c) : rows{r}, cols{c}, matrix(r * c, 0.0) {
  if (r == 0 || c == 0) {
    throw std::runtime_error("Error: invalid matrix size inserted.");
  }
}

Matrix::Matrix(size_t r, size_t c, std::vector<double> const& data)
    : rows{r}, cols{c}, matrix{data} {
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("Error: invalid matrix size inserted.");
  }
  if (data.size() != rows * cols) {
    throw std::runtime_error("Error: invalid data lenght inserted");
  }
}

double& Matrix::operator()(size_t i, size_t j) {
  assert(rows > 0 && cols > 0);
  if (i >= rows || j >= cols) {
    throw std::runtime_error("Error: index out of range.");
  }
  return matrix[i * cols + j];
}

double const& Matrix::operator()(size_t i, size_t j) const {
  assert(rows > 0 && cols > 0);
  if (i >= rows || j >= cols) {
    throw std::runtime_error("Error: index out of range.");
  }
  return matrix[i * cols + j];
}

Network::Network(std::string const& wMatrixFilePath)
    : wMatrixFilePath_(wMatrixFilePath) {}

void Network::addImage(sf::Image const& img) {
  auto size{img.getSize()};
  assert(size.x > 0u && size.y > 0u);
  trainImgs_.push_back(img);
}

void Network::addImages(std::vector<sf::Image> const& imgs) {
  for (auto img : imgs) {
    addImage(img);
  }
}

void Network::addImages(std::filesystem::path const& imgsPath) {
  if (!std::filesystem::exists(imgsPath) ||
      !std::filesystem::is_directory(imgsPath)) {
    throw std::runtime_error("Error: " + imgsPath.string() +
                             " does not exist or is not a directory");
  }
  for (auto const& file : std::filesystem::directory_iterator{imgsPath}) {
    sf::Image img{};
    img.loadFromFile(file.path());
    addImage(img);
  }
}

Matrix Network::train() {
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
        for (auto const& pattern : binaryPatterns) {
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
      file << w(j, i) << ' ';
    }
    file << '\n';
  }
  file.close();
  return w;
}

double Network::energy(PatternInt const& neuron, Matrix const& w) const {
  assert(!neuron.data.empty());
  assert(neuron.data.size() == w.cols && neuron.data.size() == w.rows);

  double energy{0.0};
  const size_t size = neuron.data.size();

  for (size_t i{0}; i != size; ++i) {
    for (size_t j{0}; j != size; ++j) {
      energy += -0.5 * w(i, j) * neuron.data[i] * neuron.data[j];
    }
  }

  return energy;
}

PatternInt Network::recall(
    PatternInt const& inPattern,
    std::function<void(size_t id, PatternInt const& newPattern,
                       double energy)> const& callback) const {
  if (validSize_ == sf::Vector2u{0u, 0u}) {
    throw std::runtime_error(
        "Error: the network has not been trained yet. "
        "Please call train() before recall().\n");
  }
  if (inPattern.size != validSize_ ||
      inPattern.data.size() != static_cast<size_t>(validSize_.x) * validSize_.y) {
    throw std::runtime_error(
        "Error: the input pattern size does not match the network size.\n");
  }

  size_t const N{static_cast<size_t>(validSize_.x) * validSize_.y};
  Matrix w{N, N};
  std::ifstream file{wMatrixFilePath_};
  if (!file.is_open()) {
    throw std::runtime_error("Error: impossible to open the file " +
                             wMatrixFilePath_.string());
  }
  for (size_t j{0}; j != N; ++j) {
    for (size_t i{0}; i != N; ++i) {
      if (!(file >> w(j, i))) {
        throw std::runtime_error("Error: the weight matrix file is corrupted "
                                 "or too short.\n");
      }
    }
  }

  PatternInt newPattern{inPattern};

  size_t id{0};
  callback(id++, newPattern, energy(newPattern, w));  // stato iniziale

  bool changed{true};
  while (changed) {
    changed = false;
    for (size_t j{0}; j != N; ++j) {
      double sum{0.0};
      for (size_t i{0}; i != N; ++i) {
        sum += w(j, i) * newPattern.data[i];
      }
      int const s{sum > 0.0 ? 1 : (sum < 0.0 ? -1 : newPattern.data[j])};
      if (s != newPattern.data[j]) {
        newPattern.data[j] = s;
        changed = true;
      }
    }
    if (changed) {
      callback(id++, newPattern, energy(newPattern, w));
    }
  }

  return newPattern;
}
}  // namespace hopfield
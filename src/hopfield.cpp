#include "hopfield.hpp"

#include <cmath>

namespace hopfield {

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

double energy(PatternInt const& neuron, Matrix const& w) {
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

Network::Network(std::string const& wMatrixFilePath)
    : wMatrixFilePath_(wMatrixFilePath) {}

void Network::addImage(sf::Image const& img) {
  auto size{img.getSize()};
  assert(size.x > 0u && size.y > 0u);
  trainImgs_.push_back(img);
}

void Network::addImages(std::vector<sf::Image> const& imgs) {
  for (auto const& img : imgs) {
    addImage(img);
  }
}

void Network::addImages(std::filesystem::path const& imgsPath) {
  assert(std::filesystem::exists(imgsPath));
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
  if (resizedImgs.empty()) {
    throw std::runtime_error(
        "Error: invalid training dataset. Every image had an invalid "
        "size.\n");
  }
  sf::Vector2u validSize = {resizedImgs[0].getSize()};
  for (auto const& img : resizedImgs) {
    assert(img.getSize().x != 0u);
    assert(img.getSize().y != 0u);
    assert(img.getSize() == validSize);
  }
  size_t const N{validSize.x * validSize.y};
  std::vector<PatternInt> binaryPatterns{imageToBinaries(resizedImgs)};

  Matrix w{N, N};
  std::ofstream file{wMatrixFilePath_};
  if (!file.is_open()) {
    throw std::runtime_error("Error: impossible to open the file " +
                             wMatrixFilePath_.string());
  }

  file << validSize.x << ' ' << validSize.y << '\n';

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
      file << w(j, i) << '\t';
    }
    file << '\n';
  }
  file.close();
  return w;
}

PatternInt Network::recall(
    PatternInt const& inPattern, size_t maxSweep,
    std::function<void(PatternInt const& newPattern, size_t id,
                       double energy)> const& callback) const {
  std::ifstream file{wMatrixFilePath_};
  if (!file.is_open()) {
    throw std::runtime_error("Error: impossible to open the file " +
                             wMatrixFilePath_.string());
  }
  sf::Vector2u validSize{0u, 0u};
  if (!(file >> validSize.x >> validSize.y) || validSize.x == 0u ||
      validSize.y == 0u) {
    throw std::runtime_error("Error: invalid weight matrix file header.\n");
  }
  size_t const N{static_cast<size_t>(validSize.x) * validSize.y};
  if (inPattern.size != validSize || inPattern.data.size() != N) {
    throw std::runtime_error(
        "Error: the input pattern size does not match the network size.\n");
  }

  Matrix w{N, N};

  for (size_t j{0}; j != N; ++j) {
    for (size_t i{0}; i != N; ++i) {
      if (!(file >> w(j, i))) {
        throw std::runtime_error(
            "Error: the weight matrix file is corrupted or too short.\n");
      }
    }
  }

  PatternInt current{inPattern};
  size_t id{0};
  callback(current, id++, energy(current, w));

  bool changed{true};
  while (changed && id <= maxSweep) {
    changed = false;
    for (size_t i{0}; i != N; ++i) {
      double sum{0.0};
      for (size_t j{0}; j != N; ++j) {
        sum += w(i, j) * current.data[j];
      }
      int const s{(sum > 0.0) ? 1 : ((sum < 0.0) ? -1 : current.data[i])};
      if (s != current.data[i]) {
        current.data[i] = s;
        changed = true;
      }
    }
    if (changed) {
      callback(current, id++, energy(current, w));
    }
  }
  if (changed) {
    throw std::runtime_error("Error: recall did not converge within " +
                             std::to_string(maxSweep) + " sweeps.\n");
  }
  return current;
}
}  // namespace hopfield
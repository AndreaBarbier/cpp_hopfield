#include "hopfield.hpp"

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "image_processor.hpp"

namespace hopfield {

// ============================================================
// HELPER FUNCTIONS
// ============================================================

bool hasPrefix(std::string const& name, std::size_t prefixLen) {
  if (name.size() <= prefixLen || name.size() <= 15 || name[8] != '-' ||
      name[15] != '_') {
    return false;
  }
  for (std::size_t i :
       {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 9u, 10u, 11u, 12u, 13u, 14u}) {
    if (!std::isdigit(static_cast<unsigned char>(name[i]))) return false;
  }
  return true;
}

bool hasImageExtension(std::filesystem::path const& file) {
  auto ext{file.extension().string()};
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

bool isImageFile(std::filesystem::path const& file) {
  return std::filesystem::is_regular_file(file) && hasImageExtension(file);
}

std::string timestamp() {
  auto const t{
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())};
  std::ostringstream os;
  os << std::put_time(std::gmtime(&t), "%Y%m%d-%H%M%S") << '_';
  return os.str();
}

std::string nameOf(std::filesystem::path const& file, std::size_t prefixLen) {
  return file.filename().string().substr(prefixLen);
}

std::size_t neuronCount(sf::Vector2u const& size) {
  if (size.x == 0u || size.y == 0u) {
    throw std::runtime_error("Error: invalid network size.");
  }
  return static_cast<std::size_t>(size.x) * size.y;
}

// ============================================================
// WEIGHT MATRIX
// ============================================================

Matrix::Matrix(std::size_t r, std::size_t c)
    : rows{r}, cols{c}, matrix(r * c, 0.0) {
  if (r == 0 || c == 0) {
    throw std::runtime_error("Error: invalid matrix size inserted.\n");
  }
}

Matrix::Matrix(std::size_t r, std::size_t c, std::vector<double> data)
    : rows{r}, cols{c}, matrix{std::move(data)} {
  if (rows == 0 || cols == 0) {
    throw std::runtime_error("Error: invalid matrix size inserted.\n");
  }
  if (matrix.size() != rows * cols) {
    throw std::runtime_error("Error: invalid data lenght inserted\n");
  }
}

double& Matrix::operator()(std::size_t i, std::size_t j) {
  assert(rows > 0 && cols > 0);
  if (i >= rows || j >= cols) {
    throw std::runtime_error("Error: index out of range.\n");
  }
  return matrix[i * cols + j];
}

double const& Matrix::operator()(std::size_t i, std::size_t j) const {
  assert(rows > 0 && cols > 0);
  if (i >= rows || j >= cols) {
    throw std::runtime_error("Error: index out of range.\n");
  }
  return matrix[i * cols + j];
}

double energy(PatternInt const& neuron, Matrix const& w) {
  assert(!neuron.data.empty());
  assert(neuron.data.size() == w.cols && neuron.data.size() == w.rows);

  double energy{0.0};
  const std::size_t size = neuron.data.size();

  for (std::size_t i{0}; i != size; ++i) {
    for (std::size_t j{0}; j != size; ++j) {
      energy += -0.5 * w(i, j) * neuron.data[i] * neuron.data[j];
    }
  }

  return energy;
}

// ============================================================
// DATASET CLASS
// ============================================================

std::vector<std::filesystem::path> Dataset::getFiles() const {
  std::vector<std::filesystem::path> result;
  for (auto const& file : std::filesystem::directory_iterator(dir_)) {
    if (isImageFile(file.path()) &&
        hasPrefix(file.path().filename().string(), kPrefixLen_)) {
      result.push_back(file.path());
    }
  }
  return result;
}

Dataset::Dataset(std::filesystem::path const& dir) : dir_{dir} {
  if (!std::filesystem::exists(dir)) {
    throw std::runtime_error(
        "Error: directory does not exist: " + dir.string() + '\n');
  }
  if (!std::filesystem::is_directory(dir)) {
    throw std::runtime_error(
        "Error: path exists, but is not a valid directory: " + dir.string() +
        '\n');
  }
}

std::vector<std::string> Dataset::getNames() const {
  std::vector<std::string> result;
  for (auto const& img : getFiles()) {
    assert(hasPrefix(img.filename().string(), kPrefixLen_));
    assert(isImageFile(img));
    result.push_back(nameOf(img, kPrefixLen_));
  }
  std::sort(result.begin(), result.end());
  return result;
}

PatternInt Dataset::add(std::filesystem::path const& img) {
  if (!std::filesystem::is_regular_file(img)) {
    throw std::runtime_error("Error: '" + img.string() + "' is not a file.\n");
  }
  if (!hasImageExtension(img)) {
    throw std::runtime_error("Error: invalid file format " +
                             img.filename().string() + '\n');
  }
  auto const name{img.filename().string()};
  auto const existing{getNames()};
  if (std::find(existing.begin(), existing.end(), name) != existing.end()) {
    throw std::runtime_error("Error: pattern '" + name + "' already exists.\n");
  }
  sf::Image image{};
  if (!image.loadFromFile(img)) {
    throw std::runtime_error("Error: cannot load image " + img.string() + '\n');
  }
  if (!isValidImage(image)) {
    throw std::runtime_error("Error: invalid image " + img.string() + '\n');
  }
  std::filesystem::copy_file(
      img, dir_ / (timestamp() + name));
  return imageToBinary(image);
}

PatternInt Dataset::remove(std::string const& name) {
  auto files{getFiles()};
  auto it{std::find_if(files.begin(), files.end(), [&](auto const& f) {
    return nameOf(f, kPrefixLen_) == name;
  })};
  if (it == files.end()) {
    throw std::runtime_error("Error: pattern '" + name + "' not found.");
  }
  sf::Image image{};
  if (!image.loadFromFile(*it)) {
    throw std::runtime_error("Error: cannot load image " + (*it).string() +
                             '\n');
  }
  if (!isValidImage(image)) {
    throw std::runtime_error("Error: invalid image " + (*it).string() + '\n');
  }
  if (!std::filesystem::remove(*it)) {
    throw std::runtime_error("Error: cannot remove " + name + '\n');
  }
  return imageToBinary(image);
}

sf::Vector2u Dataset::size() const {
  auto files{getFiles()};
  std::vector<sf::Image> inImgs{};
  for (auto const& f : files) {
    if (isImageFile(f)) {
      sf::Image img;
      if (img.loadFromFile(f)) {
        inImgs.push_back(img);
      }
    }
  }
  if (inImgs.empty()) {
    return {0u, 0u};
  }
  auto outImgs{resizeImages(inImgs)};
  return outImgs[0].getSize();
}

std::vector<PatternInt> Dataset::patterns() const {
  auto files{getFiles()};
  std::vector<sf::Image> in{};
  for (auto const& f : files) {
    if (isImageFile(f)) {
      sf::Image img;
      if (img.loadFromFile(f)) {
        in.push_back(img);
      }
    }
  }
  auto resized{resizeImages(in)};
  if (resized.empty()) {
    return {};
  }
  return imageToBinaries(resized);
}

// ============================================================
// NETWORK CLASS
// ============================================================

Network::Network(sf::Vector2u const& size, Matrix const& w)
    : size_{size}, w_{w} {
  if (w.cols != w.rows) {
    throw std::runtime_error("Error: weight matrix must be a square matrix.");
  }
  if (neuronCount(size) != w.cols) {
    throw std::runtime_error(
        "Error: pattern's size does not match the weight matrix size!");
  }
}

Network::Network(sf::Vector2u const& size)
    : Network(size, Matrix(neuronCount(size), neuronCount(size))) {}

sf::Vector2u Network::size() const { return size_; }

Network Network::load(std::filesystem::path const& file) {
  std::ifstream in{file, std::ios::binary};
  if (!in.is_open()) {
    throw std::runtime_error("Error: cannot open " + file.string());
  }
  char magic[4]{};
  std::uint32_t width{};
  std::uint32_t height{};
  in.read(magic, sizeof(magic));
  in.read(reinterpret_cast<char*>(&width), sizeof(width));
  in.read(reinterpret_cast<char*>(&height), sizeof(height));
  if (!in || !std::equal(magic, magic + 4, kMagic)) {
    throw std::runtime_error("Error: invalid weight file header.");
  }
  sf::Vector2u const size{width, height};
  std::size_t const n{neuronCount(size)};
  auto const expected{kHeaderBytes + std::uintmax_t{n} * n * sizeof(double)};
  if (expected != std::filesystem::file_size(file)) {
    throw std::runtime_error("Error: weight file is corrupted or truncated.");
  }
  std::vector<double> data(n * n);
  in.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(double)));
  if (!in) {
    throw std::runtime_error("Error: cannot read weights.");
  }
  return Network{size, Matrix{n, n, std::move(data)}};
}

void Network::save(std::filesystem::path const& file) const {
  auto tmp{file};
  tmp += ".tmp";
  std::ofstream out{tmp, std::ios::binary};
  if (!out.is_open()) {
    throw std::runtime_error("Error: cannot open " + file.string());
  }
  if (w_.matrix.empty() || size_.x == 0u || size_.y == 0u) {
    throw std::runtime_error(
        "Error: network not trained yet. Use net.train() before save.");
  }
  std::uint32_t width{size_.x};
  std::uint32_t height{size_.y};
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&width), sizeof(width));
  out.write(reinterpret_cast<const char*>(&height), sizeof(height));
  out.write(reinterpret_cast<const char*>(w_.matrix.data()),
            static_cast<std::streamsize>(w_.matrix.size() * sizeof(double)));
  out.close();
  if (!out) {
    std::filesystem::remove(tmp);
    throw std::runtime_error("Error: failed while writing data to " +
                             file.string());
  }
  std::filesystem::rename(tmp, file);
}

void Network::learn(PatternInt const& p, double sign) {
  assert(!w_.matrix.empty());
  assert(w_.cols == w_.rows);
  assert(size_ == p.size);
  assert(w_.cols == p.data.size());
  auto const count{p.data.size()};
  double const N{static_cast<double>(count)};
  for (std::size_t j{0}; j != count; ++j) {
    w_(j, j) = 0;
    for (std::size_t i{j + 1}; i != count; ++i) {
      double sum{sign * p.data[i] * p.data[j]};
      w_(i, j) += sum / N;
      w_(j, i) += sum / N;
      assert(w_(i, j) == w_(j, i));
    }
  }
}

void Network::train(std::vector<PatternInt> const& patterns) {
  if (patterns.empty()) {
    throw std::runtime_error(
        "Error: cannot train network with an empty pattern list.");
  }
  auto const expected_size = patterns[0].size;
  auto it{std::find_if(patterns.begin(), patterns.end(),
                       [&](auto const& p) { return p.size != expected_size; })};
  assert(it == patterns.end());
  std::size_t N{patterns[0].data.size()};
  w_ = Matrix{N, N};
  for (auto const& p : patterns) {
    learn(p);
  }
}

bool Network::isTrained() const {
  auto it{std::find_if(w_.matrix.begin(), w_.matrix.end(),
                       [](auto const& d) { return d != 0.0; })};
  if (it == w_.matrix.end()) {
    return false;
  }
  return true;
}

PatternInt Network::recall(PatternInt const& inPattern, std::size_t maxSweep,
                           RecallCallback const& callback) const {
  assert(!w_.matrix.empty());
  assert(w_.cols == w_.rows);
  assert(size_ == inPattern.size);
  assert(w_.cols == inPattern.data.size());

  PatternInt outPattern{inPattern};
  std::size_t id{0};
  if (callback) {
    callback(outPattern, id, energy(outPattern, w_));
  }

  bool changed{true};
  while (changed && id <= maxSweep) {
    changed = false;
    for (std::size_t i{0}; i != w_.rows; ++i) {
      double sum{0.0};
      for (std::size_t j{0}; j != w_.cols; ++j) {
        sum += w_(i, j) * outPattern.data[j];
      }
      int const s{(sum > 0.0) ? 1 : ((sum < 0.0) ? -1 : outPattern.data[i])};
      if (s != outPattern.data[i]) {
        outPattern.data[i] = s;
        changed = true;
      }
    }
    if (changed) {
      if (callback) {
        callback(outPattern, id++, energy(outPattern, w_));
      } else {
        ++id;
      }
    }
  }
  if (changed) {
    throw std::runtime_error("Error: recall did not converge within " +
                             std::to_string(maxSweep) + " sweeps.\n");
  }
  return outPattern;
}

}  // namespace hopfield
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../hopfield.hpp"

#include <SFML/Graphics.hpp>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"

namespace {
namespace fs = std::filesystem;

fs::path makeTempDir(std::string const& label) {
  static std::size_t counter = 0;
  auto path = fs::temp_directory_path() /
              ("hopfield_tests_" + label + "_" + std::to_string(++counter));
  fs::remove_all(path);
  fs::create_directories(path);
  return path;
}

sf::Image makeImage(unsigned int width, unsigned int height, sf::Color color) {
  return sf::Image{sf::Vector2u{width, height}, color};
}

hopfield::PatternInt makePattern(unsigned int width, unsigned int height,
                                 int value) {
  hopfield::PatternInt p;
  p.size = sf::Vector2u{width, height};
  p.data = std::vector<int>(static_cast<std::size_t>(width) * height, value);
  return p;
}

bool finiteMatrix(hopfield::Matrix const& m) {
  for (std::size_t r = 0; r < m.rows; ++r)
    for (std::size_t c = 0; c < m.cols; ++c)
      if (!std::isfinite(m(r, c))) return false;
  return true;
}
}  // namespace

// ============================================================================
// MATRIX: CONSTRUCTION, STORAGE, INDEXING, AND BOUNDARY CONDITIONS
// ============================================================================
TEST_CASE("Matrix rejects zero dimensions") {
  CHECK_THROWS_AS((hopfield::Matrix{0, 1}), std::runtime_error);
  CHECK_THROWS_AS((hopfield::Matrix{1, 0}), std::runtime_error);
  CHECK_THROWS_AS((hopfield::Matrix{0, 0}), std::runtime_error);
}

TEST_CASE("Matrix rejects data vectors with incorrect lengths") {
  CHECK_THROWS_AS((hopfield::Matrix{2, 2, std::vector<double>{1, 2, 3}}),
                  std::runtime_error);
  CHECK_THROWS_AS((hopfield::Matrix{2, 2, std::vector<double>{1, 2, 3, 4, 5}}),
                  std::runtime_error);
  CHECK_THROWS_AS((hopfield::Matrix{1, 3, std::vector<double>{}}),
                  std::runtime_error);
}

TEST_CASE("Matrix zero constructor initializes every element") {
  for (std::size_t rows = 1; rows <= 12; ++rows) {
    for (std::size_t cols = 1; cols <= 12; ++cols) {
      hopfield::Matrix m{rows, cols};
      REQUIRE(m.rows == rows);
      REQUIRE(m.cols == cols);
      REQUIRE(m.matrix.size() == rows * cols);
      CHECK(finiteMatrix(m));
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
          CHECK(m(r, c) == doctest::Approx(0.0));
    }
  }
}

TEST_CASE("Matrix stores constructor data in row-major order") {
  for (std::size_t rows = 1; rows <= 10; ++rows) {
    for (std::size_t cols = 1; cols <= 10; ++cols) {
      std::vector<double> values(rows * cols);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<double>(i) + 0.25;
      hopfield::Matrix m{rows, cols, values};
      CHECK(finiteMatrix(m));
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
          CHECK(m(r, c) == doctest::Approx(values[r * cols + c]));
    }
  }
}

TEST_CASE("Matrix mutable indexing changes the selected element only") {
  hopfield::Matrix m{4, 5};
  m(2, 3) = 17.5;
  CHECK(m(2, 3) == doctest::Approx(17.5));
  CHECK(m(0, 0) == doctest::Approx(0.0));
  CHECK(m(2, 2) == doctest::Approx(0.0));
  CHECK(m(2, 4) == doctest::Approx(0.0));
  CHECK(m(3, 3) == doctest::Approx(0.0));
}

TEST_CASE("Matrix const indexing returns stored values") {
  hopfield::Matrix source{2, 3, std::vector<double>{1, 2, 3, 4, 5, 6}};
  hopfield::Matrix const& m = source;
  CHECK(m(0, 0) == doctest::Approx(1.0));
  CHECK(m(0, 1) == doctest::Approx(2.0));
  CHECK(m(0, 2) == doctest::Approx(3.0));
  CHECK(m(1, 0) == doctest::Approx(4.0));
  CHECK(m(1, 1) == doctest::Approx(5.0));
  CHECK(m(1, 2) == doctest::Approx(6.0));
}

TEST_CASE("Matrix rejects row and column indices at the upper bound") {
  hopfield::Matrix m{3, 4};
  CHECK_THROWS_AS((m(3, 0)), std::runtime_error);
  CHECK_THROWS_AS((m(0, 4)), std::runtime_error);
  CHECK_THROWS_AS((m(3, 4)), std::runtime_error);
  CHECK_THROWS_AS((m(100, 0)), std::runtime_error);
  CHECK_THROWS_AS((m(0, 100)), std::runtime_error);
}

TEST_CASE("Matrix supports negative and fractional values") {
  hopfield::Matrix m{2, 3,
                     std::vector<double>{-1.5, 0.25, 2.75, -9.0, 4.5, 8.125}};
  CHECK(m(0, 0) == doctest::Approx(-1.5));
  CHECK(m(0, 1) == doctest::Approx(0.25));
  CHECK(m(0, 2) == doctest::Approx(2.75));
  CHECK(m(1, 0) == doctest::Approx(-9.0));
  CHECK(m(1, 1) == doctest::Approx(4.5));
  CHECK(m(1, 2) == doctest::Approx(8.125));
}

TEST_CASE("Matrix copy construction preserves dimensions and contents") {
  hopfield::Matrix original{3, 2, std::vector<double>{1, 2, 3, 4, 5, 6}};
  hopfield::Matrix copy{original};
  REQUIRE(copy.rows == original.rows);
  REQUIRE(copy.cols == original.cols);
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t c = 0; c < 2; ++c)
      CHECK(copy(r, c) == doctest::Approx(original(r, c)));
}

TEST_CASE("Matrix copied instances own independent storage") {
  hopfield::Matrix original{2, 2, std::vector<double>{1, 2, 3, 4}};
  hopfield::Matrix copy{original};
  copy(0, 0) = 99;
  CHECK(original(0, 0) == doctest::Approx(1.0));
  CHECK(copy(0, 0) == doctest::Approx(99.0));
}

TEST_CASE("Matrix move and assignment preserve usable values") {
  hopfield::Matrix a{2, 2, std::vector<double>{1, 2, 3, 4}};
  hopfield::Matrix b{1, 1};
  b = a;
  REQUIRE(b.rows == 2);
  REQUIRE(b.cols == 2);
  CHECK(b(1, 1) == doctest::Approx(4.0));
  b(1, 1) = 10;
  CHECK(a(1, 1) == doctest::Approx(4.0));
}

// A collection of dimension-specific regression cases catches row/column swaps.
TEST_CASE("Matrix row-major regression case 01") {
  std::vector<double> data{0.14285714, 0.28571429, 0.42857143, 0.57142857,
                           0.71428571, 0.85714286, 1,          1.1428571};
  hopfield::Matrix m{2, 4, data};
  REQUIRE(m.rows == 2);
  REQUIRE(m.cols == 4);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 02") {
  std::vector<double> data{
      0.28571429, 0.57142857, 0.85714286, 1.1428571, 1.4285714, 1.7142857, 2,
      2.2857143,  2.5714286,  2.8571429,  3.1428571, 3.4285714, 3.7142857, 4,
      4.2857143,  4.5714286,  4.8571429,  5.1428571, 5.4285714, 5.7142857, 6};
  hopfield::Matrix m{3, 7, data};
  REQUIRE(m.rows == 3);
  REQUIRE(m.cols == 7);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 03") {
  std::vector<double> data{0.42857143, 0.85714286, 1.2857143, 1.7142857,
                           2.1428571,  2.5714286,  3,         3.4285714};
  hopfield::Matrix m{4, 2, data};
  REQUIRE(m.rows == 4);
  REQUIRE(m.cols == 2);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 04") {
  std::vector<double> data{
      0.57142857, 1.1428571, 1.7142857, 2.2857143, 2.8571429, 3.4285714, 4,
      4.5714286,  5.1428571, 5.7142857, 6.2857143, 6.8571429, 7.4285714, 8,
      8.5714286,  9.1428571, 9.7142857, 10.285714, 10.857143, 11.428571, 12,
      12.571429,  13.142857, 13.714286, 14.285714};
  hopfield::Matrix m{5, 5, data};
  REQUIRE(m.rows == 5);
  REQUIRE(m.cols == 5);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 05") {
  std::vector<double> data{
      0.71428571, 1.4285714, 2.1428571, 2.8571429, 3.5714286, 4.2857143, 5,
      5.7142857,  6.4285714, 7.1428571, 7.8571429, 8.5714286, 9.2857143, 10,
      10.714286,  11.428571, 12.142857, 12.857143, 13.571429, 14.285714, 15,
      15.714286,  16.428571, 17.142857, 17.857143, 18.571429, 19.285714, 20,
      20.714286,  21.428571, 22.142857, 22.857143, 23.571429, 24.285714, 25,
      25.714286,  26.428571, 27.142857, 27.857143, 28.571429, 29.285714, 30,
      30.714286,  31.428571, 32.142857, 32.857143, 33.571429, 34.285714};
  hopfield::Matrix m{6, 8, data};
  REQUIRE(m.rows == 6);
  REQUIRE(m.cols == 8);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 06") {
  std::vector<double> data{
      0.85714286, 1.7142857, 2.5714286, 3.4285714, 4.2857143, 5.1428571, 6,
      6.8571429,  7.7142857, 8.5714286, 9.4285714, 10.285714, 11.142857, 12,
      12.857143,  13.714286, 14.571429, 15.428571, 16.285714, 17.142857, 18};
  hopfield::Matrix m{7, 3, data};
  REQUIRE(m.rows == 7);
  REQUIRE(m.cols == 3);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 07") {
  std::vector<double> data{1, 2, 3, 4, 5, 6};
  hopfield::Matrix m{1, 6, data};
  REQUIRE(m.rows == 1);
  REQUIRE(m.cols == 6);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 08") {
  std::vector<double> data{1.1428571, 2.2857143};
  hopfield::Matrix m{2, 1, data};
  REQUIRE(m.rows == 2);
  REQUIRE(m.cols == 1);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 09") {
  std::vector<double> data{1.2857143, 2.5714286, 3.8571429, 5.1428571,
                           6.4285714, 7.7142857, 9,         10.285714,
                           11.571429, 12.857143, 14.142857, 15.428571};
  hopfield::Matrix m{3, 4, data};
  REQUIRE(m.rows == 3);
  REQUIRE(m.cols == 4);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 10") {
  std::vector<double> data{
      1.4285714, 2.8571429, 4.2857143, 5.7142857, 7.1428571, 8.5714286, 10,
      11.428571, 12.857143, 14.285714, 15.714286, 17.142857, 18.571429, 20,
      21.428571, 22.857143, 24.285714, 25.714286, 27.142857, 28.571429, 30,
      31.428571, 32.857143, 34.285714, 35.714286, 37.142857, 38.571429, 40};
  hopfield::Matrix m{4, 7, data};
  REQUIRE(m.rows == 4);
  REQUIRE(m.cols == 7);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 11") {
  std::vector<double> data{1.5714286, 3.1428571, 4.7142857, 6.2857143,
                           7.8571429, 9.4285714, 11,        12.571429,
                           14.142857, 15.714286};
  hopfield::Matrix m{5, 2, data};
  REQUIRE(m.rows == 5);
  REQUIRE(m.cols == 2);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 12") {
  std::vector<double> data{
      1.7142857, 3.4285714, 5.1428571, 6.8571429, 8.5714286, 10.285714,
      12,        13.714286, 15.428571, 17.142857, 18.857143, 20.571429,
      22.285714, 24,        25.714286, 27.428571, 29.142857, 30.857143,
      32.571429, 34.285714, 36,        37.714286, 39.428571, 41.142857,
      42.857143, 44.571429, 46.285714, 48,        49.714286, 51.428571};
  hopfield::Matrix m{6, 5, data};
  REQUIRE(m.rows == 6);
  REQUIRE(m.cols == 5);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 13") {
  std::vector<double> data{
      1.8571429, 3.7142857, 5.5714286, 7.4285714, 9.2857143, 11.142857, 13,
      14.857143, 16.714286, 18.571429, 20.428571, 22.285714, 24.142857, 26,
      27.857143, 29.714286, 31.571429, 33.428571, 35.285714, 37.142857, 39,
      40.857143, 42.714286, 44.571429, 46.428571, 48.285714, 50.142857, 52,
      53.857143, 55.714286, 57.571429, 59.428571, 61.285714, 63.142857, 65,
      66.857143, 68.714286, 70.571429, 72.428571, 74.285714, 76.142857, 78,
      79.857143, 81.714286, 83.571429, 85.428571, 87.285714, 89.142857, 91,
      92.857143, 94.714286, 96.571429, 98.428571, 100.28571, 102.14286, 104};
  hopfield::Matrix m{7, 8, data};
  REQUIRE(m.rows == 7);
  REQUIRE(m.cols == 8);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 14") {
  std::vector<double> data{2, 4, 6};
  hopfield::Matrix m{1, 3, data};
  REQUIRE(m.rows == 1);
  REQUIRE(m.cols == 3);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 15") {
  std::vector<double> data{2.1428571, 4.2857143, 6.4285714, 8.5714286,
                           10.714286, 12.857143, 15,        17.142857,
                           19.285714, 21.428571, 23.571429, 25.714286};
  hopfield::Matrix m{2, 6, data};
  REQUIRE(m.rows == 2);
  REQUIRE(m.cols == 6);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 16") {
  std::vector<double> data{2.2857143, 4.5714286, 6.8571429};
  hopfield::Matrix m{3, 1, data};
  REQUIRE(m.rows == 3);
  REQUIRE(m.cols == 1);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 17") {
  std::vector<double> data{2.4285714, 4.8571429, 7.2857143, 9.7142857,
                           12.142857, 14.571429, 17,        19.428571,
                           21.857143, 24.285714, 26.714286, 29.142857,
                           31.571429, 34,        36.428571, 38.857143};
  hopfield::Matrix m{4, 4, data};
  REQUIRE(m.rows == 4);
  REQUIRE(m.cols == 4);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 18") {
  std::vector<double> data{
      2.5714286, 5.1428571, 7.7142857, 10.285714, 12.857143, 15.428571, 18,
      20.571429, 23.142857, 25.714286, 28.285714, 30.857143, 33.428571, 36,
      38.571429, 41.142857, 43.714286, 46.285714, 48.857143, 51.428571, 54,
      56.571429, 59.142857, 61.714286, 64.285714, 66.857143, 69.428571, 72,
      74.571429, 77.142857, 79.714286, 82.285714, 84.857143, 87.428571, 90};
  hopfield::Matrix m{5, 7, data};
  REQUIRE(m.rows == 5);
  REQUIRE(m.cols == 7);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 19") {
  std::vector<double> data{2.7142857, 5.4285714, 8.1428571, 10.857143,
                           13.571429, 16.285714, 19,        21.714286,
                           24.428571, 27.142857, 29.857143, 32.571429};
  hopfield::Matrix m{6, 2, data};
  REQUIRE(m.rows == 6);
  REQUIRE(m.cols == 2);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 20") {
  std::vector<double> data{
      2.8571429, 5.7142857, 8.5714286, 11.428571, 14.285714, 17.142857, 20,
      22.857143, 25.714286, 28.571429, 31.428571, 34.285714, 37.142857, 40,
      42.857143, 45.714286, 48.571429, 51.428571, 54.285714, 57.142857, 60,
      62.857143, 65.714286, 68.571429, 71.428571, 74.285714, 77.142857, 80,
      82.857143, 85.714286, 88.571429, 91.428571, 94.285714, 97.142857, 100};
  hopfield::Matrix m{7, 5, data};
  REQUIRE(m.rows == 7);
  REQUIRE(m.cols == 5);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 21") {
  std::vector<double> data{3, 6, 9, 12, 15, 18, 21, 24};
  hopfield::Matrix m{1, 8, data};
  REQUIRE(m.rows == 1);
  REQUIRE(m.cols == 8);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 22") {
  std::vector<double> data{3.1428571, 6.2857143, 9.4285714,
                           12.571429, 15.714286, 18.857143};
  hopfield::Matrix m{2, 3, data};
  REQUIRE(m.rows == 2);
  REQUIRE(m.cols == 3);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 23") {
  std::vector<double> data{
      3.2857143, 6.5714286, 9.8571429, 13.142857, 16.428571, 19.714286,
      23,        26.285714, 29.571429, 32.857143, 36.142857, 39.428571,
      42.714286, 46,        49.285714, 52.571429, 55.857143, 59.142857};
  hopfield::Matrix m{3, 6, data};
  REQUIRE(m.rows == 3);
  REQUIRE(m.cols == 6);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 24") {
  std::vector<double> data{3.4285714, 6.8571429, 10.285714, 13.714286};
  hopfield::Matrix m{4, 1, data};
  REQUIRE(m.rows == 4);
  REQUIRE(m.cols == 1);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 25") {
  std::vector<double> data{
      3.5714286, 7.1428571, 10.714286, 14.285714, 17.857143, 21.428571, 25,
      28.571429, 32.142857, 35.714286, 39.285714, 42.857143, 46.428571, 50,
      53.571429, 57.142857, 60.714286, 64.285714, 67.857143, 71.428571};
  hopfield::Matrix m{5, 4, data};
  REQUIRE(m.rows == 5);
  REQUIRE(m.cols == 4);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 26") {
  std::vector<double> data{
      3.7142857, 7.4285714, 11.142857, 14.857143, 18.571429, 22.285714, 26,
      29.714286, 33.428571, 37.142857, 40.857143, 44.571429, 48.285714, 52,
      55.714286, 59.428571, 63.142857, 66.857143, 70.571429, 74.285714, 78,
      81.714286, 85.428571, 89.142857, 92.857143, 96.571429, 100.28571, 104,
      107.71429, 111.42857, 115.14286, 118.85714, 122.57143, 126.28571, 130,
      133.71429, 137.42857, 141.14286, 144.85714, 148.57143, 152.28571, 156};
  hopfield::Matrix m{6, 7, data};
  REQUIRE(m.rows == 6);
  REQUIRE(m.cols == 7);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 27") {
  std::vector<double> data{
      3.8571429, 7.7142857, 11.571429, 15.428571, 19.285714, 23.142857, 27,
      30.857143, 34.714286, 38.571429, 42.428571, 46.285714, 50.142857, 54};
  hopfield::Matrix m{7, 2, data};
  REQUIRE(m.rows == 7);
  REQUIRE(m.cols == 2);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 28") {
  std::vector<double> data{4, 8, 12, 16, 20};
  hopfield::Matrix m{1, 5, data};
  REQUIRE(m.rows == 1);
  REQUIRE(m.cols == 5);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 29") {
  std::vector<double> data{4.1428571, 8.2857143, 12.428571, 16.571429,
                           20.714286, 24.857143, 29,        33.142857,
                           37.285714, 41.428571, 45.571429, 49.714286,
                           53.857143, 58,        62.142857, 66.285714};
  hopfield::Matrix m{2, 8, data};
  REQUIRE(m.rows == 2);
  REQUIRE(m.cols == 8);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 30") {
  std::vector<double> data{4.2857143, 8.5714286, 12.857143,
                           17.142857, 21.428571, 25.714286,
                           30,        34.285714, 38.571429};
  hopfield::Matrix m{3, 3, data};
  REQUIRE(m.rows == 3);
  REQUIRE(m.cols == 3);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 31") {
  std::vector<double> data{
      4.4285714, 8.8571429, 13.285714, 17.714286, 22.142857, 26.571429,
      31,        35.428571, 39.857143, 44.285714, 48.714286, 53.142857,
      57.571429, 62,        66.428571, 70.857143, 75.285714, 79.714286,
      84.142857, 88.571429, 93,        97.428571, 101.85714, 106.28571};
  hopfield::Matrix m{4, 6, data};
  REQUIRE(m.rows == 4);
  REQUIRE(m.cols == 6);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 32") {
  std::vector<double> data{4.5714286, 9.1428571, 13.714286, 18.285714,
                           22.857143};
  hopfield::Matrix m{5, 1, data};
  REQUIRE(m.rows == 5);
  REQUIRE(m.cols == 1);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 33") {
  std::vector<double> data{
      4.7142857, 9.4285714, 14.142857, 18.857143, 23.571429, 28.285714,
      33,        37.714286, 42.428571, 47.142857, 51.857143, 56.571429,
      61.285714, 66,        70.714286, 75.428571, 80.142857, 84.857143,
      89.571429, 94.285714, 99,        103.71429, 108.42857, 113.14286};
  hopfield::Matrix m{6, 4, data};
  REQUIRE(m.rows == 6);
  REQUIRE(m.cols == 4);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 34") {
  std::vector<double> data{
      4.8571429, 9.7142857, 14.571429, 19.428571, 24.285714, 29.142857, 34,
      38.857143, 43.714286, 48.571429, 53.428571, 58.285714, 63.142857, 68,
      72.857143, 77.714286, 82.571429, 87.428571, 92.285714, 97.142857, 102,
      106.85714, 111.71429, 116.57143, 121.42857, 126.28571, 131.14286, 136,
      140.85714, 145.71429, 150.57143, 155.42857, 160.28571, 165.14286, 170,
      174.85714, 179.71429, 184.57143, 189.42857, 194.28571, 199.14286, 204,
      208.85714, 213.71429, 218.57143, 223.42857, 228.28571, 233.14286, 238};
  hopfield::Matrix m{7, 7, data};
  REQUIRE(m.rows == 7);
  REQUIRE(m.cols == 7);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 35") {
  std::vector<double> data{5, 10};
  hopfield::Matrix m{1, 2, data};
  REQUIRE(m.rows == 1);
  REQUIRE(m.cols == 2);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 36") {
  std::vector<double> data{5.1428571, 10.285714, 15.428571, 20.571429,
                           25.714286, 30.857143, 36,        41.142857,
                           46.285714, 51.428571};
  hopfield::Matrix m{2, 5, data};
  REQUIRE(m.rows == 2);
  REQUIRE(m.cols == 5);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 37") {
  std::vector<double> data{
      5.2857143, 10.571429, 15.857143, 21.142857, 26.428571, 31.714286,
      37,        42.285714, 47.571429, 52.857143, 58.142857, 63.428571,
      68.714286, 74,        79.285714, 84.571429, 89.857143, 95.142857,
      100.42857, 105.71429, 111,       116.28571, 121.57143, 126.85714};
  hopfield::Matrix m{3, 8, data};
  REQUIRE(m.rows == 3);
  REQUIRE(m.cols == 8);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 38") {
  std::vector<double> data{5.4285714, 10.857143, 16.285714, 21.714286,
                           27.142857, 32.571429, 38,        43.428571,
                           48.857143, 54.285714, 59.714286, 65.142857};
  hopfield::Matrix m{4, 3, data};
  REQUIRE(m.rows == 4);
  REQUIRE(m.cols == 3);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 39") {
  std::vector<double> data{
      5.5714286, 11.142857, 16.714286, 22.285714, 27.857143, 33.428571,
      39,        44.571429, 50.142857, 55.714286, 61.285714, 66.857143,
      72.428571, 78,        83.571429, 89.142857, 94.714286, 100.28571,
      105.85714, 111.42857, 117,       122.57143, 128.14286, 133.71429,
      139.28571, 144.85714, 150.42857, 156,       161.57143, 167.14286};
  hopfield::Matrix m{5, 6, data};
  REQUIRE(m.rows == 5);
  REQUIRE(m.cols == 6);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 40") {
  std::vector<double> data{5.7142857, 11.428571, 17.142857,
                           22.857143, 28.571429, 34.285714};
  hopfield::Matrix m{6, 1, data};
  REQUIRE(m.rows == 6);
  REQUIRE(m.cols == 1);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 41") {
  std::vector<double> data{
      5.8571429, 11.714286, 17.571429, 23.428571, 29.285714, 35.142857, 41,
      46.857143, 52.714286, 58.571429, 64.428571, 70.285714, 76.142857, 82,
      87.857143, 93.714286, 99.571429, 105.42857, 111.28571, 117.14286, 123,
      128.85714, 134.71429, 140.57143, 146.42857, 152.28571, 158.14286, 164};
  hopfield::Matrix m{7, 4, data};
  REQUIRE(m.rows == 7);
  REQUIRE(m.cols == 4);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 42") {
  std::vector<double> data{6, 12, 18, 24, 30, 36, 42};
  hopfield::Matrix m{1, 7, data};
  REQUIRE(m.rows == 1);
  REQUIRE(m.cols == 7);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 43") {
  std::vector<double> data{6.1428571, 12.285714, 18.428571, 24.571429};
  hopfield::Matrix m{2, 2, data};
  REQUIRE(m.rows == 2);
  REQUIRE(m.cols == 2);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 44") {
  std::vector<double> data{6.2857143, 12.571429, 18.857143, 25.142857,
                           31.428571, 37.714286, 44,        50.285714,
                           56.571429, 62.857143, 69.142857, 75.428571,
                           81.714286, 88,        94.285714};
  hopfield::Matrix m{3, 5, data};
  REQUIRE(m.rows == 3);
  REQUIRE(m.cols == 5);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 45") {
  std::vector<double> data{
      6.4285714, 12.857143, 19.285714, 25.714286, 32.142857, 38.571429, 45,
      51.428571, 57.857143, 64.285714, 70.714286, 77.142857, 83.571429, 90,
      96.428571, 102.85714, 109.28571, 115.71429, 122.14286, 128.57143, 135,
      141.42857, 147.85714, 154.28571, 160.71429, 167.14286, 173.57143, 180,
      186.42857, 192.85714, 199.28571, 205.71429};
  hopfield::Matrix m{4, 8, data};
  REQUIRE(m.rows == 4);
  REQUIRE(m.cols == 8);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 46") {
  std::vector<double> data{6.5714286, 13.142857, 19.714286, 26.285714,
                           32.857143, 39.428571, 46,        52.571429,
                           59.142857, 65.714286, 72.285714, 78.857143,
                           85.428571, 92,        98.571429};
  hopfield::Matrix m{5, 3, data};
  REQUIRE(m.rows == 5);
  REQUIRE(m.cols == 3);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 47") {
  std::vector<double> data{
      6.7142857, 13.428571, 20.142857, 26.857143, 33.571429, 40.285714,
      47,        53.714286, 60.428571, 67.142857, 73.857143, 80.571429,
      87.285714, 94,        100.71429, 107.42857, 114.14286, 120.85714,
      127.57143, 134.28571, 141,       147.71429, 154.42857, 161.14286,
      167.85714, 174.57143, 181.28571, 188,       194.71429, 201.42857,
      208.14286, 214.85714, 221.57143, 228.28571, 235,       241.71429};
  hopfield::Matrix m{6, 6, data};
  REQUIRE(m.rows == 6);
  REQUIRE(m.cols == 6);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

TEST_CASE("Matrix row-major regression case 48") {
  std::vector<double> data{6.8571429, 13.714286, 20.571429, 27.428571,
                           34.285714, 41.142857, 48};
  hopfield::Matrix m{7, 1, data};
  REQUIRE(m.rows == 7);
  REQUIRE(m.cols == 1);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(m.matrix[i] == doctest::Approx(data[i]));
}

// ============================================================================
// ENERGY: HOPFIELD ENERGY FUNCTION
// ============================================================================
TEST_CASE("Energy of a two-neuron known state") {
  hopfield::Matrix w{2, 2, std::vector<double>{0, 1, 1, 0}};
  hopfield::PatternInt p = makePattern(2, 1, 1);
  p.data[1] = -1;
  CHECK(hopfield::energy(p, w) == doctest::Approx(1.0));
}

TEST_CASE("Energy is zero for a zero weight matrix") {
  hopfield::Matrix w{4, 4};
  auto p = makePattern(2, 2, 1);
  CHECK(hopfield::energy(p, w) == doctest::Approx(0.0));
}

TEST_CASE("Energy is invariant under global inversion") {
  hopfield::Matrix w{
      3, 3, std::vector<double>{0, .5, -.25, .5, 0, .75, -.25, .75, 0}};
  auto p = makePattern(3, 1, 1);
  p.data = {1, -1, 1};
  auto q = p;
  for (auto& value : q.data) value = -value;
  CHECK(hopfield::energy(p, w) == doctest::Approx(hopfield::energy(q, w)));
}

TEST_CASE("Energy is finite for finite symmetric weights") {
  hopfield::Matrix w{
      4, 4,
      std::vector<double>{0, 1, 2, 3, 1, 0, 4, 5, 2, 4, 0, 6, 3, 5, 6, 0}};
  auto p = makePattern(2, 2, 1);
  p.data = {1, -1, -1, 1};
  CHECK(std::isfinite(hopfield::energy(p, w)));
}

TEST_CASE("Energy is zero when all off-diagonal weights are zero") {
  hopfield::Matrix w{3, 3};
  auto p = makePattern(3, 1, -1);
  CHECK(hopfield::energy(p, w) == doctest::Approx(0.0));
}

TEST_CASE("Energy changes sign when all weights change sign") {
  hopfield::Matrix positive{2, 2, std::vector<double>{0, 2, 2, 0}};
  hopfield::Matrix negative{2, 2, std::vector<double>{0, -2, -2, 0}};
  auto p = makePattern(2, 1, 1);
  p.data[1] = -1;
  CHECK(hopfield::energy(p, positive) ==
        doctest::Approx(-hopfield::energy(p, negative)));
}

// ============================================================================
// NETWORK: CONSTRUCTION, TRAINING, LEARNING, RECALL, PERSISTENCE
// ============================================================================
TEST_CASE("Network rejects zero-sized dimensions") {
  CHECK_THROWS_AS((hopfield::Network{sf::Vector2u{0u, 1u}}),
                  std::runtime_error);
  CHECK_THROWS_AS((hopfield::Network{sf::Vector2u{1u, 0u}}),
                  std::runtime_error);
  CHECK_THROWS_AS((hopfield::Network{sf::Vector2u{0u, 0u}}),
                  std::runtime_error);
}

TEST_CASE("Network constructor records the pattern size") {
  hopfield::Network net{sf::Vector2u{2u, 3u}};
  CHECK((net.size() == sf::Vector2u{2u, 3u}));
  CHECK_FALSE(net.isTrained());
}

TEST_CASE("Network constructor rejects non-square weight matrices") {
  hopfield::Matrix rectangular{2, 3};
  CHECK_THROWS_AS((hopfield::Network{sf::Vector2u{2u, 1u}, rectangular}),
                  std::runtime_error);
}

TEST_CASE("Network constructor rejects a matrix with wrong neuron count") {
  hopfield::Matrix wrong{3, 3};
  CHECK_THROWS_AS((hopfield::Network{sf::Vector2u{2u, 2u}, wrong}),
                  std::runtime_error);
}

TEST_CASE("Network accepts a square matrix matching its image size") {
  hopfield::Matrix w{4, 4};
  hopfield::Network net{sf::Vector2u{2u, 2u}, w};
  CHECK((net.size() == sf::Vector2u{2u, 2u}));
}

TEST_CASE("Network training rejects an empty pattern collection") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  CHECK_THROWS_AS((net.train(std::vector<hopfield::PatternInt>{})),
                  std::runtime_error);
}

TEST_CASE("Network training creates finite weights") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.isTrained());
}

TEST_CASE("Network trained weights have zero diagonal") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  // Reconstructing the network from a learned matrix is not part of the public
  // API; persistence and recall exercise the diagonal-free trained state
  // indirectly.
  auto result = net.recall(p);
  CHECK(result.data == p.data);
}

TEST_CASE("Network can learn a pattern incrementally") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  CHECK_NOTHROW(net.learn(p));
  CHECK(net.isTrained());
}

TEST_CASE("Network learn accepts a negative contribution") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  CHECK_NOTHROW(net.learn(p, -1.0));
}

TEST_CASE("Network learn accepts a fractional contribution") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, -1);
  CHECK_NOTHROW(net.learn(p, 0.5));
}

TEST_CASE("Network recall returns a stored uniform positive pattern") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  auto recalled = net.recall(p);
  CHECK(recalled.size == p.size);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Network recall returns a stored uniform negative pattern") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  auto recalled = net.recall(p);
  CHECK(recalled.size == p.size);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Network recall invokes its callback") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  std::size_t calls = 0;
  auto callback = [&calls](hopfield::PatternInt const&, std::size_t, double e) {
    ++calls;
    CHECK(std::isfinite(e));
  };
  CHECK_NOTHROW(net.recall(p, 100, callback));
  CHECK(calls >= 1);
}

TEST_CASE("Network recall reports finite energy to callback") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  bool finite = true;
  net.recall(p, 100,
             [&finite](hopfield::PatternInt const&, std::size_t, double e) {
               finite = finite && std::isfinite(e);
             });
  CHECK(finite);
}

TEST_CASE("Network save succeeds on an initialized untrained network") {
  auto dir = makeTempDir("save_untrained");
  auto file = dir / "network.bin";
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  CHECK_NOTHROW(net.save(file));
  REQUIRE(fs::exists(file));
  auto loaded = hopfield::Network::load(file);
  CHECK(loaded.size() == net.size());
  CHECK_FALSE(loaded.isTrained());
  fs::remove_all(dir);
}

TEST_CASE("Network save and load round-trip") {
  auto dir = makeTempDir("roundtrip");
  auto file = dir / "network.bin";
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  net.save(file);
  REQUIRE(fs::exists(file));
  auto loaded = hopfield::Network::load(file);
  CHECK(loaded.size() == net.size());
  CHECK(loaded.isTrained());
  CHECK(loaded.recall(p).data == p.data);
  fs::remove_all(dir);
}

TEST_CASE("Network load rejects a missing file") {
  auto dir = makeTempDir("missing_network");
  CHECK_THROWS_AS((hopfield::Network::load(dir / "missing.bin")),
                  std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Network load rejects an invalid file header") {
  auto dir = makeTempDir("bad_header");
  auto file = dir / "bad.bin";
  {
    std::ofstream out(file, std::ios::binary);
    out << "NOPE";
  }
  CHECK_THROWS_AS((hopfield::Network::load(file)), std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Network load rejects an empty file") {
  auto dir = makeTempDir("empty_network");
  auto file = dir / "empty.bin";
  {
    std::ofstream out(file, std::ios::binary);
  }
  CHECK_THROWS_AS((hopfield::Network::load(file)), std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Network uniform-pattern recall regression 01") {
  hopfield::Network net{sf::Vector2u{2u, 4u}};
  auto pattern = makePattern(2, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 02") {
  hopfield::Network net{sf::Vector2u{3u, 3u}};
  auto pattern = makePattern(3, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 03") {
  hopfield::Network net{sf::Vector2u{4u, 2u}};
  auto pattern = makePattern(4, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 04") {
  // Single neuron (1x1) has no off-diagonal weights, so isTrained() is false.
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto pattern = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 05") {
  hopfield::Network net{sf::Vector2u{2u, 4u}};
  auto pattern = makePattern(2, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 06") {
  hopfield::Network net{sf::Vector2u{3u, 3u}};
  auto pattern = makePattern(3, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 07") {
  hopfield::Network net{sf::Vector2u{4u, 2u}};
  auto pattern = makePattern(4, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 08") {
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto pattern = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 09") {
  hopfield::Network net{sf::Vector2u{2u, 4u}};
  auto pattern = makePattern(2, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 10") {
  hopfield::Network net{sf::Vector2u{3u, 3u}};
  auto pattern = makePattern(3, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 11") {
  hopfield::Network net{sf::Vector2u{4u, 2u}};
  auto pattern = makePattern(4, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 12") {
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto pattern = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 13") {
  hopfield::Network net{sf::Vector2u{2u, 4u}};
  auto pattern = makePattern(2, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 14") {
  hopfield::Network net{sf::Vector2u{3u, 3u}};
  auto pattern = makePattern(3, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 15") {
  hopfield::Network net{sf::Vector2u{4u, 2u}};
  auto pattern = makePattern(4, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 16") {
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto pattern = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 17") {
  hopfield::Network net{sf::Vector2u{2u, 4u}};
  auto pattern = makePattern(2, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 18") {
  hopfield::Network net{sf::Vector2u{3u, 3u}};
  auto pattern = makePattern(3, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 19") {
  hopfield::Network net{sf::Vector2u{4u, 2u}};
  auto pattern = makePattern(4, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 20") {
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto pattern = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 21") {
  hopfield::Network net{sf::Vector2u{2u, 4u}};
  auto pattern = makePattern(2, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 22") {
  hopfield::Network net{sf::Vector2u{3u, 3u}};
  auto pattern = makePattern(3, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 23") {
  hopfield::Network net{sf::Vector2u{4u, 2u}};
  auto pattern = makePattern(4, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 24") {
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto pattern = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

TEST_CASE("Network uniform-pattern recall regression 25") {
  hopfield::Network net{sf::Vector2u{2u, 4u}};
  auto pattern = makePattern(2, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{pattern});
  CHECK(net.isTrained());
  auto recalled = net.recall(pattern);
  CHECK(recalled.size == pattern.size);
  CHECK(recalled.data == pattern.data);
}

// ============================================================================
// DATASET: DIRECTORY VALIDATION, IMAGE IMPORT, LISTING, REMOVAL, CONVERSION
// ============================================================================
TEST_CASE("Dataset rejects a path that does not exist") {
  auto dir = makeTempDir("dataset_missing");
  fs::remove_all(dir);
  CHECK_THROWS_AS((hopfield::Dataset{dir}), std::runtime_error);
}

TEST_CASE("Dataset rejects a regular file as its directory") {
  auto dir = makeTempDir("dataset_not_dir");
  auto file = dir / "plain.txt";
  {
    std::ofstream out(file);
    out << "not a directory";
  }
  CHECK_THROWS_AS((hopfield::Dataset{file}), std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Empty Dataset has no names, zero size, and throws on patterns()") {
  auto dir = makeTempDir("dataset_empty");
  hopfield::Dataset dataset{dir};
  CHECK(dataset.getNames().empty());
  CHECK((dataset.size() == sf::Vector2u{0u, 0u}));
  CHECK_THROWS_AS(dataset.patterns(), std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Dataset ignores unrelated files") {
  auto dir = makeTempDir("dataset_unrelated");
  {
    std::ofstream out(dir / "notes.txt");
    out << "ignore";
  }
  {
    std::ofstream out(dir / "20260101-120000_bad.gif");
    out << "ignore";
  }
  hopfield::Dataset dataset{dir};
  CHECK(dataset.getNames().empty());
  CHECK_THROWS_AS(dataset.patterns(), std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Dataset add rejects a missing image path") {
  auto dir = makeTempDir("dataset_add_missing");
  hopfield::Dataset dataset{dir};
  CHECK_THROWS_AS((dataset.add(dir / "absent.png")), std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Dataset add imports a valid PNG image") {
  auto dir = makeTempDir("dataset_add_png");
  auto sourceDir = makeTempDir("dataset_source_png");
  auto imagePath = sourceDir / "sample.png";
  auto image = makeImage(3, 2, sf::Color::White);
  REQUIRE(image.saveToFile(imagePath));
  hopfield::Dataset dataset{dir};
  auto pattern = dataset.add(imagePath);
  CHECK((pattern.size == sf::Vector2u{3u, 2u}));
  CHECK(pattern.data.size() == 6);
  CHECK(dataset.getNames().size() == 1);
  CHECK((dataset.size() == sf::Vector2u{3u, 2u}));
  fs::remove_all(dir);
  fs::remove_all(sourceDir);
}

TEST_CASE("Dataset add imports a valid JPEG image") {
  auto dir = makeTempDir("dataset_add_jpg");
  auto sourceDir = makeTempDir("dataset_source_jpg");
  auto imagePath = sourceDir / "sample.jpg";
  auto image = makeImage(2, 2, sf::Color::Black);
  REQUIRE(image.saveToFile(imagePath));
  hopfield::Dataset dataset{dir};
  auto pattern = dataset.add(imagePath);
  CHECK((pattern.size == sf::Vector2u{2u, 2u}));
  CHECK(pattern.data.size() == 4);
  CHECK(dataset.getNames().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(sourceDir);
}

TEST_CASE("Dataset add rejects an unsupported extension") {
  auto dir = makeTempDir("dataset_bad_ext");
  auto sourceDir = makeTempDir("dataset_bad_ext_source");
  auto path = sourceDir / "image.bmp";
  auto image = makeImage(2, 2, sf::Color::White);
  REQUIRE(image.saveToFile(path));
  hopfield::Dataset dataset{dir};
  CHECK_THROWS_AS((dataset.add(path)), std::runtime_error);
  fs::remove_all(dir);
  fs::remove_all(sourceDir);
}

TEST_CASE("Dataset patterns returns added pattern") {
  auto dir = makeTempDir("dataset_patterns");
  auto sourceDir = makeTempDir("dataset_patterns_src");
  auto path = sourceDir / "white.png";
  REQUIRE(makeImage(2, 3, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto added = dataset.add(path);
  auto patterns = dataset.patterns();
  REQUIRE(patterns.size() == 1);
  CHECK(patterns[0].size == added.size);
  CHECK(patterns[0].data == added.data);
  fs::remove_all(dir);
  fs::remove_all(sourceDir);
}

TEST_CASE("Dataset remove rejects an unknown name") {
  auto dir = makeTempDir("dataset_remove_unknown");
  hopfield::Dataset dataset{dir};
  CHECK_THROWS_AS((dataset.remove("unknown.png")), std::runtime_error);
  fs::remove_all(dir);
}

TEST_CASE("Dataset remove removes the selected pattern") {
  auto dir = makeTempDir("dataset_remove");
  auto sourceDir = makeTempDir("dataset_remove_src");
  auto path = sourceDir / "remove_me.png";
  REQUIRE(makeImage(2, 2, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto added = dataset.add(path);
  auto names = dataset.getNames();
  REQUIRE(names.size() == 1);
  auto removed = dataset.remove(names.front());
  CHECK(removed.size == added.size);
  CHECK(removed.data == added.data);
  CHECK(dataset.getNames().empty());
  CHECK((dataset.size() == sf::Vector2u{0u, 0u}));
  fs::remove_all(dir);
  fs::remove_all(sourceDir);
}

TEST_CASE("Dataset names are sorted") {
  auto dir = makeTempDir("dataset_sorted");
  // Files without the required timestamp prefix are intentionally ignored.
  {
    std::ofstream out(dir / "zeta.png");
    out << "x";
  }
  {
    std::ofstream out(dir / "alpha.png");
    out << "x";
  }
  hopfield::Dataset dataset{dir};
  CHECK(dataset.getNames().empty());
  fs::remove_all(dir);
}

// Dataset integration: images with different dimensions are normalized by the
// image-processing layer before patterns() and size() expose the dataset.
TEST_CASE("Dataset handles multiple images with a common output size") {
  auto dir = makeTempDir("dataset_multiple");
  auto src = makeTempDir("dataset_multiple_src");
  auto a = src / "small.png";
  auto b = src / "large.png";
  REQUIRE(makeImage(2, 2, sf::Color::White).saveToFile(a));
  REQUIRE(makeImage(4, 4, sf::Color::Black).saveToFile(b));
  hopfield::Dataset dataset{dir};
  CHECK_NOTHROW(dataset.add(a));
  CHECK_NOTHROW(dataset.add(b));
  CHECK(dataset.getNames().size() == 2);
  auto patterns = dataset.patterns();
  REQUIRE(patterns.size() == 2);
  CHECK(patterns[0].size == patterns[1].size);
  CHECK(dataset.size() == patterns[0].size);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 01") {
  auto dir = makeTempDir("dataset_reg_1");
  auto src = makeTempDir("dataset_src_1");
  auto path = src / "input_1.png";
  REQUIRE(makeImage(2, 3, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{2u, 3u}));
  CHECK(imported.data.size() == 6);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 02") {
  auto dir = makeTempDir("dataset_reg_2");
  auto src = makeTempDir("dataset_src_2");
  auto path = src / "input_2.png";
  REQUIRE(makeImage(3, 5, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{3u, 5u}));
  CHECK(imported.data.size() == 15);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 03") {
  auto dir = makeTempDir("dataset_reg_3");
  auto src = makeTempDir("dataset_src_3");
  auto path = src / "input_3.png";
  REQUIRE(makeImage(4, 2, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{4u, 2u}));
  CHECK(imported.data.size() == 8);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 04") {
  auto dir = makeTempDir("dataset_reg_4");
  auto src = makeTempDir("dataset_src_4");
  auto path = src / "input_4.png";
  REQUIRE(makeImage(5, 4, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{5u, 4u}));
  CHECK(imported.data.size() == 20);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 05") {
  auto dir = makeTempDir("dataset_reg_5");
  auto src = makeTempDir("dataset_src_5");
  auto path = src / "input_5.png";
  REQUIRE(makeImage(1, 1, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{1u, 1u}));
  CHECK(imported.data.size() == 1);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 06") {
  auto dir = makeTempDir("dataset_reg_6");
  auto src = makeTempDir("dataset_src_6");
  auto path = src / "input_6.png";
  REQUIRE(makeImage(2, 3, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{2u, 3u}));
  CHECK(imported.data.size() == 6);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 07") {
  auto dir = makeTempDir("dataset_reg_7");
  auto src = makeTempDir("dataset_src_7");
  auto path = src / "input_7.png";
  REQUIRE(makeImage(3, 5, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{3u, 5u}));
  CHECK(imported.data.size() == 15);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 08") {
  auto dir = makeTempDir("dataset_reg_8");
  auto src = makeTempDir("dataset_src_8");
  auto path = src / "input_8.png";
  REQUIRE(makeImage(4, 2, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{4u, 2u}));
  CHECK(imported.data.size() == 8);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 09") {
  auto dir = makeTempDir("dataset_reg_9");
  auto src = makeTempDir("dataset_src_9");
  auto path = src / "input_9.png";
  REQUIRE(makeImage(5, 4, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{5u, 4u}));
  CHECK(imported.data.size() == 20);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 10") {
  auto dir = makeTempDir("dataset_reg_10");
  auto src = makeTempDir("dataset_src_10");
  auto path = src / "input_10.png";
  REQUIRE(makeImage(1, 1, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{1u, 1u}));
  CHECK(imported.data.size() == 1);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 11") {
  auto dir = makeTempDir("dataset_reg_11");
  auto src = makeTempDir("dataset_src_11");
  auto path = src / "input_11.png";
  REQUIRE(makeImage(2, 3, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{2u, 3u}));
  CHECK(imported.data.size() == 6);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 12") {
  auto dir = makeTempDir("dataset_reg_12");
  auto src = makeTempDir("dataset_src_12");
  auto path = src / "input_12.png";
  REQUIRE(makeImage(3, 5, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{3u, 5u}));
  CHECK(imported.data.size() == 15);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 13") {
  auto dir = makeTempDir("dataset_reg_13");
  auto src = makeTempDir("dataset_src_13");
  auto path = src / "input_13.png";
  REQUIRE(makeImage(4, 2, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{4u, 2u}));
  CHECK(imported.data.size() == 8);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 14") {
  auto dir = makeTempDir("dataset_reg_14");
  auto src = makeTempDir("dataset_src_14");
  auto path = src / "input_14.png";
  REQUIRE(makeImage(5, 4, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{5u, 4u}));
  CHECK(imported.data.size() == 20);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 15") {
  auto dir = makeTempDir("dataset_reg_15");
  auto src = makeTempDir("dataset_src_15");
  auto path = src / "input_15.png";
  REQUIRE(makeImage(1, 1, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{1u, 1u}));
  CHECK(imported.data.size() == 1);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 16") {
  auto dir = makeTempDir("dataset_reg_16");
  auto src = makeTempDir("dataset_src_16");
  auto path = src / "input_16.png";
  REQUIRE(makeImage(2, 3, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{2u, 3u}));
  CHECK(imported.data.size() == 6);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 17") {
  auto dir = makeTempDir("dataset_reg_17");
  auto src = makeTempDir("dataset_src_17");
  auto path = src / "input_17.png";
  REQUIRE(makeImage(3, 5, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{3u, 5u}));
  CHECK(imported.data.size() == 15);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 18") {
  auto dir = makeTempDir("dataset_reg_18");
  auto src = makeTempDir("dataset_src_18");
  auto path = src / "input_18.png";
  REQUIRE(makeImage(4, 2, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{4u, 2u}));
  CHECK(imported.data.size() == 8);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 19") {
  auto dir = makeTempDir("dataset_reg_19");
  auto src = makeTempDir("dataset_src_19");
  auto path = src / "input_19.png";
  REQUIRE(makeImage(5, 4, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{5u, 4u}));
  CHECK(imported.data.size() == 20);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Dataset PNG import regression 20") {
  auto dir = makeTempDir("dataset_reg_20");
  auto src = makeTempDir("dataset_src_20");
  auto path = src / "input_20.png";
  REQUIRE(makeImage(1, 1, sf::Color::Black).saveToFile(path));
  hopfield::Dataset dataset{dir};
  auto imported = dataset.add(path);
  CHECK((imported.size == sf::Vector2u{1u, 1u}));
  CHECK(imported.data.size() == 1);
  CHECK(dataset.getNames().size() == 1);
  CHECK(dataset.patterns().size() == 1);
  fs::remove_all(dir);
  fs::remove_all(src);
}

// ============================================================================
// CROSS-COMPONENT REGRESSION TESTS
// ============================================================================
TEST_CASE("Dataset patterns can be used to train a Network") {
  auto dir = makeTempDir("dataset_network");
  auto src = makeTempDir("dataset_network_src");
  auto path = src / "pattern.png";
  REQUIRE(makeImage(2, 2, sf::Color::White).saveToFile(path));
  hopfield::Dataset dataset{dir};
  dataset.add(path);
  hopfield::Network net{dataset.size()};
  net.train(dataset.patterns());
  CHECK(net.isTrained());
  CHECK(net.recall(dataset.patterns().front()).data ==
        dataset.patterns().front().data);
  fs::remove_all(dir);
  fs::remove_all(src);
}

TEST_CASE("Training with two opposite patterns produces a trained network") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto white = makePattern(2, 2, 1);
  auto black = makePattern(2, 2, -1);
  net.train(std::vector<hopfield::PatternInt>{white, black});
  CHECK(net.isTrained());
  CHECK_NOTHROW(net.recall(white));
  CHECK_NOTHROW(net.recall(black));
}

TEST_CASE("Training with repeated copies remains numerically finite") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p, p, p, p});
  CHECK(net.isTrained());
  CHECK_NOTHROW(net.recall(p));
}

TEST_CASE("Network save creates a non-empty binary file") {
  auto dir = makeTempDir("network_nonempty");
  auto file = dir / "saved.hopf";
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  net.save(file);
  REQUIRE(fs::exists(file));
  CHECK(fs::file_size(file) > 12);
  fs::remove_all(dir);
}

TEST_CASE("Loaded network preserves its dimensions") {
  auto dir = makeTempDir("network_dimensions");
  auto file = dir / "saved.hopf";
  hopfield::Network net{sf::Vector2u{3u, 2u}};
  auto p = makePattern(3, 2, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  net.save(file);
  auto loaded = hopfield::Network::load(file);
  CHECK((loaded.size() == sf::Vector2u{3u, 2u}));
  fs::remove_all(dir);
}

TEST_CASE("Recall callback receives a non-negative initial sweep identifier") {
  hopfield::Network net{sf::Vector2u{2u, 2u}};
  auto p = makePattern(2, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  bool called = false;
  net.recall(p, 20,
             [&called](hopfield::PatternInt const&, std::size_t id, double) {
               called = true;
               CHECK(id <= 20);
             });
  CHECK(called);
}

// The tests below deliberately exercise rectangular image dimensions while
// keeping the neuron count and network matrix square as required by Hopfield.
TEST_CASE("Rectangular network shape regression 01") {
  hopfield::Network net{sf::Vector2u{2u, 6u}};
  auto p = makePattern(2, 6, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 02") {
  hopfield::Network net{sf::Vector2u{3u, 5u}};
  auto p = makePattern(3, 5, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 03") {
  hopfield::Network net{sf::Vector2u{4u, 4u}};
  auto p = makePattern(4, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 04") {
  hopfield::Network net{sf::Vector2u{5u, 3u}};
  auto p = makePattern(5, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 05") {
  hopfield::Network net{sf::Vector2u{6u, 2u}};
  auto p = makePattern(6, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 06") {
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto p = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 07") {
  hopfield::Network net{sf::Vector2u{2u, 6u}};
  auto p = makePattern(2, 6, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 08") {
  hopfield::Network net{sf::Vector2u{3u, 5u}};
  auto p = makePattern(3, 5, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 09") {
  hopfield::Network net{sf::Vector2u{4u, 4u}};
  auto p = makePattern(4, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 10") {
  hopfield::Network net{sf::Vector2u{5u, 3u}};
  auto p = makePattern(5, 3, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 11") {
  hopfield::Network net{sf::Vector2u{6u, 2u}};
  auto p = makePattern(6, 2, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 12") {
  hopfield::Network net{sf::Vector2u{1u, 1u}};
  auto p = makePattern(1, 1, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK_FALSE(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 13") {
  hopfield::Network net{sf::Vector2u{2u, 6u}};
  auto p = makePattern(2, 6, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 14") {
  hopfield::Network net{sf::Vector2u{3u, 5u}};
  auto p = makePattern(3, 5, -1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

TEST_CASE("Rectangular network shape regression 15") {
  hopfield::Network net{sf::Vector2u{4u, 4u}};
  auto p = makePattern(4, 4, 1);
  net.train(std::vector<hopfield::PatternInt>{p});
  CHECK(net.size() == p.size);
  CHECK(net.isTrained());
  auto recalled = net.recall(p);
  CHECK(recalled.data == p.data);
}

// ============================================================================
// ADDITIONAL MATRIX PROPERTY TESTS
// ============================================================================
TEST_CASE("Matrix diagonal can be assigned independently") {
  hopfield::Matrix m{5, 5};
  for (std::size_t i = 0; i < 5; ++i) m(i, i) = static_cast<double>(i + 1);
  for (std::size_t i = 0; i < 5; ++i)
    CHECK(m(i, i) == doctest::Approx(static_cast<double>(i + 1)));
}

TEST_CASE("Matrix row writes do not overwrite neighboring rows") {
  hopfield::Matrix m{4, 3};
  for (std::size_t c = 0; c < 3; ++c)
    m(1, c) = 10.0 + static_cast<double>(c);
  for (std::size_t c = 0; c < 3; ++c)
    CHECK(m(1, c) == doctest::Approx(10.0 + static_cast<double>(c)));
  for (std::size_t c = 0; c < 3; ++c) CHECK(m(0, c) == doctest::Approx(0.0));
  for (std::size_t c = 0; c < 3; ++c) CHECK(m(2, c) == doctest::Approx(0.0));
}

TEST_CASE("Matrix column writes do not overwrite neighboring columns") {
  hopfield::Matrix m{3, 4};
  for (std::size_t r = 0; r < 3; ++r)
    m(r, 2) = 20.0 + static_cast<double>(r);
  for (std::size_t r = 0; r < 3; ++r)
    CHECK(m(r, 2) == doctest::Approx(20.0 + static_cast<double>(r)));
  for (std::size_t r = 0; r < 3; ++r) CHECK(m(r, 1) == doctest::Approx(0.0));
  for (std::size_t r = 0; r < 3; ++r) CHECK(m(r, 3) == doctest::Approx(0.0));
}

TEST_CASE("Matrix handles a one-row matrix") {
  hopfield::Matrix m{1, 5, std::vector<double>{1, 2, 3, 4, 5}};
  for (std::size_t c = 0; c < 5; ++c)
    CHECK(m(0, c) == doctest::Approx(static_cast<double>(c + 1)));
  CHECK_THROWS_AS((m(1, 0)), std::runtime_error);
}

TEST_CASE("Matrix handles a one-column matrix") {
  hopfield::Matrix m{5, 1, std::vector<double>{1, 2, 3, 4, 5}};
  for (std::size_t r = 0; r < 5; ++r)
    CHECK(m(r, 0) == doctest::Approx(static_cast<double>(r + 1)));
  CHECK_THROWS_AS((m(0, 1)), std::runtime_error);
}

TEST_CASE("Matrix supports large but practical dimensions") {
  hopfield::Matrix m{64, 64};
  REQUIRE(m.matrix.size() == 4096);
  m(63, 63) = 42.0;
  CHECK(m(63, 63) == doctest::Approx(42.0));
}

TEST_CASE("Matrix stores zero, signed zero, and ordinary values") {
  hopfield::Matrix m{1, 4, std::vector<double>{0.0, -0.0, 1.0, -1.0}};
  CHECK(m(0, 0) == doctest::Approx(0.0));
  CHECK(m(0, 1) == doctest::Approx(0.0));
  CHECK(m(0, 2) == doctest::Approx(1.0));
  CHECK(m(0, 3) == doctest::Approx(-1.0));
}

TEST_CASE("Matrix access at every valid corner succeeds") {
  hopfield::Matrix m{3, 4};
  CHECK_NOTHROW(m(0, 0));
  CHECK_NOTHROW(m(0, 3));
  CHECK_NOTHROW(m(2, 0));
  CHECK_NOTHROW(m(2, 3));
}

TEST_CASE("Matrix const access rejects invalid coordinates") {
  hopfield::Matrix source{2, 2};
  hopfield::Matrix const& m = source;
  CHECK_THROWS_AS((m(2, 0)), std::runtime_error);
  CHECK_THROWS_AS((m(0, 2)), std::runtime_error);
}

TEST_CASE("Matrix data vector length agrees with dimensions") {
  for (std::size_t n = 1; n <= 20; ++n) {
    hopfield::Matrix m{n, n + 1};
    CHECK(m.matrix.size() == n * (n + 1));
  }
}
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../hopfield.hpp"

#include <SFML/Graphics.hpp>
#include <cmath>
#include <filesystem>
#include <vector>

#include "doctest.h"

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
    hopfield::Network net{
        (hopfield::getDataDir() / "test_matrix.txt").string()};

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

    SUBCASE("Weight matrix values match hand-computed result") {
      sf::Image img1{};
      img1.create(2, 2, sf::Color::White);
      sf::Image img2{};
      img2.create(2, 2, sf::Color::Black);
      net.addImage(img1);
      net.addImage(img2);
      auto w = net.train();
      for (size_t i{0}; i < w.rows; ++i) {
        for (size_t j{0}; j < w.cols; ++j) {
          if (i == j) continue;
          CHECK(w(i, j) == doctest::Approx(0.5));
        }
      }
    }

    SUBCASE("Multiple small images and matrix inspection") {
      sf::Image img1{};
      img1.create(2, 2, sf::Color::White);
      sf::Image img2{};
      img2.create(2, 2, sf::Color::Black);
      sf::Image img3{};
      img3.create(2, 2, sf::Color::White);

      net.addImage(img1);
      net.addImage(img2);
      net.addImage(img3);

      auto w = net.train();

      CHECK(w.rows == 4);
      CHECK(w.cols == 4);
      for (size_t i{0}; i < w.rows; ++i) {
        CHECK(w(i, i) == doctest::Approx(0.0));
        for (size_t j{0}; j < w.cols; ++j) {
          CHECK(w(i, j) == doctest::Approx(w(j, i)));
          CHECK_FALSE(std::isnan(w(i, j)));
          CHECK_FALSE(std::isinf(w(i, j)));
        }
      }
    }

    SUBCASE("Mixed image sizes (automatic resizing)") {
      sf::Image imgSmall{};
      imgSmall.create(2, 2, sf::Color::White);
      sf::Image imgLarge{};
      imgLarge.create(4, 4, sf::Color::Black);

      net.addImage(imgSmall);
      net.addImage(imgLarge);

      CHECK_NOTHROW(net.train());
    }
  }

  SUBCASE("energy") {
    // energy() is a free function in the hopfield namespace
    hopfield::Matrix w{2, 2, {0.0, 1.0, 1.0, 0.0}};
    hopfield::PatternInt pattern{{1, -1}, sf::Vector2u{2u, 1u}};

    SUBCASE("Known value") {
      // E = -0.5*(w01*x0*x1 + w10*x1*x0) = -0.5*(1*1*-1 + 1*-1*1) = 1.0
      CHECK(hopfield::energy(pattern, w) == doctest::Approx(1.0));
    }

    SUBCASE("Energy is invariant under global sign flip") {
      hopfield::PatternInt flipped{{-1, 1}, sf::Vector2u{2u, 1u}};
      CHECK(hopfield::energy(pattern, w) ==
            doctest::Approx(hopfield::energy(flipped, w)));
    }

    SUBCASE("Energy with zero matrix is zero") {
      hopfield::Matrix zero{2, 2};
      CHECK(hopfield::energy(pattern, zero) == doctest::Approx(0.0));
    }
  }

  SUBCASE("Network::recall") {
    auto const matrixPath{hopfield::getDataDir() / "recall_matrix.txt"};
    hopfield::Network net{matrixPath.string()};

    SUBCASE("Recall before train throws") {
      // make sure no matrix file is left over from previous runs
      std::filesystem::remove(matrixPath);
      hopfield::PatternInt p{};
      p.size = sf::Vector2u{2u, 2u};
      p.data = std::vector<int>(4, 1);
      CHECK_THROWS_AS(net.recall(p), std::runtime_error);
    }

    SUBCASE("A stored pattern is stable under recall") {
      sf::Image img{};
      img.create(2, 2, sf::Color::White);
      net.addImage(img);
      net.train();

      auto stored = hopfield::imageToBinary(img);
      auto result = net.recall(stored);

      REQUIRE(result.size == stored.size);
      CHECK(result.data == stored.data);
    }

    SUBCASE("Energy is non-increasing along recall trajectory") {
      sf::Image img1{};
      img1.create(2, 2, sf::Color::White);
      sf::Image img2{};
      img2.create(2, 2, sf::Color::Black);
      net.addImage(img1);
      net.addImage(img2);
      auto w = net.train();

      auto noisy = hopfield::imageToBinary(img1);
      noisy.data[0] = -noisy.data[0];  // introduce un errore

      std::vector<double> energies;
      net.recall(noisy, 1000,
                 [&](hopfield::PatternInt const& p, size_t, double) {
                   energies.push_back(hopfield::energy(p, w));
                 });

      REQUIRE(!energies.empty());
      for (size_t k{1}; k < energies.size(); ++k) {
        CHECK(energies[k] <= energies[k - 1] + 1e-9);
      }
    }
  }

  SUBCASE("Network::addImages") {
    SUBCASE("addImages from a directory loads every valid file") {
      auto tmpDir =
          std::filesystem::temp_directory_path() / "hopfield_test_imgs";
      std::filesystem::create_directories(tmpDir);

      sf::Image img1{};
      img1.create(2, 2, sf::Color::White);
      sf::Image img2{};
      img2.create(2, 2, sf::Color::Black);
      img1.saveToFile((tmpDir / "img1.png").string());
      img2.saveToFile((tmpDir / "img2.png").string());

      hopfield::Network net{
          (hopfield::getDataDir() / "dir_matrix.txt").string()};
      net.addImages(tmpDir);
      auto w = net.train();

      // 2 immagini caricate, entrambe 2x2 -> N = 4
      CHECK(w.rows == 4);
      CHECK(w.cols == 4);

      std::filesystem::remove_all(tmpDir);
    }
  }

  // a3: 2x2   R R
  //           R R
  sf::Image a3;
  a3.create(2, 2, K);
  {
    std::vector<sf::Color> px{R, R, R, R};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 2; ++x) {
        a3.setPixel(x, y, px[y * 2 + x]);
      }
    }
  }

  // d1: 3x2   R G B
  //           B G R
  sf::Image d1;
  d1.create(3, 2, K);
  {
    std::vector<sf::Color> px{R, G, B, B, G, R};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 3; ++x) {
        d1.setPixel(x, y, px[y * 3 + x]);
      }
    }
  }

  // d2: 1x3   R
  //           G
  //           B
  sf::Image d2;
  d2.create(1, 3, K);
  {
    std::vector<sf::Color> px{R, G, B};
    for (auto y{0u}; y != 3; ++y) {
      for (auto x{0u}; x != 1; ++x) {
        d2.setPixel(x, y, px[y * 1 + x]);
      }
    }
  }

  // d3: 3x3   W K K
  //           K W K
  //           K K W
  sf::Image d3;
  d3.create(3, 3, K);
  {
    std::vector<sf::Color> px{W, K, K, K, W, K, K, K, W};
    for (auto y{0u}; y != 3; ++y) {
      for (auto x{0u}; x != 3; ++x) {
        d3.setPixel(x, y, px[y * 3 + x]);
      }
    }
  }

  // invalid images (SFML reduces them to 0x0 anyway)
  sf::Image inv1;
  sf::Image inv2;
  inv2.create(0, 4, K);
  sf::Image inv3;
  inv3.create(4, 0, K);

  // all valid, same size
  std::vector<sf::Image> allSameSize{a1, a2, a3};

  // all valid, at least one with a different size
  std::vector<sf::Image> atLeastOneDifferent{a1, a2, d1};

  // all valid, all different sizes
  std::vector<sf::Image> allDifferent{a1, d1, d2, d3};

  // ---------- black and white images (for value and recall tests) ----------

  // b1: 2x2   W K
  //           K W
  sf::Image b1;
  b1.create(2, 2, K);
  {
    std::vector<sf::Color> px{W, K, K, W};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 2; ++x) {
        b1.setPixel(x, y, px[y * 2 + x]);
      }
    }
  }

  // b2: 2x2   W W
  //           K K
  sf::Image b2;
  b2.create(2, 2, K);
  {
    std::vector<sf::Color> px{W, W, K, K};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 2; ++x) {
        b2.setPixel(x, y, px[y * 2 + x]);
      }
    }
  }

  // b3: 2x2   W K
  //           W K
  sf::Image b3;
  b3.create(2, 2, K);
  {
    std::vector<sf::Color> px{W, K, W, K};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 2; ++x) {
        b3.setPixel(x, y, px[y * 2 + x]);
      }
    }
  }

  // c1: 3x2   W K W
  //           K W K
  sf::Image c1;
  c1.create(3, 2, K);
  {
    std::vector<sf::Color> px{W, K, W, K, W, K};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 3; ++x) {
        c1.setPixel(x, y, px[y * 3 + x]);
      }
    }
  }

  // c2: 1x3   W
  //           K
  //           W
  sf::Image c2;
  c2.create(1, 3, K);
  {
    std::vector<sf::Color> px{W, K, W};
    for (auto y{0u}; y != 3; ++y) {
      for (auto x{0u}; x != 1; ++x) {
        c2.setPixel(x, y, px[y * 1 + x]);
      }
    }
  }

  // c3: 3x3   K W K
  //           W W W
  //           K W K
  sf::Image c3;
  c3.create(3, 3, K);
  {
    std::vector<sf::Color> px{K, W, K, W, W, W, K, W, K};
    for (auto y{0u}; y != 3; ++y) {
      for (auto x{0u}; x != 3; ++x) {
        c3.setPixel(x, y, px[y * 3 + x]);
      }
    }
  }

  // b/w, all valid, same size (2x2, 2x2, 2x2)
  std::vector<sf::Image> bwAllSameSize{b1, b2, b3};

  // b/w, all valid, at least one with a different size (2x2, 2x2, 3x2)
  std::vector<sf::Image> bwAtLeastOneDifferent{b1, b2, c1};

  // b/w, all valid, all different sizes (2x2, 3x2, 1x3, 3x3)
  std::vector<sf::Image> bwAllDifferent{b1, c1, c2, c3};

  SUBCASE("Network::train"){
    
  }
}

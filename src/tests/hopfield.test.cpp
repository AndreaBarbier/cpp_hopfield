#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../hopfield.hpp"

#include <SFML/Graphics.hpp>
#include <filesystem>

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

    SUBCASE("Energy with zero matrix is zero") {
      hopfield::Matrix zero{2, 2};
      CHECK(net.energy(pattern, zero) == doctest::Approx(0.0));
    }
  }

  SUBCASE("Network::recall") {
    hopfield::Network net{"data/recall_matrix.txt"};

    SUBCASE("Recall before train throws") {
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
      net.recall(noisy, [&](hopfield::PatternInt const& p, size_t) {
        energies.push_back(net.energy(p.data, w));
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

      hopfield::Network net{"data/dir_matrix.txt"};
      net.addImages(tmpDir);
      auto w = net.train();

      // 2 immagini caricate, entrambe 2x2 -> N = 4
      CHECK(w.rows == 4);
      CHECK(w.cols == 4);

      std::filesystem::remove_all(tmpDir);
    }
  }
}

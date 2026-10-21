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
  const sf::Color R = sf::Color::Red;
  const sf::Color G = sf::Color::Green;
  const sf::Color B = sf::Color::Blue;
  const sf::Color W = sf::Color::White;
  const sf::Color K = sf::Color::Black;

  // a1: 2x2   R G
  //           B W
  sf::Image a1;
  a1.create(2, 2, K);
  {
    std::vector<sf::Color> px{R, G, B, W};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 2; ++x) {
        a1.setPixel(x, y, px[y * 2 + x]);
      }
    }
  }

  // a2: 2x2   K K
  //           K W
  sf::Image a2;
  a2.create(2, 2, K);
  {
    std::vector<sf::Color> px{K, K, K, W};
    for (auto y{0u}; y != 2; ++y) {
      for (auto x{0u}; x != 2; ++x) {
        a2.setPixel(x, y, px[y * 2 + x]);
      }
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

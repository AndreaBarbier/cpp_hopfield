#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

#include "tests/doctest.h"

namespace pf {
// ============================================================
// IMAGE UTILITIES
// ============================================================

/// @brief Resize an image using nearest-neighbor algorithm
/// @param[in] img Source image
/// @param[in] targetSize Target size: it should be <=the source image's size
/// @return Returns an image with targetSize
sf::Image resizeImage(sf::Image const& img, sf::Vector2u targetSize) {
  auto oldSize{img.getSize()};
  // Assert because it will not be a runtime error if the size is wrong
  assert(targetSize.x <= oldSize.x && targetSize.y <= oldSize.y &&
         targetSize.x > 0u && targetSize.y > 0u);
  sf::Image newImg;
  newImg.create(targetSize.x, targetSize.y);
  for (auto j{0u}; j != targetSize.y; ++j) {
    for (auto i{0u}; i != targetSize.x; ++i) {
      // To round correctly j_o and i_o, add half the denominator to the numerator
      auto j_o = ((j * oldSize.y + targetSize.y / 2) / targetSize.y);
      auto i_o = ((i * oldSize.x + targetSize.x / 2) / targetSize.x);
      newImg.setPixel(i, j, img.getPixel(i_o, j_o));
    }
  }
  return newImg;
}

bool isValidImage(sf::Image const& img){
  if(img.getSize().x == 0u && img.getSize().y == 0u){
    return false;
  }
  return true;
}

// ============================================================
// IMAGES PROCESSING
// ============================================================

/// @brief Generic container for 2D patterns
template <class T>
struct Pattern {
  std::vector<T> data;
  sf::Vector2u size;
};

/// @brief Specifc container for color patterns
using PatternRGB = Pattern<sf::Color>;

/// @brief Specific container for integer patterns
using PatternInt = Pattern<int>;

/// @brief Converts a sf::Image into a PatternRGB
/// @param[in] img Source image
/// @throws std::runtime_error if img.getSize() == (0u,0u)
/// @return Returns a PatternRGB
PatternRGB imageToColors(sf::Image const& img) {
  if(!isValidImage(img)){
    throw std::runtime_error("Error: you inserted an invald image.");
  }
  auto x{img.getSize().x};
  auto y{img.getSize().y};
  // Assert because it will not be a runtime error if the size is wrong
  assert(x > 0u && y > 0u);
  PatternRGB outPattern;
  for (auto j{0u}; j != img.getSize().y; ++j) {
    for (auto i{0u}; i != img.getSize().x; ++i) {
      outPattern.data.push_back(img.getPixel(i, j));
    }
  }
  return outPattern;
}

/// @brief Converts a PatternRGB into a PatternInt
/// @param[in] inPattern Input PatternRGB
/// @return Returns a PatternInt with +1 (brigth pixel) and -1 (dark pixel)
PatternInt colorsToBinary(PatternRGB const& inPattern) {
  // Assert because it will not be a runtime error if the size is wrong
  assert(inPattern.data.size() > 0 && inPattern.size.x > 0 &&
         inPattern.size.y > 0);
  PatternInt outPattern;
  for (sf::Color const& c : inPattern.data) {
    int grey = (static_cast<int>(c.r) + static_cast<int>(c.g) +
                static_cast<int>(c.b)) /
               3;
    if (grey > 127) {
      outPattern.data.push_back(1);
    } else {
      outPattern.data.push_back(-1);
    }
  }
  outPattern.size = inPattern.size;
  return outPattern;
}
}  // namespace pf

TEST_CASE("IMAGE UTILITIES") {
  SUBCASE("resizeImage") {
    sf::Image inImg;
    inImg.create(8u, 8u, sf::Color::Transparent);
    for (unsigned int j{0u}; j != 8; ++j) {
      for (unsigned int i{0u}; i != 8; ++i) {
        sf::Color coloreScelto;
        // Simple color map
        if (i % 2 == 0 && j % 2 == 0) {
          coloreScelto = sf::Color::Red;
        } else if (i % 2 == 0 && j % 2 != 0) {
          coloreScelto = sf::Color::Green;
        } else if (i % 2 != 0 && j % 2 == 0) {
          coloreScelto = sf::Color::Blue;
        } else {
          coloreScelto = sf::Color::Yellow;
        }

        inImg.setPixel(i, j, coloreScelto);
      }
    }
    SUBCASE("resizeImage -- n1") {
      sf::Image correctImg;
      correctImg.create(5u, 5u);
      sf::Color matriceCorretta[5][5] = {
          {sf::Color::Red, sf::Color::Red, sf::Color::Blue, sf::Color::Blue,
           sf::Color::Red},
          {sf::Color::Red, sf::Color::Red, sf::Color::Blue, sf::Color::Blue,
           sf::Color::Red},
          {sf::Color::Green, sf::Color::Green, sf::Color::Yellow,
           sf::Color::Yellow, sf::Color::Green},
          {sf::Color::Green, sf::Color::Green, sf::Color::Yellow,
           sf::Color::Yellow, sf::Color::Green},
          {sf::Color::Red, sf::Color::Red, sf::Color::Blue, sf::Color::Blue,
           sf::Color::Red}};
      for (unsigned int j{0u}; j != correctImg.getSize().y; ++j) {
        for (unsigned int i{0u}; i != correctImg.getSize().x; ++i) {
          correctImg.setPixel(i, j, matriceCorretta[j][i]);
        }
      }

      sf::Image outImg;
      outImg = pf::resizeImage(inImg, sf::Vector2u(5u, 5u));

      CHECK(outImg.getSize() == correctImg.getSize());
      for (unsigned int j{0u}; j != correctImg.getSize().y; ++j) {
        for (unsigned int i{0u}; i != correctImg.getSize().x; ++i) {
          CHECK(outImg.getPixel(i, j) == correctImg.getPixel(i, j));
        }
      }
    }
    SUBCASE("resizeImage -- n2"){
      sf::Image outImg;
      outImg = pf::resizeImage(inImg, sf::Vector2u(8u,8u));
      REQUIRE(outImg.getSize() == inImg.getSize());
      for (unsigned int j{0u}; j != inImg.getSize().y; ++j) {
        for (unsigned int i{0u}; i != inImg.getSize().x; ++i) {
          CHECK(outImg.getPixel(i, j) == inImg.getPixel(i, j));
        }
      }

    }
  }
}
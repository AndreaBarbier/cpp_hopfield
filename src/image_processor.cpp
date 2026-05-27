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

/// @brief Checks if an image has a valid size
/// @param[in] img Source image
/// @return Returns true if the image has a valid size. Else it returns false.
bool isValidImage(sf::Image const& img) {
  if (img.getSize().x == 0u || img.getSize().y == 0u) {
    return false;
  }
  return true;
}

/// @brief Checks an images vector for any that do not have a valid size
/// @param[in] inImgs Images vector
/// @return Retuns an vector of valid size images
std::vector<sf::Image> validateImages(std::vector<sf::Image> const& inImgs) {
  auto outImgs{inImgs};
  std::erase_if(outImgs, [](auto const& img) { return !isValidImage(img); });
  return outImgs;
}

/// @brief Resizes an image using nearest-neighbor algorithm
/// @param[in] img Source image
/// @param[in] targetSize Target size: it should be <=the source image's size
/// @return Returns an image with targetSize
sf::Image resizeImage(sf::Image const& img, sf::Vector2u targetSize) {
  auto oldSize{img.getSize()};
  // Assert because it will not be a runtime error if the size is wrong
  assert(targetSize.x <= oldSize.x && targetSize.y <= oldSize.y &&
         isValidImage(img));
  sf::Image newImg;
  newImg.create(targetSize.x, targetSize.y);
  for (auto j{0u}; j != targetSize.y; ++j) {
    for (auto i{0u}; i != targetSize.x; ++i) {
      // To round correctly j_o and i_o, add half the denominator to the
      // numerator
      auto j_o = ((j * oldSize.y + targetSize.y / 2) / targetSize.y);
      auto i_o = ((i * oldSize.x + targetSize.x / 2) / targetSize.x);
      newImg.setPixel(i, j, img.getPixel(i_o, j_o));
    }
  }
  return newImg;
}

/// @brief Resizes all the images inside a vector
/// @param[in] inImgs Images vector
/// @return Returns a vector with all the valid images scaled
std::vector<sf::Image> resizeImages(std::vector<sf::Image> const& inImgs) {
  auto validImgs{validateImages(inImgs)};
  if (validImgs.empty()) {
    throw std::runtime_error(
        "Error: empty vector inserted or full of invalid images.\n");
  }
  for (auto const& img : validImgs) {
    assert(isValidImage(img));
  }
  auto minWidth{
      (*std::min_element(validImgs.begin(), validImgs.end(),
                         [](sf::Image const& left, sf::Image const& right) {
                           return left.getSize().x < right.getSize().x;
                         }))
          .getSize()
          .x};
  auto minHeight{
      (*std::min_element(validImgs.begin(), validImgs.end(),
                         [](sf::Image const& left, sf::Image const& right) {
                           return left.getSize().y < right.getSize().y;
                         }))
          .getSize()
          .y};
  std::vector<sf::Image> outImgs;
  for (auto const& img : validImgs) {
    auto resizeImg{resizeImage(img, sf::Vector2u{minWidth, minHeight})};
    outImgs.push_back(resizeImg);
  }
  return outImgs;
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
  auto x{img.getSize().x};
  auto y{img.getSize().y};
  // Assert because it will not be a runtime error if the size is wrong
  assert(x > 0u && y > 0u);
  PatternRGB outPattern;
  for (auto j{0u}; j != y; ++j) {
    for (auto i{0u}; i != x; ++i) {
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
    auto grey = (c.r + c.g + c.b) / 3u;
    if (grey > 127u) {
      outPattern.data.push_back(1);
    } else {
      outPattern.data.push_back(-1);
    }
  }
  outPattern.size = inPattern.size;
  return outPattern;
}

PatternInt imageToBinary(sf::Image const& img){
  return colorsToBinary(imageToColors(img));
}
}  // namespace pf

// ============================================================
// IMAGE UTILITIES TEST CASE
// ============================================================
TEST_CASE("IMAGE UTILITIES") {
  // --- Some images ---
  sf::Image imgEmpty;
  imgEmpty.create(0u, 0u);
  sf::Image imgZeroW;
  imgZeroW.create(0u, 8u);
  sf::Image imgZeroH;
  imgZeroH.create(8u, 0u);
  sf::Image imgValid;
  imgValid.create(3u, 4u);
  sf::Image imgCyan5x5;
  imgCyan5x5.create(5u, 5u, sf::Color::Cyan);
  sf::Image imgCyan0x0;
  imgCyan0x0.create(0u, 0u, sf::Color::Cyan);
  sf::Image imgCyan0x3;
  imgCyan0x3.create(0u, 3u, sf::Color::Cyan);
  sf::Image imgCyan3x0;
  imgCyan3x0.create(3u, 0u, sf::Color::Cyan);

  // 8x8 image
  sf::Image img8x8;
  img8x8.create(8u, 8u);
  for (unsigned int j{0u}; j != 8u; ++j) {
    for (unsigned int i{0u}; i != 8u; ++i) {
      sf::Color color;
      if (i % 2 == 0 && j % 2 == 0)
        color = sf::Color::Red;
      else if (i % 2 == 0 && j % 2 != 0)
        color = sf::Color::Green;
      else if (i % 2 != 0 && j % 2 == 0)
        color = sf::Color::Blue;
      else
        color = sf::Color::Yellow;
      img8x8.setPixel(i, j, color);
    }
  }

  // Correct resized 8x8 image into a 5x5 image
  sf::Color correctColors[5][5] = {
      {sf::Color::Red, sf::Color::Red, sf::Color::Blue, sf::Color::Blue,
       sf::Color::Red},
      {sf::Color::Red, sf::Color::Red, sf::Color::Blue, sf::Color::Blue,
       sf::Color::Red},
      {sf::Color::Green, sf::Color::Green, sf::Color::Yellow, sf::Color::Yellow,
       sf::Color::Green},
      {sf::Color::Green, sf::Color::Green, sf::Color::Yellow, sf::Color::Yellow,
       sf::Color::Green},
      {sf::Color::Red, sf::Color::Red, sf::Color::Blue, sf::Color::Blue,
       sf::Color::Red}};
  sf::Image correctImg5x5;
  correctImg5x5.create(5u, 5u, sf::Color::Transparent);
  for (unsigned int j{0u}; j != 5u; ++j) {
    for (unsigned int i{0u}; i != 5u; ++i) {
      correctImg5x5.setPixel(i, j, correctColors[j][i]);
    }
  }

  // Simple image validation tests
  SUBCASE("isValidImage") {
    CHECK(!pf::isValidImage(imgEmpty));
    CHECK(!pf::isValidImage(imgZeroW));
    CHECK(!pf::isValidImage(imgZeroH));
    CHECK(pf::isValidImage(imgValid));
  }

  // Simple image vectors validation tests
  SUBCASE("validateImages") {
    std::vector<sf::Image> imgs;
    imgs.push_back(imgEmpty);
    imgs.push_back(imgZeroW);
    imgs.push_back(imgZeroH);
    imgs.push_back(imgValid);

    auto outImgs{pf::validateImages(imgs)};
    REQUIRE(outImgs.size() == 1);
    for (unsigned int j{0u}; j != outImgs[0].getSize().y; ++j) {
      for (unsigned int i{0u}; i != outImgs[0].getSize().x; ++i) {
        CHECK(outImgs[0].getPixel(i, j) == imgValid.getPixel(i, j));
      }
    }
  }

  // Single image resize tests
  SUBCASE("resizeImage") {
    SUBCASE("resizeImage -- n1") {
      sf::Image outImg = pf::resizeImage(img8x8, sf::Vector2u(5u, 5u));

      CHECK(outImg.getSize() == correctImg5x5.getSize());
      for (unsigned int j{0u}; j != correctImg5x5.getSize().y; ++j) {
        for (unsigned int i{0u}; i != correctImg5x5.getSize().x; ++i) {
          CHECK(outImg.getPixel(i, j) == correctImg5x5.getPixel(i, j));
        }
      }
    }

    SUBCASE("resizeImage -- n2") {
      sf::Image outImg = pf::resizeImage(img8x8, sf::Vector2u(8u, 8u));

      REQUIRE(outImg.getSize() == img8x8.getSize());
      for (unsigned int j{0u}; j != img8x8.getSize().y; ++j) {
        for (unsigned int i{0u}; i != img8x8.getSize().x; ++i) {
          CHECK(outImg.getPixel(i, j) == img8x8.getPixel(i, j));
        }
      }
    }
  }

  // Multiple image resize tests
  SUBCASE("resizeImages") {
    SUBCASE("resizeImages -- n1") {
      std::vector<sf::Image> inImgs;
      inImgs.push_back(img8x8);
      inImgs.push_back(imgCyan5x5);

      std::vector<sf::Image> solutions;
      solutions.push_back(correctImg5x5);
      solutions.push_back(imgCyan5x5);

      auto outImgs{pf::resizeImages(inImgs)};

      REQUIRE(!outImgs.empty());
      REQUIRE(outImgs.size() == 2);
      for (size_t k{0}; k != outImgs.size(); ++k) {
        REQUIRE(outImgs[k].getSize() == solutions[k].getSize());
        for (unsigned int j{0u}; j != solutions[k].getSize().y; ++j) {
          for (unsigned int i{0u}; i != solutions[k].getSize().x; ++i) {
            CHECK(outImgs[k].getPixel(i, j) == solutions[k].getPixel(i, j));
          }
        }
      }
    }
    SUBCASE("resizeImages -- n2") {
      std::vector<sf::Image> inImgs;
      inImgs.push_back(imgCyan0x0);
      inImgs.push_back(imgCyan0x3);
      inImgs.push_back(imgCyan3x0);
      inImgs.push_back(imgCyan5x5);

      auto outImgs{pf::resizeImages(inImgs)};

      REQUIRE(!outImgs.empty());
      REQUIRE(outImgs.size() == 1);
      for (auto const& img : outImgs) {
        REQUIRE(img.getSize() != sf::Vector2u{0u, 0u});
        for (unsigned int j{0u}; j != img.getSize().y; ++j) {
          for (unsigned int i{0u}; i != img.getSize().x; ++i) {
            CHECK(img.getPixel(i, j) == imgCyan5x5.getPixel(i, j));
          }
        }
      }
    }
  }
}
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../image_processor.hpp"

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"

namespace {

std::filesystem::path makeTempDir(std::string const& label) {
  static std::size_t counter = 0;
  auto const path =
      std::filesystem::temp_directory_path() /
      ("hopfield_imgproc_" + label + "_" + std::to_string(++counter));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

sf::Image makeSolidImage(unsigned int width, unsigned int height,
                        sf::Color color) {
  return sf::Image{sf::Vector2u{width, height}, color};
}

sf::Image makeCheckerboard(unsigned int width, unsigned int height,
                           sf::Color evenColor, sf::Color oddColor) {
  sf::Image img{sf::Vector2u{width, height}, oddColor};
  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      if (((x + y) % 2u) == 0u) {
        img.setPixel(sf::Vector2u{x, y}, evenColor);
      }
    }
  }
  return img;
}

sf::Image makeHorizontalBands(unsigned int width, unsigned int height,
                               sf::Color topColor, sf::Color bottomColor) {
  sf::Image img{sf::Vector2u{width, height}, topColor};
  for (unsigned int y = height / 2u; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      img.setPixel(sf::Vector2u{x, y}, bottomColor);
    }
  }
  return img;
}

sf::Image makeVerticalBands(unsigned int width, unsigned int height,
                              sf::Color leftColor, sf::Color rightColor) {
  sf::Image img{sf::Vector2u{width, height}, leftColor};
  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = width / 2u; x < width; ++x) {
      img.setPixel(sf::Vector2u{x, y}, rightColor);
    }
  }
  return img;
}

void checkImagesEqual(sf::Image const& actual, sf::Image const& expected) {
  REQUIRE(actual.getSize() == expected.getSize());
  for (unsigned int y = 0; y < expected.getSize().y; ++y) {
    for (unsigned int x = 0; x < expected.getSize().x; ++x) {
      CHECK(actual.getPixel(sf::Vector2u{x, y}) ==
            expected.getPixel(sf::Vector2u{x, y}));
    }
  }
}

}  // namespace

// ============================================================
// IMAGE UTILITIES TEST CASE
// ============================================================
TEST_CASE("IMAGE UTILITIES") {
  // --- Some images ---
  sf::Image imgEmpty{sf::Vector2u{0u, 0u}};
  sf::Image imgZeroW{sf::Vector2u{0u, 8u}};
  sf::Image imgZeroH{sf::Vector2u{8u, 0u}};
  sf::Image imgValid{sf::Vector2u{3u, 4u}};
  sf::Image imgCyan5x5{sf::Vector2u{5u, 5u}, sf::Color::Cyan};
  sf::Image imgCyan0x0{sf::Vector2u{0u, 0u}, sf::Color::Cyan};
  sf::Image imgCyan0x3{sf::Vector2u{0u, 3u}, sf::Color::Cyan};
  sf::Image imgCyan3x0{sf::Vector2u{3u, 0u}, sf::Color::Cyan};

  // 8x8 image
  sf::Image img8x8{sf::Vector2u{8u, 8u}};
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
      img8x8.setPixel(sf::Vector2u{i, j}, color);
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
  sf::Image correctImg5x5{sf::Vector2u{5u, 5u}, sf::Color::Transparent};
  for (unsigned int j{0u}; j != 5u; ++j) {
    for (unsigned int i{0u}; i != 5u; ++i) {
      correctImg5x5.setPixel(sf::Vector2u{i, j}, correctColors[j][i]);
    }
  }

  // Simple image validation tests
  SUBCASE("isValidImage") {
    CHECK(!hopfield::isValidImage(imgEmpty));
    CHECK(!hopfield::isValidImage(imgZeroW));
    CHECK(!hopfield::isValidImage(imgZeroH));
    CHECK(hopfield::isValidImage(imgValid));
  }

  // Simple image vectors validation tests
  SUBCASE("validateImages") {
    std::vector<sf::Image> imgs;
    imgs.push_back(imgEmpty);
    imgs.push_back(imgZeroW);
    imgs.push_back(imgZeroH);
    imgs.push_back(imgValid);

    auto outImgs{hopfield::validateImages(imgs)};
    REQUIRE(outImgs.size() == 1);
    for (unsigned int j{0u}; j != outImgs[0].getSize().y; ++j) {
      for (unsigned int i{0u}; i != outImgs[0].getSize().x; ++i) {
        CHECK(outImgs[0].getPixel(sf::Vector2u{i, j}) ==
              imgValid.getPixel(sf::Vector2u{i, j}));
      }
    }
  }

  // Single image resize tests
  SUBCASE("resizeImage") {
    SUBCASE("resizeImage -- n1") {
      sf::Image outImg = hopfield::resizeImage(img8x8, sf::Vector2u(5u, 5u));

      CHECK(outImg.getSize() == correctImg5x5.getSize());
      for (unsigned int j{0u}; j != correctImg5x5.getSize().y; ++j) {
        for (unsigned int i{0u}; i != correctImg5x5.getSize().x; ++i) {
          CHECK(outImg.getPixel(sf::Vector2u{i, j}) ==
                correctImg5x5.getPixel(sf::Vector2u{i, j}));
        }
      }
    }

    SUBCASE("resizeImage -- n2") {
      sf::Image outImg = hopfield::resizeImage(img8x8, sf::Vector2u(8u, 8u));

      REQUIRE(outImg.getSize() == img8x8.getSize());
      for (unsigned int j{0u}; j != img8x8.getSize().y; ++j) {
        for (unsigned int i{0u}; i != img8x8.getSize().x; ++i) {
          CHECK(outImg.getPixel(sf::Vector2u{i, j}) ==
                img8x8.getPixel(sf::Vector2u{i, j}));
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

      auto outImgs{hopfield::resizeImages(inImgs)};

      REQUIRE(!outImgs.empty());
      REQUIRE(outImgs.size() == 2);
      for (size_t k{0}; k != outImgs.size(); ++k) {
        REQUIRE(outImgs[k].getSize() == solutions[k].getSize());
        for (unsigned int j{0u}; j != solutions[k].getSize().y; ++j) {
          for (unsigned int i{0u}; i != solutions[k].getSize().x; ++i) {
            CHECK(outImgs[k].getPixel(sf::Vector2u{i, j}) ==
                  solutions[k].getPixel(sf::Vector2u{i, j}));
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

      auto outImgs{hopfield::resizeImages(inImgs)};

      REQUIRE(!outImgs.empty());
      REQUIRE(outImgs.size() == 1);
      for (auto const& img : outImgs) {
        REQUIRE(img.getSize() != sf::Vector2u{0u, 0u});
        for (unsigned int j{0u}; j != img.getSize().y; ++j) {
          for (unsigned int i{0u}; i != img.getSize().x; ++i) {
            CHECK(img.getPixel(sf::Vector2u{i, j}) ==
                  imgCyan5x5.getPixel(sf::Vector2u{i, j}));
          }
        }
      }
    }
    SUBCASE("resizeImages -- n3") {
      std::vector<sf::Image> inImgs{};

      REQUIRE(inImgs.empty());
      CHECK_THROWS(hopfield::resizeImages(inImgs));
    }
    SUBCASE("resizeImages -- n4") {
      std::vector<sf::Image> inImgs;
      inImgs.push_back(imgCyan0x0);
      inImgs.push_back(imgCyan0x3);
      inImgs.push_back(imgCyan3x0);

      REQUIRE(!inImgs.empty());
      CHECK_THROWS(hopfield::resizeImages(inImgs));
    }
  }
}

// ============================================================
// IMAGE PROCESSING TEST CASE
// ============================================================
TEST_CASE("IMAGE PROCESSING") {
  // ----- IMAGES -----
  sf::Image allWhite{sf::Vector2u{8u, 8u}, sf::Color::White};

  sf::Image allBlack{sf::Vector2u{8u, 8u}, sf::Color::Black};

  sf::Image blackWhiteChecker{sf::Vector2u{8u, 8u}};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      sf::Color pixelColor =
          ((j + i) % 2 == 0) ? sf::Color::White : sf::Color::Black;
      blackWhiteChecker.setPixel(sf::Vector2u{i, j}, pixelColor);
    }
  }

  const sf::Color colorA{220u, 50u, 50u};   // rosso
  const sf::Color colorB{50u, 180u, 80u};   // verde
  const sf::Color colorC{40u, 90u, 210u};   // blu
  const sf::Color colorD{210u, 160u, 30u};  // giallo-arancio

  sf::Image rgbChecker{sf::Vector2u{8u, 8u}};
  for (unsigned int j{0u}; j != 8u; ++j) {
    for (unsigned int i{0u}; i != 8u; ++i) {
      sf::Color pixelColor;
      if (i % 2 == 0 && j % 2 == 0)
        pixelColor = colorA;
      else if (i % 2 == 0 && j % 2 != 0)
        pixelColor = colorB;
      else if (i % 2 != 0 && j % 2 == 0)
        pixelColor = colorC;
      else
        pixelColor = colorD;
      rgbChecker.setPixel(sf::Vector2u{i, j}, pixelColor);
    }
  }

  // ----- COLOR PATTERNS -----
  hopfield::PatternRGB expectedAllWhiteColors;
  expectedAllWhiteColors.size = sf::Vector2u{8u, 8u};
  expectedAllWhiteColors.data = std::vector<sf::Color>(64, sf::Color::White);

  hopfield::PatternRGB expectedAllBlackColors;
  expectedAllBlackColors.size = sf::Vector2u{8u, 8u};
  expectedAllBlackColors.data = std::vector<sf::Color>(64, sf::Color::Black);

  hopfield::PatternRGB expectedBlackWhiteCheckerColors;
  expectedBlackWhiteCheckerColors.size = sf::Vector2u{8u, 8u};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      expectedBlackWhiteCheckerColors.data.push_back(
          ((i + j) % 2 == 0) ? sf::Color::White : sf::Color::Black);
    }
  }

  hopfield::PatternRGB expectedRgbCheckerColors;
  expectedRgbCheckerColors.size = sf::Vector2u{8u, 8u};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      sf::Color c;
      if (i % 2 == 0 && j % 2 == 0)
        c = colorA;
      else if (i % 2 == 0 && j % 2 != 0)
        c = colorB;
      else if (i % 2 != 0 && j % 2 == 0)
        c = colorC;
      else
        c = colorD;
      expectedRgbCheckerColors.data.push_back(c);
    }
  }

  // ----- BINARY PATTERNS -----
  hopfield::PatternInt expectedAllWhiteBinary;
  expectedAllWhiteBinary.size = sf::Vector2u{8u, 8u};
  expectedAllWhiteBinary.data = std::vector<int>(64, 1);

  hopfield::PatternInt expectedAllBlackBinary;
  expectedAllBlackBinary.size = sf::Vector2u{8u, 8u};
  expectedAllBlackBinary.data = std::vector<int>(64, -1);

  hopfield::PatternInt expectedBlackWhiteCheckerBinary;
  expectedBlackWhiteCheckerBinary.size = sf::Vector2u{8u, 8u};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      expectedBlackWhiteCheckerBinary.data.push_back(((i + j) % 2 == 0) ? 1
                                                                        : -1);
    }
  }

  hopfield::PatternInt expectedRgbCheckerBinary;
  expectedRgbCheckerBinary.size = {8u, 8u};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      expectedRgbCheckerBinary.data.push_back((i % 2 != 0 && j % 2 != 0) ? 1
                                                                         : -1);
    }
  }

  SUBCASE("imageToColors") {
    auto allWhiteColors{hopfield::imageToColors(allWhite)};
    auto allBlackColors{hopfield::imageToColors(allBlack)};
    auto blackWhiteCheckerColors{hopfield::imageToColors(blackWhiteChecker)};
    auto rgbCheckerColors{hopfield::imageToColors(rgbChecker)};
    REQUIRE(allWhiteColors.size == expectedAllWhiteColors.size);
    REQUIRE(allWhiteColors.data.size() == expectedAllWhiteColors.data.size());
    REQUIRE(allBlackColors.size == expectedAllBlackColors.size);
    REQUIRE(allBlackColors.data.size() == expectedAllBlackColors.data.size());
    REQUIRE(blackWhiteCheckerColors.size ==
            expectedBlackWhiteCheckerColors.size);
    REQUIRE(blackWhiteCheckerColors.data.size() ==
            expectedBlackWhiteCheckerColors.data.size());
    REQUIRE(rgbCheckerColors.size == expectedRgbCheckerColors.size);
    REQUIRE(rgbCheckerColors.data.size() ==
            expectedRgbCheckerColors.data.size());
    for (size_t i{0}; i != allWhiteColors.data.size(); ++i) {
      CHECK(allWhiteColors.data[i] == expectedAllWhiteColors.data[i]);
    }
    for (size_t i{0}; i != allBlackColors.data.size(); ++i) {
      CHECK(allBlackColors.data[i] == expectedAllBlackColors.data[i]);
    }
    for (size_t i{0}; i != blackWhiteCheckerColors.data.size(); ++i) {
      CHECK(blackWhiteCheckerColors.data[i] ==
            expectedBlackWhiteCheckerColors.data[i]);
    }
    for (size_t i{0}; i != rgbCheckerColors.data.size(); ++i) {
      CHECK(rgbCheckerColors.data[i] == expectedRgbCheckerColors.data[i]);
    }
  }
  SUBCASE("imageToBinary") {
    auto allWhiteBinary{hopfield::imageToBinary(allWhite)};
    auto allBlackBinary{hopfield::imageToBinary(allBlack)};
    auto blackWhiteCheckerBinary{hopfield::imageToBinary(blackWhiteChecker)};
    auto rgbCheckerBinary{hopfield::imageToBinary(rgbChecker)};
    REQUIRE(allWhiteBinary.size == expectedAllWhiteBinary.size);
    REQUIRE(allWhiteBinary.data.size() == expectedAllWhiteBinary.data.size());
    REQUIRE(allBlackBinary.size == expectedAllBlackBinary.size);
    REQUIRE(allBlackBinary.data.size() == expectedAllBlackBinary.data.size());
    REQUIRE(blackWhiteCheckerBinary.size ==
            expectedBlackWhiteCheckerBinary.size);
    REQUIRE(blackWhiteCheckerBinary.data.size() ==
            expectedBlackWhiteCheckerBinary.data.size());
    REQUIRE(rgbCheckerBinary.size == expectedRgbCheckerBinary.size);
    REQUIRE(rgbCheckerBinary.data.size() ==
            expectedRgbCheckerBinary.data.size());
    for (size_t i{0}; i != allWhiteBinary.data.size(); ++i) {
      CHECK(allWhiteBinary.data[i] == expectedAllWhiteBinary.data[i]);
    }
    for (size_t i{0}; i != allBlackBinary.data.size(); ++i) {
      CHECK(allBlackBinary.data[i] == expectedAllBlackBinary.data[i]);
    }
    for (size_t i{0}; i != blackWhiteCheckerBinary.data.size(); ++i) {
      CHECK(blackWhiteCheckerBinary.data[i] ==
            expectedBlackWhiteCheckerBinary.data[i]);
    }
    for (size_t i{0}; i != rgbCheckerBinary.data.size(); ++i) {
      CHECK(rgbCheckerBinary.data[i] == expectedRgbCheckerBinary.data[i]);
    }
  }
  SUBCASE("imageToBinaries") {
    SUBCASE("imageToBinaries -- n1") {
      std::vector<sf::Image> imgs{allWhite, allBlack, blackWhiteChecker,
                                  rgbChecker};
      auto binaries{hopfield::imageToBinaries(imgs)};
      REQUIRE(binaries.size() == 4);
      REQUIRE(binaries[0].size == expectedAllWhiteBinary.size);
      REQUIRE(binaries[0].data.size() == expectedAllWhiteBinary.data.size());
      REQUIRE(binaries[1].size == expectedAllBlackBinary.size);
      REQUIRE(binaries[1].data.size() == expectedAllBlackBinary.data.size());
      REQUIRE(binaries[2].size == expectedBlackWhiteCheckerBinary.size);
      REQUIRE(binaries[2].data.size() ==
              expectedBlackWhiteCheckerBinary.data.size());
      REQUIRE(binaries[3].size == expectedRgbCheckerBinary.size);
      REQUIRE(binaries[3].data.size() == expectedRgbCheckerBinary.data.size());
      for (size_t i{0}; i != binaries[0].data.size(); ++i) {
        CHECK(binaries[0].data[i] == expectedAllWhiteBinary.data[i]);
      }
      for (size_t i{0}; i != binaries[1].data.size(); ++i) {
        CHECK(binaries[1].data[i] == expectedAllBlackBinary.data[i]);
      }
      for (size_t i{0}; i != binaries[2].data.size(); ++i) {
        CHECK(binaries[2].data[i] == expectedBlackWhiteCheckerBinary.data[i]);
      }
      for (size_t i{0}; i != binaries[3].data.size(); ++i) {
        CHECK(binaries[3].data[i] == expectedRgbCheckerBinary.data[i]);
      }
    }
    SUBCASE("imageToBinaries -- n2") {
      std::vector<sf::Image> emptyImgs{};
      CHECK_THROWS(hopfield::imageToBinaries(emptyImgs));
    }
  }
  SUBCASE("binaryToColors") {
    auto allWhiteColors{hopfield::binaryToColors(expectedAllWhiteBinary)};
    auto allBlackColors{hopfield::binaryToColors(expectedAllBlackBinary)};
    auto blackWhiteCheckerColors{
        hopfield::binaryToColors(expectedBlackWhiteCheckerBinary)};
    auto rgbCheckerColors{hopfield::binaryToColors(expectedRgbCheckerBinary)};
    REQUIRE(allWhiteColors.size == expectedAllWhiteBinary.size);
    REQUIRE(allWhiteColors.data.size() == expectedAllWhiteBinary.data.size());
    REQUIRE(allBlackColors.size == expectedAllBlackBinary.size);
    REQUIRE(allBlackColors.data.size() == expectedAllBlackBinary.data.size());
    REQUIRE(blackWhiteCheckerColors.size ==
            expectedBlackWhiteCheckerBinary.size);
    REQUIRE(blackWhiteCheckerColors.data.size() ==
            expectedBlackWhiteCheckerBinary.data.size());
    REQUIRE(rgbCheckerColors.size == expectedRgbCheckerBinary.size);
    REQUIRE(rgbCheckerColors.data.size() ==
            expectedRgbCheckerBinary.data.size());
    for (size_t i{0}; i != allWhiteColors.data.size(); ++i) {
      CHECK(allWhiteColors.data[i] == sf::Color::White);
    }
    for (size_t i{0}; i != allBlackColors.data.size(); ++i) {
      CHECK(allBlackColors.data[i] == sf::Color::Black);
    }
    for (size_t i{0}; i != blackWhiteCheckerColors.data.size(); ++i) {
      CHECK(blackWhiteCheckerColors.data[i] ==
            expectedBlackWhiteCheckerColors.data[i]);
    }
    for (size_t i{0}; i != rgbCheckerColors.data.size(); ++i) {
      CHECK(rgbCheckerColors.data[i] == ((i % 2 != 0 && (i / 8) % 2 != 0)
                                             ? sf::Color::White
                                             : sf::Color::Black));
    }
  }
  SUBCASE("colorsToImage") {
    auto allWhiteImg{hopfield::colorsToImage(expectedAllWhiteColors)};
    auto allBlackImg{hopfield::colorsToImage(expectedAllBlackColors)};
    auto blackWhiteCheckerImg{
        hopfield::colorsToImage(expectedBlackWhiteCheckerColors)};
    REQUIRE(allWhiteImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(allBlackImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(blackWhiteCheckerImg.getSize() == sf::Vector2u{8u, 8u});
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allWhiteImg.getPixel(sf::Vector2u{i, j}) == sf::Color::White);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allBlackImg.getPixel(sf::Vector2u{i, j}) == sf::Color::Black);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(blackWhiteCheckerImg.getPixel(sf::Vector2u{i, j}) ==
              expectedBlackWhiteCheckerColors.data[j * 8u + i]);
      }
    }
  }
  SUBCASE("binaryToImage") {
    auto allWhiteImg{hopfield::binaryToImage(expectedAllWhiteBinary)};
    auto allBlackImg{hopfield::binaryToImage(expectedAllBlackBinary)};
    auto blackWhiteCheckerImg{
        hopfield::binaryToImage(expectedBlackWhiteCheckerBinary)};
    REQUIRE(allWhiteImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(allBlackImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(blackWhiteCheckerImg.getSize() == sf::Vector2u{8u, 8u});
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allWhiteImg.getPixel(sf::Vector2u{i, j}) == sf::Color::White);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allBlackImg.getPixel(sf::Vector2u{i, j}) == sf::Color::Black);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(blackWhiteCheckerImg.getPixel(sf::Vector2u{i, j}) ==
              expectedBlackWhiteCheckerColors.data[j * 8u + i]);
      }
    }
  }
}

// ============================================================
// EXTENDED VALIDATION, RESIZING & CONVERSION TESTS
// ============================================================
TEST_CASE("Image validation for 1x1 solid images") {
  auto white = makeSolidImage(1u, 1u, sf::Color::White);
  auto black = makeSolidImage(1u, 1u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 1u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{1u, 1u});
  CHECK(valid[1].getSize() == sf::Vector2u{1u, 1u});
}

TEST_CASE("Image validation for 1x2 solid images") {
  auto white = makeSolidImage(1u, 2u, sf::Color::White);
  auto black = makeSolidImage(1u, 2u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 2u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{1u, 2u});
  CHECK(valid[1].getSize() == sf::Vector2u{1u, 2u});
}

TEST_CASE("Image validation for 1x3 solid images") {
  auto white = makeSolidImage(1u, 3u, sf::Color::White);
  auto black = makeSolidImage(1u, 3u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 3u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{1u, 3u});
  CHECK(valid[1].getSize() == sf::Vector2u{1u, 3u});
}

TEST_CASE("Image validation for 1x4 solid images") {
  auto white = makeSolidImage(1u, 4u, sf::Color::White);
  auto black = makeSolidImage(1u, 4u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 4u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{1u, 4u});
  CHECK(valid[1].getSize() == sf::Vector2u{1u, 4u});
}

TEST_CASE("Image validation for 1x5 solid images") {
  auto white = makeSolidImage(1u, 5u, sf::Color::White);
  auto black = makeSolidImage(1u, 5u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 5u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{1u, 5u});
  CHECK(valid[1].getSize() == sf::Vector2u{1u, 5u});
}

TEST_CASE("Image validation for 1x6 solid images") {
  auto white = makeSolidImage(1u, 6u, sf::Color::White);
  auto black = makeSolidImage(1u, 6u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 6u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{1u, 6u});
  CHECK(valid[1].getSize() == sf::Vector2u{1u, 6u});
}

TEST_CASE("Image validation for 2x1 solid images") {
  auto white = makeSolidImage(2u, 1u, sf::Color::White);
  auto black = makeSolidImage(2u, 1u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 1u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{2u, 1u});
  CHECK(valid[1].getSize() == sf::Vector2u{2u, 1u});
}

TEST_CASE("Image validation for 2x2 solid images") {
  auto white = makeSolidImage(2u, 2u, sf::Color::White);
  auto black = makeSolidImage(2u, 2u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 2u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{2u, 2u});
  CHECK(valid[1].getSize() == sf::Vector2u{2u, 2u});
}

TEST_CASE("Image validation for 2x3 solid images") {
  auto white = makeSolidImage(2u, 3u, sf::Color::White);
  auto black = makeSolidImage(2u, 3u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 3u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{2u, 3u});
  CHECK(valid[1].getSize() == sf::Vector2u{2u, 3u});
}

TEST_CASE("Image validation for 2x4 solid images") {
  auto white = makeSolidImage(2u, 4u, sf::Color::White);
  auto black = makeSolidImage(2u, 4u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 4u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{2u, 4u});
  CHECK(valid[1].getSize() == sf::Vector2u{2u, 4u});
}

TEST_CASE("Image validation for 2x5 solid images") {
  auto white = makeSolidImage(2u, 5u, sf::Color::White);
  auto black = makeSolidImage(2u, 5u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 5u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{2u, 5u});
  CHECK(valid[1].getSize() == sf::Vector2u{2u, 5u});
}

TEST_CASE("Image validation for 2x6 solid images") {
  auto white = makeSolidImage(2u, 6u, sf::Color::White);
  auto black = makeSolidImage(2u, 6u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 6u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{2u, 6u});
  CHECK(valid[1].getSize() == sf::Vector2u{2u, 6u});
}

TEST_CASE("Image validation for 3x1 solid images") {
  auto white = makeSolidImage(3u, 1u, sf::Color::White);
  auto black = makeSolidImage(3u, 1u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 1u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{3u, 1u});
  CHECK(valid[1].getSize() == sf::Vector2u{3u, 1u});
}

TEST_CASE("Image validation for 3x2 solid images") {
  auto white = makeSolidImage(3u, 2u, sf::Color::White);
  auto black = makeSolidImage(3u, 2u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 2u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{3u, 2u});
  CHECK(valid[1].getSize() == sf::Vector2u{3u, 2u});
}

TEST_CASE("Image validation for 3x3 solid images") {
  auto white = makeSolidImage(3u, 3u, sf::Color::White);
  auto black = makeSolidImage(3u, 3u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 3u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{3u, 3u});
  CHECK(valid[1].getSize() == sf::Vector2u{3u, 3u});
}

TEST_CASE("Image validation for 3x4 solid images") {
  auto white = makeSolidImage(3u, 4u, sf::Color::White);
  auto black = makeSolidImage(3u, 4u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 4u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{3u, 4u});
  CHECK(valid[1].getSize() == sf::Vector2u{3u, 4u});
}

TEST_CASE("Image validation for 3x5 solid images") {
  auto white = makeSolidImage(3u, 5u, sf::Color::White);
  auto black = makeSolidImage(3u, 5u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 5u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{3u, 5u});
  CHECK(valid[1].getSize() == sf::Vector2u{3u, 5u});
}

TEST_CASE("Image validation for 3x6 solid images") {
  auto white = makeSolidImage(3u, 6u, sf::Color::White);
  auto black = makeSolidImage(3u, 6u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 6u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{3u, 6u});
  CHECK(valid[1].getSize() == sf::Vector2u{3u, 6u});
}

TEST_CASE("Image validation for 4x1 solid images") {
  auto white = makeSolidImage(4u, 1u, sf::Color::White);
  auto black = makeSolidImage(4u, 1u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 1u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{4u, 1u});
  CHECK(valid[1].getSize() == sf::Vector2u{4u, 1u});
}

TEST_CASE("Image validation for 4x2 solid images") {
  auto white = makeSolidImage(4u, 2u, sf::Color::White);
  auto black = makeSolidImage(4u, 2u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 2u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{4u, 2u});
  CHECK(valid[1].getSize() == sf::Vector2u{4u, 2u});
}

TEST_CASE("Image validation for 4x3 solid images") {
  auto white = makeSolidImage(4u, 3u, sf::Color::White);
  auto black = makeSolidImage(4u, 3u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 3u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{4u, 3u});
  CHECK(valid[1].getSize() == sf::Vector2u{4u, 3u});
}

TEST_CASE("Image validation for 4x4 solid images") {
  auto white = makeSolidImage(4u, 4u, sf::Color::White);
  auto black = makeSolidImage(4u, 4u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 4u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{4u, 4u});
  CHECK(valid[1].getSize() == sf::Vector2u{4u, 4u});
}

TEST_CASE("Image validation for 4x5 solid images") {
  auto white = makeSolidImage(4u, 5u, sf::Color::White);
  auto black = makeSolidImage(4u, 5u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 5u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{4u, 5u});
  CHECK(valid[1].getSize() == sf::Vector2u{4u, 5u});
}

TEST_CASE("Image validation for 4x6 solid images") {
  auto white = makeSolidImage(4u, 6u, sf::Color::White);
  auto black = makeSolidImage(4u, 6u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 6u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{4u, 6u});
  CHECK(valid[1].getSize() == sf::Vector2u{4u, 6u});
}

TEST_CASE("Image validation for 5x1 solid images") {
  auto white = makeSolidImage(5u, 1u, sf::Color::White);
  auto black = makeSolidImage(5u, 1u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 1u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{5u, 1u});
  CHECK(valid[1].getSize() == sf::Vector2u{5u, 1u});
}

TEST_CASE("Image validation for 5x2 solid images") {
  auto white = makeSolidImage(5u, 2u, sf::Color::White);
  auto black = makeSolidImage(5u, 2u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 2u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{5u, 2u});
  CHECK(valid[1].getSize() == sf::Vector2u{5u, 2u});
}

TEST_CASE("Image validation for 5x3 solid images") {
  auto white = makeSolidImage(5u, 3u, sf::Color::White);
  auto black = makeSolidImage(5u, 3u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 3u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{5u, 3u});
  CHECK(valid[1].getSize() == sf::Vector2u{5u, 3u});
}

TEST_CASE("Image validation for 5x4 solid images") {
  auto white = makeSolidImage(5u, 4u, sf::Color::White);
  auto black = makeSolidImage(5u, 4u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 4u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{5u, 4u});
  CHECK(valid[1].getSize() == sf::Vector2u{5u, 4u});
}

TEST_CASE("Image validation for 5x5 solid images") {
  auto white = makeSolidImage(5u, 5u, sf::Color::White);
  auto black = makeSolidImage(5u, 5u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 5u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{5u, 5u});
  CHECK(valid[1].getSize() == sf::Vector2u{5u, 5u});
}

TEST_CASE("Image validation for 5x6 solid images") {
  auto white = makeSolidImage(5u, 6u, sf::Color::White);
  auto black = makeSolidImage(5u, 6u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 6u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{5u, 6u});
  CHECK(valid[1].getSize() == sf::Vector2u{5u, 6u});
}

TEST_CASE("Image validation for 6x1 solid images") {
  auto white = makeSolidImage(6u, 1u, sf::Color::White);
  auto black = makeSolidImage(6u, 1u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 1u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{6u, 1u});
  CHECK(valid[1].getSize() == sf::Vector2u{6u, 1u});
}

TEST_CASE("Image validation for 6x2 solid images") {
  auto white = makeSolidImage(6u, 2u, sf::Color::White);
  auto black = makeSolidImage(6u, 2u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 2u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{6u, 2u});
  CHECK(valid[1].getSize() == sf::Vector2u{6u, 2u});
}

TEST_CASE("Image validation for 6x3 solid images") {
  auto white = makeSolidImage(6u, 3u, sf::Color::White);
  auto black = makeSolidImage(6u, 3u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 3u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{6u, 3u});
  CHECK(valid[1].getSize() == sf::Vector2u{6u, 3u});
}

TEST_CASE("Image validation for 6x4 solid images") {
  auto white = makeSolidImage(6u, 4u, sf::Color::White);
  auto black = makeSolidImage(6u, 4u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 4u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{6u, 4u});
  CHECK(valid[1].getSize() == sf::Vector2u{6u, 4u});
}

TEST_CASE("Image validation for 6x5 solid images") {
  auto white = makeSolidImage(6u, 5u, sf::Color::White);
  auto black = makeSolidImage(6u, 5u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 5u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{6u, 5u});
  CHECK(valid[1].getSize() == sf::Vector2u{6u, 5u});
}

TEST_CASE("Image validation for 6x6 solid images") {
  auto white = makeSolidImage(6u, 6u, sf::Color::White);
  auto black = makeSolidImage(6u, 6u, sf::Color::Black);
  CHECK(hopfield::isValidImage(white));
  CHECK(hopfield::isValidImage(black));
  std::vector<sf::Image> batch{white, sf::Image{sf::Vector2u{0u, 6u}}, black};
  auto valid = hopfield::validateImages(batch);
  REQUIRE(valid.size() == 2u);
  CHECK(valid[0].getSize() == sf::Vector2u{6u, 6u});
  CHECK(valid[1].getSize() == sf::Vector2u{6u, 6u});
}

TEST_CASE("Resize regression pair 01: 4x4 -> 2x2") {
  auto src = makeHorizontalBands(4u, 4u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{2u, 2u});
  CHECK(resized.getSize() == sf::Vector2u{2u, 2u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{2u, 2u});
  CHECK(bin.data.size() == static_cast<std::size_t>(2u) * 2u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 02: 6x8 -> 2x4") {
  auto src = makeHorizontalBands(6u, 8u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{2u, 4u});
  CHECK(resized.getSize() == sf::Vector2u{2u, 4u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{2u, 4u});
  CHECK(bin.data.size() == static_cast<std::size_t>(2u) * 4u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 03: 6x10 -> 3x5") {
  auto src = makeHorizontalBands(6u, 10u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{3u, 5u});
  CHECK(resized.getSize() == sf::Vector2u{3u, 5u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{3u, 5u});
  CHECK(bin.data.size() == static_cast<std::size_t>(3u) * 5u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 04: 5x5 -> 1x1") {
  auto src = makeHorizontalBands(5u, 5u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{1u, 1u});
  CHECK(resized.getSize() == sf::Vector2u{1u, 1u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{1u, 1u});
  CHECK(bin.data.size() == static_cast<std::size_t>(1u) * 1u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 05: 5x5 -> 3x2") {
  auto src = makeHorizontalBands(5u, 5u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{3u, 2u});
  CHECK(resized.getSize() == sf::Vector2u{3u, 2u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{3u, 2u});
  CHECK(bin.data.size() == static_cast<std::size_t>(3u) * 2u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 06: 6x4 -> 3x3") {
  auto src = makeHorizontalBands(6u, 4u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{3u, 3u});
  CHECK(resized.getSize() == sf::Vector2u{3u, 3u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{3u, 3u});
  CHECK(bin.data.size() == static_cast<std::size_t>(3u) * 3u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 07: 4x6 -> 2x2") {
  auto src = makeHorizontalBands(4u, 6u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{2u, 2u});
  CHECK(resized.getSize() == sf::Vector2u{2u, 2u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{2u, 2u});
  CHECK(bin.data.size() == static_cast<std::size_t>(2u) * 2u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 08: 7x7 -> 3x5") {
  auto src = makeHorizontalBands(7u, 7u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{3u, 5u});
  CHECK(resized.getSize() == sf::Vector2u{3u, 5u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{3u, 5u});
  CHECK(bin.data.size() == static_cast<std::size_t>(3u) * 5u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 09: 8x8 -> 4x6") {
  auto src = makeHorizontalBands(8u, 8u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{4u, 6u});
  CHECK(resized.getSize() == sf::Vector2u{4u, 6u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{4u, 6u});
  CHECK(bin.data.size() == static_cast<std::size_t>(4u) * 6u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 10: 6x6 -> 5x4") {
  auto src = makeHorizontalBands(6u, 6u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{5u, 4u});
  CHECK(resized.getSize() == sf::Vector2u{5u, 4u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{5u, 4u});
  CHECK(bin.data.size() == static_cast<std::size_t>(5u) * 4u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 11: 5x5 -> 5x5") {
  auto src = makeHorizontalBands(5u, 5u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{5u, 5u});
  CHECK(resized.getSize() == sf::Vector2u{5u, 5u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{5u, 5u});
  CHECK(bin.data.size() == static_cast<std::size_t>(5u) * 5u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 12: 2x3 -> 2x3") {
  auto src = makeHorizontalBands(2u, 3u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{2u, 3u});
  CHECK(resized.getSize() == sf::Vector2u{2u, 3u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{2u, 3u});
  CHECK(bin.data.size() == static_cast<std::size_t>(2u) * 3u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 13: 6x2 -> 6x2") {
  auto src = makeHorizontalBands(6u, 2u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{6u, 2u});
  CHECK(resized.getSize() == sf::Vector2u{6u, 2u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{6u, 2u});
  CHECK(bin.data.size() == static_cast<std::size_t>(6u) * 2u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 14: 4x4 -> 1x1") {
  auto src = makeHorizontalBands(4u, 4u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{1u, 1u});
  CHECK(resized.getSize() == sf::Vector2u{1u, 1u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{1u, 1u});
  CHECK(bin.data.size() == static_cast<std::size_t>(1u) * 1u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 15: 8x6 -> 4x3") {
  auto src = makeHorizontalBands(8u, 6u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{4u, 3u});
  CHECK(resized.getSize() == sf::Vector2u{4u, 3u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{4u, 3u});
  CHECK(bin.data.size() == static_cast<std::size_t>(4u) * 3u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 16: 6x8 -> 3x4") {
  auto src = makeHorizontalBands(6u, 8u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{3u, 4u});
  CHECK(resized.getSize() == sf::Vector2u{3u, 4u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{3u, 4u});
  CHECK(bin.data.size() == static_cast<std::size_t>(3u) * 4u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 17: 5x4 -> 2x2") {
  auto src = makeHorizontalBands(5u, 4u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{2u, 2u});
  CHECK(resized.getSize() == sf::Vector2u{2u, 2u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{2u, 2u});
  CHECK(bin.data.size() == static_cast<std::size_t>(2u) * 2u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 18: 4x5 -> 2x3") {
  auto src = makeHorizontalBands(4u, 5u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{2u, 3u});
  CHECK(resized.getSize() == sf::Vector2u{2u, 3u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{2u, 3u});
  CHECK(bin.data.size() == static_cast<std::size_t>(2u) * 3u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 19: 7x5 -> 3x2") {
  auto src = makeHorizontalBands(7u, 5u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{3u, 2u});
  CHECK(resized.getSize() == sf::Vector2u{3u, 2u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{3u, 2u});
  CHECK(bin.data.size() == static_cast<std::size_t>(3u) * 2u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Resize regression pair 20: 5x7 -> 2x3") {
  auto src = makeHorizontalBands(5u, 7u, sf::Color::White, sf::Color::Black);
  auto resized = hopfield::resizeImage(src, sf::Vector2u{2u, 3u});
  CHECK(resized.getSize() == sf::Vector2u{2u, 3u});
  CHECK(hopfield::isValidImage(resized));
  auto bin = hopfield::imageToBinary(resized);
  CHECK(bin.size == sf::Vector2u{2u, 3u});
  CHECK(bin.data.size() == static_cast<std::size_t>(2u) * 3u);
  for (int spin : bin.data) {
    CHECK((spin == 1 || spin == -1));
  }
}

TEST_CASE("Color and binary conversion case 01 (3x4)") {
  auto checker = makeCheckerboard(3u, 4u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{3u, 4u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{3u, 4u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 02 (4x6)") {
  auto checker = makeCheckerboard(4u, 6u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{4u, 6u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{4u, 6u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 03 (5x3)") {
  auto checker = makeCheckerboard(5u, 3u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{5u, 3u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{5u, 3u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 04 (6x5)") {
  auto checker = makeCheckerboard(6u, 5u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{6u, 5u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{6u, 5u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 05 (2x2)") {
  auto checker = makeCheckerboard(2u, 2u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{2u, 2u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{2u, 2u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 06 (3x4)") {
  auto checker = makeCheckerboard(3u, 4u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{3u, 4u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{3u, 4u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 07 (4x6)") {
  auto checker = makeCheckerboard(4u, 6u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{4u, 6u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{4u, 6u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 08 (5x3)") {
  auto checker = makeCheckerboard(5u, 3u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{5u, 3u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{5u, 3u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 09 (6x5)") {
  auto checker = makeCheckerboard(6u, 5u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{6u, 5u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{6u, 5u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 10 (2x2)") {
  auto checker = makeCheckerboard(2u, 2u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{2u, 2u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{2u, 2u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 11 (3x4)") {
  auto checker = makeCheckerboard(3u, 4u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{3u, 4u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{3u, 4u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 12 (4x6)") {
  auto checker = makeCheckerboard(4u, 6u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{4u, 6u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{4u, 6u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 13 (5x3)") {
  auto checker = makeCheckerboard(5u, 3u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{5u, 3u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{5u, 3u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 14 (6x5)") {
  auto checker = makeCheckerboard(6u, 5u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{6u, 5u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{6u, 5u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 15 (2x2)") {
  auto checker = makeCheckerboard(2u, 2u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{2u, 2u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{2u, 2u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 16 (3x4)") {
  auto checker = makeCheckerboard(3u, 4u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{3u, 4u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{3u, 4u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 17 (4x6)") {
  auto checker = makeCheckerboard(4u, 6u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{4u, 6u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{4u, 6u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 18 (5x3)") {
  auto checker = makeCheckerboard(5u, 3u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{5u, 3u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{5u, 3u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 19 (6x5)") {
  auto checker = makeCheckerboard(6u, 5u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{6u, 5u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{6u, 5u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 20 (2x2)") {
  auto checker = makeCheckerboard(2u, 2u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{2u, 2u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{2u, 2u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 21 (3x4)") {
  auto checker = makeCheckerboard(3u, 4u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{3u, 4u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{3u, 4u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(3u) * 4u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 22 (4x6)") {
  auto checker = makeCheckerboard(4u, 6u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{4u, 6u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{4u, 6u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(4u) * 6u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 23 (5x3)") {
  auto checker = makeCheckerboard(5u, 3u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{5u, 3u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{5u, 3u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(5u) * 3u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 24 (6x5)") {
  auto checker = makeCheckerboard(6u, 5u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{6u, 5u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{6u, 5u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(6u) * 5u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Color and binary conversion case 25 (2x2)") {
  auto checker = makeCheckerboard(2u, 2u, sf::Color::White, sf::Color::Black);
  auto rgb = hopfield::imageToColors(checker);
  REQUIRE(rgb.size == sf::Vector2u{2u, 2u});
  REQUIRE(rgb.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto rebuiltFromRgb = hopfield::colorsToImage(rgb);
  checkImagesEqual(rebuiltFromRgb, checker);

  auto bin = hopfield::imageToBinary(checker);
  REQUIRE(bin.size == sf::Vector2u{2u, 2u});
  REQUIRE(bin.data.size() == static_cast<std::size_t>(2u) * 2u);

  auto binColors = hopfield::binaryToColors(bin);
  REQUIRE(binColors.size == bin.size);
  REQUIRE(binColors.data.size() == bin.data.size());

  auto rebuiltFromBin = hopfield::binaryToImage(bin);
  checkImagesEqual(rebuiltFromBin, checker);
}

TEST_CASE("Image file round-trip and batch conversion 01") {
  auto dir = makeTempDir("io_batch_" + std::to_string(1));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(3u, 4u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(3u, 4u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{3u, 4u});
  CHECK(bins[1].size == sf::Vector2u{3u, 4u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(3u) * 4u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(3u) * 4u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 02") {
  auto dir = makeTempDir("io_batch_" + std::to_string(2));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(4u, 5u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(4u, 5u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{4u, 5u});
  CHECK(bins[1].size == sf::Vector2u{4u, 5u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(4u) * 5u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(4u) * 5u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 03") {
  auto dir = makeTempDir("io_batch_" + std::to_string(3));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(5u, 3u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(5u, 3u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{5u, 3u});
  CHECK(bins[1].size == sf::Vector2u{5u, 3u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(5u) * 3u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(5u) * 3u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 04") {
  auto dir = makeTempDir("io_batch_" + std::to_string(4));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(2u, 4u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(2u, 4u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{2u, 4u});
  CHECK(bins[1].size == sf::Vector2u{2u, 4u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(2u) * 4u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(2u) * 4u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 05") {
  auto dir = makeTempDir("io_batch_" + std::to_string(5));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(3u, 5u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(3u, 5u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{3u, 5u});
  CHECK(bins[1].size == sf::Vector2u{3u, 5u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(3u) * 5u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(3u) * 5u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 06") {
  auto dir = makeTempDir("io_batch_" + std::to_string(6));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(4u, 3u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(4u, 3u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{4u, 3u});
  CHECK(bins[1].size == sf::Vector2u{4u, 3u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(4u) * 3u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(4u) * 3u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 07") {
  auto dir = makeTempDir("io_batch_" + std::to_string(7));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(5u, 4u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(5u, 4u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{5u, 4u});
  CHECK(bins[1].size == sf::Vector2u{5u, 4u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(5u) * 4u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(5u) * 4u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 08") {
  auto dir = makeTempDir("io_batch_" + std::to_string(8));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(2u, 5u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(2u, 5u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{2u, 5u});
  CHECK(bins[1].size == sf::Vector2u{2u, 5u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(2u) * 5u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(2u) * 5u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 09") {
  auto dir = makeTempDir("io_batch_" + std::to_string(9));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(3u, 3u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(3u, 3u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{3u, 3u});
  CHECK(bins[1].size == sf::Vector2u{3u, 3u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(3u) * 3u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(3u) * 3u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 10") {
  auto dir = makeTempDir("io_batch_" + std::to_string(10));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(4u, 4u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(4u, 4u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{4u, 4u});
  CHECK(bins[1].size == sf::Vector2u{4u, 4u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(4u) * 4u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(4u) * 4u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 11") {
  auto dir = makeTempDir("io_batch_" + std::to_string(11));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(5u, 5u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(5u, 5u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{5u, 5u});
  CHECK(bins[1].size == sf::Vector2u{5u, 5u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(5u) * 5u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(5u) * 5u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 12") {
  auto dir = makeTempDir("io_batch_" + std::to_string(12));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(2u, 3u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(2u, 3u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{2u, 3u});
  CHECK(bins[1].size == sf::Vector2u{2u, 3u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(2u) * 3u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(2u) * 3u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 13") {
  auto dir = makeTempDir("io_batch_" + std::to_string(13));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(3u, 4u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(3u, 4u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{3u, 4u});
  CHECK(bins[1].size == sf::Vector2u{3u, 4u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(3u) * 4u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(3u) * 4u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 14") {
  auto dir = makeTempDir("io_batch_" + std::to_string(14));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(4u, 5u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(4u, 5u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{4u, 5u});
  CHECK(bins[1].size == sf::Vector2u{4u, 5u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(4u) * 5u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(4u) * 5u);

  std::filesystem::remove_all(dir);
}

TEST_CASE("Image file round-trip and batch conversion 15") {
  auto dir = makeTempDir("io_batch_" + std::to_string(15));
  auto p1 = dir / "band_a.png";
  auto p2 = dir / "band_b.png";

  auto imgA = makeHorizontalBands(5u, 3u, sf::Color::White, sf::Color::Black);
  auto imgB = makeVerticalBands(5u, 3u, sf::Color::Black, sf::Color::White);
  REQUIRE(imgA.saveToFile(p1));
  REQUIRE(imgB.saveToFile(p2));

  sf::Image loadedA;
  sf::Image loadedB;
  REQUIRE(loadedA.loadFromFile(p1));
  REQUIRE(loadedB.loadFromFile(p2));

  checkImagesEqual(loadedA, imgA);
  checkImagesEqual(loadedB, imgB);

  auto bins = hopfield::imageToBinaries({loadedA, loadedB});
  REQUIRE(bins.size() == 2u);
  CHECK(bins[0].size == sf::Vector2u{5u, 3u});
  CHECK(bins[1].size == sf::Vector2u{5u, 3u});
  CHECK(bins[0].data.size() == static_cast<std::size_t>(5u) * 3u);
  CHECK(bins[1].data.size() == static_cast<std::size_t>(5u) * 3u);

  std::filesystem::remove_all(dir);
}

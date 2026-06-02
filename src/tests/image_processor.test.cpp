#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <SFML/Graphics.hpp>

#include "../image_processor.hpp"
#include "doctest.h"

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
    SUBCASE("resizeImages -- n3") {
      std::vector<sf::Image> inImgs{};

      REQUIRE(inImgs.empty());
      CHECK_THROWS(pf::resizeImages(inImgs));
    }
    SUBCASE("resizeImages -- n4") {
      std::vector<sf::Image> inImgs;
      inImgs.push_back(imgCyan0x0);
      inImgs.push_back(imgCyan0x3);
      inImgs.push_back(imgCyan3x0);

      REQUIRE(!inImgs.empty());
      CHECK_THROWS(pf::resizeImages(inImgs));
    }
  }
}

// ============================================================
// IMAGE PROCESSING TEST CASE
// ============================================================
TEST_CASE("IMAGE PROCESSING") {
  // ----- IMAGES -----
  sf::Image allWhite;
  allWhite.create(8u, 8u, sf::Color::White);

  sf::Image allBlack;
  allBlack.create(8u, 8u, sf::Color::Black);

  sf::Image blackWhiteChecker;
  blackWhiteChecker.create(8u, 8u);
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      sf::Color pixelColor =
          ((j + i) % 2 == 0) ? sf::Color::White : sf::Color::Black;
      blackWhiteChecker.setPixel(i, j, pixelColor);
    }
  }

  const sf::Color colorA{220u, 50u, 50u};   // rosso
  const sf::Color colorB{50u, 180u, 80u};   // verde
  const sf::Color colorC{40u, 90u, 210u};   // blu
  const sf::Color colorD{210u, 160u, 30u};  // giallo-arancio

  sf::Image rgbChecker;
  rgbChecker.create(8u, 8u);
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
      rgbChecker.setPixel(i, j, pixelColor);
    }
  }

  // ----- COLOR PATTERNS -----
  pf::PatternRGB expectedAllWhiteColors;
  expectedAllWhiteColors.size = sf::Vector2u{8u, 8u};
  expectedAllWhiteColors.data = std::vector<sf::Color>(64, sf::Color::White);

  pf::PatternRGB expectedAllBlackColors;
  expectedAllBlackColors.size = sf::Vector2u{8u, 8u};
  expectedAllBlackColors.data = std::vector<sf::Color>(64, sf::Color::Black);

  pf::PatternRGB expectedBlackWhiteCheckerColors;
  expectedBlackWhiteCheckerColors.size = sf::Vector2u{8u, 8u};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      expectedBlackWhiteCheckerColors.data.push_back(
          ((i + j) % 2 == 0) ? sf::Color::White : sf::Color::Black);
    }
  }

  pf::PatternRGB expectedRgbCheckerColors;
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
  pf::PatternInt expectedAllWhiteBinary;
  expectedAllWhiteBinary.size = sf::Vector2u{8u, 8u};
  expectedAllWhiteBinary.data = std::vector<int>(64, 1);

  pf::PatternInt expectedAllBlackBinary;
  expectedAllBlackBinary.size = sf::Vector2u{8u, 8u};
  expectedAllBlackBinary.data = std::vector<int>(64, -1);

  pf::PatternInt expectedBlackWhiteCheckerBinary;
  expectedBlackWhiteCheckerBinary.size = sf::Vector2u{8u, 8u};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      expectedBlackWhiteCheckerBinary.data.push_back(((i + j) % 2 == 0) ? 1
                                                                        : -1);
    }
  }

  pf::PatternInt expectedRgbCheckerBinary;
  expectedRgbCheckerBinary.size = {8u, 8u};
  for (auto j{0u}; j != 8u; ++j) {
    for (auto i{0u}; i != 8u; ++i) {
      expectedRgbCheckerBinary.data.push_back((i % 2 != 0 && j % 2 != 0) ? 1
                                                                         : -1);
    }
  }

  SUBCASE("imageToColors") {
    auto allWhiteColors{pf::imageToColors(allWhite)};
    auto allBlackColors{pf::imageToColors(allBlack)};
    auto blackWhiteCheckerColors{pf::imageToColors(blackWhiteChecker)};
    auto rgbCheckerColors{pf::imageToColors(rgbChecker)};
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
    auto allWhiteBinary{pf::imageToBinary(allWhite)};
    auto allBlackBinary{pf::imageToBinary(allBlack)};
    auto blackWhiteCheckerBinary{pf::imageToBinary(blackWhiteChecker)};
    auto rgbCheckerBinary{pf::imageToBinary(rgbChecker)};
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
      auto binaries{pf::imageToBinaries(imgs)};
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
      CHECK_THROWS(pf::imageToBinaries(emptyImgs));
    }
  }
  SUBCASE("binaryToColors") {
    auto allWhiteColors{pf::binaryToColors(expectedAllWhiteBinary)};
    auto allBlackColors{pf::binaryToColors(expectedAllBlackBinary)};
    auto blackWhiteCheckerColors{
        pf::binaryToColors(expectedBlackWhiteCheckerBinary)};
    auto rgbCheckerColors{pf::binaryToColors(expectedRgbCheckerBinary)};
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
    auto allWhiteImg{pf::colorsToImage(expectedAllWhiteColors)};
    auto allBlackImg{pf::colorsToImage(expectedAllBlackColors)};
    auto blackWhiteCheckerImg{
        pf::colorsToImage(expectedBlackWhiteCheckerColors)};
    REQUIRE(allWhiteImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(allBlackImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(blackWhiteCheckerImg.getSize() == sf::Vector2u{8u, 8u});
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allWhiteImg.getPixel(i, j) == sf::Color::White);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allBlackImg.getPixel(i, j) == sf::Color::Black);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(blackWhiteCheckerImg.getPixel(i, j) ==
              expectedBlackWhiteCheckerColors.data[j * 8u + i]);
      }
    }
  }
  SUBCASE("binaryToImage") {
    auto allWhiteImg{pf::binaryToImage(expectedAllWhiteBinary)};
    auto allBlackImg{pf::binaryToImage(expectedAllBlackBinary)};
    auto blackWhiteCheckerImg{
        pf::binaryToImage(expectedBlackWhiteCheckerBinary)};
    REQUIRE(allWhiteImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(allBlackImg.getSize() == sf::Vector2u{8u, 8u});
    REQUIRE(blackWhiteCheckerImg.getSize() == sf::Vector2u{8u, 8u});
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allWhiteImg.getPixel(i, j) == sf::Color::White);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(allBlackImg.getPixel(i, j) == sf::Color::Black);
      }
    }
    for (auto j{0u}; j != 8u; ++j) {
      for (auto i{0u}; i != 8u; ++i) {
        CHECK(blackWhiteCheckerImg.getPixel(i, j) ==
              expectedBlackWhiteCheckerColors.data[j * 8u + i]);
      }
    }
  }
}
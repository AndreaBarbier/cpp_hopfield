#include "image_processor.hpp"

namespace hopfield {
// ============================================================
// IMAGE UTILITIES
// ============================================================

bool isValidImage(sf::Image const& img) {
  if (img.getSize().x == 0u || img.getSize().y == 0u) {
    return false;
  }
  return true;
}

std::vector<sf::Image> validateImages(std::vector<sf::Image> const& inImgs) {
  auto outImgs{inImgs};
  std::erase_if(outImgs, [](auto const& img) { return !isValidImage(img); });
  return outImgs;
}

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
// IMAGE PROCESSING
// ============================================================

PatternRGB imageToColors(sf::Image const& img) {
  auto imgSize{img.getSize()};
  // Assert because it will not be a runtime error if the size is wrong
  assert(imgSize.x > 0u && imgSize.y > 0u);
  PatternRGB outPattern;
  for (auto j{0u}; j != imgSize.y; ++j) {
    for (auto i{0u}; i != imgSize.x; ++i) {
      outPattern.data.push_back(img.getPixel(i, j));
    }
  }
  outPattern.size = imgSize;
  return outPattern;
}

PatternInt colorsToBinary(PatternRGB const& inPattern) {
  // Assert because it will not be a runtime error if the size is wrong
  assert(!inPattern.data.empty() && inPattern.size.x > 0u &&
         inPattern.size.y > 0u);
  PatternInt outPattern;
  for (auto const& c : inPattern.data) {
    auto grey = (c.r + c.g + c.b) / 3u;
    outPattern.data.push_back((grey > 127u) ? (1) : (-1));
  }
  outPattern.size = inPattern.size;
  return outPattern;
}

PatternInt imageToBinary(sf::Image const& img) {
  return colorsToBinary(imageToColors(img));
}

std::vector<PatternInt> imageToBinaries(std::vector<sf::Image> const& imgs) {
  if (imgs.empty()) {
    throw std::runtime_error(
        "Error: no images to process or every image has an invalid size.\n");
  }
  std::vector<PatternInt> outPatterns;
  for (auto const& img : imgs) {
    outPatterns.push_back(imageToBinary(img));
  }
  return outPatterns;
}

PatternRGB binaryToColors(PatternInt const& inPattern) {
  assert(!inPattern.data.empty() && inPattern.size.x > 0u &&
         inPattern.size.y > 0u);
  PatternRGB outPattern;
  for (auto i : inPattern.data) {
    assert(i == 1 || i == -1);
    outPattern.data.push_back((i == 1) ? (sf::Color::White)
                                       : (sf::Color::Black));
  }
  outPattern.size = inPattern.size;
  return outPattern;
}

sf::Image colorsToImage(PatternRGB const& inPattern) {
  auto x{inPattern.size.x};
  auto y{inPattern.size.y};
  assert(!inPattern.data.empty() && x > 0u && y > 0u);
  sf::Image outImg;
  outImg.create(x, y);

  for (auto j{0u}; j != y; ++j) {
    for (auto i{0u}; i != x; ++i) {
      outImg.setPixel(i, j, inPattern.data[j * x + i]);
    }
  }
  return outImg;
}

sf::Image binaryToImage(PatternInt const& inPattern) {
  return colorsToImage(binaryToColors(inPattern));
}
}  // namespace hopfield
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace pf {
// ###############################################################################
// UTILITIES ON IMAGES
// ###############################################################################
sf::Image resizeImage(sf::Image const& img, sf::Vector2u targetSize) {
  auto oldSize{img.getSize()};
  assert(targetSize.x > oldSize.x && targetSize.y > oldSize.y &&
         targetSize.x == 0u && targetSize.y == 0u);
  if (targetSize.x > oldSize.x || targetSize.y > oldSize.y ||
      targetSize.x == 0u || targetSize.y == 0u) {
    throw std::runtime_error("Error: target image size is invalid.\n");
  }
  sf::Image newImg;
  newImg.create(targetSize.x, targetSize.y);
  for (auto j{0u}; j != targetSize.y; ++j) {
    for (auto i{0u}; i != targetSize.x; ++i) {
      auto j_o = (j * oldSize.y / targetSize.y);
      auto i_o = (i * oldSize.x / targetSize.x);
      newImg.setPixel(i, j, img.getPixel(i_o, j_o));
    }
  }
  return newImg;
}

// ###############################################################################
// IMAGES PROCESSING
// ###############################################################################
template <class T>
struct Pattern {
  std::vector<T> data;
  sf::Vector2u size;
};

using PatternRGB = Pattern<sf::Color>;
using PatternInt = Pattern<int>;

PatternRGB imageToColors(sf::Image const& img) {
  auto x{img.getSize().x};
  auto y{img.getSize().y};
  assert(x > 0u && y > 0u);
  if (x <= 0u || y <= 0u) {
    throw std::runtime_error("Error: invalid image size.\n");
  }
  PatternRGB outPattern;
  for (auto j{0u}; j != img.getSize().y; ++j) {
    for (auto i{0u}; i != img.getSize().x; ++i) {
      outPattern.data.push_back(img.getPixel(i, j));
    }
  }
  return outPattern;
}

PatternInt colorsToBinary(PatternRGB const& inPattern) {
  assert(inPattern.data.size() > 0 && inPattern.size.x > 0 &&
         inPattern.size.y > 0);
  if (inPattern.data.size() == 0 || inPattern.size.x == 0 ||
      inPattern.size.y == 0) {
    throw std::runtime_error("Error: invalid input pattern.\n");
  }
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

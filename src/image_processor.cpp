#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace pf {
// ============================================================
// IMAGE UTILITIES
// ============================================================

/// @brief Resize an image using nearest-neighbor algorithm
/// @param[in] img Source image
/// @param[in] targetSize Target size: it should be <=the source image's size
/// @throws std::runtime_error if targetSize > img.getSize() or if it is (0u,0u)
/// @return Returns an image with targetSize
sf::Image resizeImage(sf::Image const& img, sf::Vector2u targetSize) {
  auto oldSize{img.getSize()};
  assert(targetSize.x < oldSize.x && targetSize.y < oldSize.y &&
         targetSize.x > 0u && targetSize.y > 0u);
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

/// @brief Converts a PatternRGB into a PatternInt
/// @param[in] inPattern Input PatternRGB
/// @throws std::runtime_error if inPattern.data is empty or if inPattern.size
/// is (0u,0u)
/// @return Returns a PatternInt with +1 (brigth pixel) and -1 (dark pixel)
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

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace pf {
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
    throw std::runtime_error("Error: invalid image size\n");
  }
  PatternRGB outPattern;
  for (auto j{0u}; j != img.getSize().y; ++j) {
    for (auto i{0u}; i != img.getSize().x; ++i) {
      outPattern.data.push_back(img.getPixel(i, j));
    }
  }
  return outPattern;
}

void resizePattern(PatternRGB& pattern, sf::Vector2u targetsize) {
  if (targetsize.x > pattern.size.x || targetsize.y > pattern.size.y) {
    throw std::runtime_error("Error: image size is too big.\n");
  }
  std::vector<sf::Color> newdata;
  for (auto j{0u}; j != targetsize.y; ++j) {
    for (auto i{0u}; i != targetsize.x; ++i) {
      auto j_o = j * (pattern.size.y / targetsize.y);
      auto i_o = i * (pattern.size.x / targetsize.x);
      newdata.push_back(pattern.data[j_o * pattern.size.x + i_o]);
    }
  }
  pattern.data = std::move(newdata);
  pattern.size = targetsize;
}

PatternInt binarizePattern(PatternRGB const& pattern) {
  PatternInt result;
  for (sf::Color const& c : pattern.data) {
    int grigio = (static_cast<int>(c.r) + static_cast<int>(c.g) +
                  static_cast<int>(c.b)) /
                 3;
    if (grigio > 127) {
      result.data.push_back(1);
    } else {
      result.data.push_back(-1);
    }
  }
  result.size = pattern.size;
  return result;
}
}  // namespace pf

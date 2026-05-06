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

}  // namespace pf

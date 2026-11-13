#include "plot.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <string>
#include <utility>

namespace hopfield {

sf::Vector2f Plot::toPixel(std::size_t i, double y) const {
  float fx{y_.size() > 1 ? float(i) / float(y_.size() - 1) : 0.f};
  float fy{static_cast<float>((y - yMin_) / (yMax_ - yMin_))};

  return {inner_.position.x + inner_.size.x * fx,
          inner_.position.y + inner_.size.y * (1 - fy)};
}

void Plot::setData(std::vector<double> energies) {
  assert(!energies.empty());
  y_ = std::move(energies);

  auto [lo, hi] = std::minmax_element(y_.begin(), y_.end());
  yMax_ = *hi;
  yMin_ = *lo;
  if (yMax_ == yMin_) {
    yMax_ += 1;
    yMin_ -= 1;
  }

  curve_.clear();
  frame_.clear();
  labels_.clear();
}

void Plot::place(sf::FloatRect outer, const sf::Font& font) {
  assert(!y_.empty());
  inner_ = {
      {outer.position.x + m_.left, outer.position.y + m_.top},
      {outer.size.x - m_.left - m_.right, outer.size.y - m_.top - m_.bottom}};
  curve_.clear();
  frame_.clear();
  labels_.clear();

  for (std::size_t i = 0; i != y_.size(); ++i)
    curve_.push_back({toPixel(i, y_[i]), sf::Color::Green});

  const sf::Color ax(180, 180, 180);
  auto line = [&](sf::Vector2f a, sf::Vector2f b) {
    frame_.push_back({a, ax});
    frame_.push_back({b, ax});
  };
  auto label = [&](const std::string& s, sf::Vector2f p, bool alignRight) {
    sf::Text t(font, s, 12);
    if (alignRight) p.x -= t.getGlobalBounds().size.x;
    t.setPosition(p);
    labels_.push_back(std::move(t));
  };

  auto o = inner_.position;
  auto s = inner_.size;
  line(o, {o.x, o.y + s.y});
  line({o.x, o.y + s.y}, {o.x + s.x, o.y + s.y});

  // y ticks and labels
  for (int k = 0; k != 5; ++k) {
    double v = yMin_ + (yMax_ - yMin_) * k / 4.0;
    auto p = toPixel(0, v);
    line({p.x - 4, p.y}, p);

    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", v);
    label(buf, {p.x - 8, p.y - 8}, true);
  }

  // x ticks
  std::size_t step = std::max<std::size_t>(1, (y_.size() - 1) / 5);
  for (std::size_t i = 0; i < y_.size(); i += step) {
    auto p = toPixel(i, yMin_);
    line(p, {p.x, p.y + 4});
    label(std::to_string(i), {p.x - 4, p.y + 6}, false);
  }
}

void Plot::draw(sf::RenderTarget& t, std::size_t cursor) const {
  if (curve_.empty()) {
    return;
  }

  t.draw(frame_.data(), frame_.size(), sf::PrimitiveType::Lines);
  for (const auto& l : labels_) {
    t.draw(l);
  }

  std::size_t k = std::min(cursor, curve_.size() - 1);
  t.draw(curve_.data(), k + 1, sf::PrimitiveType::LineStrip);
}

}  // namespace hopfield
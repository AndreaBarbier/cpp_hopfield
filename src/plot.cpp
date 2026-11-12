#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <random>

namespace hopfield {
struct Margins {
  float left{60.f};
  float right{15.f};
  float top{15.f};
  float bottom{30.f};
};

class Plot {
  std::vector<double> y_;
  double yMax_{};
  double yMin_{};
  Margins m_;
  sf::FloatRect inner_;

 public:
  std::vector<sf::Vertex> curve;
  std::vector<sf::Vertex> frame;
  std::vector<sf::Text> labels;

  // Taken by value: the caller can either copy or move into it
  void setData(std::vector<double> energies) {
    assert(!energies.empty());
    y_ = std::move(energies);

    // minmax_element returns iterators to the smallest and largest elements
    auto [lo, hi] = std::minmax_element(y_.begin(), y_.end());
    yMax_ = *hi;
    yMin_ = *lo;
    if (yMax_ == yMin_) {
      yMax_ += 1;
      yMin_ -= 1;
    }
  }

  sf::Vector2f toPixel(size_t i, double y) const {
    float fx{y_.size() > 1 ? float(i) / float(y_.size() - 1) : 0.f};
    float fy{static_cast<float>((y - yMin_) / (yMax_ - yMin_))};

    return {inner_.position.x + inner_.size.x * fx,
            inner_.position.y + inner_.size.y * (1 - fy)};
  }

  void place(sf::FloatRect outer, const sf::Font& font) {
    inner_ = {
        {outer.position.x + m_.left, outer.position.y + m_.top},
        {outer.size.x - m_.left - m_.right, outer.size.y - m_.top - m_.bottom}};
    curve.clear();
    frame.clear();
    labels.clear();
    assert(!y_.empty());

    for (size_t i = 0; i != y_.size(); ++i)
      curve.push_back({toPixel(i, y_[i]), sf::Color::Green});

    const sf::Color ax(180, 180, 180);
    auto line = [&](sf::Vector2f a, sf::Vector2f b) {
      frame.push_back({a, ax});
      frame.push_back({b, ax});
    };
    auto label = [&](const std::string& s, sf::Vector2f p, bool alignRight) {
      sf::Text t(font, s, 12);
      if (alignRight) p.x -= t.getGlobalBounds().size.x;
      t.setPosition(p);
      labels.push_back(std::move(t));
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
      // Converting double --> string with 1 decimal place
      std::snprintf(buf, sizeof buf, "%.1f", v);
      label(buf, {p.x - 8, p.y - 8}, true);
    }

    // x ticks: '<' (not '!=') so the loop ends even when step doesn't
    // divide size exactly
    size_t step = std::max<size_t>(1, (y_.size() - 1) / 5);
    for (size_t i = 0; i < y_.size(); i += step) {
      auto p = toPixel(i, yMin_);
      line(p, {p.x, p.y + 4});
      label(std::to_string(i), {p.x - 4, p.y + 6}, false);
    }
  }
};

void render(const Plot& p, size_t cursor, sf::RenderTarget& t) {
  assert(!p.curve.empty());
  t.draw(p.frame.data(), p.frame.size(), sf::PrimitiveType::Lines);
  for (auto& l : p.labels) {
    t.draw(l);
  }
  size_t k = std::min(cursor, p.curve.size() - 1);
  t.draw(p.curve.data(), k + 1, sf::PrimitiveType::LineStrip);
}
}  // namespace hopfield

int main() {
  std::cerr << "[DEBUG] Starting program...\n";

  // 1. Creating the window
  sf::RenderWindow window(sf::VideoMode({1400, 800}), "Hopfield - Energy Plot");

  window.setFramerateLimit(60);

  std::cerr << "[DEBUG] Window created.\n";

  // 2. Loading the font
  sf::Font font("data/STIX_Two_Math/STIXTwoMath-Regular.ttf");

  std::cerr << "[DEBUG] Font loaded.\n";

  // 3. Creating example data
  // 1000 points: exponential decay towards a minimum, plus small noise
  constexpr size_t N = 1000;
  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0.0, 0.8);
  std::vector<double> energies;
  energies.reserve(N);
  for (size_t i = 0; i != N; ++i)
    energies.push_back(-50.0 + 60.0 * std::exp(-double(i) / 200.0) +
                       noise(rng));

  // 4. Creating and initializing the plot
  hopfield::Plot plot;

  plot.setData(energies);

  std::cerr << "[DEBUG] Data initialized: " << energies.size() << " points.\n";

  // 5. Defining the plot area
  sf::FloatRect area{{0.f, 0.f}, sf::Vector2f(window.getSize())};

  plot.place(area, font);

  std::cerr << "[DEBUG] Plot placed.\n";
  std::cerr << "[DEBUG] Curve points: " << plot.curve.size() << '\n';

  // 6. Main rendering loop
  while (window.isOpen()) {
    // Handle window events
    while (const auto event = window.pollEvent()) {
      if (event->is<sf::Event::Closed>()) {
        std::cerr << "[DEBUG] Closing window.\n";
        window.close();
      }
    }

    // Clear previous frame
    window.clear(sf::Color::Black);

    // Draw the complete energy curve
    if (!plot.curve.empty()) {
      hopfield::render(plot, plot.curve.size() - 1, window);
    }

    // Display the new frame
    window.display();
  }

  std::cerr << "[DEBUG] Program terminated correctly.\n";

  return 0;
}

#include "recall_view.hpp"

#include <SFML/Graphics.hpp>
#include <SFML/System/Sleep.hpp>
#include <SFML/System/Time.hpp>
#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_processor.hpp"
#include "plot.hpp"

namespace hopfield {

namespace {

sf::Font loadPlotFont() {
  std::filesystem::path const fontPath{
      std::filesystem::path{PROJECT_ROOT} / "STIX_Two_Math" /
      "STIXTwoMath-Regular.ttf"};
  sf::Font font;
  if (!font.openFromFile(fontPath)) {
    throw std::runtime_error("Error: cannot load font from '" +
                             fontPath.string() + "'.\n");
  }
  return font;
}

}  // namespace

PatternInt runRecallView(Network const& net, PatternInt const& inputPattern,
                         std::size_t maxSteps, RecallStepFn const& onStep) {
  auto const patternSize{net.size()};
  sf::Font const font{loadPlotFont()};

  unsigned int const panelScale{
      std::max(1u, 256u / std::max(patternSize.x, patternSize.y))};
  unsigned int const paneWidth{
      std::max(280u, patternSize.x * panelScale + 40u)};
  unsigned int const topHeight{
      std::max(280u, patternSize.y * panelScale + 56u)};
  unsigned int const plotHeight{240u};

  unsigned int const winWidth{paneWidth * 2u};
  unsigned int const winHeight{topHeight + plotHeight};

  sf::RenderWindow window{sf::VideoMode{sf::Vector2u{winWidth, winHeight}},
                          "Hopfield Recall - Sweep 0"};
  window.setFramerateLimit(60);

  sf::Texture corruptedTex{sf::Vector2u{patternSize.x, patternSize.y}};
  corruptedTex.update(binaryToImage(inputPattern));
  sf::Sprite corruptedSprite{corruptedTex};
  corruptedSprite.setScale(sf::Vector2f{static_cast<float>(panelScale),
                                        static_cast<float>(panelScale)});

  sf::Texture currentTex{sf::Vector2u{patternSize.x, patternSize.y}};
  currentTex.update(binaryToImage(inputPattern));
  sf::Sprite currentSprite{currentTex};
  currentSprite.setScale(sf::Vector2f{static_cast<float>(panelScale),
                                      static_cast<float>(panelScale)});

  sf::Text leftLabel{font, "Corrupted input", 16};
  leftLabel.setFillColor(sf::Color::White);

  sf::Text rightLabel{font, "Current output", 16};
  rightLabel.setFillColor(sf::Color::White);

  sf::RectangleShape divider{sf::Vector2f{2.f, static_cast<float>(topHeight)}};
  divider.setFillColor(sf::Color(90, 90, 90));
  divider.setPosition(sf::Vector2f{static_cast<float>(paneWidth), 0.f});

  sf::RectangleShape hDivider{sf::Vector2f{static_cast<float>(winWidth), 2.f}};
  hDivider.setFillColor(sf::Color(90, 90, 90));
  hDivider.setPosition(sf::Vector2f{0.f, static_cast<float>(topHeight)});

  std::vector<double> energies;
  Plot energyPlot;

  auto rebuildLayout = [&](sf::Vector2u currentWinSize) {
    float const w{static_cast<float>(currentWinSize.x)};
    float const h{static_cast<float>(currentWinSize.y)};
    float const splitY{h * 0.58f};
    float const halfW{w * 0.5f};

    divider.setSize(sf::Vector2f{2.f, splitY});
    divider.setPosition(sf::Vector2f{halfW, 0.f});

    hDivider.setSize(sf::Vector2f{w, 2.f});
    hDivider.setPosition(sf::Vector2f{0.f, splitY});

    float const scaledW{static_cast<float>(patternSize.x * panelScale)};
    float const scaledH{static_cast<float>(patternSize.y * panelScale)};
    float const labelY{12.f};
    float const imgTop{std::max(40.f, (splitY - scaledH) * 0.5f + 8.f)};

    leftLabel.setPosition(
        sf::Vector2f{std::max(12.f, (halfW - leftLabel.getGlobalBounds().size.x) * 0.5f),
                     labelY});
    corruptedSprite.setPosition(
        sf::Vector2f{(halfW - scaledW) * 0.5f, imgTop});

    rightLabel.setPosition(sf::Vector2f{
        halfW + std::max(12.f, (halfW - rightLabel.getGlobalBounds().size.x) * 0.5f),
        labelY});
    currentSprite.setPosition(
        sf::Vector2f{halfW + (halfW - scaledW) * 0.5f, imgTop});

    if (!energies.empty()) {
      energyPlot.place(
          sf::FloatRect{{0.f, splitY + 4.f}, {w, std::max(60.f, h - splitY - 4.f)}},
          font);
    }
  };

  rebuildLayout(window.getSize());

  auto drawScene = [&](std::size_t cursor) {
    window.clear(sf::Color(24, 24, 28));
    window.draw(leftLabel);
    window.draw(rightLabel);
    window.draw(corruptedSprite);
    window.draw(currentSprite);
    window.draw(divider);
    window.draw(hDivider);
    energyPlot.draw(window, cursor);
    window.display();
  };

  auto renderPattern = [&](PatternInt const& current, std::size_t id,
                           double currentEnergy) {
    if (onStep) {
      onStep(current, id, currentEnergy);
    }

    while (std::optional const event = window.pollEvent()) {
      if (event->is<sf::Event::Closed>()) {
        window.close();
      } else if (auto const* resized = event->getIf<sf::Event::Resized>()) {
        window.setView(sf::View(sf::FloatRect{
            {0.f, 0.f},
            {static_cast<float>(resized->size.x),
             static_cast<float>(resized->size.y)}}));
        rebuildLayout(resized->size);
      }
    }

    if (!window.isOpen()) {
      return;
    }

    energies.push_back(currentEnergy);
    energyPlot.setData(energies);
    energyPlot.place(
        sf::FloatRect{{0.f, static_cast<float>(window.getSize().y) * 0.58f + 4.f},
                      {static_cast<float>(window.getSize().x),
                       std::max(60.f, static_cast<float>(window.getSize().y) * 0.42f - 4.f)}},
        font);

    currentTex.update(binaryToImage(current));
    window.setTitle("Hopfield Recall - Sweep " + std::to_string(id) +
                    " | E = " + std::to_string(currentEnergy));

    drawScene(energies.size() - 1);
    sf::sleep(sf::milliseconds(80));
  };

  PatternInt const result{net.recall(inputPattern, maxSteps, renderPattern)};

  if (window.isOpen()) {
    window.setTitle("Hopfield Recall - Converged (close window to continue)");
  }
  while (window.isOpen()) {
    while (std::optional const event = window.pollEvent()) {
      if (event->is<sf::Event::Closed>()) {
        window.close();
      } else if (auto const* key = event->getIf<sf::Event::KeyPressed>()) {
        if (key->code == sf::Keyboard::Key::Escape) {
          window.close();
        }
      } else if (auto const* resized = event->getIf<sf::Event::Resized>()) {
        window.setView(sf::View(sf::FloatRect{
            {0.f, 0.f},
            {static_cast<float>(resized->size.x),
             static_cast<float>(resized->size.y)}}));
        rebuildLayout(resized->size);
      }
    }
    drawScene(energies.empty() ? 0 : energies.size() - 1);
  }

  return result;
}

}  // namespace hopfield
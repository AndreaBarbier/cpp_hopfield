#ifndef PLOT_HPP
#define PLOT_HPP

/// @file plot.hpp
/// @brief Utilities for plotting the energy of the network over time

#include <SFML/Graphics.hpp>
#include <limits>
#include <vector>

namespace hopfield {

/// @brief Space (in pixels) between the border of the drawing area and the
/// plot itself
///
/// The left and bottom margins must leave enough room for the labels of the y
/// and x axes.
struct Margins {
  float left{60.f};    ///< Left margin (room for the y labels)
  float right{15.f};   ///< Right margin
  float top{15.f};     ///< Top margin
  float bottom{30.f};  ///< Bottom margin (room for the x labels)
};

/// @brief Line plot of a sequence of values (e.g. the energy of the network)
///
/// The x axis is the index of the data point, the y axis is its value. The
/// geometry is built in two steps, then drawn:
///  -# setData() stores the values and computes their range
///  -# place() converts values to pixels and builds curve, axes and labels
///  -# draw() renders the result on a render target
///
/// @note After every setData(), place() must be called again before draw():
/// until then draw() does nothing.
class Plot {
  std::vector<double> y_;  ///< Data points (one value per index)
  double yMax_{};          ///< Upper bound of the y axis
  double yMin_{};          ///< Lower bound of the y axis
  Margins m_;              ///< Margins around the plot area
  sf::FloatRect inner_;    ///< Plot area in pixels (drawing area minus margins)

  std::vector<sf::Vertex> curve_;  ///< Vertices of the curve (one per point)
  std::vector<sf::Vertex> frame_;  ///< Axes and ticks (each pair is a segment)
  std::vector<sf::Text> labels_;   ///< Numeric labels of the axes

  /// @brief Converts a data point into pixel coordinates
  /// @pre setData() has been called
  /// @pre inner_ has been computed (it is, inside place())
  /// @param[in] i Index of the point (x axis)
  /// @param[in] y Value of the point (y axis)
  /// @return Returns the position in pixels. The y axis is flipped, because
  /// in SFML y grows downwards: the maximum value ends up at the top.
  sf::Vector2f toPixel(std::size_t i, double y) const;

 public:
  /// @brief Creates a plot with the default margins
  Plot() = default;

  /// @brief Creates a plot with custom margins
  /// @param[in] m Margins around the plot area
  explicit Plot(Margins m) : m_(m) {}

  /// @brief Sets the data to plot and computes the range of the y axis
  ///
  /// If all values are equal the range is widened by +-1, to avoid a division
  /// by zero when converting to pixels. The previous geometry is discarded.
  ///
  /// @pre energies must not be empty
  /// @post place() must be called before draw() shows anything
  /// @param[in] energies Values to plot. Taken by value: the caller can copy
  /// or move into it.
  void setData(std::vector<double> energies);

  /// @brief Builds the geometry of the plot (curve, axes, ticks and labels)
  ///
  /// Call it after setData(), and again whenever the drawing area changes
  /// (e.g. when the window is resized).
  ///
  /// @pre setData() has been called
  /// @pre font must outlive the plot, because the labels keep a reference to
  /// it
  /// @param[in] outer Drawing area in pixels, margins included
  /// @param[in] font Font used for the labels
  void place(sf::FloatRect outer, const sf::Font& font);

  /// @brief Draws axes, labels and the curve on a render target
  ///
  /// Does nothing if place() has not been called since the last setData().
  ///
  /// @param[in,out] t Render target to draw on (e.g. a sf::RenderWindow)
  /// @param[in] cursor Index of the last point of the curve to draw: points
  /// from 0 to cursor are drawn. Values past the end are clamped, so the
  /// default draws the whole curve. Increasing it at each frame animates the
  /// curve.
  void draw(sf::RenderTarget& t,
            std::size_t cursor = std::numeric_limits<std::size_t>::max()) const;

  /// @brief Number of data points
  /// @return Returns the number of points set with setData() (0 if none)
  std::size_t size() const { return y_.size(); }
};

}  // namespace hopfield

#endif
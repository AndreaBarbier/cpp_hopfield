#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace hopfield {
// ============================================================
// IMAGE UTILITIES
// ============================================================

/// @brief Checks if an image has a valid size
/// @param[in] img Source image
/// @return Returns true if the image has a valid size. Else it returns false.
bool isValidImage(sf::Image const& img);

/// @brief Checks an images vector for any that do not have a valid size
/// @param[in] inImgs Images vector
/// @return Retuns an vector of valid size images
std::vector<sf::Image> validateImages(std::vector<sf::Image> const& inImgs);

/// @brief Resizes an image using nearest-neighbor algorithm
/// @param[in] img Source image
/// @param[in] targetSize Target size: it should be <=the source image's size
/// @return Returns an image with targetSize
sf::Image resizeImage(sf::Image const& img, sf::Vector2u targetSize);

/// @brief Resizes all the images inside a vector
/// @param[in] inImgs Images vector
/// @throws std::runtime_error if the image vector is empty or full of invalid
/// images
/// @return Returns a vector with all the valid images scaled
std::vector<sf::Image> resizeImages(std::vector<sf::Image> const& inImgs);

// ============================================================
// IMAGE PROCESSING
// ============================================================

/// @brief Generic container for 2D patterns
/// @tparam T Type of each element in the pattern
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
/// @return Returns a PatternRGB
PatternRGB imageToColors(sf::Image const& img);

/// @brief Converts a PatternRGB into a PatternInt
/// @param[in] inPattern Input PatternRGB
/// @return Returns a PatternInt with +1 (brigth pixel) and -1 (dark pixel)
PatternInt colorsToBinary(PatternRGB const& inPattern);

/// @brief Converts an image into a binary pattern
/// @param[in] img Source image
/// @return Returns a PatterInt with images data converted into binary data
PatternInt imageToBinary(sf::Image const& img);

/// @brief Converts an image vector into a binary pattern vector
/// @param[in] imgs Image vector
/// @throws std::runtime_error if the image vector is empty or full of invalid
/// images
/// @return Returns a vector of PatterInt
std::vector<PatternInt> imageToBinaries(std::vector<sf::Image> const& imgs);

/// @brief Converts a PatternInt into a PatternRGB
/// @param[in] inPattern PatternInt to convert
/// @return Returns a PatternRGB with sf::Color::White (+1) and sf::Color::Black
/// (-1)
PatternRGB binaryToColors(PatternInt const& inPattern);

/// @brief Converts a PatternRGB into an image
/// @param[in] inPattern PatternRGB -- full of sf:.Color::Black and
/// sf::Color::White -- to convert
/// @return Returns an image in grey scale
sf::Image colorsToImage(PatternRGB const& inPattern);

/// @brief Converts a PatternInt into an image
/// @param[in] inPattern PatternInt to convert
/// @return Returns an image in grey scale
sf::Image binaryToImage(PatternInt const& inPattern);
}  // namespace hopfield

#endif
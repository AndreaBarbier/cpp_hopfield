#include <SFML/Graphics.hpp>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <vector>

#include "hopfield.hpp"
#include "image_processor.hpp"

int main() {
  try {
    hopfield::Network net;
    net.addImages();  // inserire controllo in hopfield.cpp per estensioni
    net.train();

    sf::Image noisy{hopfield::corruptImage(
        std::filesystem::path{hopfield::getDataDir() /
                              "train/icons8-spongebob-squarepants-48.png"},
        0.30)};

    noisy.saveToFile(
        std::filesystem::path{hopfield::getDataDir() / "recall/spongebob.png"});
    // std::filesystem::path recallImgPath;
    // std::cout << "Insert the absolute path of the corrupt image: ";
    // std::cin >> recallImgPath;
    //
    // if (!recallImgPath.is_absolute() || recallImgPath.extension().empty() ||
    //     (recallImgPath.extension().string() != ".jpg" &&
    //      recallImgPath.extension().string() != ".png")) {
    //   std::cerr << "Error: invalid input.\n";
    //   return EXIT_FAILURE;
    // }

    // sf::Image recallImage;
    // recallImage.loadFromFile(recallImgPath);
    hopfield::PatternInt recallPatternInput{hopfield::imageToBinary(noisy)};

    std::map<size_t, hopfield::Status> statusMap;

    auto callback = [&](size_t id, hopfield::PatternInt const& pattern,
                        double energy) {
      hopfield::Status status{energy, pattern};
      statusMap.insert(std::make_pair(id, status));
      std::cout << energy << '\n';
    };
    auto pattern_right = net.recall(recallPatternInput, callback);

    sf::Image{hopfield::binaryToImage(pattern_right)}.saveToFile(
        std::filesystem::path{hopfield::getDataDir() / "recall/spongebob-right.png"});

  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
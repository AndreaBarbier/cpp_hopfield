#ifndef SESSION_HPP
#define SESSION_HPP

#include <filesystem>
#include <optional>
#include <stdexcept>

#include "hopfield.hpp"

namespace hopfield {
struct Session {
  std::filesystem::path datasetPath;
  std::filesystem::path netPath;
  std::optional<Network> loadedNet;

  Dataset dataset() const {
    if (datasetPath.empty()) {
      throw std::runtime_error("Error: no dataset selected, use 'use-dataset'");
    }
    Dataset dataset{datasetPath};
    return dataset;
  }

  Network& net() {
    if (!loadedNet) {
      throw std::runtime_error(
          "Error: no network selected, use 'init' or 'use-net'");
    }
    return *loadedNet;
  }

  const Network& net() const {
    if (!loadedNet) {
      throw std::runtime_error(
          "Error: no network selected, use 'init' or 'use-net'");
    }
    return *loadedNet;
  }

  const std::filesystem::path& netFile() const {
    if (netPath.empty()) {
      throw std::runtime_error(
          "Error: no network selected, use 'init' or 'use-net'");
    }
    return netPath;
  }

  void setNet(std::filesystem::path path, Network network) {
    netPath = std::move(path);
    loadedNet = std::move(network);
  }
};

}  // namespace hopfield

#endif
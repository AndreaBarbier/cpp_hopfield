#include "tokenizer.hpp"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace hopfield {

std::vector<std::string> tokenize(std::string const& line) {
  std::vector<std::string> tokens;
  std::string current{""};
  bool inQuotes{false};
  bool hasToken{false};
  for (auto const& c : line) {
    if (c == '"') {
      inQuotes = !inQuotes;
      hasToken = true;
    } else if (std::isspace(static_cast<unsigned char>(c)) && !inQuotes) {
      if (hasToken) {
        tokens.push_back(current);
        current.clear();
        hasToken = false;
      }
    } else {
      current += c;
      hasToken = true;
    }
  }
  if (inQuotes) {
    throw std::runtime_error("Error: unclosed quote.\n");
  }
  if (hasToken) {
    tokens.push_back(current);
  }
  return tokens;
}

std::string expandHome(std::string const& token) {
  if (token != "~" && token.rfind("~/", 0) != 0) {
    return token;
  }
  const char* home{std::getenv("HOME")};
  if (home == nullptr) {
    return token;
  }
  return std::string(home) + token.substr(1);
}

}  // namespace hopfield
#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

/// @file tokenizer.hpp
/// @brief Command-line tokenization and home directory expansion utilities

#include <string>
#include <vector>

namespace hopfield {

/// @brief Splits a command line into tokens, supporting double-quoted substrings
/// @details Whitespace outside double quotes separates tokens; empty quotes produce
///          an empty token.
/// @param[in] line Input command line to tokenize
/// @throws std::runtime_error if a double quote is left unclosed
/// @return Vector of extracted tokens
std::vector<std::string> tokenize(std::string const& line);

/// @brief Expands a leading `~` or `~/` using the `HOME` environment variable
/// @param[in] token Path or argument string to expand
/// @return Expanded path if `token` starts with `~` or `~/` and `HOME` is set;
///         otherwise returns `token` unchanged
std::string expandHome(std::string const& token);

}  // namespace hopfield

#endif
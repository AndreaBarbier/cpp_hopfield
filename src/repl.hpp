#ifndef REPL_HPP
#define REPL_HPP

/// @file repl.hpp
/// @brief Interactive read-eval-print loop for CLI commands

#include <istream>
#include <ostream>

#include "session.hpp"

namespace hopfield {

/// @brief Runs the interactive command loop reading lines from `in`
/// @details Ignores blank lines, exits on `exit`/`quit` or EOF, dispatches valid
///          commands, and prints caught `std::exception` messages to `out`.
/// @param[in,out] session Active user session state
/// @param[in,out] in Input stream providing command lines
/// @param[in,out] out Output stream for prompts, command output, and errors
void runRepl(Session& session, std::istream& in, std::ostream& out);

}  // namespace hopfield

#endif
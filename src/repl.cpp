#include "repl.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>
#include <string_view>

#include "commands.hpp"
#include "tokenizer.hpp"

namespace hopfield {

void runRepl(Session& session, std::istream& in, std::ostream& out) {
  std::string line;

  while (true) {
    out << "hopfield> ";

    if (!std::getline(in, line)) {
      out << '\n';
      break;
    }

    bool const isBlank{std::all_of(line.begin(), line.end(), [](unsigned char c) {
      return std::isspace(c);
    })};
    if (isBlank) {
      continue;
    }

    try {
      Args const tokens{tokenize(line)};
      if (tokens.empty()) {
        continue;
      }

      if (tokens[0] == "exit" || tokens[0] == "quit") {
        break;
      }

      dispatch(session, tokens, out);
    } catch (std::exception const& e) {
      out << e.what();
      if (!std::string_view{e.what()}.ends_with('\n')) {
        out << '\n';
      }
    }
  }
}

}  // namespace hopfield
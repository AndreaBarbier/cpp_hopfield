#include "commands.hpp"

#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hopfield.hpp"
#include "image_processor.hpp"
#include "recall_view.hpp"
#include "session.hpp"
#include "tokenizer.hpp"

namespace hopfield {

std::size_t parseMaxSteps(std::string const& text) {
  auto const fail{[] {
    return std::runtime_error(
        "Error: --max-steps must be a positive integer.\n");
  }};
  // stoull accepts negative numbers by wrapping them: reject them first.
  if (text.empty() || text[0] == '-') {
    throw fail();
  }
  unsigned long long val{0};
  std::size_t idx{0};
  try {
    val = std::stoull(text, &idx, 10);
  } catch (std::exception const&) {  // invalid_argument, out_of_range
    throw fail();
  }
  if (idx != text.size() || val == 0) {
    throw fail();
  }
  return static_cast<std::size_t>(val);
}

double parseNoise(std::string const& text) {
  auto const fail{[] {
    return std::runtime_error(
        "Error: --noise must be a number between 0 and 1.\n");
  }};
  double value{0.0};
  std::size_t idx{0};
  try {
    value = std::stod(text, &idx);
  } catch (std::exception const&) {
    throw fail();
  }
  // written this way so that NaN is rejected too
  if (idx != text.size() || !(value >= 0.0 && value <= 1.0)) {
    throw fail();
  }
  return value;
}

std::map<std::string, Command> const& commandsTable() {
  static std::map<std::string, Command> const commands{
      {"help",
       {"help [command]",
        "Show the list of commands, or the usage of a single one.", cmdHelp}},
      {"list",
       {"list",
        "List the patterns in the current dataset and show the common size.",
        cmdList}},
      {"use-dataset",
       {"use-dataset <directory>", "Set the current dataset directory.",
        cmdUseDataset}},
      {"add",
       {"add <image>",
        "Validate an image and copy it into the current dataset.", cmdAdd}},
      {"remove",
       {"remove <name>", "Remove a pattern from the current dataset.",
        cmdRemove}},
      {"new-net",
       {"new-net <directory> <name>",
        "Create a new untrained network in memory sized for the "
        "current dataset, and make it the current network.",
        cmdNewNet}},
      {"use-net",
       {"use-net <directory> <name>",
        "Load the network <directory>/<name>.hop and make it the current "
        "network.",
        cmdUseNet}},
      {"train",
       {"train",
        "Train the current network on the current dataset and save it.",
        cmdTrain}},
      {"recall",
       {"recall <image> [-o file] [--no-window] [--max-steps n]",
        "Run the current network on an image and report steps, convergence "
        "and energy.",
        cmdRecall}},
      {"corrupt",
       {"corrupt <image> -o <file> [--noise p]",
        "Flip each pixel of an image with probability p (default 0.25) and "
        "save the result.",
        cmdCorrupt}},
      {"info",
       {"info",
        "Show name, size, neuron count, training state and dataset "
        "compatibility of the current network.",
        cmdInfo}},
  };
  return commands;
}

void dispatch(Session& session, Args const& tokens, std::ostream& out) {
  auto it{commandsTable().find(tokens[0])};
  if (it == commandsTable().end()) {
    throw std::runtime_error("Error: unknown command. Try 'help'.");
  }
  Args args(tokens.begin() + 1, tokens.end());
  it->second.run(session, args, out);
}

void printHelp(std::map<std::string, Command> const& table, Args const& args,
               std::ostream& out) {
  if (args.size() > 1) {
    throw std::runtime_error("usage: help [command]");
  }

  if (args.size() == 1) {
    auto it{table.find(args[0])};
    if (it == table.end()) {
      throw std::runtime_error("unknown command '" + args[0] +
                               "', use 'help' to list the commands");
    }
    out << "Usage:  " << it->second.usage << "\n\n"
        << it->second.description << '\n';
    return;
  }

  size_t width{0};
  for (auto const& entry : table) {
    width = std::max(width, entry.second.usage.size());
  }

  out << "Available commands:\n\n";
  for (auto const& entry : table) {
    Command const& cmd = entry.second;
    out << "  " << cmd.usage << std::string(width - cmd.usage.size() + 2, ' ')
        << cmd.description << '\n';
  }
  out << "\nType 'help <command>' for the usage of a single command.\n"
      << "Type 'quit' or 'exit' (or press Ctrl-D) to leave.\n";
}

ParsedOptions parseOptions(Args const& args,
                           std::vector<OptionSpec> const& specs) {
  ParsedOptions result;

  for (std::size_t i{0}; i != args.size(); ++i) {
    std::string const& token = args[i];

    if (token.size() < 2 || token[0] != '-') {
      result.positional.push_back(token);
      continue;
    }

    bool const isLong = token.rfind("--", 0) == 0;
    std::string const name = token.substr(isLong ? 2 : 1);

    auto const spec =
        std::find_if(specs.begin(), specs.end(), [&](OptionSpec const& s) {
          return isLong ? s.longName == name
                        : (!s.shortName.empty() && s.shortName == name);
        });
    if (spec == specs.end()) {
      throw std::runtime_error("unknown option '" + token + "'");
    }

    if (spec->takesValue) {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("option '" + token + "' requires a value");
      }
      result.values[spec->longName] = args[++i];
    } else {
      result.flags.insert(spec->longName);
    }
  }

  return result;
}

void cmdHelp(Session&, Args const& args, std::ostream& out) {
  printHelp(commandsTable(), args, out);
}

void cmdList(Session& session, Args const& args, std::ostream& out) {
  if (!args.empty()) {
    throw std::runtime_error(
        "Error: 'list' does not accept any arguments. Try 'help list'.");
  }
  auto const names{session.dataset().getNames()};
  if (names.empty()) {
    throw std::runtime_error("Error: the dataset has no patterns.");
  }
  auto const size{session.dataset().size()};
  std::size_t const indexWidth{std::to_string(names.size()).size()};

  out << "Dataset: " << session.datasetPath.string() << '\n'
      << names.size() << (names.size() == 1 ? " pattern" : " patterns")
      << ", common size: " << size.x << "x" << size.y << '\n';

  for (std::size_t i{0}; i != names.size(); ++i) {
    std::string const idx{std::to_string(i + 1)};
    out << "  " << std::string(indexWidth - idx.size(), ' ') << idx << "  "
        << names[i] << '\n';
  }
}

void cmdUseDataset(Session& session, Args const& args, std::ostream& out) {
  if (args.size() != std::size_t{1}) {
    throw std::runtime_error(
        "Error: wrong numbers of arguments. Try 'help use-dataset'.\n");
  }
  auto const expanded{expandHome(args[0])};
  auto const path{std::filesystem::path{expanded}};
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error("Error: " + expanded +
                             " is not a valid directory.");
  }
  session.datasetPath = std::filesystem::path{path};
  out << "Dataset set to " << session.datasetPath.string() << '\n';
}

void cmdAdd(Session& session, Args const& args, std::ostream& out) {
  if (args.size() != std::size_t{1}) {
    throw std::runtime_error(
        "Error: wrong numbers of arguments. Try 'help add'.\n");
  }
  auto dataset{session.dataset()};
  dataset.add(expandHome(args[0]));
  out << "Added " << args[0] << " into dataset " << session.datasetPath.string()
      << '\n';
}

void cmdRemove(Session& session, Args const& args, std::ostream& out) {
  if (args.size() != std::size_t{1}) {
    throw std::runtime_error(
        "Error: wrong numbers of arguments. Try 'help remove'.\n");
  }
  auto dataset{session.dataset()};
  dataset.remove(args[0]);
  out << "Removed " << args[0] << " from dataset "
      << session.datasetPath.string() << '\n';
}

void cmdNewNet(Session& session, Args const& args, std::ostream& out) {
  if (args.size() != std::size_t{2}) {
    throw std::runtime_error(
        "Error: wrong numbers of arguments. Try 'help new-net'.\n");
  }
  std::filesystem::path const hop{std::filesystem::path{expandHome(args[0])} /
                                  (args[1] + ".hop")};
  auto dataset{session.dataset()};
  Network net{dataset.size()};
  session.setNet(hop, std::move(net));
  out << "Network initilized successfully.\n";
}

void cmdUseNet(Session& session, Args const& args, std::ostream& out) {
  if (args.size() != std::size_t{2}) {
    throw std::runtime_error(
        "Error: wrong numbers of arguments. Try 'help use-net'.\n");
  }
  std::filesystem::path const hop{expandHome(args[0]) + "/" + args[1] + ".hop"};
  Network net{Network::load(hop)};
  session.setNet(hop, std::move(net));
  out << "Network loaded successfully. from " << hop.string() << '\n';
}

void cmdTrain(Session& session, Args const& args, std::ostream& out) {
  if (!args.empty()) {
    throw std::runtime_error(
        "Error: wrong numbers of arguments. Try 'help train'.\n");
  }
  Dataset dataset{session.dataset()};
  auto patterns{dataset.patterns()};
  Network& net{session.net()};
  net.train(patterns);
  net.save(session.netFile());
  out << "Network trained and saved successfully.\n";
}

void cmdRecall(Session& session, Args const& args, std::ostream& out) {
  auto const parsed{parseOptions(args, {
                                           {"output", "o", true},
                                           {"no-window", "", false},
                                           {"max-steps", "", true},
                                       })};
  if (parsed.positional.size() != std::size_t{1}) {
    throw std::runtime_error(
        "Error: wrong number of arguments. Try 'help recall'.\n");
  }

  std::size_t maxSteps{1000};
  if (auto const it{parsed.values.find("max-steps")};
      it != parsed.values.end()) {
    maxSteps = parseMaxSteps(it->second);
  }

  Network const& net{session.net()};
  if (!net.isTrained()) {
    throw std::runtime_error("Error: the network is not trained.\n");
  }

  std::filesystem::path const imgPath{expandHome(parsed.positional[0])};
  if (!isImageFile(imgPath)) {
    throw std::runtime_error("Error: invalid image file '" + imgPath.string() +
                             "'.\n");
  }
  sf::Image img{};
  if (!img.loadFromFile(imgPath) || !isValidImage(img)) {
    throw std::runtime_error("Error: cannot load valid image '" +
                             imgPath.string() + "'.\n");
  }
  auto const netSize{net.size()};
  if (img.getSize() != netSize) {
    throw std::runtime_error(
        "Error: image size (" + std::to_string(img.getSize().x) + "x" +
        std::to_string(img.getSize().y) + ") does not match network size (" +
        std::to_string(netSize.x) + "x" + std::to_string(netSize.y) + ").\n");
  }

  PatternInt const inputPattern{imageToBinary(img)};

  std::size_t steps{0};
  double finalEnergy{0.0};
  auto const onStep{[&](PatternInt const&, std::size_t id, double energy) {
    steps = id;
    finalEnergy = energy;
  }};

  // Both throw std::runtime_error if the recall does not converge.
  PatternInt const result{
      parsed.flags.contains("no-window")
          ? net.recall(inputPattern, maxSteps, onStep)
          : runRecallView(net, inputPattern, maxSteps, onStep)};

  if (auto const outIt{parsed.values.find("output")};
      outIt != parsed.values.end()) {
    std::filesystem::path const outPath{expandHome(outIt->second)};
    if (!binaryToImage(result).saveToFile(outPath)) {
      throw std::runtime_error("Error: cannot save output image to '" +
                               outPath.string() + "'.\n");
    }
  }

  out << "Steps:     " << steps << '\n'
      << "Converged: yes\n"
      << "Energy:    " << finalEnergy << '\n';
}

void cmdCorrupt(Session&, Args const& args, std::ostream& out) {
  auto const parsed{parseOptions(args, {
                                           {"output", "o", true},
                                           {"noise", "", true},
                                       })};
  if (parsed.positional.size() != std::size_t{1}) {
    throw std::runtime_error(
        "Error: wrong number of arguments. Try 'help corrupt'.\n");
  }

  auto const outIt{parsed.values.find("output")};
  if (outIt == parsed.values.end()) {
    throw std::runtime_error(
        "Error: --output is required. Try 'help corrupt'.\n");
  }

  double noise{0.25};
  if (auto const it{parsed.values.find("noise")}; it != parsed.values.end()) {
    noise = parseNoise(it->second);
  }

  std::filesystem::path const inPath{expandHome(parsed.positional[0])};
  if (!isImageFile(inPath)) {
    throw std::runtime_error("Error: invalid image file '" + inPath.string() +
                             "'.\n");
  }

  std::filesystem::path const outPath{expandHome(outIt->second)};
  if (!corruptImage(inPath, noise).saveToFile(outPath)) {
    throw std::runtime_error("Error: cannot save output image to '" +
                             outPath.string() + "'.\n");
  }

  out << "Corrupted " << inPath.string() << " (noise " << noise << ").\n"
      << "Saved to " << outPath.string() << '\n';
}

void cmdInfo(Session& session, Args const& args, std::ostream& out) {
  if (!args.empty()) {
    throw std::runtime_error(
        "Error: wrong numbers of arguments. Try 'help info'.\n");
  }
  std::string name{session.netPath.stem().string()};
  Dataset dataset{session.dataset()};
  sf::Vector2u sizeData{dataset.size()};
  Network net{Network::load(session.netPath)};
  sf::Vector2u sizeNet{net.size()};
  bool trained{net.isTrained()};
  bool compatibility{sizeNet == sizeData};
  out << "Dataset:       " << session.datasetPath.string() << " (" << sizeData.x
      << 'x' << sizeData.y << ")\n"
      << "Network:       " << name << " (" << session.netPath.string() << ")\n"
      << "Network size:  " << sizeNet.x << 'x' << sizeNet.y << '\n'
      << "Trained:       " << (trained ? "yes" : "no") << '\n'
      << "Compatible:    " << (compatibility ? "yes" : "no") << '\n';
}

}  // namespace hopfield
#ifndef COMMANDS_HPP
#define COMMANDS_HPP

/// @file commands.hpp
/// @brief CLI command definitions, option parsing, and command handlers

#include <functional>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "session.hpp"

namespace hopfield {

// ============================================================
// TYPES AND DATA STRUCTURES
// ============================================================

/// @brief List of command-line arguments
using Args = std::vector<std::string>;

/// @brief Callable signature for a CLI command handler
using CommandFn = std::function<void(Session&, Args const&, std::ostream&)>;

/// @brief Metadata and execution handler for a single CLI command
struct Command {
  std::string usage;        ///< Usage syntax string shown in help messages
  std::string description;  ///< Short summary of what the command does
  CommandFn run;            ///< Function invoked to execute the command
};

/// @brief Specification of a command-line option accepted by parseOptions()
struct OptionSpec {
  std::string longName;   ///< Long option name without leading "--"
  std::string shortName;  ///< Optional short option name without leading "-"
  bool takesValue;        ///< True if the option expects a following argument
};

/// @brief Result of parsing positional arguments, valued options, and flags
struct ParsedOptions {
  Args positional;                            ///< Positional arguments in order
  std::map<std::string, std::string> values;  ///< Option values keyed by longName
  std::set<std::string> flags;                ///< Active boolean flags keyed by longName
};

// ============================================================
// COMMAND DISPATCH AND OPTION PARSING
// ============================================================

/// @brief Returns the registry mapping command names to their definitions
/// @return Constant reference to the static command lookup table
std::map<std::string, Command> const& commandsTable();

/// @brief Dispatches a tokenized command line to its matching handler
/// @pre tokens must not be empty
/// @param[in,out] session Active user session state
/// @param[in] tokens Tokenized input where tokens[0] is the command name
/// @param[in,out] out Output stream for command feedback
/// @throws std::runtime_error if the command name is not found or the handler fails
void dispatch(Session& session, Args const& tokens, std::ostream& out);

/// @brief Prints general command help or detailed usage for a single command
/// @param[in] table Command registry to inspect
/// @param[in] args Optional single-element vector naming the command to describe
/// @param[in,out] out Output stream where help text is written
/// @throws std::runtime_error if more than one argument is passed or the command is unknown
void printHelp(std::map<std::string, Command> const& table, Args const& args,
               std::ostream& out);

/// @brief Parses positional arguments, flags, and valued options according to specs
/// @param[in] args Raw argument list (excluding the command name)
/// @param[in] specs Supported option specifications
/// @throws std::runtime_error if an unknown option is encountered or a required value is missing
/// @return A ParsedOptions struct containing separated positional args, values, and flags
ParsedOptions parseOptions(Args const& args,
                           std::vector<OptionSpec> const& specs);

// ============================================================
// COMMAND HANDLERS
// ============================================================

/// @brief Displays the list of available commands or usage for a specific command
/// @param[in,out] session Active session (unused)
/// @param[in] args Optional command name
/// @param[in,out] out Output stream for help text
/// @throws std::runtime_error if arguments are invalid or the requested command does not exist
void cmdHelp(Session& session, Args const& args, std::ostream& out);

/// @brief Lists all patterns in the current dataset along with their common size
/// @param[in,out] session Active session providing the dataset
/// @param[in] args Must be empty
/// @param[in,out] out Output stream for the pattern listing
/// @throws std::runtime_error if arguments are provided or the dataset is empty
void cmdList(Session& session, Args const& args, std::ostream& out);

/// @brief Sets the active dataset directory in the session
/// @param[in,out] session Active session to update
/// @param[in] args Single-element vector containing the target directory path
/// @param[in,out] out Output stream for confirmation message
/// @throws std::runtime_error if the argument count is not 1 or the path is not a valid directory
void cmdUseDataset(Session& session, Args const& args, std::ostream& out);

/// @brief Validates an image and adds it to the current dataset
/// @param[in,out] session Active session providing the dataset
/// @param[in] args Single-element vector containing the source image path
/// @param[in,out] out Output stream for confirmation message
/// @throws std::runtime_error if the argument count is not 1 or adding the image fails
void cmdAdd(Session& session, Args const& args, std::ostream& out);

/// @brief Removes a pattern by name from the current dataset
/// @param[in,out] session Active session providing the dataset
/// @param[in] args Single-element vector containing the pattern name
/// @param[in,out] out Output stream for confirmation message
/// @throws std::runtime_error if the argument count is not 1 or removing the pattern fails
void cmdRemove(Session& session, Args const& args, std::ostream& out);

/// @brief Creates a new untrained network sized for the current dataset
/// @param[in,out] session Active session where the new network is stored
/// @param[in] args Two-element vector: `<directory>` and `<name>`
/// @param[in,out] out Output stream for confirmation message
/// @throws std::runtime_error if the argument count is not 2 or dataset size is invalid
void cmdNewNet(Session& session, Args const& args, std::ostream& out);

/// @brief Loads an existing `.hop` network file and sets it as the current network
/// @param[in,out] session Active session where the loaded network is stored
/// @param[in] args Two-element vector: `<directory>` and `<name>`
/// @param[in,out] out Output stream for confirmation message
/// @throws std::runtime_error if the argument count is not 2 or loading the network fails
void cmdUseNet(Session& session, Args const& args, std::ostream& out);

/// @brief Trains the current network on the current dataset and saves it to disk
/// @param[in,out] session Active session providing dataset and network
/// @param[in] args Must be empty
/// @param[in,out] out Output stream
/// @throws std::runtime_error if arguments are provided, dataset is empty, or saving fails
void cmdTrain(Session& session, Args const& args, std::ostream& out);

/// @brief Runs pattern recall on an input image and reports steps, convergence, and energy
/// @details Supports `-o/--output <file>`, `--no-window`, and `--max-steps <n>`.
/// @param[in,out] session Active session providing the trained network
/// @param[in] args Input image path and optional flags/values
/// @param[in,out] out Output stream for recall statistics
/// @throws std::runtime_error if options/arguments are invalid, image loading/size fails,
///         recall does not converge, or saving the output image fails
void cmdRecall(Session& session, Args const& args, std::ostream& out);

/// @brief Displays metadata, training status, and dataset compatibility of the current network
/// @param[in,out] session Active session providing dataset and network paths
/// @param[in] args Must be empty
/// @param[in,out] out Output stream for network information
/// @throws std::runtime_error if arguments are provided or loading the network fails
void cmdInfo(Session& session, Args const& args, std::ostream& out);

/// @brief Unlearns a pattern from the current network using the inverse Hebb rule
/// @param[in,out] session Active session providing the trained network
/// @param[in] args Single-element vector containing the image path to forget
/// @param[in,out] out Output stream for confirmation message
/// @throws std::runtime_error if the argument count is not 1, network is untrained,
///         or the image cannot be loaded or is smaller than the network size
void cmdForget(Session& session, Args const& args, std::ostream& out);

}  // namespace hopfield

#endif
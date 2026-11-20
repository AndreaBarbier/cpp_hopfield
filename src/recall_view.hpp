#ifndef RECALL_VIEW_HPP
#define RECALL_VIEW_HPP

/// @file recall_view.hpp
/// @brief Interactive SFML visualization for Hopfield network pattern recall and energy plot

#include <cstddef>
#include <functional>

#include "hopfield.hpp"
#include "image_processor.hpp"

namespace hopfield {

/// @brief Callback invoked on each step during recall visualization
using RecallStepFn = std::function<void(PatternInt const& pattern,
                                        std::size_t id, double energy)>;

/// @brief Runs pattern recall while displaying an SFML window with input/current images and energy plot
/// @details Divides the upper half of the window into two panels (corrupted input pattern on the left,
///          current/recalled pattern on the right) and renders the network energy curve in the lower half
///          using `Plot` and `STIXTwoMath-Regular.ttf` under `PROJECT_ROOT`. Captures every step via
///          the `Network::recall()` callback and forwards updates to `onStep`.
/// @param[in] net Trained Hopfield network used to recall the pattern
/// @param[in] inputPattern Initial binary pattern to recall from
/// @param[in] maxSteps Maximum number of recall sweeps allowed
/// @param[in] onStep Optional callback invoked on each recall step with pattern,
///                   sweep index, and energy
/// @throws std::runtime_error if recall fails to converge within maxSteps or if the font cannot be loaded
/// @return The final recalled binary pattern
PatternInt runRecallView(Network const& net, PatternInt const& inputPattern,
                         std::size_t maxSteps, RecallStepFn const& onStep);

}  // namespace hopfield

#endif
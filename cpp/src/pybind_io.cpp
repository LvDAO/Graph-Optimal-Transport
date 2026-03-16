#include "graphot/pybind_io.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace graphot::core {

void check_size(std::size_t actual, std::size_t expected, std::string_view name) {
    if (actual != expected) {
        throw std::invalid_argument(std::string(name) + " has unexpected size");
    }
}

void check_finite(const std::vector<double>& value, std::string_view name) {
    const bool all_finite = std::all_of(value.begin(), value.end(), [](double x) {
        return std::isfinite(x);
    });
    if (!all_finite) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

std::optional<State> parse_initial_state(
    py::handle initial_state_payload,
    int num_steps,
    int num_nodes,
    int num_edges
) {
    if (initial_state_payload.is_none()) {
        return std::nullopt;
    }

    const py::dict state_dict = py::cast<py::dict>(initial_state_payload);
    auto load_array = [&](std::string_view name, std::size_t expected_size) {
        const py::array_t<double, py::array::c_style | py::array::forcecast> array =
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(state_dict[py::str(name)]);
        check_size(array.size(), expected_size, name);
        std::vector<double> out(array.data(), array.data() + array.size());
        check_finite(out, name);
        return out;
    };

    State state(num_steps, num_nodes, num_edges);
    state.rho = load_array("rho", static_cast<std::size_t>(num_steps + 1) * static_cast<std::size_t>(num_nodes));
    state.m = load_array("m", static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges));
    state.vartheta = load_array(
        "vartheta",
        static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges)
    );
    state.rho_minus = load_array(
        "rho_minus",
        static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges)
    );
    state.rho_plus = load_array(
        "rho_plus",
        static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges)
    );
    state.rho_bar = load_array(
        "rho_bar",
        static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_nodes)
    );
    state.q_node = load_array(
        "q_node",
        static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_nodes)
    );
    return state;
}

}  // namespace graphot::core

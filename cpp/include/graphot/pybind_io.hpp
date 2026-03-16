#pragma once

#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "graphot/core_types.hpp"

namespace graphot::core {

namespace py = pybind11;

void check_size(std::size_t actual, std::size_t expected, std::string_view name);
void check_finite(const std::vector<double>& value, std::string_view name);

template <typename T>
py::array_t<T> make_array_from_vector(const std::vector<T>& data, std::vector<py::ssize_t> shape) {
    py::array_t<T> array(shape);
    std::memcpy(array.mutable_data(), data.data(), sizeof(T) * data.size());
    return array;
}

std::optional<State> parse_initial_state(
    py::handle initial_state_payload,
    int num_steps,
    int num_nodes,
    int num_edges
);

}  // namespace graphot::core

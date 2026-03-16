#pragma once

#include <cstddef>
#include <vector>

namespace graphot::core {

inline constexpr double kTiny = 1e-30;

inline std::size_t row_major(int row, int col, int width) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col);
}

const char* simd_name();
void configure_openmp_threads();
int configured_openmp_threads();
const char* configured_openmp_thread_source();
int graphot_default_openmp_threads();
int graphot_openmp_max_threads();
int graphot_openmp_num_procs();
bool should_parallelize(std::size_t work_items);

void project_zero_mean(std::vector<double>& value);
void vector_axpy(std::vector<double>& dst, const std::vector<double>& src, double alpha);
double vector_dot(const std::vector<double>& left, const std::vector<double>& right);
double vector_norm(const std::vector<double>& value);
int sign_of(double value);

}  // namespace graphot::core

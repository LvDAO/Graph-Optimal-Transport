#include "graphot/runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(GRAPHOT_USE_OPENMP)
#include <omp.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define GRAPHOT_HAS_TARGET_SIMD 1
#else
#define GRAPHOT_HAS_TARGET_SIMD 0
#endif

namespace graphot::core {
namespace {

constexpr std::size_t kParallelVectorOpsThreshold = 1u << 14;
constexpr std::size_t kParallelWorkThreshold = 1u << 10;

enum class SimdLevel {
    scalar,
    avx2,
    avx512,
};

double dot_scalar(const double* left, const double* right, std::size_t size) {
    double total = 0.0;
    for (std::size_t idx = 0; idx < size; ++idx) {
        total += left[idx] * right[idx];
    }
    return total;
}

void axpy_scalar(double* dst, const double* src, double alpha, std::size_t size) {
    for (std::size_t idx = 0; idx < size; ++idx) {
        dst[idx] += alpha * src[idx];
    }
}

#if GRAPHOT_HAS_TARGET_SIMD
__attribute__((target("avx2,fma"))) double dot_avx2(const double* left, const double* right, std::size_t size) {
    std::size_t idx = 0;
    __m256d acc = _mm256_setzero_pd();
    for (; idx + 4 <= size; idx += 4) {
        const __m256d lhs = _mm256_loadu_pd(left + idx);
        const __m256d rhs = _mm256_loadu_pd(right + idx);
        acc = _mm256_fmadd_pd(lhs, rhs, acc);
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, acc);
    double total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; idx < size; ++idx) {
        total += left[idx] * right[idx];
    }
    return total;
}

__attribute__((target("avx2,fma"))) void axpy_avx2(double* dst, const double* src, double alpha, std::size_t size) {
    const __m256d scale = _mm256_set1_pd(alpha);
    std::size_t idx = 0;
    for (; idx + 4 <= size; idx += 4) {
        const __m256d y = _mm256_loadu_pd(dst + idx);
        const __m256d x = _mm256_loadu_pd(src + idx);
        _mm256_storeu_pd(dst + idx, _mm256_fmadd_pd(scale, x, y));
    }
    for (; idx < size; ++idx) {
        dst[idx] += alpha * src[idx];
    }
}

__attribute__((target("avx512f,avx512vl,avx512dq,fma"))) double dot_avx512(
    const double* left,
    const double* right,
    std::size_t size
) {
    std::size_t idx = 0;
    __m512d acc = _mm512_setzero_pd();
    for (; idx + 8 <= size; idx += 8) {
        const __m512d lhs = _mm512_loadu_pd(left + idx);
        const __m512d rhs = _mm512_loadu_pd(right + idx);
        acc = _mm512_fmadd_pd(lhs, rhs, acc);
    }
    alignas(64) double lanes[8];
    _mm512_store_pd(lanes, acc);
    double total = 0.0;
    for (double lane : lanes) {
        total += lane;
    }
    for (; idx < size; ++idx) {
        total += left[idx] * right[idx];
    }
    return total;
}

__attribute__((target("avx512f,avx512vl,avx512dq,fma"))) void axpy_avx512(
    double* dst,
    const double* src,
    double alpha,
    std::size_t size
) {
    const __m512d scale = _mm512_set1_pd(alpha);
    std::size_t idx = 0;
    for (; idx + 8 <= size; idx += 8) {
        const __m512d y = _mm512_loadu_pd(dst + idx);
        const __m512d x = _mm512_loadu_pd(src + idx);
        _mm512_storeu_pd(dst + idx, _mm512_fmadd_pd(scale, x, y));
    }
    for (; idx < size; ++idx) {
        dst[idx] += alpha * src[idx];
    }
}
#endif

struct SimdOps {
    double (*dot)(const double*, const double*, std::size_t);
    void (*axpy)(double*, const double*, double, std::size_t);
    SimdLevel level;
    const char* name;
};

SimdOps resolve_simd() {
    const char* raw = std::getenv("GRAPHOT_SIMD");
    std::string forced = raw == nullptr ? "auto" : std::string(raw);
    std::transform(forced.begin(), forced.end(), forced.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    auto scalar = SimdOps{dot_scalar, axpy_scalar, SimdLevel::scalar, "scalar"};

#if GRAPHOT_HAS_TARGET_SIMD
    auto avx2 = SimdOps{dot_avx2, axpy_avx2, SimdLevel::avx2, "avx2"};
    auto avx512 = SimdOps{dot_avx512, axpy_avx512, SimdLevel::avx512, "avx512"};

    if (forced == "scalar") {
        return scalar;
    }
    if (forced == "avx2") {
        return avx2;
    }
    if (forced == "avx512") {
        return avx512;
    }
    if (forced != "auto" && !forced.empty()) {
        throw std::invalid_argument("GRAPHOT_SIMD must be one of: auto, scalar, avx2, avx512");
    }

    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl")
        && __builtin_cpu_supports("avx512dq")) {
        return avx512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return avx2;
    }
#else
    if (forced != "auto" && !forced.empty() && forced != "scalar") {
        throw std::invalid_argument("forced SIMD levels are only supported on x86_64 builds");
    }
#endif
    return scalar;
}

const SimdOps& simd_ops() {
    static const SimdOps ops = resolve_simd();
    return ops;
}

#if defined(GRAPHOT_USE_OPENMP)
int g_graphot_openmp_threads = 1;
const char* g_graphot_openmp_thread_source = "runtime_default";

int parse_positive_thread_count(const char* raw_value, std::string_view env_name) {
    if (raw_value == nullptr || raw_value[0] == '\0') {
        throw std::invalid_argument(std::string(env_name) + " must be a positive integer");
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw_value, &end, 10);
    if (end == raw_value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(env_name) + " must be a positive integer");
    }
    return static_cast<int>(parsed);
}
#else
constexpr int g_graphot_openmp_threads = 1;
const char* g_graphot_openmp_thread_source = "serial_build";
#endif

}  // namespace

const char* simd_name() {
    return simd_ops().name;
}

void configure_openmp_threads() {
#if defined(GRAPHOT_USE_OPENMP)
    const char* graphot_raw = std::getenv("GRAPHOT_NUM_THREADS");
    const char* omp_raw = std::getenv("OMP_NUM_THREADS");
    int configured_threads = graphot_default_openmp_threads();
    const char* source = "all_available_threads";
    if (graphot_raw != nullptr && graphot_raw[0] != '\0') {
        configured_threads = parse_positive_thread_count(graphot_raw, "GRAPHOT_NUM_THREADS");
        source = "GRAPHOT_NUM_THREADS";
    } else if (omp_raw != nullptr && omp_raw[0] != '\0') {
        configured_threads = parse_positive_thread_count(omp_raw, "OMP_NUM_THREADS");
        source = "OMP_NUM_THREADS";
    }
    omp_set_num_threads(configured_threads);
    g_graphot_openmp_threads = configured_threads;
    g_graphot_openmp_thread_source = source;
#endif
}

int configured_openmp_threads() {
    return g_graphot_openmp_threads;
}

const char* configured_openmp_thread_source() {
    return g_graphot_openmp_thread_source;
}

int graphot_default_openmp_threads() {
#if defined(GRAPHOT_USE_OPENMP)
    return std::max(1, omp_get_num_procs());
#else
    return 1;
#endif
}

int graphot_openmp_max_threads() {
#if defined(GRAPHOT_USE_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int graphot_openmp_num_procs() {
#if defined(GRAPHOT_USE_OPENMP)
    return omp_get_num_procs();
#else
    return 1;
#endif
}

bool should_parallelize(std::size_t work_items) {
#if defined(GRAPHOT_USE_OPENMP)
    return omp_get_max_threads() > 1 && work_items >= kParallelWorkThreshold;
#else
    (void)work_items;
    return false;
#endif
}

void project_zero_mean(std::vector<double>& value) {
    if (value.empty()) {
        return;
    }
    double total = 0.0;
    for (double x : value) {
        total += x;
    }
    const double mean = total / static_cast<double>(value.size());
    for (double& x : value) {
        x -= mean;
    }
}

void vector_axpy(std::vector<double>& dst, const std::vector<double>& src, double alpha) {
#if defined(GRAPHOT_USE_OPENMP)
    if (should_parallelize(dst.size()) && dst.size() >= kParallelVectorOpsThreshold) {
#pragma omp parallel
        {
            const int num_threads = omp_get_num_threads();
            const int thread_id = omp_get_thread_num();
            const std::size_t chunk = (dst.size() + static_cast<std::size_t>(num_threads) - 1)
                / static_cast<std::size_t>(num_threads);
            const std::size_t begin = std::min(dst.size(), static_cast<std::size_t>(thread_id) * chunk);
            const std::size_t end = std::min(dst.size(), begin + chunk);
            if (begin < end) {
                simd_ops().axpy(dst.data() + begin, src.data() + begin, alpha, end - begin);
            }
        }
        return;
    }
#endif
    simd_ops().axpy(dst.data(), src.data(), alpha, dst.size());
}

double vector_dot(const std::vector<double>& left, const std::vector<double>& right) {
#if defined(GRAPHOT_USE_OPENMP)
    if (should_parallelize(left.size()) && left.size() >= kParallelVectorOpsThreshold) {
        double total = 0.0;
#pragma omp parallel reduction(+:total)
        {
            const int num_threads = omp_get_num_threads();
            const int thread_id = omp_get_thread_num();
            const std::size_t chunk = (left.size() + static_cast<std::size_t>(num_threads) - 1)
                / static_cast<std::size_t>(num_threads);
            const std::size_t begin = std::min(left.size(), static_cast<std::size_t>(thread_id) * chunk);
            const std::size_t end = std::min(left.size(), begin + chunk);
            if (begin < end) {
                total += simd_ops().dot(left.data() + begin, right.data() + begin, end - begin);
            }
        }
        return total;
    }
#endif
    return simd_ops().dot(left.data(), right.data(), left.size());
}

double vector_norm(const std::vector<double>& value) {
    return std::sqrt(std::max(vector_dot(value, value), 0.0));
}

int sign_of(double value) {
    if (value > 0.0) {
        return 1;
    }
    if (value < 0.0) {
        return -1;
    }
    return 0;
}

}  // namespace graphot::core

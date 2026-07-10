#pragma once

#include <type_traits>

namespace tl {

// Numeric — the element types a Tensor may hold. Constraining the Tensor class
// template on this gives clear, early diagnostics ("constraint not satisfied")
// instead of deep template-instantiation errors when someone writes, e.g.,
// Tensor<std::string>.
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

} // namespace tl

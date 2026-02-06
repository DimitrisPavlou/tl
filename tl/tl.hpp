// tl/tl.hpp
#pragma once

// 1. View comes first (it's the most basic dependency)
#include "tensor_core/view.hpp"

// 2. Tensor comes second (depends on View)
#include "tensor_core/tensor.hpp"

// 3. Utils comes last (depends on Tensor and View)
#include "tensor_core/tensor_utils.hpp"

#include "linalg/linalg_utils.hpp"

#include "functional/functions.hpp"
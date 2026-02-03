// tl/tl.hpp
#pragma once

// 1. View comes first (it's the most basic dependency)
#include "view.hpp"

// 2. Tensor comes second (depends on View)
#include "tensor.hpp"

// 3. Utils comes last (depends on Tensor and View)
#include "tensor_utils.hpp"
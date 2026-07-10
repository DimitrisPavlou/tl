// tl/tl.hpp
#pragma once

// 1. View comes first (it's the most basic dependency)
#include "tensor_core/view.hpp"

// 2. Tensor comes second (depends on View)
#include "tensor_core/tensor.hpp"

// 3. Broadcasting utilities (depends on Tensor)
#include "tensor_core/broadcasting.hpp"

// 4. Utils comes last (depends on Tensor and View)
#include "tensor_core/tensor_utils.hpp"

// 5. Random factories, slicing (depend on Tensor)
#include "tensor_core/random.hpp"
#include "tensor_core/slicing.hpp"

#include "linalg/linalg_utils.hpp"

#include "functional/functions.hpp"

// 6. Axis-wise reductions + fused element-wise map (depend on Tensor)
#include "functional/reductions.hpp"
#include "functional/map.hpp"

// 7. Reverse-mode autograd (depends on Tensor, linalg, functional)
#include "autograd/autograd.hpp"

// 8. ODE integrators + heap-free SmallTensor (standalone)
#include "ode/small_tensor.hpp"
#include "ode/rk4.hpp"
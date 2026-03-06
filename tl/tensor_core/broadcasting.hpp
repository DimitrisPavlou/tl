#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>

namespace tl {

/**
 * Computes the resulting shape when broadcasting two shapes.
 * Following NumPy rules:
 * 1. Prepend 1s to the shorter shape until ranks match.
 * 2. Dimensions are compatible if they are equal or one of them is 1.
 */
inline std::vector<std::size_t> compute_broadcast_shape(
    const std::vector<std::size_t>& s1, 
    const std::vector<std::size_t>& s2) 
{
    std::size_t rank1 = s1.size();
    std::size_t rank2 = s2.size();
    std::size_t out_rank = std::max(rank1, rank2);
    
    std::vector<std::size_t> out_shape(out_rank);
    
    for (std::size_t i = 0; i < out_rank; ++i) {
        // Access dims from the end (right-aligned)
        std::size_t d1 = (i < rank1) ? s1[rank1 - 1 - i] : 1;
        std::size_t d2 = (i < rank2) ? s2[rank2 - 1 - i] : 1;
        
        if (d1 == d2) {
            out_shape[out_rank - 1 - i] = d1;
        } else if (d1 == 1) {
            out_shape[out_rank - 1 - i] = d2;
        } else if (d2 == 1) {
            out_shape[out_rank - 1 - i] = d1;
        } else {
            throw std::runtime_error("Shapes are not broadcast-compatible");
        }
    }
    
    return out_shape;
}

/**
 * Computes the strides for a shape when it is broadcasted to a target shape.
 * If a dimension is expanded (from 1 to N), its stride becomes 0.
 */
inline std::vector<std::size_t> get_broadcast_strides(
    const std::vector<std::size_t>& orig_shape,
    const std::vector<std::size_t>& orig_strides,
    const std::vector<std::size_t>& target_shape)
{
    std::size_t orig_rank = orig_shape.size();
    std::size_t target_rank = target_shape.size();
    
    std::vector<std::size_t> out_strides(target_rank, 0);
    
    for (std::size_t i = 0; i < orig_rank; ++i) {
        std::size_t d_orig = orig_shape[orig_rank - 1 - i];
        std::size_t d_target = target_shape[target_rank - 1 - i];
        
        if (d_orig == d_target) {
            out_strides[target_rank - 1 - i] = orig_strides[orig_rank - 1 - i];
        } else if (d_orig == 1) {
            out_strides[target_rank - 1 - i] = 0; // Broadcasting dimension!
        } else {
            throw std::runtime_error("Internal error: strides not broadcastable to target shape");
        }
    }
    
    return out_strides;
}

} // namespace tl

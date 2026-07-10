#pragma once
#include <vector>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <memory>

namespace tl {

// View is a NON-OWNING reference handle into a Tensor's buffer.
//
// Value semantics (important):
//   * Copy-CONSTRUCTION aliases:  `auto v = t[i];`  makes v refer to the same
//     memory as t[i] (it copies the pointers, not the elements).
//   * Copy-ASSIGNMENT writes through (NumPy-style):  `t[i] = t[j];`  copies the
//     source *elements* into this view's memory; it does not reseat the handle.
//   For an explicit, self-documenting element copy use `.copy_from(other)`.
//
// Because a View only borrows the owning Tensor's data/shape/strides, it must
// not outlive that Tensor (a full ownership redesign is planned separately).
template <typename T>
struct View {
    T* data_ptr;
    const std::size_t* shape_ptr;   // Points into the owning Tensor's shape array
    const std::size_t* strides_ptr; // Points into the owning Tensor's strides array
    std::size_t dims_left;          // Dimensions remaining in this view

    // Optional keep-alive handles. When a View is produced from a Tensor these
    // hold shared ownership of the Tensor's data/shape/strides buffers, so the
    // View stays valid even if the source Tensor is destroyed. They are null for
    // transient internal views (e.g. printing), which never outlive their Tensor.
    std::shared_ptr<const void> own_data{};
    std::shared_ptr<const void> own_shape{};
    std::shared_ptr<const void> own_strides{};

    View operator[](std::size_t index) {
        if (dims_left == 0) {
            throw std::out_of_range("Cannot index a 0-dimensional view (scalar)");
        }
        if (index >= shape_ptr[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(index) +
                " out of range for dimension of size " + std::to_string(shape_ptr[0]));
        }
        return View{
            data_ptr + (index * strides_ptr[0]),
            shape_ptr + 1,
            strides_ptr + 1,
            dims_left - 1,
            own_data, own_shape, own_strides   // propagate keep-alive
        };
    }

    // Const-correct indexing: indexing a const view yields a read-only
    // View<const T>, so writes cannot leak through a const handle.
    View<const T> operator[](std::size_t index) const {
        if (dims_left == 0) {
            throw std::out_of_range("Cannot index a 0-dimensional view (scalar)");
        }
        if (index >= shape_ptr[0]) {
            throw std::out_of_range(
                "Index " + std::to_string(index) +
                " out of range for dimension of size " + std::to_string(shape_ptr[0]));
        }
        return View<const T>{
            data_ptr + (index * strides_ptr[0]),
            shape_ptr + 1,
            strides_ptr + 1,
            dims_left - 1,
            own_data, own_shape, own_strides   // propagate keep-alive
        };
    }

    // Implicit conversion to an element reference. Only valid on a fully-indexed
    // (0-dimensional) view; converting a higher-rank view is a logic error and
    // used to silently return element 0 — now it throws.
    operator T&() const {
        if (dims_left != 0) {
            throw std::out_of_range(
                "Cannot convert a " + std::to_string(dims_left) +
                "-dimensional view to a scalar element; index it fully first");
        }
        return *data_ptr;
    }

    // Scalar assignment: tensor[i][j] = value
    T& operator=(const T& val) { *data_ptr = val; return *data_ptr; }

    // View-to-view assignment: NumPy-style write-through (see class note).
    View& operator=(const View& other) { return copy_from(other); }

    // Explicit element copy from another view into this view's memory.
    // Validates that both views cover the same total number of elements.
    //
    // Stride-aware: walks both views in row-major logical order and copies
    // element by element, honouring each view's own strides. This stays correct
    // for non-contiguous views (e.g. a transposed view or a column slice), not
    // just contiguous ones.
    View& copy_from(const View& other) {
        if (this == &other) return *this;

        std::size_t my_size = 1;
        for (std::size_t i = 0; i < dims_left; ++i) my_size *= shape_ptr[i];

        std::size_t other_size = 1;
        for (std::size_t i = 0; i < other.dims_left; ++i) other_size *= other.shape_ptr[i];

        if (my_size != other_size) {
            throw std::runtime_error(
                "View assignment size mismatch: destination has " + std::to_string(my_size) +
                " elements, source has " + std::to_string(other_size));
        }

        for (std::size_t i = 0; i < my_size; ++i) {
            std::size_t dst = offset_of(shape_ptr,       strides_ptr,       dims_left,       i);
            std::size_t src = offset_of(other.shape_ptr, other.strides_ptr, other.dims_left, i);
            data_ptr[dst] = other.data_ptr[src];
        }
        return *this;
    }

private:
    // Map a row-major flat index to a linear memory offset using shape/strides.
    static std::size_t offset_of(const std::size_t* shape, const std::size_t* strides,
                                 std::size_t dims, std::size_t flat) {
        std::size_t off = 0;
        for (std::size_t d = dims; d-- > 0; ) {
            std::size_t coord = flat % shape[d];
            flat /= shape[d];
            off += coord * strides[d];
        }
        return off;
    }
};

} // namespace tl

#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <initializer_list>
#include <utility>

namespace tl {

// SharedVec<T> — a std::vector-compatible container whose underlying buffer is
// held behind a std::shared_ptr.
//
// Why: a Tensor hands out lightweight View handles that point directly into its
// data/shape/strides buffers. If the Tensor dies while a View is alive, those
// raw pointers dangle. By storing each buffer in a shared control block, a View
// can grab a keep-alive (via share()) so the buffer outlives the Tensor.
//
// Value semantics are preserved: copying a SharedVec DEEP-COPIES the buffer
// (so copying a Tensor still yields an independent tensor). Only share() aliases.
template <typename T>
class SharedVec {
    std::shared_ptr<std::vector<T>> buf_;

public:
    using value_type     = T;
    using iterator       = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    // ── Construction ─────────────────────────────────────────────────────────
    SharedVec() : buf_(std::make_shared<std::vector<T>>()) {}
    SharedVec(std::vector<T> v)
        : buf_(std::make_shared<std::vector<T>>(std::move(v))) {}
    SharedVec(std::initializer_list<T> l)
        : buf_(std::make_shared<std::vector<T>>(l)) {}

    // Copy = deep copy (independent buffer) → preserves Tensor value semantics.
    SharedVec(const SharedVec& o)
        : buf_(std::make_shared<std::vector<T>>(*o.buf_)) {}
    SharedVec& operator=(const SharedVec& o) {
        if (this != &o) buf_ = std::make_shared<std::vector<T>>(*o.buf_);
        return *this;
    }
    SharedVec(SharedVec&&) noexcept = default;
    SharedVec& operator=(SharedVec&&) noexcept = default;
    ~SharedVec() = default;

    // Assignment from raw vectors / initializer lists (used by reshape, ctors).
    SharedVec& operator=(std::vector<T> v) {
        buf_ = std::make_shared<std::vector<T>>(std::move(v));
        return *this;
    }
    SharedVec& operator=(std::initializer_list<T> l) {
        buf_ = std::make_shared<std::vector<T>>(l);
        return *this;
    }

    // ── Element access ───────────────────────────────────────────────────────
    T&       operator[](std::size_t i)       { return (*buf_)[i]; }
    const T& operator[](std::size_t i) const { return (*buf_)[i]; }

    T*       data()       { return buf_->data(); }
    const T* data() const { return buf_->data(); }

    std::size_t size()  const { return buf_->size(); }
    bool        empty() const { return buf_->empty(); }

    iterator       begin()        { return buf_->begin(); }
    iterator       end()          { return buf_->end(); }
    const_iterator begin()  const { return buf_->begin(); }
    const_iterator end()    const { return buf_->end(); }

    // ── Mutating vector operations forwarded as-needed ───────────────────────
    void resize(std::size_t n)             { buf_->resize(n); }
    void reserve(std::size_t n)            { buf_->reserve(n); }
    void push_back(const T& v)             { buf_->push_back(v); }
    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last) {
        return buf_->insert(pos, first, last);
    }

    // ── Comparison (exact-match member beats the vector-conversion path) ──────
    bool operator==(const SharedVec& o) const { return *buf_ == *o.buf_; }
    bool operator!=(const SharedVec& o) const { return *buf_ != *o.buf_; }

    // ── Implicit vector view (lets free functions taking std::vector<T> work) ─
    operator const std::vector<T>&() const { return *buf_; }
    operator std::vector<T>&()             { return *buf_; }

    // ── Shared-ownership handle for Views (keep-alive) ───────────────────────
    std::shared_ptr<std::vector<T>> share() const { return buf_; }
};

} // namespace tl

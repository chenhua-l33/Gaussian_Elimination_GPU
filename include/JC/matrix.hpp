#pragma once

/**
 * @file matrix.hpp
 * @brief Template-based Matrix class implementation for numerical computations
 *
 * This file provides a robust matrix class implementation with:
 * - Template support for different numeric types
 * - Exception handling for matrix operations
 * - Cross-platform compatibility (Windows/Unix)
 * - STL-compatible iterators
 * - Basic matrix operations (transpose, multiply)
 * - Floating-point comparison utilities
 */

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <memory>
#ifdef _WIN32
#include <stdexcept>
#endif

namespace jc
{

    /**
     * @brief Custom exception class for matrix operations
     *
     * Provides detailed error messages for matrix-related operations
     * by prefixing all errors with "jc::Matrix: "
     */
    class MatrixException : public std::runtime_error
    {
    public:
        MatrixException(const std::string &message) : std::runtime_error("jc::Matrix: " + message) {}
    };

    /**
     * @brief Generic matrix class template for numerical computations
     *
     * Features:
     * - Row-major storage for compatibility with OpenCL
     * - Safe memory management with RAII
     * - Bounds checking on element access
     * - Support for matrix operations
     * - STL-compatible iterators
     *
     * @tparam T Numeric type for matrix elements (float, double, etc.)
     */
    template <typename T>
    class Matrix
    {
    public:
        /**
         * @brief Default constructor creates an empty 0x0 matrix
         */
        Matrix();

        /**
         * @brief Constructs a matrix with specified dimensions
         * @param rows Number of rows
         * @param cols Number of columns
         */
        Matrix(unsigned int rows, unsigned int cols);

        /**
         * @brief Constructs a matrix from existing data
         *
         * Copies the data from the provided array into a new matrix.
         * The source array must contain at least rows * cols elements.
         *
         * @param data Source array to copy from
         * @param rows Number of rows
         * @param cols Number of columns
         */
        Matrix(T *data, unsigned int rows, unsigned int cols);

        /**
         * @brief Copy constructor
         * Creates a deep copy of the source matrix
         */
        Matrix(const Matrix &);

        /**
         * @brief Destructor
         * Ensures proper cleanup of allocated memory
         */
        ~Matrix();

        /**
         * @brief Assignment operator
         *
         * IMPORTANT: Only allows assignment to uninitialized matrices
         * (where data_ == nullptr) to prevent memory leaks
         *
         * @throws MatrixException if target matrix is already initialized
         */
        Matrix &operator=(const Matrix &);

        /**
         * @brief Get number of rows
         * @return Number of rows in matrix
         */
        unsigned int rows() const { return rows_; }

        /**
         * @brief Get number of columns
         * @return Number of columns in matrix
         */
        unsigned int cols() const { return cols_; }

        /**
         * @brief Get raw data pointer
         * @return Pointer to matrix data array
         */
        T *data() { return data_; }

        /**
         * @brief Get const raw data pointer
         * @return Const pointer to matrix data array
         */
        const T *data() const { return data_; }

        /**
         * @brief Access matrix element with bounds checking
         * @throws MatrixException if indices are out of bounds
         */
        T &at(unsigned int row, unsigned int col);

        /**
         * @brief Const access to matrix element with bounds checking
         * @throws MatrixException if indices are out of bounds
         */
        T at(unsigned int row, unsigned int col) const;

        /**
         * @brief Fill matrix with a value
         * @param value Value to fill with (defaults to 0)
         */
        void fill(const T &value = 0);

        /**
         * @brief Create transposed copy of matrix
         * @return New matrix containing transposed data
         */
        Matrix transpose() const;

        /**
         * @brief Check if matrix is identity matrix
         * Uses Knuth's method for floating-point comparison
         * @return true if matrix is identity, false otherwise
         */
        bool isIdentity() const;

        /**
         * @brief STL-compatible iterator support
         * Enables use of matrix in standard algorithms
         */
        typedef T *iterator;
        typedef const T *const_iterator;
        iterator begin() { return &data_[0]; }
        iterator end() { return &data_[rows_ * cols_]; }
        const_iterator cbegin() const { return &data_[0]; }
        const_iterator cend() const { return &data_[rows_ * cols_]; }

    private:
        T *data_;
        unsigned int rows_;
        unsigned int cols_;
    };

    // implementation

    template <typename T>
    Matrix<T>::Matrix() : rows_(0), cols_(0), data_(nullptr) {}

    template <typename T>
    Matrix<T>::Matrix(unsigned int m, unsigned int n)
        : rows_(m), cols_(n)
    {
        data_ = new T[m * n];
    }

    template <typename T>
    Matrix<T>::Matrix(T *data, unsigned int m, unsigned int n)
        : rows_(m), cols_(n)
    {
        data_ = new T[m * n];
#ifdef _WIN32
        std::copy<T *>(data, data + m * n, stdext::make_checked_array_iterator(begin(), m * n));
#else
        std::copy<T *>(data, data + m * n, begin());
#endif
    }

    template <typename T>
    Matrix<T>::Matrix(const Matrix<T> &other)
        : rows_(other.rows_), cols_(other.cols_)
    {
        data_ = new T[rows_ * cols_];
#ifdef _WIN32
        // must be a bug in the windows library
        std::copy<T *>(const_cast<T *>(other.cbegin()), const_cast<T *>(other.cend()),
                       stdext::make_checked_array_iterator(begin(), rows_ * cols_));
#else
        std::copy<T *>(other.cbegin(), other.cend(), begin());
#endif
    }

    template <typename T>
    Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
    {
        if (data_ != nullptr)
        {
            throw MatrixException("Assigning to initialized matrix");
        }
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = new T[other.rows_ * other.cols_];
#ifdef _WIN32
        // must be a bug in the windows library
        std::copy<T *>(const_cast<T *>(other.cbegin()), const_cast<T *>(other.cend()),
                       stdext::make_checked_array_iterator(begin(), rows_ * cols_));
#else
        std::copy<T *>(other.cbegin(), other.cend(), begin());
#endif
        return *this;
    }

    template <typename T>
    Matrix<T>::~Matrix()
    {
        delete[] data_;
    }

    template <typename T>
    T &Matrix<T>::at(unsigned int i, unsigned int j)
    {
        if (i >= rows_ || j >= cols_)
        {
            throw MatrixException("index out of bounds");
        }
        return data_[cols_ * i + j];
    }

    template <typename T>
    T Matrix<T>::at(unsigned int i, unsigned int j) const
    {
        if (i >= rows_ || j >= cols_)
        {
            throw MatrixException("index out of bounds");
        }
        return data_[cols_ * i + j];
    }

    template <typename T>
    void Matrix<T>::fill(const T &value = 0)
    {
        for (auto &e : *this)
            e = value;
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const
    {
        Matrix<T> t(cols_, rows_);
        for (unsigned int i = 0; i < rows_; ++i)
        {
            for (unsigned int j = 0; j < cols_; ++j)
            {
                t.at(j, i) = data_[i * cols_ + j];
            }
        }
        return t;
    }

    // uses a comparison method suggested by KNUTH
    template <typename T>
    bool Matrix<T>::isIdentity() const
    {
        T e = static_cast<T>(0.000001);
        if (rows_ != cols_)
        {
            return false;
        }
        for (unsigned int i = 0; i < rows_; ++i)
        {
            for (unsigned int j = 0; j < cols_; ++j)
            {
                T u = data_[i * cols_ + j];
                if (i == j)
                {
                    if (std::abs(u - 1) > e * std::abs(u) &&
                        std::abs(u - 1) > e * 1)
                    {
                        return false;
                    }
                }
                else
                {
                    if (std::abs(u) > e * std::abs(u))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * @brief Robust floating-point comparison function
     *
     * Implementation based on guidelines from:
     * http://floating-point-gui.de/errors/comparison/
     *
     * This function handles various edge cases in floating-point comparison:
     * - Exact equality
     * - Near-zero values
     * - Very large/small number comparisons
     * - Relative error comparison for normal ranges
     *
     * @param a First value to compare
     * @param b Second value to compare
     * @param e Epsilon value for comparison tolerance
     * @return true if values are considered equal within tolerance
     */
    template <typename T>
    bool nearlyEqual(T a, T b, T e)
    {
        T abs_a = std::abs(a);
        T abs_b = std::abs(b);
        T delta = std::abs(a - b);

        if (a == b)
        {
            return true;
        }
        else if (a == 0 || b == 0 || delta < DBL_MIN)
        {
            return delta < e * DBL_MIN; // Special handling for zero/subnormal
        }
        else
        {
            return delta / (abs_a + abs_b) < e; // Relative error comparison
        }
    }

    /**
     * @brief Matrix equality comparison operator
     *
     * Uses Knuth's method for floating-point comparison via nearlyEqual().
     * Provides detailed diagnostic output on comparison failure.
     *
     * @param a First matrix to compare
     * @param b Second matrix to compare
     * @return true if matrices are equal within tolerance
     */
    template <typename T>
    bool operator==(const Matrix<T> &a, const Matrix<T> &b)
    {
        T e = static_cast<T>(0.00001); // Comparison tolerance

        // Check dimensions
        if (a.rows() != b.rows() || a.cols() != b.cols())
        {
            return false;
        }

        // Compare elements with diagnostic output
        for (unsigned int i = 0; i < a.rows(); ++i)
        {
            for (unsigned int j = 0; j < a.cols(); ++j)
            {
                T u = a.at(i, j);
                T v = b.at(i, j);
                if (!nearlyEqual<T>(u, v, e))
                {
                    std::cerr << "FAILED: @ (";
                    std::cerr << i << "," << j << "): ";
                    std::cerr << u << " vs " << v << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Matrix multiplication operator
     *
     * Performs standard matrix multiplication: C = A * B
     * Checks dimension compatibility and throws if invalid.
     *
     * @param a Left matrix operand
     * @param b Right matrix operand
     * @return Result matrix of dimensions (a.rows x b.cols)
     * @throws MatrixException if dimensions are incompatible
     */
    template <typename T>
    Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b)
    {
        if (a.cols() != b.rows())
        {
            throw MatrixException("cannot multiply matrices of these dimensions");
        }

        Matrix<T> c(a.rows(), b.cols());
        for (unsigned int i = 0; i < c.rows(); ++i)
        {
            for (unsigned int j = 0; j < c.cols(); ++j)
            {
                c.at(i, j) = 0;
                for (unsigned int x = 0; x < a.cols(); ++x)
                {
                    c.at(i, j) += a.at(i, x) * b.at(x, j);
                }
            }
        }

        return c;
    }

    /**
     * @brief Matrix output stream operator
     *
     * Formats matrix for human-readable output with space-separated values
     * and newlines between rows.
     *
     * @param oss Output stream
     * @param a Matrix to output
     * @return Modified output stream
     */
    template <typename T>
    std::ostream &operator<<(std::ostream &oss, const Matrix<T> &a)
    {
        for (unsigned int i = 0; i < a.rows(); ++i)
        {
            for (unsigned int j = 0; j < a.cols(); ++j)
            {
                oss << a.at(i, j) << " ";
            }
            oss << std::endl;
        }
        return oss;
    }
}
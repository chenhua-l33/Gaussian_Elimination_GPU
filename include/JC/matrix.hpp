#pragma once

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <memory>
#ifdef _WIN32
#include <stdexcept>
#endif

namespace jc {

    class MatrixException : public std::runtime_error {
    public:
        MatrixException(const std::string &message) :
            std::runtime_error("jc::Matrix: " + message) {}
    };

    template <typename T>
    class Matrix {
    public:

        Matrix();
        Matrix(unsigned int, unsigned int);

        // copies the data pointed to, not the pointer
        Matrix(T *data, unsigned int, unsigned int);
        Matrix(const Matrix &);
        ~Matrix();

        // only assigning to uninitialized matrices is allowed
        // an uninitialized matrix is one with data_ == nullptr
        Matrix& operator=(const Matrix &);

        unsigned int rows() const { return rows_;  }
        unsigned int cols() const { return cols_;  }
        
        T *data() { return data_;  }
        const T *data() const  { return data_; }
        
        T& at(unsigned int, unsigned int);
        T at(unsigned int, unsigned int) const;
        
        void fill(const T& value = 0);
        
        Matrix transpose() const;
        
        bool isIdentity() const;

        typedef T * iterator;
        typedef const T * const_iterator;
        iterator begin()  { return &data_[0]; }
        iterator end()    { return &data_[rows_*cols_]; }
        const_iterator cbegin() const { return &data_[0]; }
        const_iterator cend()   const { return &data_[rows_*cols_]; }

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
        data_ = new T[m*n];
    }

    template <typename T>
    Matrix<T>::Matrix(T* data, unsigned int m, unsigned int n)
        : rows_(m), cols_(n)
    {
        data_ = new T[m*n];
#ifdef _WIN32
        std::copy<T*>(data, data + m*n, stdext::make_checked_array_iterator(begin(), m*n));
#else
        std::copy<T*>(data, data + m*n, begin());
#endif
    }

    template <typename T>
    Matrix<T>::Matrix(const Matrix<T> &other)
        : rows_(other.rows_), cols_(other.cols_)
    {
        data_ = new T[rows_*cols_];
#ifdef _WIN32
        // must be a bug in the windows library
        std::copy<T*>(const_cast<T*>(other.cbegin()), const_cast<T*>(other.cend()),
            stdext::make_checked_array_iterator(begin(), rows_*cols_));
#else
        std::copy<T*>(other.cbegin(), other.cend(), begin());
#endif
    }

    template <typename T>
    Matrix<T>& Matrix<T>::operator=(const Matrix<T> &other) 
    {
        if (data_ != nullptr) {
            throw MatrixException("Assigning to initialized matrix");
        }
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = new T[other.rows_*other.cols_];
#ifdef _WIN32
        // must be a bug in the windows library
        std::copy<T*>(const_cast<T*>(other.cbegin()), const_cast<T*>(other.cend()),
            stdext::make_checked_array_iterator(begin(), rows_*cols_));
#else
        std::copy<T*>(other.cbegin(), other.cend(), begin());
#endif
        return *this;
    }

    template <typename T>
    Matrix<T>::~Matrix()
    {
        delete[] data_;
    }

    template <typename T>
    T& Matrix<T>::at(unsigned int i, unsigned int j)
    {
        if (i >= rows_ || j >= cols_) {
            throw MatrixException("index out of bounds");
        }
        return data_[cols_*i + j];
    }

    template <typename T>
    T Matrix<T>::at(unsigned int i, unsigned int j) const
    {
        if (i >= rows_ || j >= cols_) {
            throw MatrixException("index out of bounds");
        }
        return data_[cols_*i + j];
    }

    template <typename T>
    void Matrix<T>::fill(const T &value = 0)
    {
        for (auto &e : *this) e = value;
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const
    {
        Matrix<T> t(cols_, rows_);
        for (unsigned int i = 0; i < rows_; ++i) {
            for (unsigned int j = 0; j < cols_; ++j) {
                t.at(j, i) = data_[i*cols_ + j];
            }
        }
        return t;
    }

    // uses a comparison method suggested by KNUTH 
    template <typename T>
    bool Matrix<T>::isIdentity() const
    {
        T e = static_cast<T>(0.000001);
        if (rows_ != cols_) {
            return false;
        }
        for (unsigned int i = 0; i < rows_; ++i) {
            for (unsigned int j = 0; j < cols_; ++j) {
                T u = data_[i*cols_ + j];
                if (i == j) {
                    if (std::abs(u - 1) > e * std::abs(u) &&
                        std::abs(u - 1) > e * 1) {
                        return false;
                    }
                }
                else {
                    if (std::abs(u) > e * std::abs(u)) { 
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // alternative comparison
    // taken from http://floating-point-gui.de/errors/comparison/
    template <typename T>
    bool nearlyEqual(T a, T b, T e)
    {
        T abs_a = std::abs(a);
        T abs_b = std::abs(b);
        T delta = std::abs(a - b);

        if (a == b) {
            return true;
        }
        else if (a == 0 || b == 0 || delta < DBL_MIN) {
            return delta < e*DBL_MIN;
        }
        else {
            return delta / (abs_a + abs_b) < e;
        }

    }

    // used a comparison method suggested by KNUTH 
    // now uses nearlyEqual
    template <typename T>
    bool operator==(const Matrix<T> &a, const Matrix<T> &b)
    {
        T e = static_cast<T>(0.00001);
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            return false;
        }
        for (unsigned int i = 0; i < a.rows(); ++i) {
            for (unsigned int j = 0; j < a.cols(); ++j) {
                T u = a.at(i, j);
                T v = b.at(i, j);
                if (!nearlyEqual<T>(u, v, e)) {
                    std::cerr << "FAILED: @ (";
                    std::cerr << i << "," << j << "): ";
                    std::cerr << u << " vs " << v << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b)
    {
        if (a.cols() != b.rows()) {
            throw MatrixException("cannot multiply matrices of these dimensions");
        }

        Matrix<T> c(a.rows(), b.cols());
        for (unsigned int i = 0; i < c.rows(); ++i) {
            for (unsigned int j = 0; j < c.cols(); ++j) {
                c.at(i, j) = 0;
                for (unsigned int x = 0; x < a.cols(); ++x) {
                    c.at(i, j) += a.at(i, x) * b.at(x, j);
                }
            }
        }

        return c;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream &oss, const Matrix<T> &a)
    {
        for (unsigned int i = 0; i < a.rows(); ++i) {
            for (unsigned int j = 0; j < a.cols(); ++j) {
                oss << a.at(i, j) << " ";
            }
            oss << std::endl;
        }
        return oss;
    }
}
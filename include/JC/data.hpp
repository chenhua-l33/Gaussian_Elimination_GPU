#ifndef __JC_DATA_HPP_
#define __JC_DATA_HPP_

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <time.h>

#include <cstdlib>

namespace jc {

template <typename T>
class Distribution {
public:
    virtual T get_value() = 0;
}; 

template <typename T>
class ConstantDistribution : public Distribution<T> {
public:
    ConstantDistribution(T value) : value_(value) {}

    T get_value()
    {
        return value_;
    }

private:
    T value_;
};

template <typename T>
class RandomDistribution : public Distribution<T> {
public:
    RandomDistribution(T value) : max_value_(value) 
    {   
        srand ( (unsigned) time(0) );
    }

	 T get_value()
    {
        float fraction = static_cast<float>(max_value_) / RAND_MAX;
        return static_cast<T>(rand() * fraction);
    }

private:
    T max_value_;
};

template <typename T>
class Data {
public: 
    Data(int X, int Y = 1, int Z = 1) 
        : X_(X), Y_(Y), Z_(Z)
    {
        data_ = static_cast<T*>(malloc(X * Y * Z * sizeof(T)));
        if (!data_) {
            std::ostringstream oss;
            oss << "malloc failed to allocate "
                << X * Y * Z * sizeof(T) << " bytes";
            throw std::runtime_error(oss.str());
        }
    }

    ~Data()
    {
        free(data_);
    }

    int X() const
    {
        return X_;
    }

    int Y() const
    {
        return Y_;
    }

    int Z() const
    {
        return Z_;
    }

    const T& get(int x, int y = 0, int z = 0) const 
    {
        return data_[Y_ * X_ * z + X_ * y + x];
    }
  
    void set(int x, int y, int z, T val) 
    {
        data_[Y_ * X_ * z + X_ * y + x] = val;
    }

    T *data() 
    {
        return data_;
    }

    void fill(Distribution<T>& distribution)
    {
        for (int z = 0; z < Z_; ++z) {
            for (int y = 0; y < Y_; ++y) {
                for (int x = 0; x < X_; ++x) {
                    set(x, y, z, distribution.get_value());
                }
            }
        }
    }

	void clear()
    {
        for (int z = 0; z < Z_; ++z) {
            for (int y = 0; y < Y_; ++y) {
                for (int x = 0; x < X_; ++x) {
                    set(x, y, z, 0);
                }
            }
        }
    }

    bool equals(const Data<T>& other) const {
        if (X_ != other.X_ || Y_ != other.Y_ || Z_ != other.Z_) {
            return false;
        }
        for (int z = 0; z < Z_; ++z) {
            for (int y = 0; y < Y_; ++y) {
                for (int x = 0; x < X_; ++x) {
                    if (get(x, y, z) != other.get(x, y, z)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }


private:
  T *data_;
  int X_;
  int Y_;
  int Z_;

};

} 

#endif

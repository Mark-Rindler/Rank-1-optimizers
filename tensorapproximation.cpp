#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

class Tensor3D {
private:
    std::vector<double> data;
    size_t I, J, K;

public:
    Tensor3D(size_t i, size_t j, size_t k) : I(i), J(j), K(k) {
        data.resize(I * J * K, 0.0);
    }

    double& operator()(size_t i, size_t j, size_t k) {
        return data[i * J * K + j * K + k];
    }

    const double& operator()(size_t i, size_t j, size_t k) const {
        return data[i * J * K + j * K + k];
    }

    size_t dim1() const { return I; }
    size_t dim2() const { return J; }
    size_t dim3() const { return K; }
};

double norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

void normalize(std::vector<double>& v) {
    double n = norm(v);
    if (n > 1e-10) {
        for (double& val : v) {
            val /= n;
        }
    }
}

void mode1_product(const Tensor3D& A, const std::vector<double>& a, 
                   std::vector<std::vector<double>>& result) {
    size_t J = A.dim2();
    size_t K = A.dim3();
    
    result.assign(J, std::vector<double>(K, 0.0));
    
    for (size_t j = 0; j < J; ++j) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t i = 0; i < a.size(); ++i) {
                result[j][k] += A(i, j, k) * a[i];
            }
        }
    }
}

void mode2_product(const Tensor3D& A, const std::vector<double>& b,
                   std::vector<std::vector<double>>& result) {
    size_t I = A.dim1();
    size_t K = A.dim3();
    
    result.assign(I, std::vector<double>(K, 0.0));
    
    for (size_t i = 0; i < I; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < b.size(); ++j) {
                result[i][k] += A(i, j, k) * b[j];
            }
        }
    }
}

void mode3_product(const Tensor3D& A, const std::vector<double>& c,
                   std::vector<std::vector<double>>& result) {
    size_t I = A.dim1();
    size_t J = A.dim2();
    
    result.assign(I, std::vector<double>(J, 0.0));
    
    for (size_t i = 0; i < I; ++i) {
        for (size_t j = 0; j < J; ++j) {
            for (size_t k = 0; k < c.size(); ++k) {
                result[i][j] += A(i, j, k) * c[k];
            }
        }
    }
}

void update_a(const Tensor3D& A, const std::vector<double>& b, 
              const std::vector<double>& c, std::vector<double>& a) {
    size_t I = A.dim1();
    a.assign(I, 0.0);
    
    for (size_t i = 0; i < I; ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            for (size_t k = 0; k < c.size(); ++k) {
                a[i] += A(i, j, k) * b[j] * c[k];
            }
        }
    }
}

void update_b(const Tensor3D& A, const std::vector<double>& a,
              const std::vector<double>& c, std::vector<double>& b) {
    size_t J = A.dim2();
    b.assign(J, 0.0);
    
    for (size_t j = 0; j < J; ++j) {
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t k = 0; k < c.size(); ++k) {
                b[j] += A(i, j, k) * a[i] * c[k];
            }
        }
    }
}

void update_c(const Tensor3D& A, const std::vector<double>& a,
              const std::vector<double>& b, std::vector<double>& c) {
    size_t K = A.dim3();
    c.assign(K, 0.0);
    
    for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                c[k] += A(i, j, k) * a[i] * b[j];
            }
        }
    }
}

double compute_lambda(const Tensor3D& A, const std::vector<double>& a,
                      const std::vector<double>& b, const std::vector<double>& c) {
    double lambda = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            for (size_t k = 0; k < c.size(); ++k) {
                lambda += A(i, j, k) * a[i] * b[j] * c[k];
            }
        }
    }
    
    return lambda;
}

double compute_residual_norm(const Tensor3D& A, const std::vector<double>& a,
                             const std::vector<double>& b, const std::vector<double>& c,
                             double lambda) {
    double residual = 0.0;
    
    for (size_t i = 0; i < A.dim1(); ++i) {
        for (size_t j = 0; j < A.dim2(); ++j) {
            for (size_t k = 0; k < A.dim3(); ++k) {
                double diff = A(i, j, k) - lambda * a[i] * b[j] * c[k];
                residual += diff * diff;
            }
        }
    }
    
    return std::sqrt(residual);
}

struct Rank1Result {
    std::vector<double> a, b, c;
    double lambda;
    int iterations;
    double residual_norm;
};

Rank1Result rank1_approximation(const Tensor3D& A, int max_iter = 1000, 
                                double tol = 1e-16, unsigned seed = 42) {
    size_t I = A.dim1();
    size_t J = A.dim2();
    size_t K = A.dim3();
    
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    std::vector<double> a(I), b(J), c(K);
    
    for (double& val : b) val = dist(gen);
    for (double& val : c) val = dist(gen);
    normalize(b);
    normalize(c);
    
    double lambda = 0.0;
    double prev_lambda = 0.0;
    int iter = 0;
    
    for (iter = 0; iter < max_iter; ++iter) {
        update_a(A, b, c, a);
        normalize(a);
        
        update_b(A, a, c, b);
        normalize(b);
        
        update_c(A, a, b, c);
        normalize(c);
        
        lambda = compute_lambda(A, a, b, c);
        
        if (iter > 0 && std::abs(lambda - prev_lambda) / (std::abs(prev_lambda) + 1e-12) < tol) break;
        
        prev_lambda = lambda;
    }
    
    double residual = compute_residual_norm(A, a, b, c, lambda);
    
    return {a, b, c, lambda, iter + 1, residual};
}

int main() {
    size_t I = 5, J = 4, K = 3;
    Tensor3D A(I, J, K);
    
    std::mt19937 gen(123);
    std::normal_distribution<double> noise(0.0, 0.1);
    
    std::vector<double> true_a = {1.0, 0.5, -0.3, 0.8, 0.2};
    std::vector<double> true_b = {0.7, -0.4, 0.9, 0.1};
    std::vector<double> true_c = {0.6, 0.8, -0.5};
    
    normalize(true_a);
    normalize(true_b);
    normalize(true_c);
    
    double true_lambda = 10.0;
    
    for (size_t i = 0; i < I; ++i) {
        for (size_t j = 0; j < J; ++j) {
            for (size_t k = 0; k < K; ++k) {
                A(i, j, k) = true_lambda * true_a[i] * true_b[j] * true_c[k] + noise(gen);
            }
        }
    }
    
    auto result = rank1_approximation(A);
    
    std::cout << "Converged in " << result.iterations << " iterations\n";
    std::cout << "Lambda: " << std::fixed << std::setprecision(6) << result.lambda << "\n";
    std::cout << "Residual norm: " << result.residual_norm << "\n\n";
    
    std::cout << "Factor a: [";
    for (size_t i = 0; i < result.a.size(); ++i) {
        std::cout << std::setw(8) << result.a[i];
        if (i < result.a.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "Factor b: [";
    for (size_t i = 0; i < result.b.size(); ++i) {
        std::cout << std::setw(8) << result.b[i];
        if (i < result.b.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "Factor c: [";
    for (size_t i = 0; i < result.c.size(); ++i) {
        std::cout << std::setw(8) << result.c[i];
        if (i < result.c.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    return 0;
}
#include <iostream>
#include <cmath>
using namespace std;

// ----------------- BASIC UTILITIES -----------------

// Simple linear congruential generator
unsigned long long seed = 123456789ULL;
double randu() {
    seed = (6364136223846793005ULL * seed + 1ULL);
    return ((seed >> 11) & 0xFFFFFFFF) / 4294967296.0;
}

// Generate uniform(-1, 1) random number
double rand_uniform() {
    return 2.0 * randu() - 1.0;
}

// Create new random 3x3 matrix each run
void random_matrix(double A[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            A[i][j] = rand_uniform();
}

// Dot product
double dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

// Norm
double norm(const double* a, int n) {
    return sqrt(dot(a, a, n));
}

// Print 3-vector
void print_vec(const char* name, const double* v) {
    cout << name << " = [";
    for (int i = 0; i < 3; ++i) {
        cout << v[i];
        if (i < 2) cout << ", ";
    }
    cout << "]\n";
}

// Compute Frobenius error
double compute_error(double A[3][3], const double* u, const double* v) {
    double error = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double diff = A[i][j] - u[i]*v[j];
            error += diff * diff;
        }
    return sqrt(error);
}

// ----------------- MATRIX HELPERS -----------------

void mat_transpose(double M[3][3], double Mt[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Mt[j][i] = M[i][j];
}

void mat_mul(double A[3][3], double B[3][3], double C[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < 3; ++k)
                C[i][j] += A[i][k]*B[k][j];
        }
}

void matvec(double M[3][3], const double v[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = M[i][0]*v[0] + M[i][1]*v[1] + M[i][2]*v[2];
}

// ----------------- POWER ITERATION (dominant eigenpair) -----------------

void dominant_eigenpair(double A[3][3], double* eigvec, double& eigval, int max_iter=1000, double tol=1e-9) {
    eigvec[0] = 1; eigvec[1] = 1; eigvec[2] = 1;
    double nrm = norm(eigvec, 3);
    for (int i = 0; i < 3; ++i) eigvec[i] /= nrm;

    double prev_lambda = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        double y[3];
        matvec(A, eigvec, y);
        double lambda = dot(y, eigvec, 3);
        double ny = norm(y, 3);
        for (int i = 0; i < 3; ++i) eigvec[i] = y[i] / ny;

        if (fabs(lambda - prev_lambda) < tol) break;
        prev_lambda = lambda;
        eigval = lambda;
    }
}

// ----------------- GLOBAL OPTIMAL RANK-1 APPROX -----------------

void global_optimal_rank1(double A[3][3], double* u_opt, double* v_opt) {
    // Compute AtA
    double At[3][3], AtA[3][3];
    mat_transpose(A, At);
    mat_mul(At, A, AtA);

    // Get dominant eigenvector of AtA
    double lambda;
    dominant_eigenpair(AtA, v_opt, lambda);

    // Compute u = (A * v)
    matvec(A, v_opt, u_opt);
    double sigma1 = norm(u_opt, 3);
    for (int i = 0; i < 3; ++i) u_opt[i] /= sigma1;

    // Scale v by sigma1 for consistency
    for (int j = 0; j < 3; ++j) v_opt[j] *= sigma1;
}

// ----------------- ITERATIVE RANK-1 APPROX -----------------

void rank1_approximation(double A[3][3], int max_iter, double tol,
                         double* u_out, double* v_out, double& final_error) {
    double u[3] = {0.2, 0.2, 0.2};
    double v[3] = {1, 1, 1};
    for (int i = 0; i < 3; ++i) u[i] /= norm(u,3);
    for (int j = 0; j < 3; ++j) v[j] /= norm(v,3);

    double prev_error = 1e100;
    for (int iter = 0; iter < max_iter; ++iter) {
        double v_sq_sum = dot(v,v,3);
        for (int k = 0; k < 3; ++k) {
            double num = 0;
            for (int j = 0; j < 3; ++j)
                num += A[k][j] * v[j];
            u[k] = num / (v_sq_sum + 1e-12);
        }

        double u_sq_sum = dot(u,u,3);
        for (int l = 0; l < 3; ++l) {
            double num = 0;
            for (int i = 0; i < 3; ++i)
                num += A[i][l] * u[i];
            v[l] = num / (u_sq_sum + 1e-12);
        }

        double error = compute_error(A, u, v);
        if (fabs(error - prev_error)/ (prev_error + 1e-12) < tol) {
            cout << "Converged after " << iter+1 << " iterations.\n";
            final_error = error;
            break;
        }
        prev_error = error;
        final_error = error;
    }

    for (int i = 0; i < 3; ++i) u_out[i] = u[i];
    for (int j = 0; j < 3; ++j) v_out[j] = v[j];
}

// ----------------- MAIN -----------------

int main() {
    // Seed the generator with time
    seed ^= static_cast<unsigned long long>(time(0));

    double A[3][3];
    random_matrix(A);

    cout << "Original Matrix A:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            cout << A[i][j] << "\t";
        cout << "\n";
    }

    // Iterative approximation
    cout << "\n--- ITERATIVE APPROXIMATION ---\n";
    double u[3], v[3], err;
    rank1_approximation(A, 1000, 1e-9, u, v, err);
    print_vec("u_iter", u);
    print_vec("v_iter", v);
    cout << "Iterative Error = " << err << "\n";

    // Global optimal approximation
    cout << "\n--- GLOBAL OPTIMAL ---\n";
    double u_opt[3], v_opt[3];
    global_optimal_rank1(A, u_opt, v_opt);
    double opt_err = compute_error(A, u_opt, v_opt);
    print_vec("u_opt", u_opt);
    print_vec("v_opt", v_opt);
    cout << "Global Optimal Error = " << opt_err << "\n";

    // Comparison
    cout << "\n--- COMPARISON ---\n";
    cout << "Difference = " << fabs(err - opt_err) << "\n";
    if (fabs(err - opt_err) < 1e-6)
        cout << "Converged to the global optimum.\n";
    else
        cout << "Local minimum (or not fully converged).\n";

    return 0;
}

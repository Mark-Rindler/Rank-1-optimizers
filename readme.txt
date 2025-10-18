GradientAscentRank1.py

This Python program implements and visualizes a rank-1 matrix approximation using projected gradient ascent
on the unit sphere. It specifically works with 3x3 matricies but can be generalized easily. The main 
function, rank_1_approximation_pga, iteratively maximizes F(u)=(u^T)(A)(A^T)(u) by following the gradient
on the sphere, producing vectors u and v = (A^T)(u) that define the best rank-1 approximation A_0 = u(v^T).
The script includes functions to test the algorithm, compare it with the optimal SVD-based rank-1 approximation,
and visualize convergence and reconstruction quality using Matplotlib. It also verifies mathematical properites 
of the gradient projection, such as orthogonality to the current point on the sphere.

LeastSquaresRank1.cpp

This C++ program computes a rank-1 approximation of a 3x3 matrix using two methods: an iterative alternating 
least squares algorithm and the globally optimal solution obtained through the dominant eigenpair of (A^T)(A).
It includes a minimal random number generator, matrix and vector utility functions, and a Frobenius-norm error
calculator. The iterative solver alternates between updating u and v to minimize reconstruction error, while the
optimal methods applies the power iteration to find the top singular compenents analytically. The main function
generates a random matrix, runs both methods, prints results and errors, and compares convergence quality. The 
code demonstrates a numerical implementation without external libraries and a spectral approach to rank-1 approximation.

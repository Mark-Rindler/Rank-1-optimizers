import numpy as np
import matplotlib.pyplot as plt
import time

def rank1_approximation_pga(A, alpha=0.01, max_iters=1000, tol=1e-8):
    """
    Compute rank-1 approximation using Projected Gradient Ascent.
    Works for any m x n matrix.
    Returns: sigma * u * v^T form where u and v are unit vectors.
    """
    m, n = A.shape
    AAT = A @ A.T
    
    # Initialize u randomly on unit sphere in R^m
    u = np.random.randn(m)
    u = u / np.linalg.norm(u)
    
    objective_history = []
    
    for iter in range(max_iters):
        obj_val = u.T @ AAT @ u
        objective_history.append(obj_val)
        
        grad_ambient = 2 * AAT @ u
        grad_sphere = grad_ambient - (u.T @ grad_ambient) * u
        
        u_new = u + alpha * grad_sphere
        u_new = u_new / np.linalg.norm(u_new)
        
        if np.linalg.norm(u_new - u) < tol:
            u = u_new
            break
        
        u = u_new
    
    # Compute v in R^n
    v_unnormalized = A.T @ u
    sigma = np.linalg.norm(v_unnormalized)
    v = v_unnormalized / sigma if sigma > 0 else v_unnormalized
    
    A_hat = sigma * np.outer(u, v)
    
    return A_hat, sigma, objective_history


def rank1_approximation_svd(A):
    """
    Compute rank-1 approximation using SVD.
    Works for any m x n matrix.
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    sigma = S[0]
    u = U[:, 0]
    v = Vt[0, :]
    
    A_hat = sigma * np.outer(u, v)
    
    return A_hat, sigma


def test_rank1_approximation(m, n, alpha=0.01):
    """
    Test the rank-1 approximation for an m x n matrix and compare with SVD.
    """
    A = np.random.rand(m, n)
    
    print(f"\nRANK-1 APPROXIMATION: {m}×{n} matrix")
    print("=" * 50)
    
    # Compute both approximations
    A_hat_pga, sigma_pga, obj_history = rank1_approximation_pga(A, alpha=alpha)
    A_hat_svd, sigma_svd = rank1_approximation_svd(A)
    
    # Display comparison
    print(f"Singular Value:")
    print(f"  PGA: {sigma_pga:.8f}")
    print(f"  SVD: {sigma_svd:.8f}")
    print(f"  Difference: {abs(sigma_pga - sigma_svd):.2e}")
    
    print(f"\nConverged in {len(obj_history)} iterations")
    
    # Visualize only for small matrices
    if m <= 10 and n <= 10:
        visualize_results(A, A_hat_pga, obj_history, m, n)
    
    return sigma_pga, sigma_svd


def visualize_results(A, A_hat_pga, obj_history, m, n):
    """
    Visualize convergence and matrices (only for small matrices).
    """
    plt.figure(figsize=(12, 4))
    
    # Convergence plot
    plt.subplot(1, 3, 1)
    plt.plot(obj_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective F(u)')
    plt.title('PGA Convergence')
    plt.grid(True, alpha=0.3)
    
    # Original matrix
    plt.subplot(1, 3, 2)
    vmin = min(A.min(), A_hat_pga.min())
    vmax = max(A.max(), A_hat_pga.max())
    plt.imshow(A, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(fraction=0.046)
    plt.title(f'Original Matrix A ({m}×{n})')
    
    # PGA approximation
    plt.subplot(1, 3, 3)
    plt.imshow(A_hat_pga, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(fraction=0.046)
    plt.title('PGA Rank-1 Approximation')
    
    plt.tight_layout()
    plt.show()


def test_multiple_sizes():
    """
    Test various matrix sizes to show PGA works well for small matrices.
    """
    print("\n" + "=" * 50)
    print("TESTING VARIOUS MATRIX SIZES")
    print("=" * 50)
    print("(Testing PGA accuracy across different dimensions)\n")
    
    test_cases = [
        (2, 2, 0.01),
        (3, 3, 0.01),
        (5, 5, 0.01),
        (3, 5, 0.01),
        (5, 3, 0.01),
        (10, 10, 0.005),  # Smaller step size for larger matrix
    ]
    
    for m, n, alpha in test_cases:
        A = np.random.rand(m, n)
        A_hat_pga, sigma_pga, _ = rank1_approximation_pga(A, alpha=alpha)
        A_hat_svd, sigma_svd = rank1_approximation_svd(A)
        
        sigma_diff = abs(sigma_pga - sigma_svd)
        method_diff = np.linalg.norm(A_hat_pga - A_hat_svd, 'fro')
        
        print(f"{m}×{n} matrix (α={alpha}): σ_diff = {sigma_diff:.2e}, "
              f"method_diff = {method_diff:.2e}")


def test_consistency(m, n, n_trials=5, alpha=0.01):
    """
    Run multiple trials for a specific matrix size.
    """
    print("\n" + "=" * 50)
    print(f"CONSISTENCY TEST: {m}×{n} matrix ({n_trials} trials)")
    print("=" * 50)
    print("(σ_diff = difference between PGA and SVD singular values)")
    print("(method_diff = ||A_hat_PGA - A_hat_SVD||, should be ~0)\n")
    
    for trial in range(n_trials):
        A = np.random.rand(m, n)
        A_hat_pga, sigma_pga, _ = rank1_approximation_pga(A, alpha=alpha)
        A_hat_svd, sigma_svd = rank1_approximation_svd(A)
        
        sigma_diff = abs(sigma_pga - sigma_svd)
        method_diff = np.linalg.norm(A_hat_pga - A_hat_svd, 'fro')
        
        print(f"Trial {trial + 1}: σ_diff = {sigma_diff:.2e}, method_diff = {method_diff:.2e}")


def benchmark_performance(n_values, n_trials=3):
    """
    Benchmark PGA vs SVD performance for different matrix sizes.
    """
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)
    print("(Measuring computation time for n×n matrices)\n")
    
    pga_times = []
    svd_times = []
    
    for n in n_values:
        # Adaptive step size for larger matrices
        if n <= 5:
            alpha = 0.01
        elif n <= 10:
            alpha = 0.005
        elif n <= 50:
            alpha = 0.001
        else:
            alpha = 0.0005
        
        pga_trial_times = []
        svd_trial_times = []
        
        print(f"Testing n={n:4d}...", end=" ", flush=True)
        
        for trial in range(n_trials):
            A = np.random.rand(n, n)
            
            # Time PGA
            start = time.time()
            rank1_approximation_pga(A, alpha=alpha, max_iters=2000)
            pga_time = time.time() - start
            pga_trial_times.append(pga_time)
            
            # Time SVD
            start = time.time()
            rank1_approximation_svd(A)
            svd_time = time.time() - start
            svd_trial_times.append(svd_time)
        
        avg_pga = np.mean(pga_trial_times)
        avg_svd = np.mean(svd_trial_times)
        
        pga_times.append(avg_pga)
        svd_times.append(avg_svd)
        
        print(f"PGA = {avg_pga*1000:8.2f} ms, SVD = {avg_svd*1000:8.2f} ms, "
              f"Ratio = {avg_pga/avg_svd:5.2f}x")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.plot(n_values, [t*1000 for t in pga_times], 'o-', 
             linewidth=2, markersize=8, label='PGA', color='blue')
    plt.plot(n_values, [t*1000 for t in svd_times], 's-', 
             linewidth=2, markersize=8, label='SVD', color='red')
    
    plt.xlabel('Matrix Size (n×n)', fontsize=12)
    plt.ylabel('Computation Time (ms)', fontsize=12)
    plt.title('PGA vs SVD Performance Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Use log scale if range is large
    if max(n_values) / min(n_values) > 10:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return pga_times, svd_times


if __name__ == "__main__":
    np.random.seed(42)
    
    # Show one example with visualization
    print("=" * 50)
    print("EXAMPLE: 5×5 Matrix Convergence")
    print("=" * 50)
    test_rank1_approximation(5, 5)
    
    n_values = np.arange(1, 501, 50)
    benchmark_performance(n_values, n_trials=3)

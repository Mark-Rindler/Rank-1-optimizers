import numpy as np
import matplotlib.pyplot as plt

# This program works specifically for 3x3 matricies, but the mathematical framework would make it 
# incredibly easy to generalize

def rank1_approximation_pga(A, alpha=0.01, max_iters=1000, tol=1e-8, verbose=False):
    
    # Compute AA^T 
    AAT = A @ A.T
    
    # Initialize u randomly on unit sphere
    u = np.random.randn(3)
    u = u / np.linalg.norm(u)
    
    objective_history = []
    
    for iter in range(max_iters):
        # gradient ascent
        obj_val = u.T @ AAT @ u
        objective_history.append(obj_val)
        
        grad_ambient = 2 * AAT @ u
        
        grad_sphere = grad_ambient - (u.T @ grad_ambient) * u
        
        # Update 
        u_new = u + alpha * grad_sphere
        
        # Normalize to stay on unit sphere
        u_new = u_new / np.linalg.norm(u_new)
        
        # Check convergence
        change = np.linalg.norm(u_new - u)
        if change < tol:
            if verbose:
                print(f"Converged at iteration {iter}")
            u = u_new
            break
        
        u = u_new
        
        if verbose and iter % 100 == 0:
            print(f"Iteration {iter}: F(u) = {obj_val:.6f}, ||u_new - u|| = {change:.8f}")
    
    
    # Compute v = A^T u
    v = A.T @ u
    
    # Compute rank-1 approximation A_hat = u v^T
    A_hat = np.outer(u, v)
    
    return A_hat, u, v, objective_history


def test_rank1_approximation():
    """
    Test the rank-1 approximation with a sample 3x3 matrix
    and compare with SVD-based approach.
    """
    
    # Create a test 3x3 matrix
    A = np.random.rand(3, 3)
    
    print("Original Matrix A:")
    print(A)
    print(f"\nFrobenius norm of A: {np.linalg.norm(A, 'fro'):.6f}")
    
    # Compute rank-1 approximation using projected gradient ascent
    A_hat_pga, u_pga, v_pga, obj_history = rank1_approximation_pga(
        A, alpha=0.01, max_iters=1000, verbose=True
    )
    
    print("\n" + "="*50)
    print("Projected Gradient Ascent Results:")
    print("="*50)
    print(f"\nConverged u vector: {u_pga}")
    print(f"Computed v vector (A^T u): {v_pga}")
    print(f"\nRank-1 Approximation (PGA):")
    print(A_hat_pga)
    
    # Compute approximation error
    error_pga = np.linalg.norm(A - A_hat_pga, 'fro')
    print(f"\nApproximation error (Frobenius norm): {error_pga:.6f}")
    print(f"Final objective value F(u): {obj_history[-1]:.6f}")
    
    # Compare with SVD-based rank-1 approximation
    U_svd, S_svd, Vt_svd = np.linalg.svd(A)
    A_hat_svd = S_svd[0] * np.outer(U_svd[:, 0], Vt_svd[0, :])
    
    print("\n" + "="*50)
    print("SVD-based Rank-1 Approximation (for comparison):")
    print("="*50)
    print(A_hat_svd)
    error_svd = np.linalg.norm(A - A_hat_svd, 'fro')
    print(f"\nApproximation error (SVD): {error_svd:.6f}")
    print(f"Largest singular value: {S_svd[0]:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(obj_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective F(u)')
    plt.title('Convergence of Objective Function')
    plt.grid(True, alpha=0.3)
    
    # Visualize the matrices
    plt.subplot(1, 2, 2)
    vmin = min(A.min(), A_hat_pga.min())
    vmax = max(A.max(), A_hat_pga.max())
    
    # Create side-by-side comparison
    comparison = np.hstack([A, np.ones((3, 1))*np.nan, A_hat_pga])
    
    im = plt.imshow(comparison, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title('Original A | Rank-1 Approximation')
    plt.xticks([1, 4], ['Original', 'Approx'])
    plt.yticks([0, 1, 2])
    
    # Add text annotations for values
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center', fontsize=10)
            plt.text(j+4, i, f'{A_hat_pga[i,j]:.2f}', ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Test different step sizes
    print("\n" + "="*50)
    print("Testing different step sizes:")
    print("="*50)
    
    alphas = [0.001, 0.01, 0.05, 0.1]
    for alpha in alphas:
        A_hat, _, _, obj_hist = rank1_approximation_pga(
            A, alpha=alpha, max_iters=1000, verbose=False
        )
        error = np.linalg.norm(A - A_hat, 'fro')
        print(f"α = {alpha:5.3f}: Converged in {len(obj_hist)-1:4d} iterations, "
              f"Error = {error:.6f}, Final F(u) = {obj_hist[-1]:.6f}")


def verify_orthogonality_property():
    print("\n" + "="*50)
    print("Verifying Gradient Projection Properties:")
    print("="*50)
    
    # Create a random matrix and u vector
    A = np.random.randn(3, 3)
    u = np.random.randn(3)
    u = u / np.linalg.norm(u)
    
    # Compute gradients
    AAT = A @ A.T
    grad_ambient = 2 * AAT @ u
    grad_sphere = grad_ambient - (u.T @ grad_ambient) * u
    
    # Check orthogonality
    dot_product = u.T @ grad_sphere
    print(f"u^T · grad_sphere = {dot_product:.10f} (should be ≈ 0)")
    print(f"||u|| = {np.linalg.norm(u):.10f} (should be = 1)")
    
    # Verify that gradient points in direction of increasing F
    alpha = 0.001
    u_new = u + alpha * grad_sphere
    u_new = u_new / np.linalg.norm(u_new)
    
    F_old = u.T @ AAT @ u
    F_new = u_new.T @ AAT @ u_new
    print(f"\nF(u) = {F_old:.6f}")
    print(f"F(u_new) = {F_new:.6f}")
    print(f"Increase = {F_new - F_old:.8f} (should be > 0)")


if __name__ == "__main__":
    # Run the main test
    test_rank1_approximation()
    test_rank1_approximation()
    test_rank1_approximation()
    test_rank1_approximation()
    test_rank1_approximation()
    
    # Verify gradient projection properties
    verify_orthogonality_property()
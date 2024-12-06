

**Algorithm:**  

Repeat until convergence:  
1. **Compute $\text{prox}_{tf}(x_k)$ in $\mathbb{R}^n$:**  
   - Use Monte Carlo sampling with a large number of samples.  

2. **Define the descent direction $d_k$:**  
   - Compute $d_k = \frac{x_k - \text{prox}_{tf}(x_k)}{\|x_k - \text{prox}_{tf}(x_k)\|_{2}}$.  

3. **Optimize in the subspace$L$:**  
   - Compute $\text{prox}_{tf}(x_k)$ for the function restricted to the subspace $L$ (spanned by $d_k$ and $x_k$).  
   - Use Gauss-Hermite quadrature for this step, with a relatively small number of samples and keeping$t$constant from the previous iteration.  

4. **Update $\text{prox}_{tf}(x_k)$:**  
   - If $f(\text{prox}_{tf}(x_k) \text{ in } L) < f(\text{prox}_{tf}(x_k) \text{ in } \mathbb{R}^n)$, update $\text{prox}_{tf}(x_k)$ to the subspace solution.  

5. **Perform gradient descent:**  
   - Update $x_k$ using the computed gradient.  

6. **Update$t_k$:**  
   - Adjust $t_k$ using HJ-MAD condition for updating $tk$ based on current and previous gradients. 

 # EP4Orth+ beta 1.0
 A MATLAB software for solving optimization problems with orthogonal nonnegative constraints.

 # Problems and solvers
The package contains codes for optimization problems with orthogonal nonnegative constraints: 
   $$\min \,f(X) \,\, s.t. \,\,X\in \mathcal{S}^{n,k}_+, $$

where $\mathcal{S}^{n,k}_+=\{X\in\mathbb{R}^{n\times k},\,X^\top X=I_k,\,X\ge 0\}$ and $f$ is twice continuously differentiable.

Applications have been solved by the solver:

+ Projections onto $\mathcal{S}^{n,k}_+$: $\min\,\|X-C\|_F^2\,\,s.t.\,\,X\in\mathcal{S}^{n,k}_+$

- Trace minimization with nonnegative orthogonality constraints: $\min\,\mathrm{Tr}(X^\top MX)\,\,s.t.\,\,X\in\mathcal{S}^{n,k}_+$
- Orthogonal nonnegative matrix factorization (ONMF): $\min \,\|A-XY^\top\|_F^2, s.t.\,\,X\in\mathcal{S}^{n,k}_+,\,\,Y\in \mathbb R^{n,k}_+$
- K-indicators model: $ \min\,\|UY-X\|_F^2\,\, s.t.\,\,X\in\mathcal{S}^{n,k}_+,\,\,\|X_{i,:}\|=1,\,\,Y^\top Y=I_k$
- Discriminative nonnegative spectral clustering
- Nonnegative principal component analysis (NPCA)

 # References
- [Bo Jiang, Xiang Meng, Zaiwen Wen, Xiaochun Chen. An exact penalty approach for optimization with nonnegative orthogonality constraints](https://arxiv.org/abs/1907.12424v2)
- [Jiang Hu, Bo Jiang, Lin Lin, Zaiwen Wen, Yaxiang Yuan. Structured Quasi-Newton Methods for Optimization with Orthogonality Constraints. SIAM Journal on Scientific Computing, Vol. 41, No. 4, pp. A2239-A2269](https://arxiv.org/abs/1809.00452)
- [J. Hu, A. Milzarek, Z. Wen, and Y. Yuan, Adaptive quadratically regularized Newton method for Riemannian optimization, SIAM J. Matrix Anal. Appl., 39 (2018), pp. 1181-1207.](https://arxiv.org/abs/1708.02016)
- [X. Xiao, Y. Li, Z. Wen, and L. Zhang, A regularized semi-smooth Newton method with projection steps for composite convex programs, J. Sci. Comput., 76 (2016), pp. 364-389.](https://link.springer.com/article/10.1007/s10915-017-0624-3)
- [J. Bolte, S. Sabach, and M. Teboulle, Proximal alternating linearized minimization for nonconvex and nonsmooth problems, Math. Program., 146 (2014), pp. 459-494.](https://link.springer.com/article/10.1007/s10107-013-0701-9)
- [F. Pompili, N. Gillis, P.-A. Absil, and F. Glineur, Two algorithms for orthogonal non-negative matrix factorization with application to clustering, Neurocomputing, 141 (2014), pp. 15-25.](https://arxiv.org/abs/1201.0901)




 # The Authors
 We hope that the package is useful for your application.  If you have any bug reports or comments, please feel free to email one of the toolbox authors:

 * Xiang Meng, 1700010614 at pku.edu.cn
 * Bo Jiang, jiangbo at njnu.edu.cn
 * Zaiwen Wen, wenzw at pku.edu.cn

 # Installation
 `>> startup`  

 `>> cd Examples` 

 `>> test_onmf` 
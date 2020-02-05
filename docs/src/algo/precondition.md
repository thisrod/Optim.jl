# Preconditioning

The `GradientDescent`, `ConjugateGradient` and `LBFGS` methods support preconditioning. A preconditioner
is a linear operator `P` that
can be thought of as a change of coordinates
from `x = Py` to `y`
to improve the condition of the Hessian. With a
good preconditioner substantially improved convergence is possible.

More precisely, the solvers that support preconditioning look along a line.  Given a guess `x₀` for the minimizer, they search for a better minimizer `x = x₀ + a*g`, where `g` is a direction vector, and `a` is a distance along the line.  Typically, `g` is the gradient of the cost function `f` at `x₀`.  When a preconditioner is specified, they search along the line `x = x₀ + a*P⁻¹*g` instead.  How this relates to the Hessian is explained below.

A preconditioner `P` can be of any type as long as the following two methods are
implemented:

* `A_ldiv_B!(pgr, P, gr)` : apply `P` to a vector `gr` and store in `pgr`
      (intuitively, `pgr = P \ gr`)
* `dot(x, P, y)` : the inner product induced by `P`
      (intuitively, `dot(x, P * y)`)

Precisely what these operations mean, depends on how `P` is stored. Commonly, we store a matrix `P` which
approximates the Hessian in some vague sense. In this case,

* `A_ldiv_B!(pgr, P, gr) = copyto!(pgr, P \ A)`
* `dot(x, P, y) = dot(x, P * y)`

Finally, it is possible to update the preconditioner as the state variable `x`
changes. This is done through  `precondprep!` which is passed to the
optimizers as kw-argument, e.g.,
```jl
   method=ConjugateGradient(P = precond(100), precondprep! = precond(100))
```
though in this case it would always return the same matrix.
(See `fminbox.jl` for a more natural example.)

Apart from preconditioning with matrices, `Optim.jl` provides
a type `InverseDiagonal`, which represents a diagonal matrix by
its inverse elements.

## Example
Below, we see an example where a function is minimized without and with a preconditioner
applied.
```jl
using ForwardDiff, Optim, SparseArrays
initial_x = zeros(100)
plap(U; n = length(U)) = (n-1)*sum((0.1 .+ diff(U).^2).^2 ) - sum(U) / (n-1)
plap1(x) = ForwardDiff.gradient(plap,x)
precond(n) = spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1)) * (n+1)
f(x) = plap([0; x; 0])
g!(G, x) = copyto!(G, (plap1([0; x; 0]))[2:end-1])
result = Optim.optimize(f, g!, initial_x, method = ConjugateGradient(P = nothing))
result = Optim.optimize(f, g!, initial_x, method = ConjugateGradient(P = precond(100)))
```
The former optimize call converges at a slower rate than the latter. Looking at a
 plot of the 2D version of the function shows the problem.

![plap](./plap.png)

The contours are shaped like ellipsoids, but we would rather want them to be circles, so that the gradient points to the minimizer.
Using the preconditioner effectively changes the coordinates such that the contours
becomes less ellipsoid-like. Benchmarking shows that using preconditioning provides
 an approximate speed-up factor of 15 in this 100 dimensional case.

Looking at the contours in this example, it is apparent that the
minimum lies in the direction where, as we move, we keep crossing
contours that point the same way, and the gradient does not rotate.
Let `H` be the Hessian of the cost function, and suppose we search
along a line `x = x₀ + a*d`.  To avoid rotating the gradient, we
seek a direction `d` for which `H*d = λ*g`, so that the change in
the gradient, `H*d`, is parallel to the gradient `g`.  This direction
`d = H⁻¹*g` is the one obtained by preconditioning with the Hessian.
In the case that the cost is a quadratic form, it points straight
at the minimizer.

It is rarely possible to compute `H⁻¹*g` exactly.  However, it is
often possible to find a `P` that approximates `H` in the directions
where `x*H*x` is large, where the gradient changes rapidly.  In
this case, preconditioning with `P` will select a direction where
the gradient rotates gradually, and the cost function keeps decreasing
for a long way.  This allows a larger step to be taken.

## References

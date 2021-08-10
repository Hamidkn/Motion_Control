# Estimators

## Kalman Filter

use kalman state observer, to estimate car's position and velocity.

# Controllers

## LQ (Linear-Quadratic controller)

### Introduction
The Linear Quadratic Regulator (LQR) is a well-known method that provides optimally controlled feedback gains to enable the closed-loop stable and high performance design of systems.

### Full-State Feedback
For the derivation of the linear quadratic regulator we consider a linear system state-space representation:
```
x˙=Ax+Bu
y˙=Cx, C=In×n
```
which essentially means that full state feedback is available (all n states are measurable).

The feedback gain is a matrix K and the feedback control action takes the form:
```
u=K(xref−x)
```
The closed-loop system dynamics are then written:
```
x˙=(A−BK)x+BKxref
```
where xref represents the vector of desired states, and serves as the external input to the closed-loop system. The “A-matrix” of the closed-loop systems is (A−BK), while its B−matrix is BK. The closed-loop system has exactly the same amount of inputs and outputs -n. The column dimension of B equals the number of channels available in u, and must match the row dimension of K. Pole-placement is the process of placing the poles of (A−BK) in stable, suitably-damped locations in the complex plane.

### The Maximum Principle
Towards a generic procedure for solving optimal control problems, we derive a methodology based on the calculus of variations. The problem statement for a fixed end time tf is:
```
choose u(t) to minimize J=ψ(x(tf))+∫tft0L(x(t),u(t),t)dt
subject to x˙=f(x(t),u(t),t)x(t_0) = x_0$
```
where ψ(x(tf),tf) is the terminal cost; the total cost J is a sum of the terminal cost and an integral along the way. We assume that L(x(t),u(t),t) is nonnegative. The first step is to augment the cost using the costate vector λ(t)
```
J¯=ψ(x(tf))+∫tft0(L+λT(f−x˙))dt
```
As understood, λ(t) may be an arbitrary expression we choose, since it multiplies f−x˙=0. Along the optimum trajectory, variations in J and hence J¯ should vanish. This follows from the fact that J is chosen to be continuous in x, u, and t. We write the variation as:
```
δJ¯=ψxδx(tf)+∫tft0[Lxδx+LuδuλTfxδx+λTfuδu−λTδx˙dt
```
where subscripts denote partial derivatives. The last term above can be evaluated using integration by parts as:
```
−∫tft0λTx˙dt=−λT(tf)δx(tf)+λT(t0)δx(t0)+∫tft0λ˙Tδxdt,
δJ¯=ψx(x(tf))δx(tf)+∫tft0(Lu+λTfu)δudt+∫tft0(Lx+λTfx+λ˙T)δxdt−λT(tf)δx(tf)+λT(t0)δx(t0).
```
The last term is zero, since we cannot vary the initial of the state by changing something later in time. This writing of J¯ indicates that there are three components of the variation that must independently be zero:
```
Lu+λTfu=0
Lx+λTfx+λ˙T=0
ψx(x(tf))−λT(tf)=0
```

The second and third requirements are met by explicitly setting:
```
λ˙T=−Lx−λTfx
$ \lambda _^T (t_f) = \psi _x (x(t_f)).
```

The evolution of λ is given in reverse time, from a final state to the initial. Hence we see the primary difficulty of solving optimal control problems: the state propagates forward in time, while the costate propagates backward. The state and costate are coordinated through the above equations.

Gradient Method Solution for the General Case
Numerical solutions to the general problem are iterative, and the simplest approach is the gradient method. Its steps are as follows:

For a given x0, pick a control history u(t).
Propagate x˙=f(x,u,t) forward in time to create a state trajectory.
Evaluate ψx(x(tf)), and propagate the costate backward in time from tf to t0.
At each time step, choose δu=−K(Lu+λTfu), where K is a positive scalar or a positive definite matrix in the case of multiple input channels.
Let u=u+δu.
Go back to step 2 and repeat loop until solution has converged.
The first three steps are consistent in the sense that x is computed directly from x(t0) and u and λ is computed from x and x(tf). All of δJ¯ except the integral with δu is therefore eliminated explicitly. The choice of δu in step 4 then achieves δJ¯<0 unless δu=0, in which case the problem is solved.

### LQR Solution
In the case of the Linear Quadratic Regulator (with zero terminal cost), we set ψ=0, and
```
L=12xTQx+12uTRu
```
where the requirement that L≥0 implies that both Q and R are positive definite. In the case of linear plant dynamics also, we have:
```
Lx=xTQ
Lu=uTR
fx=A
fu=B
```
so that:
```
x˙=Ax+Bu
x(t0)=x0
λ˙=−Qx−ATλ
λ(tf)=0
Ru+BTλ=0.
```
Since the systems are clearly linear, we try a connection λ=Px. Inserting this into λ˙ equation, and then using the x˙ equation, and a substitution for u, we obtain:
```
PAx+ATPx+Qx−PBR−1BTPx+P˙=0
```
This has to hold for all x, so in fact it is a matrix equation, the matrix Riccatti equation. The steady-state solution is given satisfies:
```
PA+ATP+Q−PBR−1BTP=0
```
### Optimal Full-State Feedback
This equation is the Matrix Algebraic Riccati Equation (MARE), whose solution P is needed to compute the optimal feedback gain K. The MARE is easily solved by standard numerical tools in linear algebra. The equation Ru+BTλ=0 gives the feedback law:
```
u=−R−1BTPx
```
Properties and Use of the LQR
Static Gain: The LQR generates a static gain matrix K, which is not a dynamical system. Hence, the order of the closed-loop system is the same as that of the plan.
Robustness: The LQR achieves infinite gain margin.
Output Variables: When we want to conduct output regulation (and not state regulation), we set Q=CTQ′C.

# Systems

contains classes used to describe discrete time systems.
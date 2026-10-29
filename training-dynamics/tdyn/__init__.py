"""tdyn: shared [AI] code for the training-dynamics course.

Stable across assignments. Assignments import from its modules; you do not edit it.

    tdyn.integrate   Euler / RK4 fixed-step integrators, solve_ivp wrapper (dense trajectories)
    tdyn.optim       gd_trajectory, sgd_trajectory (with a label-noise hook)
    tdyn.hessian     hvp, power_iteration, hutchinson_trace, exact_hessian (staff versions; torch)
    tdyn.models      the anchor model L(a,b) = (ab-1)^2/2 in numpy and torch; a tiny torch MLP
    tdyn.plotting    one matplotlib style; phase_portrait
    tdyn.results     results.json save / load / compare (python -m tdyn.results OURS REF)
    tdyn.data        offline loaders for data/mnist.npz and data/cifar10_5k.npz
    tdyn.swap        imports a stub or its reference solution (TDYN_SOLUTIONS=1)

Submodules are imported explicitly (from tdyn.integrate import rk4), so `import tdyn` is cheap.
"""
__version__ = "0.1.0"

import numpy as np
from heap_class import Heap

def gauss_southwell(fast_updates_f, safe_updates_f, x0, num_iters=1000, tol=1e-6, f_priorities=None, stepsize_strat = None):
    # Initialize the solution
    x = x0.copy()
    n = len(x)
    # Create a max heap for storing the gradients
    fx, grad = safe_updates_f(x)
    heap_gradient = Heap(grad, f_priorities)
    f_history = [fx]
    for _ in range(num_iters):
        #TODO: implement Gauss-Southwell using the heap
        pass
    return x, f_history
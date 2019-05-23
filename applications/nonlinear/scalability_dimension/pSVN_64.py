from model_lognormal_64 import *

import time

# check the stein/options to see all possible choices
options["is_projection"] = True
options["true_Hessian"] = True
options["coefficient_dimension"] = 50
options["add_dimension"] = 0

options["number_particles"] = 8
options["number_particles_add"] = 0
options["add_number"] = 0
options["add_step"] = 5
options["add_rule"] = 1

options["type_Hessian"] = "lumped"

options["type_scaling"] = 1
options["type_metric"] = "posterior_average"

options["low_rank_Hessian"] = 2
options["rank_Hessian"] = 50
options["rank_Hessian_tol"] = 1.e-1
options["low_rank_Hessian_average"] = 2
options["rank_Hessian_average"] = 50
options["rank_Hessian_average_tol"] = 1.e-1
options["gauss_newton_approx"] = True  # if error of unable to solve linear system occurs, use True

options["step_tolerance"] = 1e-2
options["step_projection_tolerance"] = 1e-1
options["max_iter"] = 10
options["line_search"] = True
options["max_backtracking_iter"] = 10
options["cg_coarse_tolerance"] = 0.5e-2
options["print_level"] = -1
options["plot"] = True

# generate particles
particle = Particle(model, options, comm)

# evaluate the variation (gradient, Hessian) of the negative log likelihood function at given particles
variation = Variation(model, particle, options, comm)

# evaluate the kernel and its gradient at given particles
kernel = Kernel(model, particle, variation, options, comm)

t0 = time.time()

solver = NewtonSeparated(model, particle, variation, kernel, options, comm)

solver.solve()

print("NewtonSeparated solving time = ", time.time() - t0)

from model_lognormal import *

import os
import time

# check the stein/options to see all possible choices
options["type_optimization"] = "newtonSeparated"
options["is_projection"] = True
options["type_projection"] = "hessian"
options["tol_projection"] = 1.e-1
options["coefficient_dimension"] = 10
options["add_dimension"] = 2

options["number_particles"] = 8
options["number_particles_add"] = 0
options["add_number"] = 0
options["add_step"] = 5
options["add_rule"] = 1

options["type_Hessian"] = "lumped"

options["type_scaling"] = 1
options["type_metric"] = "posterior_average"

options["low_rank_Hessian"] = True
options["rank_Hessian"] = 20
options["rank_Hessian_tol"] = 1.e-1
options["low_rank_Hessian_average"] = True
options["rank_Hessian_average"] = 20
options["rank_Hessian_average_tol"] = 1.e-1
options["gauss_newton_approx"] = True  # if error of unable to solve linear system occurs, use True

options["step_tolerance"] = 1.e-16
options["step_projection_tolerance"] = 1.e-2
options["max_iter"] = 20
options["line_search"] = True
options["max_backtracking_iter"] = 10
options["cg_coarse_tolerance"] = 0.5e-2
options["print_level"] = -1
options["save_number"] = 0
options["plot"] = True

# generate particles
particle = Particle(model, options, comm)

# filename = "data/laplace.p"
filename = "data/mcmc_dili_sample.p"
if os.path.isfile(filename):
    print("set reference for mean and variance")
    data_save = pickle.load(open(filename, 'rb'))
    mean = model.generate_vector(PARAMETER)
    mean.set_local(data_save["mean"])
    variance = model.generate_vector(PARAMETER)
    variance.set_local(data_save["variance"])
    particle.mean_posterior = mean
    particle.variance_posterior = variance

# evaluate the variation (gradient, Hessian) of the negative log likelihood function at given particles
variation = Variation(model, particle, options, comm)

# evaluate the kernel and its gradient at given particles
kernel = Kernel(model, particle, variation, options, comm)

t0 = time.time()

solver = NewtonSeparated(model, particle, variation, kernel, options, comm)

solver.solve()

print("NewtonSeparated solving time = ", time.time() - t0)

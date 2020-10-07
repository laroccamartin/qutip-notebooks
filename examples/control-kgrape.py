import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmaz, rand_herm,basis
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo

example_name = 'K-GRAPE'

# Set Dimension
N = 20

# Drift Hamiltonian
H_d = rand_herm(N)
# The (single) control Hamiltonian
H_c = [rand_herm(N)]
# start point for the gate evolution
psi_0 = basis(N,0)
# Target for the gate evolution Hadamard gate
psi_targ = basis(N,N-1)

#%%

# Min number of params
min_n_ts = 2*N-2

# Param Factor
param_factr = 4

# Number of time slots
n_ts = param_factr*min_n_ts

# Number of controls
n_ctrls = len(H_c)

# Timestep
dt = .1

# Time allowed for the evolution
evo_time = n_ts*dt

#%%

# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20

#%% DYNAMICS PARAMS
Nt=10
cut=2
dyn_params = dict(Nt=Nt,cut=cut)
dyn_type = "KRYLOV"

#%% CHOOSE PULSE
from numpy.random import random as rnd
c= rnd(n_ts)

#%%

optim = cpo.create_pulse_optimizer(H_d, H_c, psi_0, psi_targ, n_ts, evo_time,
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                dyn_params=dyn_params,dyn_type=dyn_type,
                max_iter=max_iter, max_wall_time=max_wall_time, 
                log_level=log_level, gen_stats=True)

dyn = optim.dynamics
    
init_amps=np.array(c).reshape(n_ts,n_ctrls)
dyn.initialize_controls(init_amps)

result = optim.run_optimization()

print(result.termination_reason)

final_amps=result.final_amps

#%%


result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time)))

#%%
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
#ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
ax1.step(result.time,
         np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
         where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
ax2.step(result.time,
         np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
         where='post')
plt.tight_layout()
plt.show()


############################################################################################
# Constants
"RNG for sampling based solutions"
const _rng = Random.MersenneTwister(1)   # shorthand
Random.seed!(_rng, 1)

"Tolerance for stochastic solutions"
const _TOL = 1.0e-6

"Number of samples"
N = 10^3
N_SMC2 = 10^2

############################################################################################
# Kernel and AD backends
include("mcmc.jl")
include("smc.jl")

############################################################################################
# Constants
"RNG for sampling based solutions"
const _rng = Random.MersenneTwister(1)   # shorthand
Random.seed!(_rng, 1)

"Tolerance for stochastic solutions"
const _TOL = 1.0e-6

"Number of samples"
N = 10^3

############################################################################################
# Kernel and AD backends
data_uv = randn(_rng, 1000)

############################################################################################
# Initiate Base Model to check sampler

######################################## Model 1
struct MyBaseModel <: ModelName end
myparameter = (μ = Param(0.0, Normal()), σ = Param(1.0, Gamma()))
mymodel = ModelWrapper(MyBaseModel(), myparameter)

#Create objective for both μ and σ and define a target function for it
myobjective = Objective(mymodel, data_uv, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{MyBaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
myobjective(myobjective.model.val)

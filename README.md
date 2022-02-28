# Baytes

<!---
![logo](docs/src/assets/logo.svg)
[![CI](xxx)](xxx)
[![arXiv article](xxx)](xxx)
-->

[![Documentation, Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paschermayr.github.io/Baytes.jl/)
[![Build Status](https://github.com/paschermayr/Baytes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/paschermayr/Baytes.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/paschermayr/Baytes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/paschermayr/Baytes.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Baytes.jl is a sampling library to perform Monte Carlo proposal steps. It consists of several sub-libraries, such as [BaytesMCMC.jl](https://github.com/paschermayr/BaytesMCMC.jl), [BaytesFilters.jl](https://github.com/paschermayr/BaytesFilters.jl), [BaytesPMCMC.jl](https://github.com/paschermayr/BaytesPMCMC.jl) and [BaytesSMC.jl](https://github.com/paschermayr/BaytesSMC.jl), and provides an interface to combine kernels from these libraries.

## Introduction

In order to start, we have to define parameter and an objective function first. Let us use the model initially defined in the [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl) introduction:

``` julia
using ModelWrappers, Baytes
using Distributions, Random, UnPack
_rng = Random.GLOBAL_RNG

#Create initial model and data
myparameter = (μ = Param(2.0, Normal()), σ = Param(3.0, Gamma()))
mymodel = ModelWrapper(myparameter)
data = randn(1000)
#Create objective for both μ and σ and define a target function for it
myobjective = Objective(mymodel, data, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
```

Sampling this model is straightforward. For instance, we can jointly estimate μ and σ via NUTS:
``` julia
trace1, algorithm1 = sample(_rng, mymodel, data, NUTS((:μ, :σ)))
```

`Trace.val` contains samples stored as `NamedTuple`. `Trace.diagnostics` contains diagnostics that were returned when sampling the corresponding parameter. `Trace.info` contains useful summary information printing and summarizing the parameter estimates. `Algorithm` returns the kernel that was used and tuned during the sampling process. Note that a new kernel is initiated for each chain separately.

Alternatively, you might want to sample parameter with different kernels. This is usually useful when working with state space models.
``` julia
trace2, algorithm2 = sample(_rng, mymodel, data, NUTS((:μ, )), Metropolis( (:σ,) ))
```

## Advanced usage

Similar to other Baytes packages, sampling arguments may be tweaked via a `Default` struct.

``` julia
sampledefault = SampleDefault(;
    dataformat=Batch(),
    tempering=IterationTempering(Float64, UpdateFalse(), 1.0, 1000),
    chains=4,
    iterations=2000,
    burnin=max(1, Int64(floor(2000/10))),
    thinning = 1,
    safeoutput=false,
    printoutput=true,
    printdefault=PrintDefault(),
    report=ProgressReport(),
)
trace3, algorithm3 = sample(_rng, mymodel, data, NUTS((:μ, :σ)); default = sampledefault)
```
Note that hyperparameter that are specific to any kernel will have to be assigned in the MCMC constructor itself, i.e.:
``` julia
mcmc4 = HMC((:μ,); proposal = ConfigProposal(; metric = MDense()), GradientBackend = :ReverseDiff,)
trace4, algorithm4 = sample(_rng, mymodel, data, mcmc4; default = sampledefault)
```

You can reuse returned `algorithm` container to sample again with the pre-tuned kernels. In this this case, on can call `sample!`. The information in `trace` will assure that sampling is continued with the correct specifications. After `sample!` is finished, a new trace is returned that contains sampling information.
``` julia
iterations = 10^3
trace5, algorithm4 = sample!(iterations, _rng, mymodel, data, trace4, algorithm4)
```

## Inference
Per default, `sample` and `sample!` return summary information of the chain and diagnostics using MCMCDiagnosticTools.jl, unless `printdefault = false`. For further inference, one can manually convert `trace.val` into a 3D-array that is compatible with MCMCChains.jl:
``` julia
burnin = 0
thinning = 1
tagged = Tagged(mymodel, (:μ,:σ) )
array_3dim = Baytes.trace_to_3DArray(trace5, mymodel, tagged, burnin, thinning)

using MCMCChains
MCMCChains.Chains(array_3dim, trace5.info.sampling.paramnames)
```

## Going Forward

This package is still highly experimental - suggestions and comments are always welcome!

<!---
# Citing Baytes.jl

If you use Baytes.jl for your own research, please consider citing the following publication: ...
-->

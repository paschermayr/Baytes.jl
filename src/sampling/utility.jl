############################################################################################
"""
$(TYPEDEF)

Contains several useful information for sampled parameter.

# Fields
$(TYPEDFIELDS)
"""
struct PrintedParameter{A<:Tuple, B<:Tuple}
    "Unique tagged parameter of all kernels."
    tagged::A
    "Parameter names based on their dimension"
    printed::B
    function PrintedParameter(
        tagged::A,
        printed::B
    ) where {A<:Tuple, B<:Tuple}
        return new{A,B}(tagged, printed)
    end
end

############################################################################################
"""
$(TYPEDEF)

Contains several useful information for constructing sampler.

# Fields
$(TYPEDFIELDS)
"""
struct SamplingInfo{A<:PrintedParameter, U<:BaytesCore.UpdateBool, B<:BaytesCore.UpdateBool}
    "Parameter settings for printing."
    printedparam::A
    "Total number of sampling iterations."
    iterations::Int64
    "Burnin iterations."
    burnin::Int64
    "Number of consecutive samples taken for diagnostics output."
    thinning::Int64
    "Number of algorithms used while sampling."
    Nalgorithms::Int64
    "Number of chains used while sampling."
    Nchains::Int64
    "Boolean if sampler need to be updated after a proposal run. This is the case for, e.g. PMCMC., or for adaptive tempering."
    captured::U
    "Boolean if temperature is adapted for target function."
    tempered::B
    function SamplingInfo(
        printedparam::A,
        iterations::Int64,
        burnin::Int64,
        thinning::Int64,
        Nalgorithms::Int64,
        Nchains::Int64,
        captured::U,
        tempered::B
    ) where {A<:PrintedParameter, U<:BaytesCore.UpdateBool, B<:BaytesCore.UpdateBool}
        return new{A, U, B}(printedparam, iterations, burnin, thinning, Nalgorithms, Nchains, captured, tempered)
    end
end

############################################################################################
"""
$(SIGNATURES)
Check if we can capture results from last proposal step. Typically only possible if a single MCMC or SMC step applied.

# Examples
```julia
```

"""
function update(datatune::DataTune, temperingadaption::B, smc::SMCConstructor) where {B<:BaytesCore.UpdateBool}
    #!NOTE: If a single SMC constructor is used, we can capture previous results and do not need to update sampler before new iteration.
    #!NOTE: SMC only updates parameter if UpdateTrue(). Log-target/gradients will always be updated if jitter step applied.
    return BaytesCore.UpdateFalse()
end
function update(datatune::DataTune{<:B}, temperingadaption::BaytesCore.UpdateFalse, mcmc::MCMCConstructor) where {B<:Batch}
    #!NOTE: If a MCMC constructor is used and no tempering is applied, we can capture previous results and do not need to update sampler before new iteration.
    #!NOTE: This only holds for Batch data in MCMC case
    #!NOTE: MCMC updates log target evaluation and eventual gradients if UpdateTrue()
    return BaytesCore.UpdateFalse()
end
function update(datatune::DataTune, temperingadaption, args...)
    #!NOTE: In all other cases, we have to update kernels for proposal before new propagation
    return BaytesCore.UpdateTrue()
end

############################################################################################
"""
$(SIGNATURES)
Obtain maximum number of iterations based on datatune.
This will always be user input, except if rolling/expanding data is used, in which case iterations are capped.

# Examples
```julia
```

"""
function maxiterations(datatune::DataTune, iterations::Integer)
    return iterations
end
function maxiterations(
    datatune::DataTune{<:D}, iterations::Integer
) where {D<:Union{Expanding,Rolling}}
    iter = maximum(datatune.config.size) - datatune.structure.index.current
    println(
        "Iteration chosen to be of length ", iter, " as datatune set to Expanding/Rolling."
    )
    return iter
end

############################################################################################
"""
$(SIGNATURES)
Obtain all parameter where output diagnostics can be printed.

# Examples
```julia
```

"""
function printedparam(datatune::DataTune, sym, algorithm...)
    return sym
end
function printedparam(
    datatune::DataTune{<:D}, sym, smc::SMCConstructor
) where {D<:Expanding}
    #!TDO: If SMC is applied on Expanding data sequence, latent data is expanding over time - no output diagnostics for this yet.
    return if smc.kernel == SMC2
        Tuple(sym, (smc.kernel.propagation.sym,))
    else
        sym
    end
end

"""
$(SIGNATURES)
Return all unique targetted parameter, and parameter where diagnostics will be printed.

# Examples
```julia
```

"""
function showparam(model::ModelWrapper, datatune::DataTune, constructor...)
    # Get all tracked symbols in constructor
    sym = BaytesCore.to_Tuple.(map(BaytesCore.get_sym, constructor))
    unique_sym = unique(collect(Iterators.flatten(sym)))
    #println("All unique tagged parameter: ", unique_sym)
    # Check if all tracked symbols are not fixed in model
    ArgCheck.@argcheck all(haskey(model.val, symbol) for symbol in unique_sym)
    # Check if one of the parameter has increasing dimension (SMC2)
    printed_sym = printedparam(datatune, unique_sym, constructor)
    # Check if any of the remaining parameter is fixed, and we can thus avoid printing
    fixed = [model.info.constraint[sym] isa ModelWrappers.Fixed for sym in printed_sym]
    # Return unique symbols and uniqute symbols that are not fixed
    return Tuple(unique_sym), Tuple(printed_sym[.!fixed])
end

############################################################################################
function update(datatune::DataTune, data)
    return DataTune(datatune.structure, ArrayConfig(data), datatune.miss)
end
function update(datatune::Vector{<:DataTune}, data)
    return DataTune(datatune[begin].structure, ArrayConfig(data), datatune[begin].miss)
end

############################################################################################
"""
$(SIGNATURES)
Infer types of all Diagnostics container.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    algorithmsᵛ::AbstractVector,
    model::ModelWrapper,
    data::D,
) where {D}
    diagtypes = map(
        algorithm -> infer(_rng, diagnostics, algorithm, model, data), algorithmsᵛ[begin]
    )
    return diagtypes
end

############################################################################################
"""
$(SIGNATURES)
Return buffer vector of type `diagtypes`. Comes from function `infer`.

# Examples
```julia
```

"""
function diagnosticsbuffer(
    diagtypes::D, iterations::Integer, Nchains::Integer, args...
) where {D}
    return [
        map(diag -> Vector{diag}(undef, iterations), diagtypes) for _ in Base.OneTo(Nchains)
    ]
end
function diagnosticsbuffer(
    diagtypes::D, iterations::Integer, Nchains::Integer, smc::SMC
) where {D}
    return Vector{diagtypes}(undef, iterations)
end

############################################################################################
#export
export SamplingInfo, update, infer

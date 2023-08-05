################################################################################
"""
$(TYPEDEF)

Contains arguments for trace to extract parameter. Used to construct a 'TraceTransform'.

# Fields
$(TYPEDFIELDS)
"""
struct TransformInfo
    "Chain indices that are used for output diagnostics."
    chains                  ::  Vector{Int64}
    "Algorithm indices that are used for output diagnostics."
    algorithms              ::  Vector{Int64}
    "Number of burnin steps before output diagnostics are taken."
    burnin                  ::  Int64
    "Number of steps that are set between 2 consecutive samples."
    thinning                ::  Int64
    "Maximum number of iterations to be collected for each chain."
    maxiterations           ::  Int64
    "StepRange for indices of effective samples"
    effective_iterations    ::  StepRange{Int64, Int64}
    function TransformInfo(
        chains::Vector{Int64},
        algorithms::Vector{Int64},
        burnin::Int64,
        thinning::Int64,
        maxiterations::Int64
    )
        ArgCheck.@argcheck maxiterations >= burnin >= 0
        ArgCheck.@argcheck thinning > 0
        ArgCheck.@argcheck maxiterations > 0
        #Assign indices for subsetting trace
        effective_iterations = (burnin+1):thinning:maxiterations
        return new(chains, algorithms, burnin, thinning, maxiterations, effective_iterations)
    end
end
function TransformInfo(
    chains::Vector{Int64},
    algorithms::Vector{Int64},
    effective_iterations::StepRange{Int64, Int64}
)
    burnin = effective_iterations.start-1
    thinning = effective_iterations.step
    iterations = effective_iterations.stop
    return TransformInfo(
        chains,
        algorithms,
        burnin,
        thinning,
        iterations
)
end

################################################################################
"""
$(TYPEDEF)

Contains arguments for trace to extract parameter from a 'Trace'.

# Fields
$(TYPEDFIELDS)
"""
struct TraceTransform{T<:Tagged, P}
    "Contains parameter where output information is printed."
    tagged                  ::  T
    "Parameter names based on tagged model parameter."
    paramnames              ::  P
    "Chain indices that are used for output diagnostics."
    chains                  ::  Vector{Int64}
    "Algorithm indices that are used for output diagnostics."
    algorithms              ::  Vector{Int64}
    "Number of burnin steps before output diagnostics are taken."
    burnin                  ::  Int64
    "Number of steps that are set between 2 consecutive samples."
    thinning                ::  Int64
    "Maximum number of iterations to be collected for each chain."
    maxiterations           ::  Int64
    "StepRange for indices of effective samples"
    effective_iterations    ::  StepRange{Int64, Int64}
    function TraceTransform(
        tagged::T,
        paramnames::P,
        chains::Vector{Int64},
        algorithms::Vector{Int64},
        burnin::Int64,
        thinning::Int64,
        maxiterations::Int64
    ) where {
        T<:Tagged,P
    }
        ArgCheck.@argcheck maxiterations >= burnin >= 0
        ArgCheck.@argcheck thinning > 0
        ArgCheck.@argcheck maxiterations > 0
        #Assign indices for subsetting trace
        effective_iterations = (burnin+1):thinning:maxiterations
        return new{T,P}(tagged, paramnames, chains, algorithms, burnin, thinning, maxiterations, effective_iterations)
    end
end

function TraceTransform(
    trace::Trace,
    model::ModelWrapper,
    tagged::Tagged = Tagged(model, trace.summary.info.printedparam.printed),
    info::TransformInfo = TransformInfo(
        collect(Base.OneTo(trace.summary.info.Nchains)),
        collect(Base.OneTo(trace.summary.info.Nalgorithms)),
        trace.summary.info.burnin,
        trace.summary.info.thinning,
        trace.summary.info.iterations
    )
)
    @unpack chains, algorithms, burnin, thinning, maxiterations = info
    paramnames = ModelWrappers.paramnames(
        tagged.info.reconstruct.default, tagged.info.transform.constraint, subset(model.val, tagged.parameter)
    )

    return TraceTransform(
        tagged,
        paramnames,
        chains, algorithms, burnin, thinning, maxiterations
)
end

################################################################################
"""
$(SIGNATURES)
Change trace.val to 3d Array that is consistent with MCMCCHains dimensons.
First dimension is iterations, second number of parameter, third number of chains.

# Examples
```julia
```

"""
function trace_to_3DArray(
    trace::Trace,
    transform::TraceTransform
)
    ## Get trace information
    @unpack tagged, chains, effective_iterations = transform
    ## Preallocate array
    mcmcchain = zeros(length(effective_iterations), length(chains), length_constrained(tagged))
    ## Flatten corresponding parameter
    #!NOTE: Commented-out loop below is threadsave, but chain is not flattened in correct order, which might be troublesome for MCMC chain analysis.
#    Threads.@threads for (idx, chain) in collect(enumerate(chains))
    for (idx, chain) in collect(enumerate(chains))
        #!NOTE: idx and chain are separated in case not all chains are used for summary
        for (iter0, iterburnin) in enumerate(effective_iterations)
            #!NOTE: New way of structuring parameter: (ndraws, nchains, nparams)
            mcmcchain[iter0, idx, :] .= flatten(tagged.info.reconstruct, subset(trace.val[chain][iterburnin], tagged.parameter))
        end
    end
    ## Return MCMCChain
    return mcmcchain
end

"""
$(SIGNATURES)
Change trace.val to 3d Array in unconstrained space that is consistent with MCMCCHains dimensons. First dimension is iterations, second number of parameter, third number of chains.

# Examples
```julia
```

"""
function trace_to_3DArrayᵤ(
    trace::Trace,
    transform::TraceTransform
)
    ## Get trace information
    @unpack tagged, chains, effective_iterations = transform
    ## Preallocate array
    mcmcchain = zeros(length(effective_iterations), length(chains), length_unconstrained(tagged))
    ## Flatten corresponding parameter
    #!NOTE: This is threadsave, but chain is not flattened in correct order, which might be troublesome for MCMC chain analysis.
#    Threads.@threads for (idx, chain) in collect(enumerate(chains))
    for (idx, chain) in collect(enumerate(chains))
        for (iter0, iterburnin) in enumerate(effective_iterations)
#            mcmcchain[iter0, :, idx] .=
            mcmcchain[iter0, idx, :] .=
#                flatten(tagged.info.reconstruct, unconstrain(tagged.info.transform, subset(trace.val[chain][iterburnin], tagged.parameter) ) )
                ModelWrappers.unconstrain_flatten(tagged.info, subset(trace.val[chain][iterburnin], tagged.parameter))
        end
    end
    ## Return MCMCChain
    return mcmcchain
end

################################################################################
"""
$(SIGNATURES)
Change trace.val to 2d Array. First dimension is iterations*chains, second number of parameter.

# Examples
```julia
```

"""
function trace_to_2DArray(
    trace::Trace,
    transform::TraceTransform
)
    ## Get trace information
    @unpack tagged, chains, effective_iterations = transform
    ## Preallocate array
    mcmcchain = zeros(length(effective_iterations) * length(chains), length_constrained(tagged))
    ## Flatten corresponding parameter
    #!NOTE: This is threadsave, but chain is not flattened in correct order, which might be troublesome for MCMC chain analysis.
#    Threads.@threads for (idx, chain) in collect(enumerate(chains))
    iter = 0
    for (idx, chain) in collect(enumerate(chains))
        for (iter0, iterburnin) in enumerate(effective_iterations)
            iter += 1
            mcmcchain[iter, :] .= flatten(tagged.info.reconstruct, subset(trace.val[chain][iterburnin], tagged.parameter))
        end
    end
    ## Return MCMCChain
    return mcmcchain
end
"""
$(SIGNATURES)
Change trace.val to 2d Array in unconstrained space. First dimension is iterations*chains, second number of parameter.

# Examples
```julia
```

"""
function trace_to_2DArrayᵤ(
    trace::Trace,
    transform::TraceTransform
)
    ## Get trace information
    @unpack tagged, chains, effective_iterations = transform
    ## Preallocate array
    mcmcchain = zeros(length(effective_iterations) * length(chains), length_unconstrained(tagged))
    ## Flatten corresponding parameter
    #!NOTE: This is threadsave, but chain is not flattened in correct ordered, which might be troublesome for MCMC chain analysis.
#    Threads.@threads for (idx, chain) in collect(enumerate(chains))
    iter = 0
    for (idx, chain) in collect(enumerate(chains))
        for (iter0, iterburnin) in enumerate(effective_iterations)
            iter += 1
            mcmcchain[iter, :] .= 
#                flatten(tagged.info.reconstruct, unconstrain(tagged.info.transform, subset(trace.val[chain][iterburnin], tagged.parameter) ) )
                ModelWrappers.unconstrain_flatten(tagged.info, subset(trace.val[chain][iterburnin], tagged.parameter))
        end
    end
    ## Return MCMCChain
    return mcmcchain
end

################################################################################
"""
$(SIGNATURES)
Change trace.val to 3d Array and return Posterior mean as NamedTuple and as Vector

# Examples
```julia
```

"""
function trace_to_posteriormean(
    mod_array::AbstractArray,
    transform::TraceTransform
)
    @unpack tagged = transform
    mod_array_mean = map(iter -> mean(view(mod_array, :, :, iter)), Base.OneTo(size(mod_array, 3)))
    mod_nt_mean = ModelWrappers.unflatten(tagged.info.reconstruct, mod_array_mean)
    return mod_array_mean, mod_nt_mean
end
function trace_to_posteriormean(
    trace::Trace,
    transform::TraceTransform
)
    return trace_to_posteriormean(
        trace_to_3DArray(trace, transform), transform
    )
end

############################################################################################
"""
$(SIGNATURES)
Return a view of a specific index 'chain' for Vector of Parameter chains of NamedTuples with index 'effective_iterations'.

# Examples
```julia
```

"""
function get_chainvals(val::F, chain::Integer, effective_iterations::StepRange{Int64}) where {F<:Vector{<:Vector{<:NamedTuple}} }
    return @view(val[chain][effective_iterations])
end

function get_chainvals(trace::Trace, transform::TraceTransform)
    @unpack tagged, chains, effective_iterations = transform
    ArgCheck.@argcheck length(chains) <= length(trace.val)
    return [map(x -> subset(x, tagged.parameter), get_chainvals(trace.val, chain, effective_iterations)) for chain in chains]
end

"""
$(SIGNATURES)
Merge Vector of Parameter chains of NamedTuples into a single vector.

# Examples
```julia
```

"""
function merge_chainvals(trace::Trace, transform::TraceTransform)
    return reduce(vcat, get_chainvals(trace, transform))
end

"""
$(SIGNATURES)
Flatten Vector of Parameter NamedTuples into a Matrix, where each row represents draws for a single parameter.

# Examples
```julia
```

"""
function flatten_chainvals(
    trace::Trace,
    transform::TraceTransform
)
    ## Get trace information
    @unpack tagged, chains, effective_iterations = transform
    ## Preallocate array
    mcmcchain = [ [ zeros(tagged.info.reconstruct.default.output, length_constrained(tagged)) for _ in eachindex(effective_iterations) ] for _ in eachindex(chains) ]
    ## Flatten corresponding parameter
    #!NOTE: This is threadsave, but chain is not flattened in correct order, which might be troublesome for MCMC chain analysis, hence we opt out of it.
#    Threads.@threads for (idx, chain) in collect(enumerate(chains))
    for (idx, chain) in collect(enumerate(chains))
        for (iter0, iterburnin) in enumerate(effective_iterations)
            mcmcchain[idx][iter0] .= flatten(tagged.info.reconstruct, subset(trace.val[chain][iterburnin], tagged.parameter))
        end
    end
    ## Return MCMCChain
    return mcmcchain
end

################################################################################
"""
$(SIGNATURES)
Obtain parameter diagnostics from trace at chain `chain`, excluding first `burnin` samples.

# Examples
```julia
```

"""
function get_chaindiagnostics(diagnostics, chain::Integer, Nalgorithm::Integer, effective_iterations::StepRange{Int64})
    return @view(diagnostics[chain][Nalgorithm][effective_iterations])
end

function get_chaindiagnostics(trace::Trace, transform::TraceTransform)
    @unpack chains, algorithms, effective_iterations = transform
    return [ map(algorithm -> get_chaindiagnostics(trace.diagnostics, chain, algorithm, effective_iterations), algorithms) for chain in chains]
end


################################################################################
# !Note: Add a couple of functions so that one can work with propose!() output only from a single kernel call
############################################################################################

function TraceTransform(
    vals::AbstractVector{<:NamedTuple},
    tagged::Tagged,
    info::TransformInfo = TransformInfo(
        [1],
        [1],
        0,
        1,
        length(vals)
    )
)
    @unpack chains, algorithms, burnin, thinning, maxiterations = info
    paramnames = ModelWrappers.paramnames(
        tagged.info.reconstruct.default, tagged.info.transform.constraint, subset(vals[begin], tagged.parameter)
    )
    return TraceTransform(
        tagged,
        paramnames,
        chains, algorithms, burnin, thinning, maxiterations
    )
end

function val_to_2DArray(
    vals::AbstractVector{<:NamedTuple},
    transform::TraceTransform
)
    ## Get trace information
    @unpack tagged, chains, effective_iterations = transform
    ## Preallocate array
    mcmcchain = zeros(length(effective_iterations), length_constrained(tagged))
    ## Flatten corresponding parameter
    for iter in effective_iterations
        mcmcchain[iter,:] .= flatten(tagged.info.reconstruct, subset(vals[iter], tagged.parameter))
    end
    ## Return MCMCChain
    return mcmcchain
end

function Array2D_to_NamedTuple(
    vals2d::Matrix{<:Real},
    tagged::Tagged
)
    # Assign range and length of parameter
    _lengths = tagged.info.reconstruct.unflatten.strict._unflatten.lngth
    _sz = tagged.info.reconstruct.unflatten.strict._unflatten.sz
    _range = [(_sz[i] - _lengths[i] + 1):_sz[i] for i in eachindex(_sz)]
    _names = keys(tagged.parameter)
    # Create NamedTuple
    nt2d = NamedTuple{_names}(map(iter -> view(vals2d, :, _range[iter]), eachindex(_range)))
    # Return Output
    return nt2d
end

function val_to_2DArrayᵤ(
    vals::AbstractVector{<:NamedTuple},
    transform::TraceTransform
)
    ## Get trace information
    @unpack tagged, chains, effective_iterations = transform
    ## Preallocate array
    mcmcchain = zeros(length(effective_iterations), length_unconstrained(tagged))
    ## Flatten corresponding parameter
    for iter in effective_iterations
        mcmcchain[iter,:] .= ModelWrappers.unconstrain_flatten(tagged.info, subset(vals[iter], tagged.parameter))
    end
    ## Return MCMCChain
    return mcmcchain
end

############################################################################################
#export
export
    TransformInfo,
    TraceTransform,

    trace_to_3DArray,
    trace_to_2DArray,
    trace_to_3DArrayᵤ,
    trace_to_2DArrayᵤ,
    trace_to_posteriormean,

    get_chainvals,
    get_chaindiagnostics,
    merge_chainvals,
    flatten_chainvals,

    val_to_2DArray,
    val_to_2DArrayᵤ,
    Array2D_to_NamedTuple


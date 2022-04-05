############################################################################################
"""
$(SIGNATURES)
Change trace.val to 3d Array that is consistent with MCMCCHains dimensons. First dimension is iterations, second number of parameter, third number of chains.

# Examples
```julia
```

"""
function trace_to_3DArray(
    trace::Trace,
    model::ModelWrapper,
    tagged::Tagged,
    burnin::Integer,
    thinning::Integer
)
    ## Get trace information
    Nparams = length(tagged)
    @unpack Nchains, iterations = trace.info.sampling
    @unpack flattendefault = model.info
    effective_iterations = (burnin+1):thinning:iterations
    ## Preallocate array
    mcmcchain = zeros(length(effective_iterations), Nparams, Nchains)
    ## Flatten corresponding parameter
    Threads.@threads for chain in Base.OneTo(Nchains)
        for (iter, index) in enumerate(effective_iterations)
            mcmcchain[iter, :, chain] .= first(
                ModelWrappers.flatten(
                    flattendefault,
                    subset(trace.val[chain][index], tagged.parameter),
                    tagged.info.constraint,
                ),
            )
        end
    end
    ## Return MCMCChain
    return mcmcchain
end

############################################################################################
"""
$(SIGNATURES)
Change trace.val to 3d Array and return Posterior mean as NamedTuple and as Vector

# Examples
```julia
```

"""
function trace_to_posteriormean(
    mod_array::AbstractArray,
    model::ModelWrapper,
    tagged::Tagged,
    burnin::Integer,
    thinning::Integer
)
    mod_array_mean = map(iter -> mean(view(mod_array, :, iter, :)), Base.OneTo(size(mod_array, 2)))
    mod_nt_mean = ModelWrappers.unflatten(model, tagged, mod_array_mean)
    return mod_array_mean, mod_nt_mean
end
function trace_to_posteriormean(
    trace::Trace,
    model::ModelWrapper,
    tagged::Tagged,
    burnin::Integer,
    thinning::Integer
)
    return trace_to_posteriormean(
        trace_to_3DArray(trace, model, tagged, burnin, thinning),
        model, tagged, burnin, thinning
    )
end

############################################################################################
#export
export trace_to_3DArray, trace_to_posteriormean

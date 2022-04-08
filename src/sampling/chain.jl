############################################################################################
"""
$(SIGNATURES)
Compute cross-chain diagnostics for parameter.

# Examples
```julia
```

"""
function chainparamdiagnostics(arr::Array{T,3}, computingtime::F) where {T<:Real,F<:Real}
    ess, rhat = MCMCDiagnosticTools.ess_rhat(arr)
    gelmandiagnostic = MCMCDiagnosticTools.gelmandiag(arr)
    vals = map(
        iter -> (
            ESS=ess[iter],
            ESSperSec=ess[iter] / computingtime,
            Rhat=rhat[iter],
            GelmanPSRF=gelmandiagnostic.psrf[iter],
        ),
        eachindex(ess),
    )
    return vals
end

function singlechaindiagnostics(arr::Array{T,3}, Nparams::Integer) where {T<:Real}
    vals = map(
        iter -> (
            ESS=NaN,
            ESSperSec=NaN,
            Rhat=NaN,
            GelmanPSRF=NaN),
        Base.OneTo(Nparams)
    )
    return vals
end

"""
$(SIGNATURES)
Compute in-chain diagnostics for parameter.

# Examples
```julia
```

"""
function paramdiagnostics(vec::AbstractVector{T}) where {T<:Real}
    vals = (
        Mean=Statistics.mean(vec),
        MCSE=MCMCDiagnosticTools.mcse(vec),
        StdDev=Statistics.std(vec),
    )
    return vals
end
function paramdiagnostics(arr::Array{T,3}) where {T<:Real}
    return map(
        iter -> paramdiagnostics(vec(view(arr, :, iter, :))), Base.OneTo(size(arr, 2))
    )
end

############################################################################################
"""
$(SIGNATURES)
Compute quantiles provided by `printdefault` for parameter.

# Examples
```julia
```

"""
function paramquantiles(vec::AbstractVector{T}, printdefault::PrintDefault) where {T<:Real}
    @unpack Ndigits, quantiles = printdefault
    vals = (Quantiles=Statistics.quantile(vec, quantiles),)
    return vals
end
function paramquantiles(arr::Array{T,3}, printdefault::PrintDefault) where {T<:Real}
    return map(
        iter -> paramquantiles(vec(view(arr, :, iter, :)), printdefault),
        Base.OneTo(size(arr, 2)),
    )
end

############################################################################################
"""
$(SIGNATURES)
Merge all statistics for all parameter.

# Examples
```julia
```

"""
function mergediagnostics(paramdiagnostic, chainparamdiagnostic, paramquantiles)
    Nparam = size(paramdiagnostic, 1)
    return map(
        param -> merge(
            paramdiagnostic[param], chainparamdiagnostic[param], paramquantiles[param]
        ),
        Base.OneTo(Nparam),
    )
end

############################################################################################
"""
$(SIGNATURES)
Return summary for trace parameter chains. `Model` defines flattening type of parameter,
`sym` defines parameter to be flattened, `backend` may be Val(:text), or Val(:latex).

# Examples
```julia
```

"""
function chainsummary(
    trace::Trace,
    model::ModelWrapper,
    sym::S,
    backend, #i.e., Val(:text), or Val(:latex)
    burnin::Integer,
    thinning::Integer,
    printdefault::PrintDefault=PrintDefault();
    kwargs...,
) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    ## Assign utility values
    @unpack Ndigits, quantiles = printdefault
    @unpack default = model.info.reconstruct
    @unpack progress = trace.info
    tagged = Tagged(model, sym)
    Nparams = length(tagged)
    paramnames = ModelWrappers.paramnames(
        default, tagged.info.constraint, subset(model.val, tagged.parameter)
    )
    computingtime = progress.enabled ? (progress.tlast - progress.tinit) : NaN
    arr3D = trace_to_3DArray(trace, model, tagged, burnin, thinning)
    ## Compute summary statistics
    #!NOTE If more than 1 chain used, can use cross-chain diagnostics
    if trace.info.sampling.Nchains > 1
        #NOTE: Most of the computing time of the function happens here
        chainparamdiagnostic = chainparamdiagnostics(arr3D, computingtime)
    else
        chainparamdiagnostic = singlechaindiagnostics(arr3D, Nparams)
    end
    paramdiagnostic = paramdiagnostics(arr3D)
    paramquantile = paramquantiles(arr3D, printdefault)
    diag = mergediagnostics(paramdiagnostic, chainparamdiagnostic, paramquantile)
    ## Assign row and header variables for table
    statsnames = union(keys(paramdiagnostic[begin]), keys(chainparamdiagnostic[begin]))
    quantilenames = string.("Q", round.(quantiles .* 100; digits=1))
    tablenames = union(string.(statsnames), quantilenames)
    Nstats = length(tablenames)
    ## Create table
    _reconstruct = ModelWrappers.ReConstructor(diag)
    diag_flattened = flatten(_reconstruct, diag)
    table = round.(reshape(diag_flattened, Nstats, Nparams)'; digits=Ndigits)
    ## Print table
    PrettyTables.pretty_table(
        table, backend=backend, header=tablenames, row_names=paramnames, kwargs...
    )
    ## Return table arguments
    return table, tablenames, paramnames
end

############################################################################################
#export
export chainsummary

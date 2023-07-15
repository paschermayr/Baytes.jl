############################################################################################
"""
$(SIGNATURES)
Compute cross-chain diagnostics for parameter.

# Examples
```julia
```

"""
function chainparamdiagnostics(arr::Array{T,3}, computingtime::F) where {T<:Real,F<:Real}
    Nparams = size(arr, 3)
    Nchains = size(arr, 2)
    #!NOTE If more than 1 chain used, can use cross-chain diagnostics
    if Nchains > 1
        ess_bulk = MCMCDiagnosticTools.ess(arr; kind = :bulk)
        ess_tail = MCMCDiagnosticTools.ess(arr; kind = :tail)
        rhat = MCMCDiagnosticTools.rhat(arr; kind = :rank)
    else
        ess_bulk = repeat([NaN], Nparams)
        ess_tail = repeat([NaN], Nparams)
        rhat = repeat([NaN], Nparams)
    end
    vals = map(
        iter -> (
            ESS_bulk=ess_bulk[iter],
            ESS_tail=ess_tail[iter],
            Rhat=rhat[iter],
        ),
        Base.OneTo(Nparams),
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
function paramdiagnostics(arr::Array{T,3}) where {T<:Real}
    # Compute statistics
    _mean = map(iter -> Statistics.mean(view(arr, :, :, iter)), Base.OneTo(size(arr, 3)) )
    _std = map(iter -> Statistics.std(view(arr, :, :, iter)), Base.OneTo(size(arr, 3)) )
    _mcse = MCMCDiagnosticTools.mcse(arr)
    # Assign NT
    return map(iter -> (
        Mean=_mean[iter],
        MCSE=_std[iter],
        StdDev=_mcse[iter],
        ), eachindex(_mean)
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
        iter -> paramquantiles(vec(view(arr, :, :, iter)), printdefault),
        Base.OneTo(size(arr, 3)),
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
function mergediagnostics(paramdiagnostic, paramquantiles, chainparamdiagnostic)
    Nparam = size(paramdiagnostic, 1)
    return map(
        param -> merge(
            paramdiagnostic[param], paramquantiles[param], chainparamdiagnostic[param]
        ),
        Base.OneTo(Nparam),
    )
end

############################################################################################

"""
$(SIGNATURES)
Check if any parameter has been stuck at each iteration in any chain, in which case chainsummary will skip computations.

# Examples
```julia
```

"""
function is_stuck(arr3D::AbstractArray)
    # Loop through chains and parameter to check if first parameter is equal to all samples == chain stuck
    for Nparams in Base.OneTo( size(arr3D, 3) )
        for Nchains in Base.OneTo( size(arr3D, 2) )
            _benchmark = arr3D[begin, Nchains, Nparams]
            stuck = all(val -> val == _benchmark, @view( arr3D[:, Nchains, Nparams] ))
            if stuck
                return true, (Nparams, Nchains)
            end
        end
    end
    return false, (0,0)
end

############################################################################################
"""
$(SIGNATURES)
Return summary for trace parameter chains. 'printdefault' defines quantiles and number of digits for printing.

# Examples
```julia
```

"""
function chainsummary(
    trace::Trace,
    transform::TraceTransform,
    printdefault::PrintDefault=PrintDefault()
)
    ## Assign utility values
    @unpack Ndigits, quantiles = printdefault
    @unpack progress = trace.summary
    @unpack tagged, paramnames = transform
    Nparams = length_constrained(tagged)
    Nchains = length(transform.chains)
    ## Flatten parameter to 3D array
    computingtime = progress.enabled ? (progress.tlast - progress.tinit) : NaN
    arr3D = trace_to_3DArray(trace, transform)
#=    
#NOTE: New Rhat/ESS implementation should just default to NaN if parameter are fixed, hence we can keep them in the output diagnostics    
    ## Check if any MCMC sampler was stuck in any chain, in which case chainsummary will be skipped
    stuck, paramchain = is_stuck(arr3D)
    if stuck
        return stuck, paramchain, nothing
    end
=#
    ## Compute summary statistics
    chainparamdiagnostic = chainparamdiagnostics(arr3D, computingtime)
    paramdiagnostic = paramdiagnostics(arr3D)
    paramquantile = paramquantiles(arr3D, printdefault)
    diag = mergediagnostics(paramdiagnostic, paramquantile, chainparamdiagnostic)
    ## Assign row and header variables for table
    paramstatsnames = keys(paramdiagnostic[begin])
    chainstatsnames = keys(chainparamdiagnostic[begin])
    quantilenames = string.("Q", round.(quantiles .* 100; digits=1))
    tablenames = union(string.(paramstatsnames), quantilenames, string.(chainstatsnames),)
    Nstats = length(tablenames)
    ## Create table
    _reconstruct = ModelWrappers.ReConstructor(diag)
    diag_flattened = flatten(_reconstruct, diag)
    table = round.(reshape(diag_flattened, Nstats, Nparams)'; digits=Ndigits)
    ## Return table arguments
    return table, tablenames, paramnames
end

############################################################################################
"""
$(SIGNATURES)
Print summary for trace parameter chains. `backend` may be Val(:text), or Val(:latex).

# Examples
```julia
```

"""
function printchainsummary(
    trace::Trace,
    transform::TraceTransform,
    backend, #i.e., Val(:text), or Val(:latex)
    printdefault::PrintDefault=PrintDefault();
    kwargs...,
)
    table, tablenames, paramnames = chainsummary(trace, transform, printdefault)
    if table isa Bool
        println(
            "#####################################################################################",
        )
        println("Chain is first stuck in (Nparam, Nchain) = ", tablenames, " - skipping chainsummary.")
        return nothing, nothing, nothing
    ## Print table
    else
        PrettyTables.pretty_table(
            table, backend=backend, header=tablenames, row_labels=paramnames, kwargs...
        )
    end
end

"""
$(SIGNATURES)
Add a ModelWrapper struct 'model' as a function argument to print model.val as "true" parameter in table.

# Examples
```julia
```

"""
function printchainsummary(
    model::ModelWrapper,
    trace::Trace,
    transform::TraceTransform,
    backend, #i.e., Val(:text), or Val(:latex)
    printdefault::PrintDefault=PrintDefault();
    kwargs...,
)
    table, tablenames, paramnames = chainsummary(trace, transform, printdefault)
    if table isa Bool
        println(
            "#####################################################################################",
        )
        println("Chain is first stuck in (Nparam, Nchain) = ", tablenames, " - skipping chainsummary.")
        return nothing, nothing, nothing
    ## Print table
    else
        #Obtain true parameter from model
        θ_true = round.(flatten(model, transform.tagged); digits = printdefault.Ndigits)
        PrettyTables.pretty_table(
            hcat(θ_true, table), backend=backend, header=vcat("True", tablenames), row_labels=paramnames, kwargs...
        )
    end
end

############################################################################################
"""
$(SIGNATURES)
Save summary for trace parameter chains as html file.

# Examples
```julia
```

"""
function savechainsummary(
    trace::Trace,
    transform::TraceTransform,
    printdefault::PrintDefault=PrintDefault(),
    name = join((
        "modeldiagnostics_",
        Dates.today(),
        "_H",
        Dates.hour(Dates.now()),
        "M",
        Dates.minute(Dates.now()),
        "_Nchains",
        trace.summary.info.Nchains,
        "_Iter",
        trace.summary.info.iterations,
        "_Burnin",
        trace.summary.info.burnin,
    ))
)
    # Compute diagnostics
    table, tablenames, paramnames = chainsummary(trace, transform, printdefault)
    # Return output as html document
    open(join((name, ".html")), "w") do f
        pretty_table(
            f,
            table,
            header=tablenames, row_labels=paramnames,
            backend=Val(:html),
            alignment = :c,
            tf = PrettyTables.tf_html_simple,
            standalone = true
        )
    end
end

"""
$(SIGNATURES)
Add a ModelWrapper struct 'model' as a function argument to save model.val as "true" parameter in table.

# Examples
```julia
```

"""
function savechainsummary(
    model::ModelWrapper,
    trace::Trace,
    transform::TraceTransform,
    printdefault::PrintDefault=PrintDefault(),
    name = join((
        "modeldiagnostics_",
        Dates.today(),
        "_H",
        Dates.hour(Dates.now()),
        "M",
        Dates.minute(Dates.now()),
        "_Nchains",
        trace.summary.info.Nchains,
        "_Iter",
        trace.summary.info.iterations,
        "_Burnin",
        trace.summary.info.burnin,
    ))
)
    # Compute diagnostics
    table, tablenames, paramnames = chainsummary(trace, transform, printdefault)
    #Obtain true parameter from model
    θ_true = round.(flatten(model, transform.tagged); digits = printdefault.Ndigits)
    # Return output as html document
    open(join((name, ".html")), "w") do f
        pretty_table(
            f,
            hcat(θ_true, table),
            header=vcat("True", tablenames), row_labels=paramnames,
            backend=Val(:html),
            alignment = :c,
            tf = PrettyTables.tf_html_simple,
            standalone = true
        )
    end
end

############################################################################################
#export
export chainsummary, printchainsummary, savechainsummary

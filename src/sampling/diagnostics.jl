############################################################################################
"""
$(SIGNATURES)
Obtain parameter diagnostics from trace at chain `chain`, excluding first `burnin` samples.

# Examples
```julia
```

"""
function get_chaindiagnostics(
    trace::Trace, chain::Integer, Nalgorithm::Integer, burnin::Integer, thinning::Integer
)
    return @view(trace.diagnostics[chain][Nalgorithm][(1 + burnin):thinning:end])
end

############################################################################################
"""
$(SIGNATURES)
Return summary for trace parameter diagnostics, `backend` may be Val(:text), or Val(:latex).

# Examples
```julia
```

"""
function diagnosticssummary(
    trace::Trace,
    algorithmᵛ::SMC,
    backend::Nothing, #i.e., Val(:text), or Val(:latex)
    burnin::Integer,
    thinning::Integer,
    printdefault::PrintDefault=PrintDefault();
    kwargs...,
)
    ## Assign utility variables
    @unpack Ndigits, quantiles = printdefault
    @unpack Nchains, Nalgorithms, burnin = trace.info.sampling
    ## Print diagnostics for each sampler for each chain
    println(
        "#####################################################################################",
    )
    return results(
        @view(trace.diagnostics[(1 + burnin):thinning:end]), algorithmᵛ, Ndigits, quantiles
    )
end

function diagnosticssummary(
    trace::Trace,
    algorithmᵛ::AbstractVector,
    backend::Nothing, #i.e., Val(:text), or Val(:latex)
    burnin::Integer,
    thinning::Integer,
    printdefault::PrintDefault=PrintDefault();
    kwargs...,
)
    ## Assign utility variables
    @unpack Ndigits, quantiles = printdefault
    @unpack Nchains, Nalgorithms = trace.info.sampling
    ## Print diagnostics for each sampler for each chain
    for Nalgorithm in Base.OneTo(Nalgorithms)
        println(
            "#####################################################################################",
        )
        for Nchain in Base.OneTo(Nchains)
            println("########################################## Chain ", Nchain, ":")
            println(Base.nameof(typeof(algorithmᵛ[Nchain][Nalgorithm])), " Diagnostics: ")
            results(
                get_chaindiagnostics(trace, Nchain, Nalgorithm, thinning, burnin),
                algorithmᵛ[Nchain][Nalgorithm],
                Ndigits,
                quantiles,
            )
        end
    end
end

############################################################################################
#export
export diagnosticssummary

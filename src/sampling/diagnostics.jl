############################################################################################
"""
$(SIGNATURES)
Return summary for trace parameter diagnostics, `backend` may be Val(:text), or Val(:latex).

# Examples
```julia
```

"""
function printdiagnosticssummary(
    trace::Trace,
    algorithmᵛ::SMC,
    transform::TraceTransform,
    backend::Nothing, #i.e., Val(:text), or Val(:latex)
    printdefault::PrintDefault=PrintDefault();
    kwargs...,
)
    ## Assign utility variables
    @unpack effective_iterations = transform
    @unpack Ndigits, quantiles = printdefault
    @unpack Nchains, Nalgorithms, burnin = trace.info.sampling
    ## Print diagnostics for each sampler for each chain
    println(
        "#####################################################################################",
    )
    return results(
        @view(trace.diagnostics[effective_iterations]), algorithmᵛ, Ndigits, quantiles
    )
end

function printdiagnosticssummary(
    trace::Trace,
    algorithmᵛ::AbstractVector,
    transform::TraceTransform,
    backend::Nothing, #i.e., Val(:text), or Val(:latex)
    printdefault::PrintDefault=PrintDefault();
    kwargs...,
)
    ## Assign utility variables
    @unpack Ndigits, quantiles = printdefault
    @unpack chains, algorithms, effective_iterations = transform
    ## Print diagnostics for each sampler for each chain
    for Nalgorithm in algorithms
        println(
            "#####################################################################################",
        )
        for Nchain in chains
            println("########################################## Chain ", Nchain, ":")
            println(Base.nameof(typeof(algorithmᵛ[Nchain][Nalgorithm])), " Diagnostics: ")
            results(
                get_chaindiagnostics(trace.diagnostics, Nchain, Nalgorithm, effective_iterations),
                algorithmᵛ[Nchain][Nalgorithm],
                Ndigits,
                quantiles,
            )
        end
    end
end

############################################################################################
#export
export printdiagnosticssummary

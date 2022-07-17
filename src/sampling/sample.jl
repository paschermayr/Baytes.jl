############################################################################################
"""
$(SIGNATURES)
Print summary of parameter chain and diagnostics to REPL.

# Examples
```julia
```

"""
function summary(
    trace::Trace,
    algorithmᵛ,
    model::ModelWrapper,
    sym=keys(model.val),
    burnin::Integer=trace.info.sampling.burnin,
    thinning::Integer = 1,
    printdefault::PrintDefault=PrintDefault(),
) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    ## Print Diagnostics summary
    diagnosticssummary(trace, algorithmᵛ, nothing, burnin, thinning, printdefault)
    ## Print Chain summary
    chainsummary(trace, model, sym, Val(:text), burnin, thinning, printdefault)
    ## Return
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Sample `model` parameter given `data` with `args` algorithm. Default sampling arguments given in `default` keyword.

# Examples
```julia
```

"""
function sample(
    _rng::Random.AbstractRNG, model::M, data::D, args...; default=SampleDefault()
) where {M<:ModelWrapper,S<:SampleDefault,D}
    @unpack dataformat, chains, tempering, iterations, burnin, thinning, safeoutput,
    printoutput, printdefault, report = default
    ## Check if datatune can be created
    datatune = DataTune(data, dataformat)
    ## Check if iterations have to be adjusted if sequential data is used
    iterations = maxiterations(datatune, iterations)
    ArgCheck.@argcheck iterations > burnin "Burnin set higher than number of iterations."
    ## Check if we can capture previous samples
    updatesampler = update(datatune, tempering.adaption, args...)
    ## Construct SamplingInfo and ProgressLog
    printedparameter = PrintedParameter(showparam(model, datatune, args...)...)
    info = SamplingInfo(printedparameter, iterations, burnin, thinning, length(args), chains, updatesampler, tempering.adaption)
    progressmeter = progress(report, info)
    ## Initialize algorithms
    println("Constructing new sampler...")
    modelᵛ, algorithmᵛ, tempertune, datatuneᵛ = construct(
        _rng, model, data, default, tempering, datatune, chains, args...
    )
    ## Initialize trace
    trace = Trace(
        _rng, algorithmᵛ, model, BaytesCore.adjust(datatune, data),
        TraceInfo(tempertune, datatuneᵛ, default, info, progressmeter)
    )
    ## Loop through iterations
    println("Sampling starts...")
    propose!(_rng, trace, algorithmᵛ, modelᵛ, data)
    ## Print diagnostics
    println("Sampling finished, printing diagnostics and saving trace.")
    if printoutput
        ## Assign relevant parameter for printing and print summary to REPL.
        sym = trace.info.sampling.printedparam.printed
        summary(trace, algorithmᵛ, model, sym, trace.info.sampling.burnin, thinning, printdefault)
    end
    ## Save output
    if safeoutput
        println("Saving trace, initial model and algorithm.")
        savetrace(trace, model, algorithmᵛ, chains, iterations, burnin)
    end
    ## Return trace and algorithm
    return trace, algorithmᵛ
end

################################################################################
"""
$(SIGNATURES)
Continue sampling with all algorithms in `algorithm`.

# Examples
```julia
```

"""
function sample!(iterations::Integer,
    _rng::Random.AbstractRNG, model::M, data::D,
    trace::Trace, algorithmᵛ
) where {M<:ModelWrapper,D}
    @unpack tempertune, datatune, sampling, default = trace.info
    @unpack Nalgorithms, Nchains, burnin, thinning, captured, tempered = sampling
    @unpack safeoutput, printoutput, printdefault, report = default
    ## Create new DataTune struct, taking into account current Index and data dimension
    datatune_new = update(datatune, data)
    ## Check if iterations have to be adjusted if sequential data is used
    iterations = maxiterations(datatune_new, iterations)
    ArgCheck.@argcheck iterations > burnin "Burnin set higher than number of iterations."
    info = SamplingInfo(sampling.printedparam, iterations, burnin, thinning, Nalgorithms, Nchains, captured, tempered)
    progressmeter = progress(report, info)
    ## Construct new models for algorithms
    modelᵛ, datatuneᵛ = construct(model, datatune_new, Nchains, algorithmᵛ)
    ## Construct new trace to store new samples
    trace_new = Trace(
        _rng, algorithmᵛ, model, BaytesCore.adjust(datatune_new, data),
        TraceInfo(tempertune, datatuneᵛ, default, info, progressmeter)
    )
    ## Loop through iterations
    println("Sampling starts...")
    propose!(_rng, trace_new, algorithmᵛ, modelᵛ, data)
    ## Print diagnostics
    println("Sampling finished, printing diagnostics.")
    if printoutput
        ## Assign relevant parameter for printing and print summary to REPL.
        sym = trace_new.info.sampling.printedparam.printed
        summary(trace_new, algorithmᵛ, model, sym, trace_new.info.sampling.burnin, thinning, printdefault)
    end
    ## Save output
    if safeoutput
        println("Saving trace, initial model and algorithm.")
        savetrace(trace_new, model, algorithmᵛ, Nchains, iterations, burnin)
    end
    ## Return trace and algorithm
    return trace_new, algorithmᵛ
end

################################################################################
#export
export SampleDefault, summary, sample

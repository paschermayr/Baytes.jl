############################################################################################
"""
$(TYPEDEF)

Contains useful information for post-sampling analysis. Also allows to continue sampling with given sampler.

# Fields
$(TYPEDFIELDS)
"""
struct TraceInfo{
    A<:TemperingMethod,
    B<:Union{DataTune, Vector{<:DataTune}},
    D<:BaytesCore.SampleDefault,
    S<:SamplingInfo
    }
    "Tuning container for temperature tempering"
    tempertune::A
    "Tuning container for data tempering"
    datatune::B
    "Default information used for sample function"
    default::D
    "Information about trace used for postprocessing."
    sampling::S
    "Progress Log while sampling."
    progress::ProgressMeter.Progress
    function TraceInfo(
        tempertune::A,
        datatune::B,
        default::D,
        sampling::S,
        progress::ProgressMeter.Progress,
    ) where {
        A<:TemperingMethod,
        B<:Union{DataTune, Vector{<:DataTune}},
        D<:BaytesCore.SampleDefault,
        S<:SamplingInfo
    }
        ## Return info
        return new{A,B,D,S}(tempertune, datatune, default, sampling, progress)
    end
end

############################################################################################
"""
$(TYPEDEF)

Contains sampling chain and diagnostics for given algorithms.

# Fields
$(TYPEDFIELDS)
"""
struct Trace{C<:TraceInfo, A<:NamedTuple,B}
    "Model samples ~ out vector for corresponding chain, inner vector for iteration"
    val::Vector{Vector{A}}
    "Algorithm diagnostics ~ out vector for corresponding chain, inner vector for iteration"
    diagnostics::Vector{B}
    "Information about trace used for postprocessing."
    info::C
    function Trace(
        val::Vector{Vector{A}},
        diagnostics::Vector{B},
        info::C,
    ) where {A,B,C<:TraceInfo}
        ## Return trace
        return new{C,A,B}(val, diagnostics, info)
    end
end

function Trace(
    _rng::Random.AbstractRNG,
    algorithmáµ›::A,
    model::ModelWrapper,
    data::D,
    info::TraceInfo,
) where {A,D}
    @unpack iterations, Nchains = info.sampling
    ## Create Model Parameter buffer
    val = [Vector{typeof(model.val)}(undef, iterations) for _ in Base.OneTo(Nchains)]
    ## Create Diagnostics buffer for each algorithm used
    diagtypes = infer(_rng, AbstractDiagnostics, algorithmáµ›, model, data)
    diagnostics = diagnosticsbuffer(diagtypes, iterations, Nchains, algorithmáµ›)
    ## Return trace
    return Trace(val, diagnostics, info)
end

############################################################################################
#3 Proposal step
"""
$(SIGNATURES)
Propose new parameter for each model in `modeláµ›` given `data` with algorithms `algorithmáµ›`. A separate model for each chain is provided to avoid pointer issues.
If `args` is a single `SMC` algorithm, chains will not be separately allocated but instead used within the algorithm.
Note that smc still works as intended if used alongside other mcmc sampler in `args`.

# Examples
```julia
```

"""
function propose!(
    _rng::Random.AbstractRNG,
    trace::Trace{<:TraceInfo{<:BaytesCore.IterationTempering}},
    algorithmáµ›::AbstractVector,
    modeláµ›::Vector{M},
    data::D,
) where {M<:ModelWrapper,D}
    @unpack default, tempertune, datatune, sampling, progress = trace.info
    @unpack iterations, Nchains, Nalgorithms, captured = sampling
    @unpack log = default.report
    ## Propagate through data
    Base.Threads.@threads for Nchain in Base.OneTo(Nchains)
        ## Compute initial temperature
        temperature = BaytesCore.initial(tempertune)
        for iter in Base.OneTo(iterations)
            ## Update data iteration and current representation
            BaytesCore.update!(_rng, datatune[Nchain])
            dataâ‚ś = BaytesCore.adjust(datatune[Nchain], data)
            ## Propose next step
            for Nalgorithm in Base.OneTo(Nalgorithms)
                _, trace.diagnostics[Nchain][Nalgorithm][iter] = propose!(
                    _rng,
                    algorithmáµ›[Nchain][Nalgorithm],
                    modeláµ›[Nchain],
                    dataâ‚ś,
                    temperature,
                    captured
                )
                update!(progress, log, trace.diagnostics[Nchain][Nalgorithm][iter])
            end
            trace.val[Nchain][iter] = modeláµ›[Nchain].val
            ## Compute new temperature
            temperature = update!(tempertune, trace, algorithmáµ›, iter)
        end
    end
    return nothing
end

function propose!(
    _rng::Random.AbstractRNG,
    trace::Trace{<:TraceInfo{<:BaytesCore.JointTempering}},
    algorithmáµ›::AbstractVector,
    modeláµ›::Vector{M},
    data::D,
) where {M<:ModelWrapper,D}
    @unpack default, tempertune, datatune, sampling, progress = trace.info
    @unpack iterations, Nchains, Nalgorithms, captured = sampling
    @unpack log = default.report
    ## Compute initial temperature
    temperature = BaytesCore.initial(tempertune)
    ## Propagate through data
    for iter in Base.OneTo(iterations)
        Base.Threads.@threads for Nchain in Base.OneTo(Nchains)
            ## Update data iteration and current representation
            BaytesCore.update!(_rng, datatune[Nchain])
            dataâ‚ś = BaytesCore.adjust(datatune[Nchain], data)
            ## Propose next step
            for Nalgorithm in Base.OneTo(Nalgorithms)
                _, trace.diagnostics[Nchain][Nalgorithm][iter] = propose!(
                    _rng,
                    algorithmáµ›[Nchain][Nalgorithm],
                    modeláµ›[Nchain],
                    dataâ‚ś,
                    temperature,
                    captured
                )
                update!(progress, log, trace.diagnostics[Nchain][Nalgorithm][iter])
            end
            trace.val[Nchain][iter] = modeláµ›[Nchain].val
        end
        ## Compute new temperature
        temperature = update!(tempertune, trace, algorithmáµ›, iter)
    end
    return nothing
end

function propose!(
    _rng::Random.AbstractRNG,
    trace::Trace,
    algorithmáµ›::SMC,
    modeláµ›::M,
    data::D,
) where {T<:Trace,M<:ModelWrapper,D}
    @unpack default, tempertune, datatune, sampling, progress = trace.info
    @unpack iterations, Nchains, Nalgorithms, captured = sampling
    @unpack log = default.report
    ## Compute initial temperature
    temperature = BaytesCore.initial(tempertune)
    ## Propagate through data
    for iter in Base.OneTo(iterations)
        ## Update data iteration and current representation
        BaytesCore.update!(_rng, datatune)
        dataâ‚ś = BaytesCore.adjust(datatune, data)
        ## Propagate through data
        _, trace.diagnostics[iter] = propose!(_rng, algorithmáµ›, modeláµ›, dataâ‚ś, temperature, captured)
        for Nchain in eachindex(algorithmáµ›.particles.model)
            trace.val[Nchain][iter] = algorithmáµ›.particles.model[Nchain].val
            update!(progress, log, trace.diagnostics[iter])
        end
        temperature = update!(tempertune, trace, algorithmáµ›, iter)
    end
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Update tempering tune based on output diagnostics of last kernels in each chain.

# Examples
```julia
```

"""
function update!(tempertune::IterationTempering, trace::Trace, algorithm, iter::Integer)
    return BaytesCore.update!(tempertune, iter)
end

function update!(tempertune::JointTempering, trace::Trace, algorithm::AbstractVector, iter::Integer)
    # Update log weights in weights buffer
    for Nchain in eachindex(tempertune.weights)
        tempertune.weights[Nchain] = BaytesCore.untemper(trace.diagnostics[Nchain][end][iter].base)
    end
    # Update and return temperature - tempertune.weights will be normalized weights in original scale.
    update!(tempertune, tempertune.weights)
    return tempertune.val.current
end
function update!(tempertune::JointTempering, trace::Trace, algorithm::SMC, iter::Integer)
    # Update weights
    scaling = 1.0/trace.diagnostics[iter].base.temperature
    for Nchain in eachindex(tempertune.weights)
        tempertune.weights[Nchain] = scaling * trace.diagnostics[iter].â„“weights[Nchain]
    end
    # Update and return temperature
    update!(tempertune, tempertune.weights)
    return tempertune.val.current
end

############################################################################################
"""
$(SIGNATURES)
Safe `trace`, `model` and `algorithm` to current working directory.

# Examples
```julia
```

"""
function savetrace(trace::Trace, model::ModelWrapper, algorithm)
    @unpack iterations, burnin, Nchains = trace.info.sampling
    JLD2.jldsave(
        join((
            Base.nameof(typeof(model.id)),
            "_",
            Base.nameof(typeof(algorithm)),
            "_",
            Dates.today(),
            "_H",
            Dates.hour(Dates.now()),
            "M",
            Dates.minute(Dates.now()),
            "_Nchains",
            Nchains,
            "_Iter",
            iterations,
            "_Burnin",
            burnin,
            ".jld2",
        ));
        trace=trace,
        model=model,
        algorithm=algorithm,
    )
    return nothing
end

############################################################################################
#export
export TraceInfo, Trace, propose!, savetrace

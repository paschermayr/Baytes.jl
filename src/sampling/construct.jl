############################################################################################
"""
$(SIGNATURES)
Construct a Tempering tuning struct with the appropriate type for model parameter.

# Examples
```julia
```

"""
function construct(
    tempering::BaytesCore.IterationTempering,
    model::ModelWrapper,
    chains::Integer
)
    MType = model.info.reconstruct.default.output
    @unpack L, k, x₀ = tempering.parameter
    ## Adjust types for parameter and initial value
    parameter = BaytesCore.TemperingParameter(MType(L), MType(k), MType(x₀))
    val₀ = BaytesCore.ValueHolder(MType(tempering.initial.current))
    ## Return correct Tempering struct
    tempertune = BaytesCore.IterationTempering(tempering.adaption, val₀, parameter)
    return tempertune, val₀.current
end
function construct(
    tempering::BaytesCore.JointTempering,
    model::ModelWrapper,
    chains::Integer
)
    tempertune = BaytesCore.JointTempering(
        model.info.reconstruct.default.output,
        tempering.adaption,
        tempering.val.current,
        tempering.ESSTarget.current,
        chains
    )
    return tempertune, tempertune.val.current
end

############################################################################################
"""
$(SIGNATURES)
Construct separate models and data tuning container for each chain.

# Examples
```julia
```

"""
function construct(model::ModelWrapper, datatune::DataTune, chains::Integer, args...)
    ## Initialize separate models that work with each chain
    modelᵛ = map(
        iter -> ModelWrapper(deepcopy(model.val), deepcopy(model.arg), model.info, model.id), Base.OneTo(chains)
    )
    datatuneᵛ = map(iter -> deepcopy(datatune), Base.OneTo(chains))
    return modelᵛ, datatuneᵛ
end
function construct(model::ModelWrapper, datatune::DataTune, chains::Integer, smc::S) where {S<:Union{SMC, SMCConstructor}}
    return model, datatune
end

############################################################################################
"""
$(SIGNATURES)
Construct `chains` MCMC chains and initiate all algorithms stated in `args` separately. A separate model for each chain is provided to avoid pointer issues.
If `args` is a single `SMC` algorithm, chains will not be separately allocated but instead used within the algorithm.
Note that smc still works as intended if used alongside other mcmc sampler in `args`.

# Examples
```julia
```

"""
function construct(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    default::BaytesCore.SampleDefault,
    tempering::BaytesCore.TemperingMethod,
    datatune::DataTune,
    chains::Integer,
    updatesampler::BaytesCore.UpdateBool,
    args...;
) where {D}
    ## Initiate Tempering Tune with correct model parameter type.
    temperingtune, temperature₀ = construct(tempering, model, default.chains)
    ## Initialize separate models and data tunes that work with each chain
    modelᵛ, datatuneᵛ = construct(model, datatune, chains, args...)
    ## Initialize sampler combination of first chain to deduce algorithm types
    algorithms = map(
        algorithm -> algorithm(
            _rng,
            modelᵛ[begin],
            BaytesCore.adjust(datatuneᵛ[begin], data),
            BaytesCore.ProposalTune(temperature₀, updatesampler, datatuneᵛ[begin]),
            default
        ),
        args,
    )
    algorithmsᵛ = Vector{typeof(algorithms)}(undef, chains)
    algorithmsᵛ[1] = algorithms
    ## Initialize algorithms from args... chains-1 times
    Threads.@threads for Nchain in 2:chains
        algorithmsᵛ[Nchain] = map(
            algorithm -> algorithm(
                _rng, modelᵛ[Nchain],
                BaytesCore.adjust(datatuneᵛ[Nchain], data),
                BaytesCore.ProposalTune(temperature₀, updatesampler, datatuneᵛ[Nchain]),
                default
            ),
            args,
        )
    end
    return modelᵛ, algorithmsᵛ, temperingtune, datatuneᵛ
end
function construct(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    default::BaytesCore.SampleDefault,
    tempering::BaytesCore.TemperingMethod,
    datatune::DataTune,
    chains::Integer,
    updatesampler::BaytesCore.UpdateBool,
    smc::SMCConstructor;
) where {D}
    ## Assign tempering struct with correct temperature type
    temperingtune, temperature₀ = construct(tempering, model, default.chains)
    ## Assign model for chains
    modelᵛ, datatuneᵛ = construct(model, datatune, chains, smc)
    ## Assign algorithm
    algorithmsᵛ = smc(_rng,
        model,
        BaytesCore.adjust(datatune, data),
        BaytesCore.ProposalTune(temperature₀, updatesampler, datatune), 
        default
    )
    return modelᵛ, algorithmsᵛ, temperingtune, datatuneᵛ
end

############################################################################################
#export
export construct

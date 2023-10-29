############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter1, (;), FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter1, (;), FlattenDefault(; output = Float32)), data_uv)
]
Nchains = 4
tempermethods = [
    IterationTempering(Float64, UpdateFalse(), 1.0, 1000),
    IterationTempering(Float64, UpdateTrue(), 1.0, 1000),
    JointTempering(Float64, UpdateFalse(), .5, Float64(Nchains), Nchains),
    JointTempering(Float64, UpdateTrue(), .5, Float64(Nchains), Nchains)
]

#=
iter = 2
tempermethod = tempermethods[iter]
=#

## Add custom Step for propagate
using BaytesDiff
import BaytesOptim: BaytesOptim, propagate
## Extend Custom Method
function propagate(
    _rng::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective{<:ModelWrapper{MyBaseModel}})
    logobjective = BaytesDiff.ℓDensityResult(objective)
    #logobjective.θᵤ[1] = 5
    logobjective.θᵤ[1] = rand()
    return logobjective
end

############################################################################################
@testset "Sampling, type conversion" begin
    for tempermethod in tempermethods
        for iter in eachindex(objectives)
            #println(tempermethod, " ", iter)
                sampledefault = SampleDefault(;
                    dataformat=Batch(),
                    tempering=deepcopy(tempermethod), #IterationTempering(Float64, UpdateFalse(), 1.0, 1000),
                    chains=4,
                    iterations=100,
                    burnin=max(1, Int64(floor(10/10))),
                    thinning = 1,
                    safeoutput=false,
                    printoutput=false,
                    printdefault=PrintDefault(),
                    report=ProgressReport(;
                        bar=false,
                        log=SilentLog()
                    ),
                )
                temperupdate = sampledefault.tempering.adaption
                _obj = deepcopy(objectives[iter])
                _flattentype = _obj.model.info.reconstruct.default.output

                # Create Custom Algorithm
                def = CustomAlgorithmDefault(; 
                    generated=UpdateTrue()
                )
                opt = CustomAlgorithm(
                    _rng,
                    _obj,
                    def,
                )

                ## Sample on its own
                customconstruct = CustomAlgorithm(:μ) #CustomAlgorithm(keys(_obj.model.val))
                trace, algorithms = sample(_rng, _obj.model, _obj.data, customconstruct ; default = deepcopy(sampledefault))
                trace.val

                ## Combine with MCMC
                mcmc = MCMC(NUTS,(:σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
                trace, algorithms = sample(_rng, _obj.model, _obj.data, customconstruct, mcmc ; default = deepcopy(sampledefault))
                trace.val

                ## Use as Propagation Kernel in SMC
                ibis = SMCConstructor(customconstruct, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
                trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
                trace.val
                ## Always update Gradient Result if new data is added
                    #!NOTE: But after first iteration, can capture results
                    @test isa(trace.summary.info.captured, UpdateFalse)
                ## Continue sampling
                newdat = randn(_rng, length(_obj.data)+100)
                trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
                    #!NOTE: But after first iteration, can capture results
                @test isa(trace2.summary.info.captured, UpdateFalse)
        end
    end
end

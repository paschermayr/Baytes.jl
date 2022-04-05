############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter1, FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter1, FlattenDefault(; output = Float32)), data_uv)
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
############################################################################################
@testset "Sampling, type conversion" begin
    for tempermethod in tempermethods
        for iter in eachindex(objectives)
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
            _flattentype = _obj.model.info.flattendefault.output
    #MCMC
            ## Assign kernels
            mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc ; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, typeof(temperupdate))
            ## Continue sampling
            trace2, algorithms2 = sample!(100, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.info.sampling.captured, typeof(temperupdate))
            postmean = trace_to_posteriormean(trace, _obj.model, _obj.tagged, 0, 1)
    #SMC
            ibis = SMCConstructor(mcmc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, UpdateFalse)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+100)
            trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.info.sampling.captured, UpdateFalse)
    # Combinations
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc, ibis; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, UpdateTrue)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+100)
            trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.info.sampling.captured, UpdateTrue)
        end
    end
end

############################################################################################
## Make model for several parameter types
@testset "Sampling, printing and ConsoleLog" begin
    for (iter, tempermethod) in enumerate(tempermethods)
            ## Set SampleDefault
            sampledefault = SampleDefault(;
                dataformat=Batch(),
                tempering=deepcopy(tempermethod),
                chains=4,
                iterations=100,
                burnin=max(1, Int64(floor(100/10))),
                thinning = 1,
                safeoutput=false,
                printoutput=true,
                printdefault=PrintDefault(),
                report=ProgressReport(;
                    bar=true,
                    log=ConsoleLog()
                ),
            )
            temperupdate = sampledefault.tempering.adaption
            _obj = deepcopy(objectives[1])
    #MCMC
            ## Assign kernels
            mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc ; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, typeof(temperupdate))
            ## Continue sampling
            trace2, algorithms2 = sample!(100, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.info.sampling.captured, typeof(temperupdate))
    #SMC
            ibis = SMCConstructor(mcmc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, UpdateFalse)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+100)
            trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.info.sampling.captured, UpdateFalse)
    # Combinations
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc, ibis; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, UpdateTrue)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+100)
            trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.info.sampling.captured, UpdateTrue)
    end
end

############################################################################################
#SMC
@testset "Sampling, Sequential Estimation" begin
    for (iter, tempermethod) in enumerate(tempermethods)
            ## Set SampleDefault
            sampledefault = SampleDefault(;
                dataformat=Expanding(100),
                tempering= deepcopy(tempermethod),
                chains=4,
                iterations=500,
                burnin=0,
                thinning = 1,
                safeoutput=false,
                printoutput=true,
                printdefault=PrintDefault(),
                report=ProgressReport(;
                    bar=false,
                    log=SilentLog()
                ),
            )
            temperupdate = sampledefault.tempering.adaption
            _obj = deepcopy(objectives[1])
    #MCMC
            ## Assign kernels
            mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc ; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            #!NOTE: If Expanding/Increasing data, update always true
            @test isa(trace.info.sampling.captured, UpdateTrue)
            ## Continue sampling
            trace2, algorithms2 = sample!(100, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.info.sampling.captured, UpdateTrue)
    #SMC
            ibis = SMCConstructor(mcmc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, UpdateFalse)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+100)
            trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.info.sampling.captured, UpdateFalse)
    # SMC2
        _obj = deepcopy(myobjective_mcmc)
        _tagged_pf = myobjective_pf.tagged
        _tagged_mcmc = myobjective_mcmc.tagged
        pfdefault_propagate = ParticleFilterDefault(referencing = Marginal(),)
        pfdefault_pmcmc = ParticleFilterDefault(referencing = Ancestral(),)
        mcmcdefault = MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        pmcmc = ParticleGibbs(
            ParticleFilterConstructor(keys(_tagged_pf.parameter), pfdefault_pmcmc),
            MCMCConstructor(NUTS, keys(_tagged_mcmc.parameter), mcmcdefault)
        )
        pf = ParticleFilterConstructor(keys(_tagged_pf.parameter), pfdefault_propagate)
        smc2 = SMCConstructor(SMC2Constructor(pf, pmcmc), SMCDefault(jitterthreshold=0.75, resamplingthreshold=0.75))
        trace, algorithms = sample(_rng, _obj.model, data, smc2; default = deepcopy(sampledefault))
        ## If single mcmc kernel assigned, can capture previous results
        @test isa(trace.info.sampling.captured, UpdateFalse)
        ## Continue sampling
        newdat = randn(_rng, length(data)+100)
        trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
        @test isa(trace2.info.sampling.captured, UpdateFalse)
        ##Check posterior parameter

    end
end

############################################################################################
#Utility
@testset "Utility, maxiterations" begin
    Nstart = 10^2
    Niter1 = 10^4
    Niter2 = 10^1
    Ndata = length(data) - Nstart

    datatune = DataTune(data, Expanding(Nstart))
    @test Baytes.maxiterations(datatune, Niter1) == Ndata
    @test Baytes.maxiterations(datatune, Niter2) == Niter2
end

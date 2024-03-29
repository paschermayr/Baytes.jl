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
        #MCMC
                ## Assign kernels
                mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
                trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc ; default = deepcopy(sampledefault))
                ## If single mcmc kernel assigned, can capture previous results
                @test isa(trace.summary.info.captured, typeof(temperupdate))
                ## Continue sampling
                trace2, algorithms2 = sample!(100, _rng, _obj.model, _obj.data, trace, algorithms)
                @test isa(trace2.summary.info.captured, typeof(temperupdate))
                ## Inference Section
                transform = Baytes.TraceTransform(trace, _obj.model)
                postmean = trace_to_posteriormean(trace, transform)
                trace_to_crosschainmean(trace, transform)

                post3D = trace_to_3DArray(trace, transform)
                post3Dᵤ = trace_to_3DArrayᵤ(trace, transform)
                @test size(post3D) == size(post3Dᵤ)
                post2D = trace_to_2DArray(trace, transform)
                post2Dᵤ = trace_to_2DArrayᵤ(trace, transform)
                @test size(post2D) == size(post2Dᵤ)
                # Check if TransformInfo can be inferred from trace
                TransformInfo(trace)

###                
                #Check if we can also work with single chain values
                _vals = trace.val[1]
                _tagged = _obj.tagged
                _transform = TraceTransform(_vals, _tagged)
                _vals2d = val_to_2DArray(_vals, _transform)
                _vals2dᵤ = val_to_2DArrayᵤ(_vals, _transform)
                _tup2d = Array2D_to_NamedTuple(_vals2d, _tagged)
                
###

                #Check trace transforms
                g_vals = get_chainvals(trace, transform)
                m_vals = merge_chainvals(trace, transform)
                f_vals = flatten_chainvals(trace, transform)
                @test sum( map(val -> length(val), g_vals) ) == length(m_vals) == sum( map(val -> length(val), f_vals) )

                get_chaindiagnostics(trace, transform)

                #Check printing commands
                printchainsummary(trace, transform, Val(:text))
                printchainsummary(_obj.model, trace, transform, Val(:text))
        #SMC
                ibis = SMCConstructor(mcmc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
                trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
                ## If single mcmc kernel assigned, can capture previous results
                @test isa(trace.summary.info.captured, UpdateFalse)
                ## Continue sampling
                newdat = randn(_rng, length(_obj.data)+100)
                trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
                @test isa(trace2.summary.info.captured, UpdateFalse)
        # Combinations
                trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc, ibis; default = deepcopy(sampledefault))
                ## If single mcmc kernel assigned, can capture previous results
                @test isa(trace.summary.info.captured, UpdateTrue)
                ## Continue sampling
                newdat = randn(_rng, length(_obj.data)+100)
                trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
                @test isa(trace2.summary.info.captured, UpdateTrue)
        end
    end
end

############################################################################################
chains = [1, Nchains, 1, Nchains]
temperchainmethods = [
    IterationTempering(Float64, UpdateTrue(), float(chains[1]), 1000),
    IterationTempering(Float64, UpdateTrue(), float(chains[2]), 1000),
    JointTempering(Float64, UpdateTrue(), .5, float(chains[1]), chains[1]),
    JointTempering(Float64, UpdateTrue(), .5, float(chains[2]), chains[2])
]

@testset "Sampling, Chain Management" begin
    for (iter, tempermethod) in enumerate(temperchainmethods)
            ## Set SampleDefault
            sampledefault = SampleDefault(;
                dataformat=Batch(),
                tempering=deepcopy(tempermethod),
                chains=chains[iter],
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
            @test isa(trace.summary.info.captured, typeof(temperupdate))
            ## Continue sampling
            trace2, algorithms2 = sample!(100, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.summary.info.captured, typeof(temperupdate))
    # Combinations
            mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc, mcmc ; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.summary.info.captured, typeof(temperupdate))
            ## Continue sampling
            trace2, algorithms2 = sample!(100, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.summary.info.captured, typeof(temperupdate))
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
            @test isa(trace.summary.info.captured, typeof(temperupdate))
            ## Continue sampling
            trace2, algorithms2 = sample!(100, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.summary.info.captured, typeof(temperupdate))
    #SMC
            ibis = SMCConstructor(mcmc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.summary.info.captured, UpdateFalse)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+100)
            trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.summary.info.captured, UpdateFalse)
    # Combinations
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc, ibis; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.summary.info.captured, UpdateTrue)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+100)
            trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.summary.info.captured, UpdateTrue)
    end
end

############################################################################################
#SMC
@testset "Sampling, Sequential Estimation" begin
    for (iter, tempermethod) in enumerate(tempermethods)
    #        println(iter)
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
#            (iter, tempermethod) = collect( enumerate(tempermethods) )[4]
            temperupdate = sampledefault.tempering.adaption
            _obj = deepcopy(objectives[1])
    #MCMC
            ## Assign kernels
            mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc ; default = deepcopy(sampledefault))
            ## If single mcmc kernel assigned, can capture previous results
            #!NOTE: If Expanding/Increasing data, update always true
            @test isa(trace.summary.info.captured, UpdateTrue)
            ## Continue sampling
            trace2, algorithms2 = sample!(500, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.summary.info.captured, UpdateTrue)
    #SMC
            ibis = SMCConstructor(mcmc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
#            transform = TraceTransform(trace, _obj.model)
#            summary(trace, algorithms,transform,PrintDefault(),)
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.summary.info.captured, UpdateFalse)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+500)
            trace2, algorithms2 = sample!(500, _rng, _obj.model, newdat, trace, algorithms)
            @test isa(trace2.summary.info.captured, UpdateFalse)
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
        @test isa(trace.summary.info.captured, UpdateFalse)
        #Check if correct parameter are printed
        allparam, printparam = Baytes.showparam(_obj.model, trace.summary.datatune, smc2)
        @test allparam == keys(_obj.model.val)
        @test printparam == (:μ, :σ, :p)
        ## Continue sampling
        newdat = randn(_rng, length(data)+500)
        trace2, algorithms2 = sample!(500, _rng, _obj.model, newdat, trace, algorithms)
        @test isa(trace2.summary.info.captured, UpdateFalse)
    end
end

############################################################################################
#tempermethod = tempermethods[1]
#iter = length(objectives)
using Optim, NLSolversBase
@testset "Sampling, BaytesOptim" begin
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
## Optimization
                def = OptimDefault(; 
                kernel = (;
                    magnitude_penalty = 1e-4,
                    iterations = 1000
                    )
                )
                opt = Optimizer(
                _rng,
                OptimLBFG,
                _obj,
                def,
                )
## Test Constructor
                mcmc = MCMC(NUTS,(:σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
                _oc = OptimConstructor(OptimLBFG, :μ, 
                OptimDefault(; 
                    kernel = (;
                        iterations = 123)
                ) 
                )
                Optimizer(OptimLBFG, :μ)
                trace, algorithms = sample(_rng, _obj.model, _obj.data, _oc; default = deepcopy(sampledefault))
                ## If single optimizer kernel assigned, can capture previous results
                @test isa(trace.summary.info.captured, typeof(temperupdate))
## Inference Section
                transform = Baytes.TraceTransform(trace, _obj.model)
                postmean = trace_to_posteriormean(trace, transform)
                trace_to_crosschainmean(trace, transform)

                post3D = trace_to_3DArray(trace, transform)
                post3Dᵤ = trace_to_3DArrayᵤ(trace, transform)
                @test size(post3D) == size(post3Dᵤ)
                post2D = trace_to_2DArray(trace, transform)
                post2Dᵤ = trace_to_2DArrayᵤ(trace, transform)
                @test size(post2D) == size(post2Dᵤ)        
##
                #Check trace transforms
                g_vals = get_chainvals(trace, transform)
                m_vals = merge_chainvals(trace, transform)
                f_vals = flatten_chainvals(trace, transform)
                @test sum( map(val -> length(val), g_vals) ) == length(m_vals) == sum( map(val -> length(val), f_vals) )

                #Check printing commands
                printchainsummary(trace, transform, Val(:text))
                printchainsummary(_obj.model, trace, transform, Val(:text))

## SMC via IBIS
                ibis = SMCConstructor(_oc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
                trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = deepcopy(sampledefault))
                ## Always update Gradient Result if new data is added
                    #!NOTE: But after first iteration, can capture results
                @test isa(trace.summary.info.captured, UpdateFalse)
                ## Continue sampling
                newdat = randn(_rng, length(_obj.data)+100)
                trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
                    #!NOTE: But after first iteration, can capture results
                @test isa(trace2.summary.info.captured, UpdateFalse)
                
## Combinations
                trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc, _oc; default = deepcopy(sampledefault))
                transform = Baytes.TraceTransform(trace, _obj.model)
                printchainsummary(trace, transform, Val(:text))
                m_vals = merge_chainvals(trace, transform)
                ## Continue sampling
                newdat = randn(_rng, length(_obj.data)+100)
                trace2, algorithms2 = sample!(100, _rng, _obj.model, newdat, trace, algorithms)
            end
    end
end

############################################################################################
# Check if Custom Sampler works



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

@testset "Utility, check if chain stuck" begin
    _Ndraws = 1000
    _Nparams = 10
    _Nchains = 4
    _chain = randn(_rng, _Ndraws, _Nchains, _Nparams)
    _chain2 = deepcopy(_chain)
    param_stuck = 7
    chain_stuck = 3
    _chain2[:, chain_stuck, param_stuck] .= 1.0

    stuck, paramchain = Baytes.is_stuck(_chain)
    stuck2, paramchain2 = Baytes.is_stuck(_chain2)

    @test stuck == false
    @test stuck2 == true
    @test paramchain2[1] == param_stuck
    @test paramchain2[2] == chain_stuck
end

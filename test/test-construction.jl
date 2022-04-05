############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter1, FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter1, FlattenDefault(; output = Float32)), data_uv)
]
temperupdates = [UpdateFalse(), UpdateTrue()]
booleanupdates = [true, false]

#=
iter = 1
temperupdate = UpdateTrue()
=#

## Make model for several parameter types
for iter in eachindex(objectives)
    for temperupdate in temperupdates
        ## Set SampleDefault
        sampledefault = SampleDefault(;
            dataformat=Batch(),
            tempering=IterationTempering(Float64, temperupdate, 1.0, 1000),
            chains=4,
            iterations=1000,
            burnin=max(1, Int64(floor(1000/10))),
            thinning = 1,
            safeoutput=false,
            printoutput=booleanupdates[iter],
            printdefault=PrintDefault(),
            report=ProgressReport(;
                bar=true,
                log=ConsoleLog()
            ),
        )
        _obj = deepcopy(objectives[iter])
        _flattentype = _obj.model.info.flattendefault.output
        @testset "Sampling, all models" begin
            ## Assign kernels
            mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc ; default = sampledefault)
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, typeof(temperupdate))
            #summary(trace, algorithms, _obj.model, keys(_obj.model.val), 0)
            ## Continue sampling
            trace2, algorithms2 = sample!(1000, _rng, _obj.model, _obj.data, trace, algorithms)
            @test isa(trace2.info.sampling.captured, typeof(temperupdate))
        end
    end
end

############################################################################################
#SMC
for iter in eachindex(objectives)
    for temperupdate in temperupdates
        ## Set SampleDefault
        sampledefault = SampleDefault(;
            dataformat=Expanding(100),
            tempering=JointTempering(Float64, temperupdate, .5, Float64(4), 4),
            chains=10,
            iterations=500,
            burnin=max(1, Int64(floor(10/10))),
            thinning = 1,
            safeoutput=false,
            printoutput= booleanupdates[iter],
            printdefault=PrintDefault(),
            report=ProgressReport(;
                bar=false,
                log=SilentLog()
            ),
        )
        _obj = deepcopy(objectives[iter])
        _flattentype = _obj.model.info.flattendefault.output
        @testset "Sampling, IBIS - New data" begin
#            println(iter, " ", temperupdate, " ", booleanupdates[iter])
            ## Assign kernels
            mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
            ibis = SMCConstructor(mcmc, SMCDefault(jitterthreshold=0.99, resamplingthreshold=1.0))
            trace, algorithms = sample(_rng, _obj.model, _obj.data, ibis; default = sampledefault)
            ## If single mcmc kernel assigned, can capture previous results
            @test isa(trace.info.sampling.captured, UpdateFalse)
            #summary(trace, algorithms, _obj.model, keys(_obj.model.val), 0)
            ## Continue sampling
            newdat = randn(_rng, length(_obj.data)+500)
            trace2, algorithms2 = sample!(500, _rng, _obj.model, newdat, trace, algorithms)
#            @test isa(trace2.info.sampling.captured, UpdateFalse)
        end
    end
end

############################################################################################
#SMC2

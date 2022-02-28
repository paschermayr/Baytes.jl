############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault(; output = Float32)), data_uv)
]

sampledefault = SampleDefault(;
    dataformat=Batch(),
    tempering=IterationTempering(Float64, UpdateFalse(), 1.0, 1000),
    chains=4,
    iterations=2000,
    burnin=max(1, Int64(floor(2000/10))),
    thinning = 1,
    safeoutput=false,
    printoutput=false,
    printdefault=PrintDefault(),
    report=ProgressReport(;
        bar=false,
        log=SilentLog()
    ),
)
## Make model for several parameter types
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.flattendefault.output
    @testset "Sampling, all models" begin
        ## Assign kernels
        mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        trace, algorithms = sample(_rng, _obj.model, _obj.data, mcmc ; default = sampledefault)
        ## If single mcmc kernel assigned, can capture previous results
        @test isa(trace.info.sampling.captured, UpdateFalse)
        #!TODO Trace.val initialized based on first model.val, so will always store type of initial model.val, not flattendefault
        #@test typeof(trace.val[begin][begin].μ) == _flattentype
    end
end

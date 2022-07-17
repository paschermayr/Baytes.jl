var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Baytes","category":"page"},{"location":"#Baytes","page":"Home","title":"Baytes","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Baytes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Baytes]","category":"page"},{"location":"#Baytes.Baytes","page":"Home","title":"Baytes.Baytes","text":"Bayesian inference on state space models\n\n\n\n\n\n","category":"module"},{"location":"#Baytes.PrintedParameter","page":"Home","title":"Baytes.PrintedParameter","text":"struct PrintedParameter{A<:Tuple, B<:Tuple}\n\nContains several useful information for sampled parameter.\n\nFields\n\ntagged::Tuple\nUnique tagged parameter of all kernels.\nprinted::Tuple\nParameter names based on their dimension\n\n\n\n\n\n","category":"type"},{"location":"#Baytes.SamplingInfo","page":"Home","title":"Baytes.SamplingInfo","text":"struct SamplingInfo{A<:Baytes.PrintedParameter, U<:UpdateBool, B<:UpdateBool}\n\nContains several useful information for constructing sampler.\n\nFields\n\nprintedparam::Baytes.PrintedParameter\nParameter settings for printing.\niterations::Int64\nTotal number of sampling iterations.\nburnin::Int64\nBurnin iterations.\nthinning::Int64\nNumber of consecutive samples taken for diagnostics output.\nNalgorithms::Int64\nNumber of algorithms used while sampling.\nNchains::Int64\nNumber of chains used while sampling.\ncaptured::UpdateBool\nBoolean if sampler need to be updated after a proposal run. This is the case for, e.g. PMCMC., or for adaptive tempering.\ntempered::UpdateBool\nBoolean if temperature is adapted for target function.\n\n\n\n\n\n","category":"type"},{"location":"#Baytes.Trace","page":"Home","title":"Baytes.Trace","text":"struct Trace{C<:TraceInfo, A<:NamedTuple, B}\n\nContains sampling chain and diagnostics for given algorithms.\n\nFields\n\nval::Array{Vector{A}, 1} where A<:NamedTuple\nModel samples ~ out vector for corresponding chain, inner vector for iteration\ndiagnostics::Vector\nAlgorithm diagnostics ~ out vector for corresponding chain, inner vector for iteration\ninfo::TraceInfo\nInformation about trace used for postprocessing.\n\n\n\n\n\n","category":"type"},{"location":"#Baytes.TraceInfo","page":"Home","title":"Baytes.TraceInfo","text":"struct TraceInfo{A<:TemperingMethod, B<:Union{DataTune, Vector{<:DataTune}}, D<:SampleDefault, S<:SamplingInfo}\n\nContains useful information for post-sampling analysis. Also allows to continue sampling with given sampler.\n\nFields\n\ntempertune::TemperingMethod\nTuning container for temperature tempering\ndatatune::Union{DataTune, Vector{<:DataTune}}\nTuning container for data tempering\ndefault::SampleDefault\nDefault information used for sample function\nsampling::SamplingInfo\nInformation about trace used for postprocessing.\nprogress::ProgressMeter.Progress\nProgress Log while sampling.\n\n\n\n\n\n","category":"type"},{"location":"#Base.summary-Union{Tuple{S}, Tuple{Trace, Any, ModelWrappers.ModelWrapper}, Tuple{Trace, Any, ModelWrappers.ModelWrapper, Any}, Tuple{Trace, Any, ModelWrappers.ModelWrapper, Any, Integer}, Tuple{Trace, Any, ModelWrappers.ModelWrapper, Any, Integer, Integer}, Tuple{Trace, Any, ModelWrappers.ModelWrapper, Any, Integer, Integer, PrintDefault}} where S<:Union{Symbol, Tuple{Vararg{Symbol, k}} where k}","page":"Home","title":"Base.summary","text":"summary(trace, algorithmᵛ, model)\nsummary(trace, algorithmᵛ, model, sym)\nsummary(trace, algorithmᵛ, model, sym, burnin)\nsummary(trace, algorithmᵛ, model, sym, burnin, thinning)\nsummary(trace, algorithmᵛ, model, sym, burnin, thinning, printdefault)\n\n\nPrint summary of parameter chain and diagnostics to REPL.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.chainparamdiagnostics-Union{Tuple{F}, Tuple{T}, Tuple{Array{T, 3}, F}} where {T<:Real, F<:Real}","page":"Home","title":"Baytes.chainparamdiagnostics","text":"chainparamdiagnostics(arr, computingtime)\n\n\nCompute cross-chain diagnostics for parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.chainsummary-Union{Tuple{S}, Tuple{Trace, ModelWrappers.ModelWrapper, S, Any, Integer, Integer}, Tuple{Trace, ModelWrappers.ModelWrapper, S, Any, Integer, Integer, PrintDefault}} where S<:Union{Symbol, Tuple{Vararg{Symbol, k}} where k}","page":"Home","title":"Baytes.chainsummary","text":"chainsummary(trace, model, sym, backend, burnin, thinning)\nchainsummary(trace, model, sym, backend, burnin, thinning, printdefault; kwargs...)\n\n\nReturn summary for trace parameter chains. Model defines flattening type of parameter, sym defines parameter to be flattened, backend may be Val(:text), or Val(:latex).\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.construct-Tuple{IterationTempering, ModelWrappers.ModelWrapper, Integer}","page":"Home","title":"Baytes.construct","text":"construct(tempering, model, chains)\n\n\nConstruct a Tempering tuning struct with the appropriate type for model parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.construct-Tuple{ModelWrappers.ModelWrapper, DataTune, Integer, Vararg{Any}}","page":"Home","title":"Baytes.construct","text":"construct(model, datatune, chains, args)\n\n\nConstruct separate models and data tuning container for each chain.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.construct-Union{Tuple{D}, Tuple{Random.AbstractRNG, ModelWrappers.ModelWrapper, D, SampleDefault, TemperingMethod, DataTune, Integer, Vararg{Any}}} where D","page":"Home","title":"Baytes.construct","text":"construct(_rng, model, data, default, tempering, datatune, chains, args)\n\n\nConstruct chains MCMC chains and initiate all algorithms stated in args separately. A separate model for each chain is provided to avoid pointer issues. If args is a single SMC algorithm, chains will not be separately allocated but instead used within the algorithm. Note that smc still works as intended if used alongside other mcmc sampler in args.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.diagnosticsbuffer-Union{Tuple{D}, Tuple{D, Integer, Integer, Vararg{Any}}} where D","page":"Home","title":"Baytes.diagnosticsbuffer","text":"diagnosticsbuffer(diagtypes, iterations, Nchains, args)\n\n\nReturn buffer vector of type diagtypes. Comes from function infer.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.diagnosticssummary","page":"Home","title":"Baytes.diagnosticssummary","text":"diagnosticssummary(trace, algorithmᵛ, backend, burnin, thinning)\ndiagnosticssummary(trace, algorithmᵛ, backend, burnin, thinning, printdefault; kwargs...)\n\n\nReturn summary for trace parameter diagnostics, backend may be Val(:text), or Val(:latex).\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#Baytes.get_chaindiagnostics-Tuple{Trace, Integer, Integer, Integer, Integer}","page":"Home","title":"Baytes.get_chaindiagnostics","text":"get_chaindiagnostics(trace, chain, Nalgorithm, burnin, thinning)\n\n\nObtain parameter diagnostics from trace at chain chain, excluding first burnin samples.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.maxiterations-Tuple{DataTune, Integer}","page":"Home","title":"Baytes.maxiterations","text":"maxiterations(datatune, iterations)\n\n\nObtain maximum number of iterations based on datatune. This will always be user input, except if rolling/expanding data is used, in which case iterations are capped.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.mergediagnostics-Tuple{Any, Any, Any}","page":"Home","title":"Baytes.mergediagnostics","text":"mergediagnostics(paramdiagnostic, chainparamdiagnostic, paramquantiles)\n\n\nMerge all statistics for all parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.paramdiagnostics-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T<:Real","page":"Home","title":"Baytes.paramdiagnostics","text":"paramdiagnostics(vec)\n\n\nCompute in-chain diagnostics for parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.paramquantiles-Union{Tuple{T}, Tuple{AbstractVector{T}, PrintDefault}} where T<:Real","page":"Home","title":"Baytes.paramquantiles","text":"paramquantiles(vec, printdefault)\n\n\nCompute quantiles provided by printdefault for parameter.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.printedparam-Tuple{DataTune, Any, Vararg{Any}}","page":"Home","title":"Baytes.printedparam","text":"printedparam(datatune, sym, algorithm)\n\n\nObtain all parameter where output diagnostics can be printed. Separate step to showparam in case parameter is increasing over time and cannot be printed\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.progress-Tuple{ProgressReport, SamplingInfo}","page":"Home","title":"Baytes.progress","text":"progress(report, info)\n\n\nReturn Progress struct with arguments from info for sampling session.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.savetrace-Tuple{Trace, ModelWrappers.ModelWrapper, Any}","page":"Home","title":"Baytes.savetrace","text":"savetrace(trace, model, algorithm)\n\n\nSafe trace, model and algorithm to current working directory.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.showparam-Tuple{ModelWrappers.ModelWrapper, DataTune, Vararg{Any}}","page":"Home","title":"Baytes.showparam","text":"showparam(model, datatune, constructor)\n\n\nReturn all unique targetted parameter, and parameter where diagnostics will be printed. Separate to get_sym.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.trace_to_3DArray-Tuple{Trace, ModelWrappers.ModelWrapper, ModelWrappers.Tagged, Integer, Integer}","page":"Home","title":"Baytes.trace_to_3DArray","text":"trace_to_3DArray(trace, model, tagged, burnin, thinning)\n\n\nChange trace.val to 3d Array that is consistent with MCMCCHains dimensons. First dimension is iterations, second number of parameter, third number of chains.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.trace_to_posteriormean-Tuple{AbstractArray, ModelWrappers.ModelWrapper, ModelWrappers.Tagged, Integer, Integer}","page":"Home","title":"Baytes.trace_to_posteriormean","text":"trace_to_posteriormean(mod_array, model, tagged, burnin, thinning)\n\n\nChange trace.val to 3d Array and return Posterior mean as NamedTuple and as Vector\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#Baytes.update-Union{Tuple{B}, Tuple{DataTune, B, SMCConstructor}} where B<:UpdateBool","page":"Home","title":"Baytes.update","text":"update(datatune, temperingadaption, smc)\n\n\nCheck if we can capture results from last proposal step. Typically only possible if a single MCMC or SMC step applied.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Type{BaytesCore.AbstractDiagnostics}, AbstractVector, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, diagnostics, algorithmsᵛ, model, data)\n\n\nInfer types of all Diagnostics container.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{D}, Tuple{M}, Tuple{Random.AbstractRNG, Trace{<:TraceInfo{<:IterationTempering}}, AbstractVector, Vector{M}, D}} where {M<:ModelWrappers.ModelWrapper, D}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, trace, algorithmᵛ, modelᵛ, data)\n\n\nPropose new parameter for each model in modelᵛ given data with algorithms algorithmᵛ. A separate model for each chain is provided to avoid pointer issues. If args is a single SMC algorithm, chains will not be separately allocated but instead used within the algorithm. Note that smc still works as intended if used alongside other mcmc sampler in args.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Tuple{IterationTempering, Trace, Any, Integer}","page":"Home","title":"BaytesCore.update!","text":"update!(tempertune, trace, algorithm, iter)\n\n\nUpdate tempering tune based on output diagnostics of last kernels in each chain.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.sample!-Union{Tuple{D}, Tuple{M}, Tuple{Integer, Random.AbstractRNG, M, D, Trace, Any}} where {M<:ModelWrappers.ModelWrapper, D}","page":"Home","title":"StatsBase.sample!","text":"sample!(iterations, _rng, model, data, trace, algorithmᵛ)\n\n\nContinue sampling with all algorithms in algorithm.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#StatsBase.sample-Union{Tuple{D}, Tuple{S}, Tuple{M}, Tuple{Random.AbstractRNG, M, D, Vararg{Any}}} where {M<:ModelWrappers.ModelWrapper, S<:SampleDefault, D}","page":"Home","title":"StatsBase.sample","text":"sample(_rng, model, data, args; default)\n\n\nSample model parameter given data with args algorithm. Default sampling arguments given in default keyword.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"}]
}

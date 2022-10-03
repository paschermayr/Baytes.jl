"Bayesian inference on state space models"
module Baytes

################################################################################
#Import modules
import BaytesCore: BaytesCore, update!, infer, results, init, init!, propose, propose!, propagate!
using BaytesCore:
    BaytesCore,
    AbstractAlgorithm,
    AbstractTune,
    AbstractConfiguration,
    AbstractDiagnostics,
    AbstractConstructor,
    Updater,
    Iterator,
    subset,
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    DataTune,
    ArrayConfig,
    DataStructure,
    Expanding,
    Rolling,
    Batch,
    SubSampled,
    adjust,
    generate_showvalues,
    TemperingMethod,
    IterationTempering,
    JointTempering,
    ValueHolder,
    PrintDefault,
    ProgressLog,
    SilentLog,
    ConsoleLog,
    ProgressReport,
    SampleDefault,
    ProposalTune

import ModelWrappers:
    ModelWrappers,
    sample,
    sample!

using ModelWrappers:
    ModelWrappers,
    ModelWrapper,
    Tagged,
    Objective,
#=    DiffObjective,
    AbstractDifferentiableTune,
    ℓObjectiveResult,
    ℓDensityResult,
    ℓGradientResult,
=#
    flatten,
    unconstrain,
    paramnames,
    FlattenDefault,
    FlattenTypes,
    FlattenAll,
    FlattenContinuous,
    UnflattenTypes,
    UnflattenStrict,
    UnflattenFlexible

using BaytesMCMC, BaytesFilters, BaytesPMCMC, BaytesSMC

#Utility tools
import Base: Base, summary
using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack, @pack!
using Random: Random, AbstractRNG, GLOBAL_RNG, randexp
using ProgressMeter: ProgressMeter, @showprogress, Progress, next!
# Post-Processing
using MCMCDiagnosticTools: MCMCDiagnosticTools, ess_rhat, mcse, gelmandiag
using Statistics: Statistics, mean, std, sqrt, quantile, var
using Dates: Dates, today, hour, minute, now
using JLD2: JLD2, jldsave
using PrettyTables: PrettyTables, pretty_table

################################################################################
#Abstract types to be dispatched in Examples section
include("sampling/Sampling.jl")

################################################################################
export
    update!,
    init,
    init!,
    sample,
    sample!,
    propose,
    propose!,
    propagate!,

    ## ModelWrappers ~ For now explicitly import if used with Baytes.jl
#    dynamics,
#    predict,
#    generate,

    #Re-export relevant types/structs from other modules
    ## BaytesCore
    Updater,
    Iterator,
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    DataTune,
    ArrayConfig,
    DataStructure,
    Expanding,
    Rolling,
    Batch,
    SubSampled,
    TemperingMethod,
    IterationTempering,
    JointTempering,
    ValueHolder,
    PrintDefault,
    ProgressLog,
    SilentLog,
    ConsoleLog,
    ProgressReport,
    SampleDefault,

    ## ModelWrappers
    flatten,
    FlattenDefault,
    FlattenTypes,
    FlattenAll,
    FlattenContinuous,
    UnflattenTypes,
    UnflattenStrict,
    UnflattenFlexible,

    ## BaytesFilters
    ParticleFilter,
    ParticleFilterDefault,
    ParticleFilterConstructor,
    ParticleFilterMemory,

    Markov,
    SemiMarkov,
    SemiMarkovInitiation,
    SemiMarkovTransition,
    Bootstrap,
    Systematic,
    Stratified,
    Residual,
    Multinomial,
    Ancestral,
    Marginal,
    Conditional,

    ## BaytesMCMC
    MCMC,
    MCMCDefault,
    MCMCConstructor,
    NUTS,
    ConfigNUTS,
    HMC,
    ConfigHMC,
    MALA,
    ConfigMALA,
    Metropolis,
    ConfigMetropolis,
    Custom,
    ConfigCustom,

    ConfigStepnumber,
    StepNumberTune,
    KineticEnergy,
    ConfigTuningWindow,
    ConfigStepsize,
    ConfigProposal,
    MatrixMetric,
    MDense,
    MDiagonal,
    MUnit,
    Warmup,
    Adaptionˢˡᵒʷ,
    Adaptionᶠᵃˢᵗ,
    Exploration,

    ## BaytesPMCMC
    PMCMC,
    PMCMCDefault,
    PMCMCConstructor,
    ParticleMetropolis,
    ParticleGibbs,

    ## BaytesSMC
    SMC,
    SMCDefault,
    SMCConstructor,

    SMC2,
    SMC2Constructor,
    SMC2Kernel,
    SMCweight,
    SMCreweight

end

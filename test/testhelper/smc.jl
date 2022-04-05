######################################## Model 2 ~ PMCMC/SMC2
# Parameter
μ = [-2., 2.]
σ = [1., 1.]
p = [.05, .95]
# Latent data
latent = rand(_rng, Categorical(p), N)
data = [rand(_rng, Normal(μ[iter], σ[iter])) for iter in latent]

# Create ModelWrapper struct, assuming we do not know latent
latent_init = rand(_rng, Categorical(p), N_SMC2)
data_init = data[1:N_SMC2]

myparameter = (;
    μ = Param(μ, [Normal(-2., 5), Normal(2., 5)]),
    σ = Param(σ, [Gamma(2.,2.), Gamma(2.,2.)]),
    p = Param(p, Dirichlet(2, 2)),
    latent = Param(latent_init, [Categorical(p) for _ in Base.OneTo(N_SMC2)]),
)
mymodel = ModelWrapper(myparameter)
myobjective = Objective(mymodel, data_init)

# Assign an objective for both a particle filter and an mcmc kernel:
myobjective_pf = Objective(mymodel, data_init, :latent)
myobjective_mcmc = Objective(mymodel, data_init, (:μ, :σ, :p))

@test length(myobjective_mcmc.model.val.latent) == N_SMC2

# Assign Model dynamics
import ModelWrappers: dynamics
function ModelWrappers.dynamics(objective::Objective{<:ModelWrapper{BaseModel}})
    @unpack model, data = objective
    @unpack μ, σ, p = model.val

    initial_latent = Categorical(p)
    transition_latent(particles, iter) = initial_latent
    transition_data(particles, iter) = Normal(μ[particles[iter]], σ[particles[iter]])

    return Markov(initial_latent, transition_latent, transition_data)
end
dynamics(myobjective)

# Assign log target
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
    @unpack model, data, tagged = objective
    @unpack μ, σ, p, latent = θ
## Prior -> a faster shortcut without initializing the priors again
    lprior = log_prior(tagged.info.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    dynamicsᵉ = [Normal(μ[iter], σ[iter]) for iter in eachindex(μ)]
    dynamicsˢ = Categorical(p)
    ll = 0.0
#FOR PMCMC ~ target p(θ ∣ latent_1:t, data_1:t)
    for iter in eachindex(data)
        ll += logpdf(dynamicsᵉ[latent[iter]], data[iter])
        ll += logpdf(dynamicsˢ, latent[iter] )
    end
#=
# FOR MCMC ~ target p(θ ∣ data_1:t) by integrating out latent_1:t
    for time in eachindex(data)
        ll += logsumexp(logpdf(dynamicsˢ, iter) + logpdf(dynamicsᵉ[iter], grab(data, time)) for iter in eachindex(dynamicsᵉ))
    end
=#
    return ll + lprior
end

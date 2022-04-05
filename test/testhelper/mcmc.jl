######################################## Model 1
struct MyBaseModel <: ModelName end
myparameter1 = (μ = Param(0.0, Normal()), σ = Param(10.0, Gamma()))
mymodel1 = ModelWrapper(MyBaseModel(), myparameter1)

data_uv = rand(_rng, Normal(mymodel1.val.μ, mymodel1.val.σ), N)
#Create objective for both μ and σ and define a target function for it
myobjective1 = Objective(mymodel1, data_uv, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{MyBaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
myobjective1(myobjective1.model.val)

function ModelWrappers.generate(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MyBaseModel}})
    @unpack model, data = objective
    @unpack μ, σ = model.val
    return Float16(μ[1])
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MyBaseModel}})
    @unpack model, data = objective
    @unpack μ, σ = model.val
	return rand(_rng, Normal(μ, σ))
end

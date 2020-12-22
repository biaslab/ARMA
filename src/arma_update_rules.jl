import LinearAlgebra: I, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType
include("util.jl")

export ruleVariationalARMAOutNPPPP,
       ruleVariationalARMAIn1PNPPP,
       ruleVariationalARMAIn2PPNPP,
       ruleVariationalARMAIn3PPPNP,
	   ruleVariationalARMAIn4PPPPN


function ruleVariationalARMAOutNPPPP(marg_y :: Nothing,
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_z :: ProbabilityDistribution{Multivariate},
									 marg_r :: ProbabilityDistribution{Multivariate},
									 marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	mθ = unsafeMean(marg_θ)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	mτ = unsafeMean(marg_τ)

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=mθ'*[mz; mr], w=mτ)
end

function ruleVariationalARMAIn1PNPPP(marg_y :: ProbabilityDistribution{Univariate},
								     marg_θ :: Nothing,
								     marg_z :: ProbabilityDistribution{Multivariate},
									 marg_r :: ProbabilityDistribution{Multivariate},
									 marg_τ :: ProbabilityDistribution{Univariate})

    # Extract moments of beliefs
	my = unsafeMean(marg_y)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	mτ = unsafeMean(marg_τ)

	# Parameters
	Φ = [mz; mr]*[mz; mr]'
	ϕ = [mz; mr]*my

	# Set message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalARMAIn2PPNPP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_z :: Nothing,
									 marg_r :: ProbabilityDistribution{Multivariate},
									 marg_τ :: ProbabilityDistribution{Univariate})
	
	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalARMAIn3PPPNP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_z :: ProbabilityDistribution{Multivariate},
									 marg_r :: Nothing,
									 marg_τ :: ProbabilityDistribution{Univariate})

	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalARMAIn4PPPPN(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
								     marg_z :: ProbabilityDistribution{Multivariate},
									 marg_r :: ProbabilityDistribution{Multivariate},
  									 marg_τ :: Nothing)

	# Extract moments of beliefs
	my = unsafeMean(marg_y)
	mθ = unsafeMean(marg_θ)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	Vθ = unsafeCov(marg_θ)

	# Parameters
	a = 3. / 2.
	b = ([mz; mr]'*(mθ*mθ' + Vθ)*[mz; mr] -2*mθ'*[mz; mr]*my + my^2) / 2.

	# Set message
	return Message(Univariate, Gamma, a=a, b=b)
end


using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov
export AutoRegressiveMovingAverage, ARMA

"""
Description:

    An AutoRegressive model with Moving Average (ARMA)

    y_k = θ'⋅[y_k-1, …, y_k-M1, e_k-1, … e_k-M2] + e_k

    where M1 is the number of previous observations and M2 the number of previous residuals.
    These histories are stored as the following vectors:
    - z_k-1 = [y_k-1, …, y_k-M1]'
    - r_k-1 = [e_k-1, …, e_k-M2]'.

    Assume y_k, z_k-1 and r_k-1 are observed and e_k ~ N(0, τ^-1).

Interfaces:

    1. y (output)
    2. θ (function coefficients)
    3. z (previous observations vector)
    6. r (previous residuals)
    7. τ (precision)

Construction:

    AutoRegressiveMovingAverage(y, θ, z, r, τ, id=:some_id)
"""

mutable struct AutoRegressiveMovingAverage <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function AutoRegressiveMovingAverage(y, θ, z, r, τ; id=generateId(AutoRegressiveMovingAverage))
        @ensureVariables(y, θ, z, r, τ)
        self = new(id, Array{Interface}(undef, 5), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)
        self.i[:r] = self.interfaces[4] = associate!(Interface(self), r)
        self.i[:τ] = self.interfaces[5] = associate!(Interface(self), τ)
        return self
    end
end

slug(::Type{AutoRegressiveMovingAverage}) = "ARMA"

function averageEnergy(::Type{AutoRegressiveMovingAverage},
                       marg_y::ProbabilityDistribution{Univariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_z::ProbabilityDistribution{Multivariate},
                       marg_r::ProbabilityDistribution{Multivariate},
                       marg_τ::ProbabilityDistribution{Univariate})

        my = unsafeMean(marg_y)
        mz = unsafeMean(marg_z)
        mr = unsafeMean(marg_r)
        x = [mz; mr]

        mθ = unsafeMean(marg_θ)
        Vθ = unsafeCov(marg_θ)
        mτ = unsafeMean(marg_τ)

        aτ = marg_τ.params[:a]
        bτ = marg_τ.params[:b]

        temp1 = -(1. / 2.)*digamma(aτ).-(1. / 2.)*log(bτ).+(1. / 2.)*log(2*pi) # 0.5 log(2 pi) ~= 0.4
        temp2 = (1. / 2.).* mτ.*(my.^2 .-2 .*my.*mθ'*x.+x'*(Vθ.+mθ*mθ')*x)
        return (temp1.+temp2)[1]
end

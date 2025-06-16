module VBInferSimplified

export vb_infer, unpack_params, create_theta0, safe_predict

using LinearAlgebra
using ForwardDiff

# Canonical hemodynamic parameters
const τs_canon = 1.0
const τf_canon = 2.0
const τ0_canon = 1.0
const E0_canon = 0.34
const α_canon  = 0.33

function create_theta0(n::Int, C::Vector{Float64}, B::Matrix{Float64}, A::Matrix{Float64}; use_B::Bool=true)
    if use_B
        return vcat(vec(A), vec(B), C)
    else
        return vcat(vec(A), C)
    end
end

function unpack_params(θ::AbstractVector, n::Int, use_B::Bool)
    offset = 0
    A = reshape(θ[offset+1:offset+n^2], n, n); offset += n^2
    B = use_B ? reshape(θ[offset+1:offset+n^2], n, n) : zeros(n, n); offset += use_B ? n^2 : 0
    C = θ[offset+1:offset+n]; offset += n
    return A, B, C
end

function safe_predict(predict_func, params, t)
    try
        return predict_func(params, t)
    catch e
        @warn "simulate_dcm failed. Returning zero output." exception = e
        return zeros(params.n_regions, length(t))
    end
end

function vb_infer(θ0::Vector{Float64}, y_obs::AbstractMatrix, t::Vector{Float64},
                  n::Int, u_func::Function, use_B::Bool;
                  predict_func::Function, prior_var::Vector{Float64})

    μ = copy(θ0)
    Σ = Matrix{Float64}(I, length(θ0), length(θ0)) * 1e-3  # initial posterior covariance

    function free_energy(θ_)
        try
            A, B, C = unpack_params(θ_, n, use_B)
            params = (; A, B, C,
                      τs=fill(τs_canon, n),
                      τf=fill(τf_canon, n),
                      τ0=fill(τ0_canon, n),
                      E0=fill(E0_canon, n),
                      α=fill(α_canon, n),
                      e₀=fill(E0_canon, n),
                      n_regions=n,
                      u=u_func)
            y_pred = safe_predict(predict_func, params, t)
            y_pred === nothing && return -1e12

            err = vec(y_obs) - vec(y_pred)
            ll = -0.5 * sum(err.^2) / 0.01
            lp = -0.5 * sum(((θ_ .- θ0).^2) ./ prior_var)
            return ll + lp
        catch e
            @warn "Exception during free energy computation: $e"
            return -1e12
        end
    end

    for iter in 1:5
        g = ForwardDiff.gradient(free_energy, μ)
        H = ForwardDiff.hessian(free_energy, μ)
        H = -H + I * 10.0  # make Hessian positive definite
        Σ = diagm(0 => 1.0 ./ (diag(H) .+ 10.0))  # diagonal approx
        μ -= Σ * g  # update
    end

    return μ, free_energy(μ), Σ
end

end

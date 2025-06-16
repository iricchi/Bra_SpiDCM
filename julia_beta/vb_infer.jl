
module VBInfer
export vb_infer, unpack_params, create_theta0

using LinearAlgebra
using FiniteDiff

function create_theta0(n::Int, C::Vector{Float64}, B::Matrix{Float64},A::Matrix{Float64})
    #A = zeros(n, n)
    #A = -0.5I(n)  # enforce negative self-connections
    τ = fill(0.98, n)
    α = fill(0.33, n)
    return vcat(vec(A), vec(B), C, τ, α)
end

function unpack_params(θ, n, use_B)
    offset = 0
    A = reshape(θ[offset+1:offset+n^2], n, n); offset += n^2
    B = use_B ? reshape(θ[offset+1:offset+n^2], n, n) : zeros(n, n); offset += use_B ? n^2 : 0
    C = θ[offset+1:offset+n]; offset += n
    τ = θ[offset+1:offset+n]; offset += n
    α = θ[offset+1:offset+n]
    return A, B, C, τ, α
end

function vb_infer(θ0, y_obs, t, n, u_func, use_B; predict_func, prior_var)
    function free_energy(θ)
        A, B, C, τ, α = unpack_params(θ, n, use_B)
        params = (; A=A, B=B, C=C, τ=τ, α=α, e₀=fill(0.34, n), n_regions=n, u=u_func)
        y_pred = predict_func(params, t)
        #@show size(y_obs)
        #@show size(y_pred)
        err = vec(y_obs) - vec(y_pred)
        prior_mean = copy(θ0)  # assume θ0 encodes prior means (incl. self-inhibition)
        ll = -0.5 * sum(err.^2) / 0.01 # Gaussian with variance of 0.01 
        #lp = -0.5 * sum((θ .- 0).^2) / 0.1
        lp = -0.5 * sum((θ .- prior_mean).^2) ./ prior_var # include non-negative diagonal

        return ll + lp
    end

    θ = copy(θ0)
    for _ in 1:8
        g = FiniteDiff.finite_difference_gradient(free_energy, θ)
        h = diagm(FiniteDiff.finite_difference_hessian(θ -> free_energy(θ), θ) |> diag)
        Σ = inv(-h + I * 10.0)
        θ -= Σ * g
    end

    
    return Vector{Float64}(θ), free_energy(θ) #θ, free_energy(θ)
end

end
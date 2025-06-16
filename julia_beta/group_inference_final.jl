module GroupInference

using StatsBase
using Combinatorics
using Statistics
using LinearAlgebra

export compare_models, run_peb, run_bmr_greedy, bmr_stability_analysis, compute_fe

function compare_models(f_dict)
    logFs = hcat([f_dict[m] for m in keys(f_dict)]...)
    rel_logF = logFs .- maximum(logFs, dims=2)
    probs = exp.(rel_logF)
    probs = probs ./ sum(probs, dims=2)
    return probs
end

# PEB with full covariance integration
function run_peb(θs::Matrix{Float64}, Σs::Vector{Matrix{Float64}})
    μ = mean(θs, dims=1)[:]
    Σ_avg = mean(Σs)  # crude average of covariances
    return μ, Σ_avg
end

function compute_fe(μ::Vector{Float64}, Σ::Matrix{Float64}; λ::Float64=1e-2)
    n = length(μ)
    penalty = λ * n
    variances = diag(Σ)
    return -0.5 * sum(log.(variances) .+ (μ.^2 ./ variances)) - penalty
end

function run_bmr_greedy(μ::Vector{Float64}, Σ::Matrix{Float64}, labels::Vector{String}; max_steps::Int=10, λ::Float64=1e-2)
    n = length(μ)
    active = trues(n)
    best_FE = compute_fe(μ[active], Σ[active, active]; λ=λ)

    println("Initial Free Energy: $best_FE")
    for step in 1:max_steps
        best_i = 0
        best_test_FE = best_FE
        for i in findall(active)
            test_mask = copy(active)
            test_mask[i] = false
            test_FE = compute_fe(μ[test_mask], Σ[test_mask, test_mask]; λ=λ)
            if test_FE > best_test_FE
                best_test_FE = test_FE
                best_i = i
            end
        end
        if best_i == 0
            println("No improvement at step $step. Stopping.")
            break
        else
            println("Step $step: Removing param $(labels[best_i]) → FE improved to $best_test_FE")
            active[best_i] = false
            best_FE = best_test_FE
        end
    end

    return active  # mask of retained parameters
end

function bmr_stability_analysis(μ::Vector{Float64}, Σ::Matrix{Float64}, labels::Vector{String};
                                 n_runs::Int = 20, max_steps::Int = 10, λ::Float64 = 1e-2)
    n = length(μ)
    retain_counts = zeros(Int, n)

    for r in 1:n_runs
        mask = run_bmr_greedy(copy(μ), copy(Σ), labels; max_steps=max_steps, λ=λ)
        retain_counts .+= mask
    end

    retain_freq = retain_counts ./ n_runs
    return retain_freq
end
end


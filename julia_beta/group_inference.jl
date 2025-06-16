module GroupInference

using StatsBase
using Combinatorics
using Statistics
using LinearAlgebra

export compare_models, run_peb, run_bmr, run_bmr_greedy

function compare_models(f_dict)
    logFs = hcat([f_dict[m] for m in keys(f_dict)]...)
    rel_logF = logFs .- maximum(logFs, dims=2)
    probs = exp.(rel_logF)
    probs = probs ./ sum(probs, dims=2)
    return probs
end

function run_peb(θs)
    μ = mean(θs, dims=1)
    σ² = var(θs, dims=1)
    return μ, σ²
end

function run_bmr(μ::Vector{Float64}, σ²::Vector{Float64}, labels::Vector{String}; max_subset_size::Int=6)
    n = length(μ)
    best_FE = -Inf
    best_mask = falses(n)

    for r in 1:max_subset_size
        for subset in combinations(1:n, r)
            m = μ[subset]
            v = σ²[subset]
            FE = -0.5 * sum(log .(v) .+ (m.^2 ./ v))
            if FE > best_FE
                best_FE = FE
                best_mask .= falses(n)
                best_mask[subset] .= true
            end
        end
    end
    return best_mask
end

function compute_fe(μ::Vector{Float64}, σ²::Vector{Float64}; λ::Float64=1e-2)
    penalty = λ * length(μ)
    return -0.5 * sum(log.(σ²) .+ (μ.^2 ./ σ²)) - penalty
end 

function run_bmr_greedy(μ::Vector{Float64}, σ²::Vector{Float64}, labels::Vector{String}; max_steps::Int=10, λ::Float64=1e-2)
    n = length(μ)
    active = trues(n)
    best_FE = compute_fe(μ[active], σ²[active]; λ=λ)

    println("Initial Free Energy: $best_FE")
    for step in 1:max_steps
        best_i = 0
        best_test_FE = best_FE
        for i in findall(active)
            test_mask = copy(active)
            test_mask[i] = false
            test_FE = compute_fe(μ[test_mask], σ²[test_mask]; λ=λ)
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

    return active  # equivalent to best_mask
end

function bmr_stability_analysis(μ::Vector{Float64}, σ²::Vector{Float64}, labels::Vector{String};
                                 n_runs::Int = 20, max_steps::Int = 10, λ::Float64 = 1e-2)
    n = length(μ)
    retain_counts = zeros(Int, n)

    for r in 1:n_runs
        # Optional: reintroduce slight noise or variation here if needed
        mask = run_bmr_greedy(copy(μ), copy(σ²), labels; max_steps=max_steps, λ=λ)
        retain_counts .+= mask
    end

    # Compute frequency of selection
    retain_freq = retain_counts ./ n_runs

    return retain_freq
end


end


using OrdinaryDiffEq
using LinearAlgebra

# Task input function
function u(t)
    return (t > 5.0 && t < 15.0) ? 1.0 : 0.0
end

# Core DCM ODE system
function dcm_ode!(du, ustate, p, t)
    n = p.n_regions
    A, B, C, τ, α, e₀ = p.A, p.B, p.C, p.τ, p.α, p.e₀

    x = view(ustate, 1:n)
    h = view(ustate, n+1:length(ustate))
    du_x = (A + B * u(t)) * x + C * u(t)

    for i in 1:n
        s, f, v, q = h[4(i-1)+1:4i]
        z = x[i]
        f = max(f, 1e-4)
        v = max(v, 1e-4)
        αi = max(α[i], 1e-2)
        τi = max(τ[i], 1e-3)
        e₀i = clamp(e₀[i], 1e-5, 0.99)

        ds = z - τi * s - f + 1
        df = s
        dv = (f - v^(1/αi)) / τi
        dq = (f * (1 - (1 - e₀i)^(1/f)) / e₀i - q * v^(1/αi - 1)) / τi

        du[n + 4(i-1)+1 : n + 4(i-1)+4] .= [ds, df, dv, dq]
    end

    du[1:n] .= du_x
end

# BOLD signal model
function bold_signal(sol, e₀, n)
    n_t = length(sol.t)
    y = zeros(n_t, n)
    for i in 1:n
        for (j, ustate) in enumerate(sol.u)
            s, f, v, q = ustate[n + 4(i-1)+1 : n + 4(i-1)+4]
            f = clamp(f, 1e-4, 5.0)
            v = clamp(v, 1e-4, 5.0)
            e = e₀[i]
            y[j, i] = 0.02 * (7e * (1 - q) + 2 * (1 - q / v) - 0.2 * (1 - v))
        end
    end
    return y
end

# Simulate DCM forward given parameters
function simulate_dcm(params, t; use_B=true)
    n = params.n_regions
    x0 = zeros(n + 4n)
    prob = ODEProblem(dcm_ode!, x0, (t[1], t[end]), params)
    sol = solve(prob, Tsit5(), saveat=t, reltol=1e-4, abstol=1e-6)
    return bold_signal(sol, params.e₀, n)
end

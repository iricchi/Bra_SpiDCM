using DSP

module DCMRealData

export make_input_function, simulate_dcm

function make_input_function(task::Vector{Float64}, tvec::Vector{Float64})
    return t -> task[clamp(findlast(x -> x ≤ t, tvec), 1, length(task))]
end

# function simulate_dcm(params, t)
#     T = length(t)
#     n = params.n_regions
#     y = zeros(T, n)
#     for i in 1:n
#         y[:, i] .= sin.(0.1 .* t .+ 0.05 * randn())
#     end
#     return y
# end


function canonical_hrf(tr; duration=32.0)
    t = 0:tr:duration
    hrf = (t .^ 8.6) .* exp.(-t / 0.547) .- 0.35 * (t .^ 9) .* exp.(-t / 0.9)
    hrf ./= sum(hrf)  # normalize
    return hrf
end



function simulate_dcm(params, tvec)
    A, B, C = params.A, params.B, params.C
    u = params.u
    n = params.n_regions
    τ, α = params.τ, params.α
    #e₀ = params.e₀  # baseline oxygen extraction fraction

    dt = tvec[2] - tvec[1]
    T = length(tvec)

    # === Neural activity ===
    z = zeros(n)
    neural = zeros(T, n)
    for i in 1:T
        t = tvec[i]
        u_t = u(t)
        dz = A * z + B * z * u_t + C * u_t
        z += dt * dz
        neural[i, :] .= z
    end

    # === Hemodynamic parameters ===
    τs = fill(1.0, n)
    τf = fill(2.0, n)
    τ0 = fill(1.0, n)
    E0 = fill(0.34, n)
    V0 = 0.02
    k1 = 7.0 .* E0
    k2 = 2.0
    k3 = 2.0 .* E0 .- 0.2

    # === State initialization ===
    s = zeros(n)
    f = ones(n)
    v = ones(n)
    q = ones(n)

    y_bold = zeros(T, n)

    for i in 1:T
        sn = neural[i, :]

        ds = sn .- τs .* s .- τf .* (f .- 1.0)
        df = s

        # ensure positive values before exponentiation
        v_safe = clamp.(v, 1e-6, Inf)
        f_safe = clamp.(f, 1e-6, Inf)
        q_safe = clamp.(q, 1e-6, Inf)

        dv = (f .- v_safe .^ (1.0 ./ α)) ./ τ0
        dq = (f_safe .* (1 .- (1 .- E0) .^ (1.0 ./ f_safe)) ./ E0 .- q_safe .* v_safe .^ (1.0 ./ α .- 1)) ./ τ0

        # Euler integration
        s .+= dt .* ds
        f .+= dt .* df
        v .+= dt .* dv
        q .+= dt .* dq

        for r in 1:n
            y_bold[i, r] = V0 * (k1[r] * (1.0 - q[r]) + k2 * (1.0 - q[r]/v[r]) + k3[r] * (1.0 - v[r]))
        end
    end

    return y_bold
end




end


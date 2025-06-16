module DCMRealData
export make_input_function, simulate_dcm
using DSP

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
    dt = tvec[2] - tvec[1]
    T = length(tvec)

    # === Neural Activity ===
    z = zeros(n)
    neural = zeros(T, n)
    for i in 1:T
        t = tvec[i]
        u_t = u(t)
        dz = A * z + B * z * u_t + C * u_t
        z += dt * dz
        neural[i, :] .= z
    end

    # === Balloon-Windkessel Parameters ===
    τs = 1.0     # signal decay
    τf = 2.0     # autoregulation
    τ0 = 1.0     # mean transit time
    α = 0.32     # Grubb’s exponent
    E0 = 0.34    # resting oxygen extraction

    V0 = 0.02
    k1 = 7 * E0
    k2 = 2
    k3 = 2 * E0 - 0.2

    # === Initialize hemodynamic states ===
    s = zeros(n)
    f = ones(n)
    v = ones(n)
    q = ones(n)

    y_bold = zeros(T, n)

    for i in 1:T
        sn = neural[i, :]  # neural drive

        ds = sn .- τs .* s .- τf .* (f .- 1)
        df = s
        dv = (f .- v .^ (1 ./ α)) ./ τ0
        dq = (f .* (1 .- (1 .- E0) .^ (1 ./ f)) ./ E0 .- q .* v .^ (1 ./ α - 1)) ./ τ0

        s .+= dt .* ds
        f .+= dt .* df
        v .+= dt .* dv
        q .+= dt .* dq

        # Compute BOLD
        for r in 1:n
            y_bold[i, r] = V0 * (k1 * (1 - q[r]) + k2 * (1 - q[r]/v[r]) + k3 * (1 - v[r]))
        end
    end

    return y_bold
end



end


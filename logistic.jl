using Zygote

function l(w, x)
    return dot(w, x)
end
l(w::Vector{Float64}, x::Vector{Float64}) = sum(w .* x)
l(w::Float64, x::Float64) = w * x
l(w::Array{Float64, 2}, x::Vector{Float64}) = w * x
l(w::Array{Float64, 2}, x::Float64) = w * x
l(w::Array{Float64, 2}, x::Array{Float64}) = w * x'
l(w, x::Vector{Vector{Float64}}) = map(u -> l(w, u), x)

function ∇l_w(w, x)
    return x
end

function σ(u)
    return 1. / (1. + exp(-u))
end
σ(u::Float64) = 1. / (1. + exp(-u))
σ(u::Vector{Float64}) = 1. ./ (1. .+ exp.(-u))
σ(u::Array{Float64, 2}) = 1. ./ (1. .+ exp.(-u))

function ∇σ(u)
    return gradient(σ, u)
end
∇σ(u::Float64) = gradient(σ, u)[1]
∇σ(u::Vector{Float64}) = map(v -> ∇σ(v), u)

function p(w, b, x)
    return σ(l(w, x) .+ b)
end

function ∇p_w(w, b, x)
    return ∇σ(l(w, x) .+ b) .* ∇l_w(w, x)'
end

function ∇p_b(w, b, x)
    return ∇σ(l(w, x) .+ b)
end

function CE(y::Int64, p::Float64)
    return y * log(p) + (1 - y) * log(1 - p)
end
CE(y::Vector{Int64}, p::Vector{Float64}) = CE.(y, p)

function ∇CE_p(y::Int64, p::Float64)
    return y / p - (1 - y) / (1 - p)
end

function LL(y::Vector{Any}, p::Vector{Any})
    return sum(CE.(y, p))
end
function ∇LL_p(y, p)
    return ∇CE_p.(y, p)
end
LL(y::Int64, p::Float64) = CE(y, p)
LL(y, w, b, x) = LL(y, p(w, b, x))

∇LL_w(y, w, b, x) = sum(∇LL_p(y, p(w, b, x)) .* ∇p_w(w, b, x), dims=1)[1]
∇LL_b(y, w, b, x) = sum(∇LL_p(y, p(w, b, x)) .* ∇p_b(w, b, x), dims=1)[1]

loss(kwargs...) = -LL(kwargs...)
∇loss_w(kwargs...) = -∇LL_w(kwargs...)
∇loss_b(kwargs...) = -∇LL_b(kwargs...)

function step!(y, w, b, x, α=1e-3)
    w = w - α * ∇loss_w(y, w, b, x)
    b = b - α * ∇loss_b(y, w, b, x)
    return w, b
end

function network(Ws::Vector{Matrix{Float64}}, Bs::Vector{Vector{Float64}}, x)
    input = x
    ∇Ss, ∇Ws, ∇Bs = [], [], []
    for idx in 1:length(Ws)
        ∇s, ∇w, ∇b = ∇σ(p(Ws[idx], Bs[idx], input)), ∇p_w(Ws[idx], Bs[idx], input), ∇p_b(Ws[idx], Bs[idx], input)
        input = p(Ws[idx], Bs[idx], input)  
        ∇Ss = append!(∇Ss, [∇s])
        ∇Ws = append!(∇Ws, [∇w])
        ∇Bs = append!(∇Bs, [∇b])
    end
    return input, ∇Ss, ∇Ws, ∇Bs
end
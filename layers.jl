using Zygote

abstract type Layer end

mutable struct FullyConnected <: Layer
    W::Matrix{Float64}
    b::Vector{Float64}
    size::Tuple{Int64, Int64}
    activation::Function
    x::Vector{Any}
    forward::Vector{Float64}
    gradient::Vector{Float64}
    Wgradient::Matrix{Float64}
    bgradient::Vector{Float64}
end
FullyConnected(;size::Tuple{Int64, Int64}, activation::Function=ReLU) = FullyConnected(rand(size[1], size[2]), rand(size[1]), size, activation, [], [], [], rand(size[1], size[2]), rand(size[1]))

mutable struct Convolutional <: Layer
    W::Array{Float64, 3}
    b::Array{Float64, 3}
    kernel::Tuple{Int64, Int64}
    filters::Int64
    activation::Function
    x::Matrix{Any}
    forward::Array{Float64, 3}
    gradient::Array{Float64, 3}
    Wgradient::Array{Float64, 3}
    bgradient::Array{Float64, 3}
    padding::Bool
end
Convolutional(;kernel::Tuple{Int64, Int64}, filters::Int64, activation::Function=ReLU, padding::Bool=false) = Convolutional(rand(kernel..., filters), zeros(kernel..., filters), kernel, filters, activation, zeros(1, 1), zeros(1, 1, 1), zeros(1, 1, 1), zeros(1, 1, 1), zeros(1, 1, 1), padding)

mutable struct Flatten <: Layer
    x::Array{Float64, 3}
    forward::Vector{Float64}
    gradient::Bool
end
Flatten() = Flatten(zeros(1, 1, 1), [], false)

mutable struct Network
    layers::Vector{Layer}
    x
    forward::Vector{Any}
    backward::Vector{Any}
    depth::Int64
    LL::Function
    y::Int64
    loss::Float64
    gradloss::Float64
end
Network(layers, LL=crossentropy) = Network(layers, [], [], [], length(layers), LL, 0, 0., 0.)

function forward(x, network)
end
function forward(x::Vector{Float64}, layer::FullyConnected)
    if layer.size[2] != length(x)
        throw(DomainError("forward: Dimensions of x and layer do not match"))
    end
    if size(layer.W)[1] != length(layer.b)
        throw(DomainError("FullyConnected: Dimension mismatch between W and b"))
    end
    σ, W, b = layer.activation, layer.W, layer.b
    s = W * x + b
    ∇σ = map(v -> gradient(σ, v)[1], s)
    layer.x = x
    layer.forward = σ(s)
    layer.gradient = ∇σ
    return σ(W * x + b)
end

function forward(x::Matrix{Float64}, layer::Convolutional)
    if any(layer.kernel .> size(x))
        throw(DomainError("forward: input image smaller than kernel"))
    end
    if all(layer.b .== 0.)
        if layer.padding
            layer.b = rand(size(x)..., layer.filters)
        else
            layer.b = rand((size(x) .- 2)..., layer.filters)
        end
    end
    is = size(x)
    ks = layer.kernel
    if layer.padding
        out = zeros(size(x)..., layer.filters)
        padded = zeros(Float64, size(x) .+ 2)
        padded[2:size(x)[1] + 1, 2:size(x)[2] + 1] = x
        for kidx in 1:layer.filters
            kernel = layer.W[:, :, kidx]
            for i in 1:is[1]
                for j in 1:is[2]
                    subimage = padded[i:i + ks[1] - 1, j:j + ks[2] - 1]
                    entry = sum(subimage .* kernel)
                    out[i, j, kidx] = entry
                end
            end
        end
    else
        out = zeros((size(x) .- 2)..., layer.filters)
        for kidx in 1:layer.filters
            kernel = layer.W[:, :, kidx]
            for i in 1:is[1] - 2
                for j in 1:is[2] - 2
                    subimage = x[i:i + ks[1] - 1, j:j + ks[2] - 1]
                    entry = sum(subimage .* kernel)
                    out[i, j, kidx] = entry
                end
            end
        end
    end
    σ, W, b = layer.activation, layer.W, layer.b
    s = out .+ b
    ∇σ = map(v -> gradient(σ, v)[1], s)
    layer.x = x
    layer.forward = σ(s)
    layer.gradient = ∇σ
    return σ(s)
end

function forward(x, layer::Flatten)
    layer.x = x
    y = reshape(x, prod(size(x)))
    layer.forward = y
    return y
end

function forward(x::Vector{Float64}, net::Network)
    net.x = x
    net.forward = []
    for layer in net.layers
        x = forward(x, layer)
        net.forward = append!(net.forward, [x])
    end
    return x
end
function forward(x::Matrix{Float64}, net::Network)
    net.x = x
    net.forward = []
    for layer in net.layers
        x = forward(x, layer)
        net.forward = append!(net.forward, [x])
    end
    return x
end
function forward(X::Vector{Any}, net::Network)
    Y = []
    for x in X
        y = forward(x, net)
        Y = append!(Y, y)
    end
    return Y
end


function backward(y, network)
end
function backward(y::Int64, net::Network)
    Wgradients = []
    bgradients = []
    net.y = y
    p = net.forward[net.depth]
    net.loss = -net.LL(y, p[1])
    tograd(u::Float64) = -net.LL(y, u)
    net.gradloss = gradient(tograd, p[1])[1]
    layer = net.layers[net.depth]
    W, b = layer.W, layer.b
    delta = net.gradloss .* layer.gradient
    layer.Wgradient = delta .* layer.x'
    layer.bgradient = delta
    Wgradients = append!(Wgradients, [layer.Wgradient])
    bgradients = append!(bgradients, [layer.bgradient])
    for step in 1:net.depth - 1
        layer = net.layers[net.depth - step]
        if typeof(layer.gradient) != Bool
            delta = (W' * delta) .* layer.gradient
            layer.Wgradient = delta .* layer.x'
            layer.bgradient = delta
            Wgradients = append!(Wgradients, [layer.Wgradient])
            bgradients = append!(bgradients, [layer.bgradient])
            W, b = layer.W, layer.b
        end
    end
    return net.loss, [Wgradients, bgradients]
end

function backward(Yt::Vector{Any}, Xt::Vector{Any}, net::Network, batch::Int64)
    nums = ceil.(rand(1:length(Xt), batch))
    X = Xt[nums]
    Y = Yt[nums]
    Wgradients = []
    bgradients = []
    losses = []
    for idx in 1:length(X)
        x, y = X[idx], Y[idx]
        forward(x, net)
        loss, grads = backward(y, net)
        losses = append!(losses, loss)
        Wgradients = append!(Wgradients, [grads[1]])
        bgradients = append!(bgradients, [grads[2]])
    end
    for step in 1:net.depth
        Wgradient = sum([Wgradients[i][step] for i in 1:length(X)])
        bgradient = sum([bgradients[i][step] for i in 1:length(X)])
        layer = net.layers[net.depth - step + 1]
        layer.Wgradient = Wgradient
        layer.bgradient = bgradient
    end
    return sum(losses)
end


function update(network, α)
end
function update(layer::Layer, α::Float64=1e-4)
    if length(layer.forward) == 0
        throw(RunTimeError("update: Need to do a forward and backward pass of network"))
    end
    layer.W = layer.W .- α * layer.Wgradient
    layer.b = layer.b .- α * layer.bgradient
    return layer
end

function update(net::Network, α::Float64=1e-4)
    for layer in net.layers
        update(layer, α)
    end
end

function train(network, X, Y, kwargs...)
end
function train(net::Network, X::Vector{Any}, Y::Vector{Any}, α::Float64=1e-4, batch::Int64=64, epochs=100)
    if length(X) != length(Y)
        throw(DomainError("train: X and Y must be the same length"))
    end
    losses = []
    for i in 1:epochs
        loss = backward(Y, X, net, batch)
        losses = append!(losses, loss)
        update(net, α)
    end
    return losses
end

function predict(network, X)
end
function predict(net::Network, X::Vector{Any})
    Y = []
    for x in X
        p = forward(x, net)
        if p[1] > 0.5
            Y = append!(Y, 1)
        else
            Y = append!(Y, 0)
        end
    end
    return Y
end

# Section: Loss Functions

function crossentropy(y::Int64, p::Float64)
    return y * log(p) + (1 - y) * log(1 - p)
end

function crossentropy(y::Int64, p::Vector{Float64})
    return y * log.(p) + (1 - y) * log.(1. .- p)
end


# Section: Activation Functions

function sigmoid(u)
    return 1. / (1. + exp.(-u))
end
sigmoid(u::Float64) = 1. / (1. + exp(-u))
sigmoid(u::Vector{Float64}) = 1. ./ (1. .+ exp.(-u))
sigmoid(u::Matrix{Float64}) = 1. ./ (1. .+ exp.(-u))
sigmoid(u::Array{Float64, 3}) = 1. ./ (1. .+ exp.(-u))

function ReLU(u)
    return max(0., u)
end
ReLU(u::Float64) = max(0., u)
ReLU(u::Vector{Float64}) = map(v -> max(0., v), u)
ReLU(u::Matrix{Float64}) = map(v -> max(0., v), u)
RelU(u::Array{Float64, 3}) = map(v -> max(0., v), u)

function LeakyReLU(u)
    α = 0.1
    return max(α * u, u)
end
LeakyReLU(u::Float64) = max(0.1 * u, u)
LeakyReLU(u::Vector{Float64}) = map(v -> max(0.1 * v, v), u)
LeakyReLU(u::Matrix{Float64}) = map(v -> max(0.1 * v, v), u)
LeakyRelU(u::Array{Float64, 3}) = map(v -> max(0.1 * v, v), u)

function softmax(h)
    return exp(-h) / sum(exp(-h))
end
softmax(h::Float64) = 1.
softmax(h::Vector{Float64}) = exp.(-h) / sum(exp.(-h))
softmax(h::Matrix{Float64}) = exp.(-h) / sum(exp.(-h))
softmax(h::Array{Float64, 3}) = exp.(-h) / sum(exp.(-h))

# Section: Utils

function color(y)
    if y == 1
        return :red
    else
        return :blue
    end
end

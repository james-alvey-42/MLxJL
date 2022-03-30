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
FullyConnected(;size::Tuple{Int64, Int64}, activation::Function=sigmoid) = FullyConnected(rand(size[1], size[2]), rand(size[1]), size, activation, [], [], [], rand(size[1], size[2]), rand(size[1]))

mutable struct Network
    layers::Vector{Layer}
    x::Vector{Any}
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

function forward(x::Vector{Float64}, net::Network)
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
        delta = (W' * delta) .* layer.gradient
        layer.Wgradient = delta .* layer.x'
        layer.bgradient = delta
        Wgradients = append!(Wgradients, [layer.Wgradient])
        bgradients = append!(bgradients, [layer.bgradient])
        W, b = layer.W, layer.b
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
    return 1. / (1. + exp(-u))
end
sigmoid(u::Float64) = 1. / (1. + exp(-u))
sigmoid(u::Vector{Float64}) = 1. ./ (1. .+ exp.(-u))
sigmoid(u::Matrix{Float64}) = 1. ./ (1. .+ exp.(-u))

function ReLU(u)
    return max(0., u)
end
ReLU(u::Float64) = max(0., u)
ReLU(u::Vector{Float64}) = map(v -> max(0., v), u)
ReLU(u::Matrix{Float64}) = map(v -> max(0., v), u)

# Section: Utils

function color(y)
    if y == 1
        return :red
    else
        return :blue
    end
end

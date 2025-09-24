rand([10,20])
rand([10,20])
rand([10,20,30])

A = rand(2,2)

A[:]

randn(2,2)

flip = rand() < 0.5

flips = zeros(Bool,100) #Where to store the flips
for i in 1:100
    flips[i] = rand() < 0.5 #check to see if the i th flip is heads
end
println(flips)

#same as
flips = rand(100) .< 0.5
a,b,c = rand(10),rand(10),rand(10)
@. a^2 + b^2 + c^2
a^2 + b^2 + c^2


function drawDiscrete(p)
    @assert sum(p) == 1
    S = length(p)
    r = rand()
    cump = cumsum(p)
    for s in 1:S
        if r <= cump[s]
            return s
        end
    end
end

drawDiscrete([0.1,0.2,0.3,0.2])

function drawNDiscrete(p,N)
    ret = zeros(Int,N)
    for i in 1:N
        ret[i] = drawDiscrete(p)
    end
    #same as
    ret =  [drawDiscrete(p) for _ in 1:N]
    return ret
end

drawNDiscrete([0.1,0.2,0.3,0.2],10)

[i^2 for i in 1:10]

function f(x)
    return x^2
end

function f(x,y)
    return x^2 + y^2
end

function g(x::Float64)
    return x^2
end

function g(x::Int)
    return 2x
end

g(3.)
g(3)

using Distributions

dist = LogNormal(0,2)
dist = Pareto(1,1)

rand(dist,10)

using Statistics

"""
    flipNcoins(N,p=0.5)

Flips N coins with probability p of returning 1 (heads)
"""
function flipNcoins(N,p=0.5)
    return rand(N) .< p
end
mean(flipNcoins(50))

using Plots
meantosses = [mean(flipNcoins(N,0.5)) for N in 1:1000]
scatter(1:1000,meantosses)


numheads = [sum(flipNcoins(15,0.5)) for k in 1:100_000]


p = 0.3
numheads = [sum(flipNcoins(15,p)) for k in 1:1000]
histogram(numheads,bins=25,normalize=:probability,xlabel="# of Heads",ylabel="Probability")
plot!(0:15,pdf.(Binomial(15,p),0:15))


using Random
Random.seed!(12345) #can put any integer here
rand()
rand()
rand()


Random.seed!(12345) #can put any integer here
rand()
rand()
rand()


function drawNPareto(α,N)
    return rand(Pareto(α,1),N)
end


function empiricalCDF_pareto(α,x,N=1000)
    X =drawNPareto(α,N)
    return mean(X.>=x)
end

function empiricalCDF_pareto(α,x,N=1000)
    X = drawNPareto(α,N)
    return mean(X .>= x)
end

α = 1
xvec = LinRange(1,10,500)
plot(xvec,empiricalCDF_pareto.(α,xvec,100_000),label="Empirical")
plot!(xvec,xvec.^(-α),label="Theoretical")


function empiricalCDF_pareto(x,N=1000)
    X =drawNPareto(α,N)
    return mean(X.>=x)
end

using Plots
"""
   simulateAR1(ρ,σ,T)

Simulates an AR(1) with mean μ, persistence ρ, and standard deviation σ for 
T periods with initial value x0
"""
function simulateAR1(μ,ρ,σ,x0,T)
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
    end
    return x[2:end]
end

plot(1:100,simulateAR1(10.,0.9,1.,0,100),xlabel="Time",ylabel="AR(1)")


mutable struct AR1
    μ::Float64 #Mean of the AR(1)
    ρ::Float64 #persistence of the AR(1)
    σ::Float64 #standard deviaiton of the AR(1)
    x0::Float64 #initial value of the AR(1)
end

ar1 = AR1(0.,0.8,1.,0.)
ar1alt = AR1(0.,0.1,1.,0.)

ar1.μ = 10.
ar1


"""
   simulateAR1(ar1,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar1,T)
    x = zeros(T+1)# initialize
    x[1] = ar1.x0
    for t in 1:T
        x[t+1] = (1-ar1.ρ)*ar1.μ + ar1.ρ*x[t] + ar1.σ*randn()
    end
    return x[2:end]
end
ar1.x0 = 10.
plot(1:100,simulateAR1(ar1,100),xlabel="Time",ylabel="AR(1)")


using Parameters
"""
   simulateAR1(ar1,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar1,T)
    @unpack x0,σ,μ,ρ = ar1 #note order doesn't matter
    #META PROGRAMMING REPLACES ABOVE WITH
    x0 = ar1.x0
    σ = ar1.σ
    μ = ar1.μ
    ρ = ar1.ρ
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
    end
    return x[2:end]
end

ar1.x0 = 10.
ar1.ρ  = 0.5
plot(1:100,simulateAR1(ar1,100),xlabel="Time",ylabel="AR(1)")

T = 50
N = 1000
X = zeros(T,N)
ar1.x0 = 2.
ar1.ρ = 0.9
for i in 1:N
    X[:,i] .= simulateAR1(ar1,T)
end

plot(1:T,mean(X,dims=2),ylabel="Mean",layout=(2,1),subplot=1)
plot!(1:T,std(X,dims=2),xlabel="Time",ylabel="Standard Deviation",subplot=2)


@with_kw mutable struct VAR1
    A::Matrix{Float64} = [0.9;;]
    C::Matrix{Float64} = [1.;;]
    X0::Vector{Float64} = [0.;]
end

var1 = VAR1()

function simulateVAR1(var1,T)
    @unpack A,C,X0 = var1
    N = length(X0)
    X = zeros(N,T+1)
    X[:,1] = X0
    for t in 1:T
        X[:,t+1] = A*X[:,t] + C*randn(N)
    end
    return X[:,2:end]
end

var1.A = [0.5 -0.3; 1. 0.]
var1.C = [1. 0; 0. 0.]
var1.X0 = [0.;0.]

simulateVAR1(var1,10)


using QuantEcon

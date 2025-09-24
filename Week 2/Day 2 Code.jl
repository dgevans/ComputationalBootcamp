using QuantEcon

P = [0.6 0.4;
     0.4 0.6]
s = simulate(MarkovChain(P),100,init=1)
println(s)

mc_ar1 = rouwenhorst(5,0.9999,0.014)

X = zeros(15,10000)
for i in 1:10000
    X[:,i] = simulate(mc_ar1,15,init=1)
end
println(mean(X[15,:]))

P,X̄ = mc_ar1.p,mc_ar1.state_values
println((P^14*X̄)[1])



using LinearAlgebra
P = mc_ar1.p
D,V = eigen(P')  #should be left unit eigenvector
πstar = V[:,isapprox.(D,1)][:]
πstar = V[:,end]
πstar ./= sum(πstar)#Need to normalize for probability

πstar'*P - πstar'

πstar2 = (P^2000000)[1,:]


using Parameters
@with_kw mutable struct RBCModel
    N::Int = 50 #number of grid points
    kgrid::Vector{Float64} = LinRange(0.01,0.5,N)
    A::Float64 = 1.
    α::Float64 = 0.33
    β::Float64 = 0.95
end
@unpack A,α,β = params
E_infty = (1-β)^(-1) * (log(A*(1-β*α))+
         (α*β)/(1-α*β)*log(A*β*α))
F_infty = α/(1-β*α)

params = RBCModel(N=500)

V = zeros(params.N)
"""
   RBCbellmanmap(params,V)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `params` parameters of the model
* `V` Vector of values for each capital level in kgrid
"""
function RBCbellmanmap(params,V)
    @unpack kgrid,A,α,β = params
    N = length(kgrid)
    V_out = zeros(N) #new value function
    n_pol = zeros(Int,N) #policy rule for grid points
    k_pol = zeros(N) #policy rule for capital
    obj = zeros(N) #objective to be maximized
    for n in 1:N #iterate for each initial capital
        for nprime in 1:N #iterate for choice of capital this period
            c = A*kgrid[n]^α - kgrid[nprime] #compute consumption
            if c <= 0
                obj[nprime] = -Inf #penalty if consumption <0
            else
                obj[nprime] = log(c)+β*V[nprime] #otherwise objective from RHS of bellman equation
            end
        end
        V_out[n],n_pol[n] = findmax(obj) #find optimal value and the choice that gives it
        k_pol[n] = kgrid[n_pol[n]] #record capital policy
    end
    return V_out,n_pol,k_pol
end

V_out,n_pol,k_pol = RBCbellmanmap(params,V)
println(V_out)
println(n_pol)
println(k_pol)


"""
    RBCsolve_bellman(params,V0,[,ϵ=1e-6])

Solves the bellman equation by iterating until convergence

# Arguments
* `params` parameters of the model
* `V0` Initial vector of values for each capital level in kgrid
* `ϵ` Convergence criteria
"""
function RBCsolve_bellman(params,V0;ϵ=1e-6)
    @unpack kgrid,A,α,β = params
    diff = 1.
    V,n_pol,k_pol = RBCbellmanmap(params,V0)
    while diff > ϵ
        V_new,n_pol,k_pol = RBCbellmanmap(params,V)
        diff = norm(V_new-V,Inf)
        V = V_new 
    end
    return V,n_pol,k_pol
end


V,n_pol,k_pol = RBCsolve_bellman(params,zeros(params.N))
println(V)
println(n_pol)
println(k_pol)

V,n_pol,k_pol =RBCsolve_bellman(params,zeros(params.N),ϵ=1e-8)

V,n_pol,k_pol = RBCsolve_bellman(params,zeros(N) )
using Plots
plot(k->E_infty+F_infty*log(k),.01,.5,color=:red,label="Analytical",legend=true)
scatter!(kgrid,V,label="Approximation",color=:lightblue,xlabel="Capital",ylabel="Value Function")

plot(k->(params.β*F_infty)/(1+params.β*F_infty)*params.A*k^params.α,.01,.5,color=:red,label="Analytical",legend=true)
scatter!(params.kgrid,k_pol,label="Approximation",color=:lightblue,xlabel="Capital",ylabel="Capital Next Period")


"""
    simulate_k(n_0,T,n_pol,kgrid)

Simulates the path of capital given policy rule n_pol and 
initial capital state kgrid[n_0] for T periods
"""
function simulate_kc(params,n_0,T,n_pol,kgrid)
    @unpack A,α,β = params
    k = zeros(T+1) # capital stock
    n = zeros(Int,T+1) #index of the capital stock
    c = zeros(T) #consumption
    n[1] = n_0
    k[1] = kgrid[n_0]
    for t in 1:T
        #enter with state k[t],n[t], exit with k[t+1],n[t+1],c[t]
        n[t+1] = n_pol[n[t]] #get the policy rule for the index
        k[t+1] = kgrid[n[t+1]]
        c[t] = A*k[t]^α - k[t+1] #compute consumption
    end

    return (k=k,c=c)
end

plot(simulate_kc(params,1,25,n_pol,kgrid).k,xlabel="Time",ylabel="Capital Stock")
plot(simulate_kc(params,1,25,n_pol,kgrid).c,xlabel="Time",ylabel="Consumption")


@unpack c,k = simulate_kc(params,1,25,n_pol,kgrid)




### with δ

@with_kw mutable struct RBCModelδ
    N::Int = 50 #number of grid points
    kgrid::Vector{Float64} = LinRange(0.01,0.5,N)
    A::Float64 = 1.
    α::Float64 = 0.33
    β::Float64 = 0.95
    δ::Float64 = 0.1
end

params = RBCModelδ(N=500)

V = zeros(params.N)
"""
   RBCbellmanmapδ(params,V)

Iterates on the bellman equation for the standard neoclassical growth model
with depreciation rate δ
# Arguments
* `params` parameters of the model
* `V` Vector of values for each capital level in kgrid
"""
function RBCbellmanmapδ(params,V)
    @unpack kgrid,A,α,β,δ = params
    N = length(kgrid)
    V_out = zeros(N) #new value function
    n_pol = zeros(Int,N) #policy rule for grid points
    k_pol = zeros(N) #policy rule for capital
    obj = zeros(N) #objective to be maximized
    for n in 1:N #iterate for each initial capital
        for nprime in 1:N #iterate for choice of capital this period
            c = (1-δ)*kgrid[n] + A*kgrid[n]^α - kgrid[nprime] #compute consumption
            if c <= 0
                obj[nprime] = -Inf #penalty if consumption <0
            else
                obj[nprime] = log(c)+β*V[nprime] #otherwise objective from RHS of bellman equation
            end
        end
        V_out[n],n_pol[n] = findmax(obj) #find optimal value and the choice that gives it
        k_pol[n] = kgrid[n_pol[n]] #record capital policy
    end
    return V_out,n_pol,k_pol
end

V_out,n_pol,k_pol = RBCbellmanmapδ(params,V)
println(V_out)
println(n_pol)
println(k_pol)


"""
    RBCsolve_bellman(params,V0,[,ϵ=1e-6])

Solves the bellman equation by iterating until convergence

# Arguments
* `params` parameters of the model
* `V0` Initial vector of values for each capital level in kgrid
* `ϵ` Convergence criteria
"""
function RBCsolve_bellmanδ(params,V0;ϵ=1e-6)
    @unpack kgrid,A,α,β = params
    diff = 1.
    V,n_pol,k_pol = RBCbellmanmapδ(params,V0)
    while diff > ϵ
        V_new,n_pol,k_pol = RBCbellmanmapδ(params,V)
        diff = norm(V_new-V,Inf)
        V = V_new 
    end
    return V,n_pol,k_pol
end

paramsδ = RBCModelδ(N=500,δ=1)
params = RBCModel(N=500)

Vδ,_,_ = RBCsolve_bellmanδ(paramsδ,zeros(paramsδ.N))
V,_,_ = RBCsolve_bellman(params,zeros(params.N))



"""
    RBCsolve_steadystate(params)

Solves the steady state of the model

# Arguments
* `params` parameters of the model
"""
function RBCsolve_steadystate(params)
    @unpack A, α, β = params
    δ = hasfield(typeof(params), :δ) ? getfield(params, :δ) : 1.0

    Rbar = 1/β
    kbar = ((Rbar - 1 + δ) / (α * A))^(1/(α - 1))
    cbar = A * kbar^α - δ * kbar

    @assert α*A*kbar^(α-1) + 1 - δ ≈ Rbar

    return cbar, kbar
end


#Stochastic state
@with_kw mutable struct RBCModelStochastic
    N::Int = 50 #number of grid points
    kgrid::Vector{Float64} = LinRange(0.01,0.5,N)
    A::Vector{Float64} = [0.97,1.03] #TFP values
    Π::Matrix{Float64} = [0.6 0.4;0.4 0.6] #Transition matrix
    α::Float64 = 0.33
    β::Float64 = 0.95
end
params = RBCModelStochastic(N=500)
"""
    RBCbellmanmap_stochastic(params,V)

Iterates on the bellman equation for the standard neoclassical growth model

# Arguments
* `params` parameters of the model
* `V` Vector of values for each capital level in kgrid
"""
function RBCbellmanmap_stochastic(params,V)
    @unpack kgrid,A,Π,α,β = params
    N = length(kgrid) #Number of gridpoints of capital
    S = length(A) #Number of stochastic states
    V_new = zeros(S,N) #New Value function
    n_pol = zeros(Int,S,N) #New policy rule for grid points
    k_pol = zeros(S,N) #New policy rule for capital
    obj = zeros(N) #objective to be maximized
    EV = Π*V #precompute expected value for speed
    for n in 1:N
        for s in 1:S
            for nprime in 1:N
                c = A[s]*kgrid[n]^α - kgrid[nprime] #compute consumption
                if c <= 0
                    obj[nprime] = -Inf #punish if c <=0
                else
                    obj[nprime] = log(c) + β*EV[s,nprime] #otherwise compute objective
                end
            end
            #find optimal value and policy
            V_new[s,n],n_pol[s,n] = findmax(obj)
            k_pol[s,n] = kgrid[n_pol[s,n]]
        end
    end
    return V_new,n_pol,k_pol
end;

"""
    RBCsolve_bellman_stochastic(params,V0,[,ϵ=1e-6])

Solves the bellman equation by iterating until convergence

# Arguments
* `params` parameters of the model
* `V0` Initial vector of values for each capital level in kgrid
* `ϵ` Convergence criteria
"""
function RBCsolve_bellman_stochastic(params,V0;ϵ=1e-6)
    diff = 1.
    V,n_pol,k_pol = RBCbellmanmap_stochastic(params,V0)
    while diff > ϵ
        V_new,n_pol,k_pol = RBCbellmanmap_stochastic(params,V)
        diff = norm(V_new-V,Inf)
        V = V_new
    end
    return V,n_pol,k_pol
end;

params = RBCModelStochastic(N=500)
V,n_pol,k_pol = RBCsolve_bellman_stochastic(params,zeros(2,params.N));


using QuantEcon
using DataFrames
"""
    simulate_k_stochastic(n_0,T,n_pol,kgrid,Π)

Simulates the path of capital given policy rule n_pol and 
initial capital state kgrid[n_0] and aggregate state s_0 for T periods
"""
function simulate_k_stochastic(params,n_0,s0,T,n_pol)
    @unpack kgrid,A,Π,α = params
    k = zeros(T+1) # capital stock
    n = zeros(Int,T+1) #index of the capital stock
    s = simulate_indices(MarkovChain(Π),T;init=s0)
    c = zeros(T)
    y = zeros(T)
    At = zeros(T)
    n[1] = n_0
    k[1] = kgrid[n_0]
    for t in 1:T
        n[t+1] = n_pol[s[t],n[t]] #get the policy rule for the index
        k[t+1] = kgrid[n[t+1]]
        c[t] = A[s[t]]*k[t]^α - k[t+1]
        y[t] = A[s[t]]*k[t]^α
        At[t] = A[s[t]]
    end

    df = DataFrame(time=1:T,k=k[1:T],c=c,y=y,At=At)

    return df
end

df = simulate_k_stochastic(params,100,1,1000,n_pol)
df = df[100:end,:]

plot(df.time,df.k,xlabel="Time",ylabel="Capital Stock")
plot(df.time,df.c,xlabel="Time",ylabel="Consumption")
plot(df.time,df.y,xlabel="Time",ylabel="Output")
plot(df.time,df.At,xlabel="Time",ylabel="Technology")
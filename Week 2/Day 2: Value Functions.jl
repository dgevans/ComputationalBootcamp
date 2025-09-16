using Plots,LinearAlgebra
default(linewidth=2,legend=false,margin=10Plots.mm)
A = 1
α = 0.33
β = 0.95
J = 200 #number of iterations

E = zeros(J)
F = zeros(J)
E[1] = log(A)
F[1] = α
for j in 2:J
    #apply formula for E
    E[j] = β*E[j-1] + log(A/(1+β*F[j-1]))+
           β*F[j-1]*log(β*F[j-1]/(1+β*F[j-1])*A)
    #apply formula for F
    F[j] = α + α*β*F[j-1]
end
E_infty = (1-β)^(-1) * (log(A*(1-β*α))+
         (α*β)/(1-α*β)*log(A*β*α))
F_infty = α/(1-β*α)

#Now plot value functions
plot( k->0 ,0.01,0.5,color=:lightblue,xlabel="Capital",ylabel="Value Function")
for j in 1:100
    #Add value function to plot for each iteration j in 1,2,..,100
    plot!(k->E[j]+F[j]*log(k),0.01,0.5,color=:lightblue)
end
#Add limiting value function
plot!(k->E_infty+F_infty*log(k),0.01,0.5,color=:red)

plot( k->0 ,0.01,0.5,color=:lightblue,xlabel="Capital",ylabel="Next Period Capital")
#NOTE only 10 iterations
for j in 1:10
    #Add policy rules to plot for each iteration j in 1,2,..,10
    plot!(k->(β*F[j])/(1+β*F[j])*A*k^α,0.01,0.5,color=:lightblue)
end
#Add limiting value function, note order = 1 puts it on top
plot!(k->(β*F_infty)/(1+β*F_infty)*A*k^α,0.01,0.5,color=:red)
#add line representing kprime = k
plot!(k->k,0.01,0.5,color=:black)

using Parameters
@with_kw mutable struct McCallSearchModel #The @with_kw allows us to given default values
    S::Int = 40 #number of grid points
    w::Vector{Float64} = LinRange(1,10,S)
    p::Vector{Float64} = ones(S)/S
    β::Float64 = 0.96 #lower β will mean code will converge faster
    c::Float64 = 3
end

"""
    mccallbellmanmap(v,para)

Iterates the McCall search model bellman equation for with value function v.
Returns the new value function.

# Arguments
* `v` vector of values for each wage
* `params` parameters of the model

"""
function mccallbellmanmap(v, params)
    @unpack w,p,β,c = params
    #first compute value of rejecting the offer
    v_reject = c + β * dot(p,v) #note that this a Float (single number)
    #now compute value of accepting the wage offer
    v_accept = w/(1-β)
    
    #finally compare the two
    v_out = max.(v_reject,v_accept)
    #this is equivalent to
    S = length(w)
    for s in 1:S
        v_out[s] = max(v_reject,w[s]/(1-β))
    end
    
    return v_out
end

params = McCallSearchModel()

J = 100 #iterate code J times
V = zeros(J,params.S)
v0 = zeros(params.S)
V[1,:] = mccallbellmanmap(v0,params)
for j in 2:J
    V[j,:] = mccallbellmanmap(V[j-1,:],params)
end

plot()
for j in 1:J-1
    scatter!(params.w,V[j,:],color=:lightblue)
end
scatter!(params.w,V[J,:],color=:red)

"""
    solvemccall(params,[,ϵ])

Iterates the McCall search model bellman equation until convergence criterion 
ϵ is reached

# Arguments
* `params` parameters of the model
* `ϵ' Stopping criteria (default 1e-6)
"""
function solvemccall(params,ϵ=1e-6)
    @unpack w,p,β,c = params
    #initialize
    v = w/(1-β)
    diff = 1.
    #check if stoping criteria is reached
    while diff > ϵ
        v_new = mccallbellmanmap(v,params)
        #use supremum norm
        diff = norm(v-v_new,Inf)
        v = v_new #reset v
    end
    return v
end

v = solvemccall(params)
println(v - mccallbellmanmap(v,params))

scatter(params.w,v,xlabel="Wage",ylabel="Value Function")

@with_kw mutable struct RBCModel
    N::Int = 50 #number of grid points
    kgrid::Vector{Float64} = LinRange(0.01,0.5,N)
    A::Float64 = 1.
    α::Float64 = 0.33
    β::Float64 = 0.95
end

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

params = RBCModel()
V,n_pol,k_pol = RBCbellmanmap(params,zeros(params.N))
println(V)

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

V,n_pol,k_pol = RBCsolve_bellman(params,zeros(N) )
plot(k->E_infty+F_infty*log(k),.01,.5,color=:red,label="Analytical",legend=true)
scatter!(kgrid,V,label="Approximation",color=:lightblue,xlabel="Capital",ylabel="Value Function")

plot(k->(params.β*F_infty)/(1+params.β*F_infty)*params.A*k^params.α,.01,.5,color=:red,label="Analytical",legend=true)
scatter!(kgrid,k_pol,label="Approximation",color=:lightblue,xlabel="Capital",ylabel="Capital Next Period")

"""
    simulate_k(n_0,T,n_pol,kgrid)

Simulates the path of capital given policy rule n_pol and 
initial capital state kgrid[n_0] for T periods
"""
function simulate_k(n_0,T,n_pol,kgrid)
    k = zeros(T+1) # capital stock
    n = zeros(Int,T+1) #index of the capital stock
    n[1] = n_0
    k[1] = kgrid[n_0]
    for t in 1:T
        n[t+1] = n_pol[n[t]] #get the policy rule for the index
        k[t+1] = kgrid[n[t+1]]
    end

    return k
end

plot(simulate_k(1,25,n_pol,kgrid),xlabel="Time",ylabel="Capital Stock")

params = RBCModel(N=50)
@time RBCsolve_bellman(params,zeros(params.N));

params = RBCModel(N=500)
@time RBCsolve_bellman(params,zeros(params.N));

"""
    RBCbellmanmap_howard(params,V,nprime)

Iterates on the bellman equation for the standard neoclassical growth model using policies nprime,
rather than computing the optimal policies

# Arguments
* `params` parameters of the model
* `V` Vector of values for each capital level in kgrid
* `nprime` policy rules k[n_pol[n]] is the capital choice when previous period capital is k[n] 
"""
function RBCbellmanmap_howard(params,V,nprime)
    @unpack kgrid,A,α,β = params
    N = length(kgrid)
    V_new = zeros(N)
    for n in 1:N
        #use given policy 
        c = A*kgrid[n]^α - kgrid[nprime[n]]
        V_new[n] = log(c) + β*V[nprime[n]]
    end
    return V_new
end;

"""
    RBCsolve_bellman_howard(params,V0,H,[,ϵ=1e-6])

Solves the bellman equation by iterating until convergence. Uses howard improvement algorithm:
only solves for optimal policy every H iteration

# Arguments
* `params` parameters of the model
* `V0` Initial vector of values for each capital level in kgrid
* `H` Controls how frequently optimal policy is solved, H=1 implies every period
* `ϵ` Convergence criteria
"""
function RBCsolve_bellman_howard(params,V0,H;ϵ=1e-6)
    @unpack kgrid,A,α,β = params
    diff = 1.
    V,n_pol,k_pol = RBCbellmanmap(params,V0)
    #do 5 or so iterations first to allow policys to converge
    for j in 1:5
        V_new,n_pol,k_pol = RBCbellmanmap(params,V)
        V = V_new
    end
    #Now apply the Howard Improvement Algorithm
    while diff > ϵ
        V_old = V 
        for h in 1:H
            V_new = RBCbellmanmap_howard(params,V,n_pol)
            V = V_new
        end
        #perform one iteration updating policies
        V_new,n_pol,k_pol = RBCbellmanmap(params,V)
        diff = norm(V_new-V_old,Inf)
        V = V_new
    end
    return V,n_pol,k_pol
end;

#evaluate once to compile
params = RBCModel(N=50)
RBCsolve_bellman_howard(params,zeros(params.N),100);
#Test Timing
@time RBCsolve_bellman_howard(params,zeros(params.N),100);
@time RBCsolve_bellman(params,zeros(params.N));

params = RBCModel(N=500)
#Test Timing
@time RBCsolve_bellman_howard(params,zeros(params.N),100);
@time RBCsolve_bellman(params,zeros(params.N));

@with_kw mutable struct RBCModelStochastic
    N::Int = 50 #number of grid points
    kgrid::Vector{Float64} = LinRange(0.01,0.5,N)
    A::Vector{Float64} = [0.97,1.03] #TFP values
    Π::Matrix{Float64} = [0.6 0.4;0.4 0.6] #Transition matrix
    α::Float64 = 0.33
    β::Float64 = 0.95
end


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

plot(legend=true)
for s in 1:2
    plot!(k->params.α*params.β*params.A[s]*k^params.α,.01,.5,color=:red,label="Analytical")
    scatter!(params.kgrid,k_pol[s,:],color=:lightblue,label="Approximation")
end
xlabel!("Capital")
ylabel!("Capital Next Period")

using QuantEcon
"""
    simulate_k_stochastic(n_0,T,n_pol,kgrid,Π)

Simulates the path of capital given policy rule n_pol and 
initial capital state kgrid[n_0] and aggregate state s_0 for T periods
"""
function simulate_k_stochastic(n_0,s0,T,n_pol,kgrid,Π)
    k = zeros(T+1) # capital stock
    n = zeros(Int,T+1) #index of the capital stock
    s = simulate_indices(MarkovChain(Π),T;init=s0)
    n[1] = n_0
    k[1] = kgrid[n_0]
    for t in 1:T
        n[t+1] = n_pol[s[t],n[t]] #get the policy rule for the index
        k[t+1] = kgrid[n[t+1]]
    end

    return k
end

scatter(simulate_k_stochastic(1,1,25,n_pol,kgrid,Π),xlabel="Time",ylabel="Capital Stock")
using Roots,Distributions,Parameters,BasisMatrices
@with_kw mutable struct TaxParams
    σ::Float64 = 2. #Standard
    γ::Float64 = 2. #Targets Frisch elasticity of 0.5
    σ_α::Float64 = sqrt(0.147) #Taken from HSV
    N::Int64 = 1000
    alphaDist::Distribution = Normal(-σ_α^2/2,σ_α)
    αvec::Vector{Float64} = rand(alphaDist,N)
end
params = TaxParams()

"""
    approximate_household_labor(params,NlŴ,NT)

Approximates HH policies as a function of log after tax wage and transfers.
"""
function approximate_household_labor(params,NlŴ,NT)
    @unpack σ_α,σ,γ = params
    lŴbasis = ChebParams(NlŴ,-5*σ_α+log(1-.8),5*σ_α)
    Tbasis = ChebParams(NT,0.,2.) #we know optimal tax will always be positive
    basis = Basis(lŴbasis,Tbasis)
    X = nodes(basis)[1]
    N = size(X,1) #How many nodes are there?
    c,h = zeros(N),zeros(N)
    for i in 1:N 
        Ŵ,T = exp(X[i,1]),X[i,2]
        res(h) = (Ŵ*h+T)^(-σ)*Ŵ-h^γ
        min_h = max(0,(.0000001-T)/Ŵ) #ensures c>.0001
        h[i] = fzero(res,min_h,20000.) #find hours that solve HH problem
        c[i] = Ŵ*h[i]+T
    end
    U = @. c^(1-σ)/(1-σ)-h^(1+γ)/(1+γ)
    return (cf=Interpoland(basis,c),hf=Interpoland(basis,h),Uf=Interpoland(basis,U))
end;

"""
    budget_residual(params,policies,τ,T)

Computes the residual of the HH budget constraint given policy (τ,T)
"""
function budget_residual(params,policies,τ,T)
    @unpack αvec = params
    @unpack hf = policies
    N = length(αvec)
    X = [αvec .+ log(1-τ)  T*ones(N)]
    tax_income = sum(hf(X).*exp.(αvec).*τ)/N
    return tax_income - T
end;

"""
    government_welfare(params,policies,τ,T)

Solves for government welfare given tax rate τ
"""
function government_welfare(params,policies,τ,T)
    @unpack αvec = params
    @unpack Uf = policies
    N = length(αvec)
    X = [αvec .+ log(1-τ)  T*ones(N)]
    return sum(Uf(X))/N
end;

using NLopt
""" 
    find_optimal_policy(αvec,Uf,hf)

Computes the optimal policy given policy fuctions hf and indirect utility Uf
"""
function find_optimal_policy(params,policies)
    @unpack αvec = params
    @unpack Uf,hf = policies
    opt = Opt(:LN_COBYLA, 2)
    lower_bounds!(opt, [0., 0.])
    upper_bounds!(opt, [0.5,Inf])
    ftol_rel!(opt,1e-8)

    min_objective!(opt, (x,g)->-government_welfare(params,policies,x[1],x[2]))
    equality_constraint!(opt, (x,g) -> -budget_residual(params,policies,x[1],x[2]))

    minf,minx,ret = NLopt.optimize(opt, [0.3, 0.3])
    if ret == :FTOL_REACHED
        return minx
    end
end;

policies = approximate_household_labor(params,10,10)
find_optimal_policy(params,policies)
@time find_optimal_policy(params,policies)
#Remember the old code took 2 seconds

policies = approximate_household_labor(params,5,5)
find_optimal_policy(params,policies)

policies = approximate_household_labor(params,10,10)
find_optimal_policy(params,policies)

policies = approximate_household_labor(params,20,20)
find_optimal_policy(params,policies)
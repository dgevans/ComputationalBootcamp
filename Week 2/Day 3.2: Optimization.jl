using Plots
default(linewidth=2,legend=false,margin=5Plots.mm)

using Optim #Our optimization libarary
f_univ(x) = 2x^2+3x+1
plot(f_univ,-2.,1)

#Finds global minimum
res = optimize(f_univ,-2.0,1.0,GoldenSection())
Optim.minimizer(res)

#Lower bound binds
res = optimize(f_univ,-0.5,1.0,GoldenSection())
Optim.minimizer(res)

optimize(f_univ,-2.0,1.0,Brent()) #run once to precompile
@time res = optimize(f_univ,-2.0,1.0,Brent());

@time res = optimize(f_univ,-2.0,1.0,GoldenSection());

#rosenbrock function
f_ros(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
optimize(f_ros,10*ones(2),NelderMead())

optimize(f_ros,zeros(2),BFGS())

@time optimize(f_ros,zeros(2),NelderMead())
@time optimize(f_ros,zeros(2),SimulatedAnnealing(),Optim.Options(iterations=10^7))
@time optimize(f_ros,zeros(2),BFGS());

using NLopt
function myfunc!(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end

    sqrt(x[2])
end

function myconstraint!(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
lower_bounds!(opt, [-Inf, 0.])
ftol_rel!(opt,1e-6)

min_objective!(opt, myfunc!)
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,2,0))
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,-1,1))

(minf,minx,ret) = NLopt.optimize(opt, [3., 1.])
println("got $minf at $minx after $count iterations (returned $ret)")

opt = Opt(:LD_SLSQP, 2)
lower_bounds!(opt, [-Inf, 0.])
ftol_rel!(opt,1e-4)

min_objective!(opt, myfunc!)
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,2,0))
inequality_constraint!(opt, (x,g) -> myconstraint!(x,g,-1,1))

(minf,minx,ret) = NLopt.optimize(opt, [3., 1.])
println("got $minf at $minx after $count iterations (returned $ret)")

using Parameters,Distributions
@with_kw mutable struct TaxParams
    σ::Float64 = 2.  #Standard
    γ::Float64 = 2.  #Targets Frisch elasticity of 0.5
    σ_α::Float64 = sqrt(0.147) #Taken from HSV
    N::Int64 = 1000 #Number of agents
    αvec::Vector{Float64} = rand(Normal(-σ_α^2/2,σ_α),N)
end

params = TaxParams()

using Roots
"""
    household_labor(params,α,τ,T)

Solves for HH labor choice given policy and preferences
"""
function household_labor(params,α,τ,T)
    @unpack σ,γ = params
    Ŵ = (1-τ)*exp(α) #after tax wages
    res(h) = (Ŵ*h+T)^(-σ)*Ŵ-h^γ
    min_h = max(0,(.0000001-T)/(1-τ)*exp(α)) #ensures c>.0001
    h = fzero(res,min_h,20000.) #find hours that solve HH problem
    c = Ŵ*h+T
    U = c^(1-σ)/(1-σ)-h^(1+γ)/(1+γ)
    return c,h,U
end

"""
    budget_residual(params,τ,T)

Computes the residual of the HH budget constraint given policy (τ,T) given model
parameters params
"""
function budget_residual(params,τ,T)
    @unpack αvec,σ,γ = params
    tax_income = 0.
    N = length(αvec)
    for i in 1:N
        c,h,U = household_labor(params,αvec[i],τ,T)
        tax_income += τ*h*exp(αvec[i])
    end
    return tax_income/N - T
end

"""
    government_welfare(params,τ,T)

Solves for government welfare given tax rate τ
"""
function government_welfare(params,τ)
    @unpack αvec,σ,γ = params
    f(T) = budget_residual(params,τ,T)
    T =  fzero(f,0.) #Find transfers that balance budget
    
    welfare = 0.
    N = length(αvec)
    for i in 1:N
        #compute HH welfare given tax rate
        c,h,U = household_labor(params,αvec[i],τ,T)
        welfare += U #Aggregate welfare
    end
    return welfare/N
end

plot(τ->government_welfare(params,τ),0.,0.8,ylabel="Welfare",xlabel="Tax Rate")

@time minx_optim = Optim.optimize(τ->-government_welfare(params,τ),0.,0.8)
println(minx_optim)
using Plots

"""
    B1(x,k,xvec)

Constructs the kth linear B-splines using knot points xvec.
"""
function B1(x,k,xvec)
    n = length(xvec)
    #check if first condition is satisfied
    if x <= xvec[k] && k>1 && x >= xvec[k-1]
        return (x-xvec[k-1])/(xvec[k]-xvec[k-1])
    end
    #check if second condition
    if x >= xvec[k] && k < n && x <= xvec[k+1]
        return (xvec[k+1]-x)/(xvec[k+1]-xvec[k])
    end
    #otherwise return 0.
    return 0.
end

#Using linear B-Splines
f(x) = log.(x.+0.1) # our function to interpolate
xvec = LinRange(0,1,5) #knots for the B splines
xvec = [0.,0.1,0.3,0.6,1.0]
function fhat(x,f,xvec)
    K = length(xvec)
    ret = 0.
    for k in 1:K
        ret += f(xvec[k])*B1(x,k,xvec)
    end
    return ret
end
plot(x->f(x),0,1,label="True Function",legend=true)
plot!(x->fhat(x,f,xvec),0,1,label="Approximation")



using BasisMatrices
#Most basic usage
xvec = LinRange(0,1,4) #break points for the B splines
basis1 = SplineParams(xvec,0,1)
basis2 = SplineParams(xvec,0,2)
basis3 = SplineParams(xvec,0,3)
fhat1 = Interpoland(basis1,f) #linear Interpolation
fhat2 = Interpoland(basis2,f) #cubic Interpolation
fhat3 = Interpoland(basis3,f) #cubic Interpolation

plot(x->f(x),0,1,label="True Function",legend=true)
plot!(x->fhat1(x),0,1,label="Linear Approx.")
plot!(x->fhat2(x),0,1,label="Quadratic Approx.")
plot!(x->fhat3(x),0,1,label="Cubic Approx.",ylabel="f(x)",xlabel="x")



xvec = LinRange(0,1,5)
qbasis = SplineParams(xvec,0,2)
Φ = BasisMatrix(Basis(qbasis),Direct(),LinRange(0,1,1000)).vals[1]'
plt = scatter(nodes(qbasis),0*nodes(qbasis),legend=false)
for i in 1:size(Φ,1)
    plot!(LinRange(0,1,1000),Φ[i,:])
end

plt


xvec = LinRange(0,1,3)

basis1 = Basis(SplineParams(xvec,0,1))
basis2 = Basis(SplineParams(xvec,0,2))
basis3 = Basis(SplineParams(xvec,0,3))

g(x) = exp.(x)

ghat1 = Interpoland(basis1,g)
ghat2 = Interpoland(basis2,g)
ghat3 = Interpoland(basis3,g)

plot(x->g(x),0,1,label="True Function",legend=true)
plot!(x->ghat1(x),0,1,label="Linear Approx.")
plot!(x->ghat2(x),0,1,label="Quadratic Approx.")
plot!(x->ghat3(x),0,1,label="Cubic Approx.",ylabel="g(x)",xlabel="x")


#using BasisMatrix code

using LinearAlgebra
n = 9
Phi = zeros(n,n)
xvec = LinRange(-1,1,n)
f_runge(x) = 1 ./(1 .+ 25 .* x.^2) #The Runge Function


chebbasis = Basis(ChebParams(20,-1,1))
xnodes = nodes(chebbasis)[1] 
f_cheb = Interpoland(chebbasis,f_runge)

plot(f_runge,-1,1,label="Runge",legend=true)
plot!(x->f_cheb(x),-1,1,label="Chebyshev",xlabel="x",ylabel="f(x)")


xgrid = LinRange(-1,1,20) #use 9 grid points like chebyshev
f_spline = Interpoland(SplineParams(xgrid,0,3),f_runge)
plot(f_runge,-1,1,label="Runge",legend=true)
plot!(x->f_spline(x),-1,1,label="Cubic Spline")
plot!(x->f_cheb(x),-1,1,label="Chebyshev",xlabel="x",ylabel="f(x)")


using Parameters
@with_kw mutable struct NCParameters
    A::Float64 = 1.
    α::Float64 = 0.3
    β::Float64 = 0.96
    kgrid::Vector{Float64} = LinRange(0.05,0.5,20)
    spline_order::Int = 3
end

para = NCParameters()


using Optim

V0 = k -> 0.0
k = 0.5

function bellmanmap(para::NCParameters,Vprime,k)
    @unpack A,α,β,kgrid = para
    k_bounds = [kgrid[1],kgrid[end]]
    f_obj = kprime -> - (log(A*k^α-kprime) + β * Vprime(kprime) )
    res = optimize(f_obj,k_bounds[1],k_bounds[2])
    kprime = res.minimizer
    V = -res.minimum
    return (kprime = kprime,V = V)
end

TV = k -> bellmanmap(para,V0,k).V
TTV = k -> bellmanmap(para,TV,k).V
TTTV = k -> bellmanmap(para,TTV,k).V
TTTTV = k -> bellmanmap(para,TTTV,k).V
TTTTTV = k -> bellmanmap(para,TTTTV,k).V
TTTTTTV = k -> bellmanmap(para,TTTTTTV,k).V
TTTTTTTV = k -> bellmanmap(para,TTTTTTTV,k).V
TTTTTTTTV = k -> bellmanmap(para,TTTTTTTTV,k).V
TTTTTTTTTV = k -> bellmanmap(para,TTTTTTTTTV,k).V
TTTTTTTTTTV = k -> bellmanmap(para,TTTTTTTTTTV,k).V
TTTTTTTTTTTV = k -> bellmanmap(para,TTTTTTTTTTTV,k).V
TTTTTTTTTTTTV = k -> bellmanmap(para,TTTTTTTTTTTTV,k).V
TTTTTTTTTTTTTV = k -> bellmanmap(para,TTTTTTTTTTTTTV,k).V

function approxbellmanmap(para::NCParameters,Vprime::Interpoland)
    kbasis = Vprime.basis
    #sometimes it's helpful to tell julia what type a variable is
    knodes = nodes(kbasis)[1]::Vector{Float64}
    V = [bellmanmap(para,Vprime,k).V for k in knodes]
    return Interpoland(kbasis,V)
end

"""
    getV0(para::NCParameters)

Initializes V0(k) = 0 using the kgrid of para
"""
function getV0(para::NCParameters)
    @unpack kgrid,spline_order = para

    kbasis = Basis(SplineParams(kgrid,0,spline_order))

    return Interpoland(kbasis,k->0 .*k)
end

V0 = getV0(para)
V0(0.3)
V1 = approxbellmanmap(para,V0)
V2 = approxbellmanmap(para,V1)
V3 = approxbellmanmap(para,V2)
V4 = approxbellmanmap(para,V3)
V5 = approxbellmanmap(para,V4)
V6 = approxbellmanmap(para,V5)
V7 = approxbellmanmap(para,V6)
V8 = approxbellmanmap(para,V7)
V9 = approxbellmanmap(para,V8)
V10 = approxbellmanmap(para,V9)
V11 = approxbellmanmap(para,V10)
V12 = approxbellmanmap(para,V11)

@time V0(0.)
@time V12(0.)

using LinearAlgebra
"""
    solvebellman(para::NCParameters,V0::Interpoland)

Solves the bellman equation for a given V0
"""
function solvebellman(para::NCParameters,V0::Interpoland)
    diff = 1
    #Iterate of Bellman Map until difference in coefficients goes to zero
    while diff > 1e-6
        V = approxbellmanmap(para,V0)
        diff = norm(V.coefs-V0.coefs,Inf)
        V0 = V
    end
    kbasis = V0.basis
    knodes = nodes(kbasis)[1]::Vector{Float64}
    #remember optimalpolicy also returns the argmax
    kprime = [bellmanmap(para,V0,k).kprime for k in knodes]
    #Now get policies
    return Interpoland(kbasis,kprime),V0
end;


solvebellman(para,V0)
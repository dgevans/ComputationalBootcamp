#   Stochastic Processes
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   Introduction
#   ============
# 
#     •  Stochastic processes are the foundation of economic models
#        • Capture risk individual agents face
#        • Aggregate risk
#        • Estimation etc.
# 
#     •  We'll study how to construct and simulate on a computer
#        • Introduce some basic programming constructs
# 
#     •  Start with simple random numbers

#   How To Generate Random numbers
#   ==============================
# 
#     •  Generating random number is as simple as calling rand()
# 
#     •  Generates a number in [0,1] uniformly distributed

rand()

#     •  Can even choose between elements of a set

rand([10,20])

#     •  If you give it dimensions it will give a vector or matrix of
#        random variables

rand(2,2)

#   Constructing Any Random Variable
#   ================================
# 
#     •  Note any random variable can be constructed from the uniform on
#        [0,1]
# 
#     •  Let x be a random variable on \mathbb R
#        • let F(x) be its CDF
# 
#     •  If \xi is a r.v. uniform on [0,1]
#        • then F^{-1}(\xi) has the same distribution as x
# 
#     •  We can draw pretty much any type of random variable we like
#        • randn is how we draw the standard normal

#   A Simple Monte Carlo
#   ====================
# 
#     •  Let's say we wanted to flip a 100 times how would we do that
# 
#     •  Start simple: flip a coin 1 time and record 1 as heads 0 as tails

flip = rand() < 0.5

#     •  To do 100 times let's use a for loop and a vector

flips = zeros(100) #Where to store the flips
r = rand(100) #A vector of 100 uniform random numbers on [0,1]
for i in 1:100
    flips[i] = r[i] < 0.5 #check to see if the i th flip is heads
end
println(flips)

#   An Even Simpler Way
#   ===================
# 
#     •  Often times we want to vectorize an operation
#        • works in place of a for loop
# 
#     •  Using . before an operation means apply this to all elements
#        • 'broadcasts' across dimensions (we'll look at this
#        later)
# 
#     •  Often very efficient (doesn't create a bunch of new variables)
#        • because we work with a lot of vectors we'll do this a
#        lot
# 
#     •  For example our coin flipping exercise becomes

flips = rand(100) .< 0.5 #note the . won't work without it
println(flips)

#   Functions
#   =========
# 
#     •  Let's create a function to do the flipping for us

"""
    flipNcoins(N,p=0.5)

Flips N coins with probability p of returning 1 (heads)
"""
function flipNcoins(N,p=0.5)
    return rand(N) .< p
end

#     •  Check that it works

println(flipNcoins(15))

#     •  Can change weighting of the coins

println(flipNcoins(15,1.))

#   Exercise 1
#   ==========
# 
#   Let p\in\mathbb R^S be a vector of probabilities where p_s represents the
#   probability of s.
# 
#     •  Write a function drawDiscrete(p) to draw a rand element
#        s\in\{1,2,3,\ldots,S\} from the discrete distribution with
#        probability vector p
# 
#     •  Write a function drawNDiscrete(p,N) to do that N times

#   The Distribution Package
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   Log Normal Example
#   ==================
# 
#     •  The distribution package contains the ability to draw from a lot
#        of distributions

using Distributions
dist  = LogNormal(0,1)

#     •  Has lots of helpful functions

println(mean(dist))
println(std(dist))
println(quantile(dist,0.5))
println(cdf(dist,1.))
println(pdf(dist,1.))
println(rand(dist))

#   Some Monte Carlo
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   Fliping 50 coins
#   ================
# 
#     •  Let's flip 50 coins what would be the mean number of heads

using Statistics
mean(flipNcoins(50))

#     •  For an unfair coin

mean(flipNcoins(50,0.3))

#   Plotting
#   ========
# 
#     •  The next few slides are going to generate plots
# 
#     •  I'm going to use Plots as my plotting library
#        • Uses language of graphics which makes it really
#        convenient

using Plots

#     •  I'm going to use some default values to make the pictures look
#        better (usually I'll hide these)

default(linewidth=2,legend=false,margin=5Plots.mm)

#   Some More Monte Carlo
#   =====================
# 
#     •  How does the average number of coins behave with the number of
#        tosses N?

meantosses = [mean(flipNcoins(N,0.5)) for N in 1:100]; #This is called a list compression

#     •  Can plot to visualize

scatter(1:100,meantosses,xlabel="Tosses",ylabel="Average # Heads")

#   Some More Monte Carlo
#   =====================
# 
#     •  Can do it for eaven longer simulation to make sure it settles down

toss_range = 5:1000
meantosses = [mean(flipNcoins(N,0.5)) for N in toss_range]; #This is called a list compression
stdtosses = [std(flipNcoins(N,0.5)) for N in toss_range]; #This is called a list compression
plot(toss_range,meantosses,layout = (2,1),subplot=1,ylabel="Average # Heads")
plot!(toss_range,stdtosses,layout = (2,1),subplot=2,xlabel="Tosses",ylabel="STD # Heads")

#   What Does the Distribution Look Like
#   ====================================
# 
#     •  We can get an idea for what the distribution heads looks like for
#        15 tosses
# 
#     •  And compare to truth (Binomial)

numheads = [sum(flipNcoins(15,0.5)) for k in 1:100_000]
histogram(numheads,bins=25,normalize=:probability,xlabel="# of Heads",ylabel="Probability")
plot!(0:15,pdf.(Binomial(15,0.5),0:15))

#   Pseudo Random variables
#   =======================
# 
#     •  One important thing to know is that all random variables on a
#        computer are not really random
#        • They are actually a deterministic sequence
#        • But behave random
#        • Depends on what seed is set. We can change this with
#        Random.seed!()
# 
#     •  The following will print 0.5627138851056968

using Random
Random.seed!(12345) #can put any integer here
rand()

#     •  Will behave the same way again

Random.seed!(12345) #can put any integer here
rand()

#   Exercise 2
#   ==========
# 
#   A Pareto distribution is often used to model thick right tails like those
#   found in wealth distributions. If X is distributed according to a Pareto
#   distribution with shape parameter \alpha then :$ \text{Pr}(X\geq x) =
#   \begin{cases}x^{-\alpha} & \text{if x\geq1}\ 1 & \text{if x<1}\end{cases} :$
# 
#     •  Write a function drawNPareto(alpha,N) to sample N draws from the
#        Pareto distribution with shape parameter \alpha. Hint: use the
#        Distributions package
# 
#     •  Verify via monte carlo that the relationship above holds

#   Continuous Processes
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   An AR(1)
#   ========
# 
#     •  Let's model and AR1 processes
# 
# :$
# 
#   x_t =(1-\rho)μ + \rho x_{t-1} + \epsilon_t
# 
# :$
# 
#   where \epsilon_t\sim\mathcal N(0,\sigma)
# 
#     •  Depends on two key parameters: \rho and \sigma
# 
#     •  Can easily write a function to simulate

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

#   Plotting and An AR1
#   ===================
# 
#     •  We can see what an AR(1) looks like by plotting an example path

plot(1:100,simulateAR1(0.,0.8,1.,0,100),xlabel="Time",ylabel="AR(1)")

#   Structs
#   =======
# 
#     •  Many times in julia (and programming) it's construct an Type that
#        represents an object
# 
#     •  For example, our AR1 is represented by three variables: \rho,\mu,
#        and \sigma
# 
#     •  We can make an mutable struct which represents this AR(1)

mutable struct AR1
    μ::Float64 #Mean of the AR(1)
    ρ::Float64 #persistence of the AR(1)
    σ::Float64 #standard deviaiton of the AR(1)
end

#     •  Can then construct an instance of that type and access elements
#        through .

ar1 = AR1(0.,0.8,1.) #Note order matters here
println(ar1.ρ)

#   Adapting function
#   =================
# 
#     •  We can now adapt our AR(1) function to use that AR1 type

"""
   simulateAR1(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar,x0,T)
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ar.ρ)*ar.μ + ar.ρ*x[t] + ar.σ*randn()
    end
    return x[2:end]
end
plot(1:100,simulateAR1(0.,0.8,1.,0,100),xlabel="Time",ylabel="AR(1)")

#   Unpacking parameters
#   ====================
# 
#     •  OK, that . notation is annoying. Is there an easier way?
#        • Of course! We can use a clever package called Parameters

using Parameters
"""
   simulateAR1(ar,x0,T)

Simulates an AR(1) ar for T periods with initial value x0
"""
function simulateAR1(ar,x0,T)
    @unpack σ,μ,ρ = ar #note order doesn't matter
    x = zeros(T+1)# initialize
    x[1] = x0
    for t in 1:T
        x[t+1] = (1-ρ)*μ + ρ*x[t] + σ*randn()
    end
    return x[2:end]
end
plot(1:100,simulateAR1(ar1,0.,100),xlabel="Time",ylabel="AR(1)")

#   Some Monte Carlo Experiments
#   ============================
# 
#     •  Let's simulate a large sample of AR1 processes

T = 50
N = 1000
X = zeros(T,N)
for i in 1:N
    X[:,i] .= simulateAR1(ar1,2.,T)
end

#     •  Can visualize all the paths

plot(1:T,X,xlabel="Time")

#   Mean and Standard Deviation
#   ===========================

#mean(X,dims=2) takes average across rows
plot(1:T,mean(X,dims=2),ylabel="Mean",layout=(2,1),subplot=1)
plot!(1:T,std(X,dims=2),xlabel="Time",ylabel="Standard Deviation",subplot=2)

#   Exercise 3
#   ==========
# 
#     •  Write some code to simulate an arbitrary VAR process
# 
# :$
# 
#   Xt = A X{t-1} + C\mathcal E_t :$
# 
#     •  Use that code to simulate a process
# 
# :$
# 
#   x_t = 0.5 x_{t-1} - 0.3 x_{t-2} + \epsilon_t
# 
# :$
# 
#   Hint: set X_t=[x_t,x_{t-1}]. What is A? What is C?

#   Finite State Processes
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   Markov Chains
#   =============
# 
#     •  Finite State Markov Chains are the building blocks of modern macro
# 
#     •  Idea: random variable X_t takes values in a finite set X̄ of length
#        S
#        • Index values of \bar X by s=1,2,\ldots,S
#        • Satisfies markov property
#        :$ \text{Pr}\left(Xt|X{t-1},X{t-2},\ldots\right) =
#        \text{Pr}\left(Xt|X_{t-1}\right) :$
# 
#     •  Therefore X_t is defined entirely by
#        • \bar X
#        : A vector of values for each states
#        • \pi_0
#        : An initial probability distribution for X_0
#        • P
#        : A transition matrix that records
#        :$ P{ij} = \text{Pr}\left(Xt=\bar X[j]|X_t = \bar X[i]\right) :$

#   Properties of Markov Chains
#   ===========================
# 
#     •  Conditional expectations are matrix multiplacation
# 
# :$
# 
#   \mathbb E\left[X_{t+1}|X_t=\bar X[i]\right] = (P\bar X)_i
# 
# :$
# 
#   and :$ \mathbb E\left[X{t+k}|Xt=\bar X[i]\right] = (P^k\bar X)_i :$
# 
#     •  Conditional probabilities are easy to compute
# 
# :$
# 
#   \text{Pr}\left[X_{t+k}=\bar X[j]|X_t=\bar X[i]\right] = (P^k)_{ij}
# 
# :$
# 
#     •  Computing unconditional distirbution is matrix multiplication
# 
# :$
# 
#   \pi_t' = \pi_0'P^t
# 
# :$
# 
#     •  Stationary distributions \pi^* are left-eigenvectors
# 
# :$
# 
#   (\pi^*)'=(\pi^*)'P
# 
# :$

#   Simulating Markov Chain
#   =======================
# 
#     •  We could spend time creating our own function to simulate markov
#        Chains
# 
#     •  Or we could use something prepackaged (don't reinvent the wheel)
# 
#     •  QuantEcon has a lot of helpful libraries

using QuantEcon

#     •  In this case we can use simulate

P = [0.6 0.4;
     0.4 0.6]
s = simulate(MarkovChain(P),100,init=1)
println(s)

#   Rouwenhorst
#   ===========
# 
#     •  QuantEcon also has a pre-packaged Rouwenhorst method
#        • Approximates an AR(1) (see homework)

mc_ar1 = rouwenhorst(51,0.9,0.014)

#     •  Let's try some monte carlo

X = zeros(15,1000)
for i in 1:1000
    X[:,i] = simulate(mc_ar1,15,init=1)
end
println(mean(X[15,:]))

#     •  Compare to formula

P,X̄ = mc_ar1.p,mc_ar1.state_values
println((P^14*X̄)[1])

#   Long Run Stationary Distribution
#   ================================
# 
#     •  We can compute the stationary distribution of this Chain
# 
#     •  Two ways:

using LinearAlgebra
D,V = eigen(P')  #should be left unit eigenvector
πstar = V[:,isapprox.(D,1)][:]
πstar ./= sum(πstar)#Need to normalize for probability

#or 

πstar2 = (P^200)[15,:] #probability distribution 1000 periods in the future
println(norm(πstar -πstar2))

#     •  Surprisingly the second can often be faster (need to check how
#        many periods)

@time D,V = eigen(P');
@time πstar2 = (P^200)[15,:];

#   Compare to Monte Carlo
#   ======================

s_end = zeros(Int,10000)
for i in 1:10000
    s_end[i] = simulate_indices(mc_ar1,200,init=1)[end]
end

histogram(s_end,bins=51,normalize=:probability,xlabel="State",ylabel="Probability")
plot!(1:51,πstar2)
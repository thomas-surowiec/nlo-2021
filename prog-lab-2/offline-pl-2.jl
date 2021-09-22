# using DataFrames
# showln(x) = (show(x); println())
#
# # A DataFrame is an in-memory database
# df = DataFrame(A = [1, 2], B = [ℯ, π], C = ["xx", "xy"])
# showln(df)

cd(dirname(@__FILE__()))

using Plots
using PlotThemes
using Measures
using LinearAlgebra
using SparseArrays


#=

Part I: A Simple Implementation of Newton's Method

=#

################################################################################
# Defining Constructors (Types)
################################################################################

#=
We wish to solve F(x) == 0. For this we need to
- Evaluate F(x)
- Approximate or Calcualte F'(x)
- Report || F(x) || : For some problems, ||⋅|| is not the Euclidan norm

We thus define a simple constructor (object) with three fields:
=#
mutable struct Residual
    evalF::Function     # F(x)
    jac::Function       # F'(x)
    Fnrm::Function      # || F(x) ||
end

# We make Residual a callable type that acts as a function but encapsulates
# data or other important functions.
function (F::Residual)(x::Array{Float64,1})
    return F.evalF(x)
end

################################################################################
# Basic Newton's Method
################################################################################

# This is the basic structure of a classic version of Newton's method for
# nonlinear equations. We will slightly adapt it blow to fit the applications.
function newton_nd(F::Residual,x_0::Array{Float64,1},
                   atol::Float64,rtol::Float64,maxit::Int64)

    x   = copy(x_0)
    it  = 0
    res = F.Fnrm(F(x))
    r_0 = res
    println("||F(x)|| = ", res)
    while res > rtol*r_0 + atol && it < maxit
        dx  = -F.jac(x)\F(x)
        x   = x + dx
        it += 1
        res = F.Fnrm(F(x))
        println("||F(x)|| = ", res)
    end
    return x, it
end


#=

Part II: A Parameter ID Problem
         See the notes and video for a description of this application.
=#

################################################################################
# The Objective Function
################################################################################

# The nonlinear least squares objective function
function nonlinear_ls(v_n::Array{Float64,1},
                      t_n::LinRange{Float64},
                        x::Array{Float64,1})
    F = 0.0
    n = length(v_n)
    if n != length(t_n)
        error("v_n and t_n must be the same length")
    end

    it = 1
    while it <= n
        F += (v_n[it] - exp(-x[1]*t_n[it])*(
                             x[3]*sin(x[2]*t_n[it]) +
                             x[4]*cos(x[2]*t_n[it])))^2
        it += 1
    end

    return F
end

################################################################################
# Problem Data
################################################################################

#=
We generate a problem instance
- ex1, ex2, ex3, ex4 represent the "true" unknown parameters
- t is the discretized time interval [0,40] with 800 time steps
- v is are the "true" measurements using the model v(t,x)
=#

# Generate "Data"
ex1 = 0.15
ex2 = 2.0
ex3 = 1.0
ex4 = 3.0

# Generate 800 steps from time 0 to 40
t = LinRange(0, 40, 800)

# v is the vector of time-series data
v = exp.(-ex1.*t).*(ex3*sin.(ex2.*t)+ex4.*cos.(ex2.*t))

theme(:wong)
p1 = plot(t,v,
          titlefont = font(18, "Helvetica"),
          title= "Motion of weight attached to damped spring",
          xlabel="\$t\$",
          ylabel="\$v(t)\$",
          titlefontsize = 24,
          label = false,#["abs_err"],
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=24,
          legend=true,
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          # legendfont = font(14, "Helvetica"),
          lw=3)

savefig(p1,"true_solution_v.pdf")

# To make the problem more interesting, we add noise to the measurments
v = exp.(-ex1.*t).*(ex3*sin.(ex2.*t)+ex4.*cos.(ex2.*t)) + randn(800)/50
plot!(t,v,label = false)

################################################################################
# Defining constructors for the objective
################################################################################

# An objective function type
mutable struct Objective
    data_1::Array{Float64,1}
    data_2::LinRange{Float64}
    obj_fn::Function
end

# Evaluate the objective without  re-passing the fixed data vectors v_n and t_n
function (F::Objective)(x::Array{Float64})
    return F.obj_fn(F.data_1,F.data_2,x)
end

# Create an instance of Objective using the data (v,t)
nls_obj = Objective(v,t,nonlinear_ls)

################################################################################
# Approximating Gradients and Hessians using Finite Differences
################################################################################

# A central difference function for finding partial derivatives
function CD(x::Array{Float64},d::Array{Float64},h::Float64,f::Objective)
    return (f(x+d) - f(x-d))/(2*h)
end

# A function for approximating the gradient using finite differences
function CD_grad(x::Array{Float64},h::Float64,f::Objective)
    n  = length(x)
    it = 1

    gradF = zeros(n)
    while it <= n
        gradF[it] = CD(x,h*(1:n .== it),h,f)
        it += 1
    end

    return gradF
end

# Random point
CD_grad(rand(4),1e-6,nls_obj)
# Value at optimal solution
CD_grad([ex1,ex2,ex3,ex4],1e-6,nls_obj)

# A function for approximating the Hessians using finite differences
function CD_hess(x::Array{Float64,1},h::Float64,f::Objective)
     n = length(x)
     i = 1
     j = 1

    HessF = zeros(n,n)
    while i <= n
        while j <= n
            HessF[i,j] = (CD_grad(x+h*(1:n .== j),h,f)[i] -
                          CD_grad(x-h*(1:n .== j),h,f)[i] +
                          CD_grad(x+h*(1:n .== i),h,f)[j] -
                          CD_grad(x-h*(1:n .== i),h,f)[j])/(4.0*h)
            j+=1
        end
        j  = 1
        i += 1
    end

    return HessF
end

# Random point
CD_hess(rand(4),1e-6,nls_obj)

# Value at optimal solution
CD_hess([ex1,ex2,ex3,ex4],1e-6,nls_obj)


################################################################################
# Defining constructors for gradients and hessians
################################################################################
mutable struct Grad
    f::Objective
    step::Float64
end

function (G::Grad)(x::Array{Float64,1})
    return CD_grad(x,G.step,G.f)
end

mutable struct Hess
    f::Objective
    step::Float64
end

function (H::Hess)(x::Array{Float64,1})
    return CD_hess(x,H.step,H.f)
end

# A composite type for optimization problems
mutable struct OptProb
    obj::Objective
    grd::Grad
    hess::Hess
end

nls_obj  = Objective(v,t,nonlinear_ls)
grad_obj = Grad(nls_obj,1e-8)
hess_obj = Hess(nls_obj,1e-8)

nls_res = OptProb(nls_obj,grad_obj,hess_obj)


# An adaptation of Newton's method for optimization
function newton_nd(F::OptProb,x_0::Array{Float64,1},
                   atol::Float64,rtol::Float64,maxit::Int64)
    x   = copy(x_0)
    it  = 0
    rhs = F.grd(x)
    res = norm(rhs)
    r_0 = res
    println("||F(x)|| = ", res)
    # while res > tol && it < maxit
    while res > rtol*r_0 + atol && it < maxit
        dx  = -F.hess(x)\rhs
        x   = x + dx
        # alpha = 1/(1 + 10*norm(dx)) # Damped version
        # x   = x+alpha*dx
        it += 1
        rhs = F.grd(x)
        res = norm(rhs)
        println("||F(x)|| = ", res)
    end
    println("Number of Iterations = ", it)
    return x, it
end

# Solve the problem!
extn, nit = newton_nd(nls_res,[ex1,ex2,ex3,ex4].+rand(4)/50,1e-8,1e-8,50)

################################################################################
# Plot and compare the approximation
################################################################################
v = exp.(-ex1.*t).*(ex3*sin.(ex2.*t)+ex4.*cos.(ex2.*t))
v_newt = exp.(-extn[1].*t).*(extn[3]*sin.(extn[2].*t)+extn[4].*cos.(extn[2].*t))

p2 = plot(t,v_newt,
          titlefont = font(18, "Helvetica"),
          title= "Reconstructed Motion",
          xlabel="\$t\$",
          ylabel="\$v(t)\$",
          titlefontsize = 24,
          label = false,#["abs_err"],
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=24,
          legend=true,
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          # legendfont = font(14, "Helvetica"),
          lw=3)

savefig(p2,"reconstructed_solution_v.pdf")

err = log.(abs.(v-v_newt))

p3 = plot(t,err,
          titlefont = font(18, "Helvetica"),
          title= "Log error",
          xlabel="\$t\$",
          ylabel="\$\\|v_{true}(t) - v_{newt}(t)|\$",
          titlefontsize = 24,
          label = false,#["abs_err"],
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=24,
          legend=true,
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          # legendfont = font(14, "Helvetica"),
          lw=3)

savefig(p3,"log_error_noisy_data_v.pdf")

p4 = plot(t,(v-v_newt),
          titlefont = font(18, "Helvetica"),
          title= "Absolute error",
          xlabel="\$t\$",
          ylabel="\$\\|v_{true}(t) - v_{newt}(t)|\$",
          titlefontsize = 24,
          label = false,#["abs_err"],
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=24,
          legend=true,
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          # legendfont = font(14, "Helvetica"),
          lw=3)

savefig(p4,"abs_error_noisy_data_v.pdf")

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

#=

Part III: Variational Denoising of Signals
          See the notes and video for a description of this application.
=#

################################################################################
# Generate Residual F(x)
################################################################################
global ts = 1e3      # Penalty of misfit (Try not to use global constants...)

# The nonlinear term in the ODE
function bulk(u::Array{Float64,1},reg::Float64)
    n = length(u)
    F = zeros(n)
    for i in 2:(n-1)
        F[i] = (2*u[i]^3 - 2*u[i]^2 + (1+reg*ts)*u[i])/reg
    end
    return F
end

# This evaluates the full nonlinear residual
function nl_residual(u::Array{Float64,1},
                     g::Array{Float64,1},
                     reg::Float64,
                     L::SparseMatrixCSC)

    return L*u + bulk(u,reg) - [0 ; ts*g ; 0]
end

# Test
# nl_residual(abs.(rand(nnd+2)),abs.(rand(nnd)),reg,L)

################################################################################
# Generate Jacobian
################################################################################
# The nonlinearity is simple enough to directly calcualte and iplement the
# Jacobian
function diff_bulk(u::Array{Float64,1},reg::Float64)
    n = length(u)
    F = zeros(n)
    for i in 2:(n-1)
        F[i] = (6*u[i]^2 - 4*u[i] + (1+ts*reg))/reg
    end
    return F
end


################################################################################
# Defining constructors
################################################################################
# We once again slightly adapt our Newton algorithm by introducing a custom type
mutable struct SigRes
    evalF::Function
    jac::Function
    sig::Array{Float64,1}
    reg::Float64
    L::SparseMatrixCSC
end

# We make Residual a callable type that acts as a function but encapsulates
# data or other important functions.
function (F::SigRes)(x::Array{Float64,1})
    return F.evalF(x,F.sig,F.reg,F.L)
end

# This is the true Hessian for the nonlinear residual
function sig_hess(x::Array{Float64,1},r::Float64,L::SparseMatrixCSC)
    dF  = spdiagm(0 => diff_bulk(x,r))
    return L + dF
end


function newton_dn(F::SigRes,x_0::Array{Float64,1},
                   atol::Float64,rtol::Float64,maxit::Int64)

    x   = copy(x_0)
    it  = 0
    res = norm(F(x))
    r_0 = res
    println("||F(x)|| = ", res)
    while res > rtol*r_0 + atol && it < maxit
        H   = F.jac(x,F.reg,F.L)
        dx  = -H\F(x)
        x   = x + dx
        it += 1
        res = norm(F(x))
        println("||F(x)|| = ", res)
    end
    return x
end

# Solve
nde = 12
nnd = 2^nde          # Number of abscissae
reg = 0.001           # "Smoothing" Parameter for Ginzburg-Landau Function
stp = 1/(2^nde - 1)  # Fixed length of intervals

################################################################################
# Generate a Random Signal Function
################################################################################
function step_function(r_0::Float64,r_1::Float64,s::Float64,x::Float64)
    if x < r_0
        return 0.0
    elseif x >= r_0 && x < r_1
        return s
    else
        return 0.0
    end
end
step_function(0.0,0.25,1.0,-1.0)

function simple_step_func(R::Array{Float64,1},S::Array{Float64,1},x::Float64)
    l = length(R)
    f = 0.0
    for j in 1:(l-1)
        f = f + step_function(R[j],R[j+1],S[j],x)
    end
    return f
end
# Test
R = collect(0:0.1:1)
S = rand((length(R)))
simple_step_func(R,S,0.95)

# Plot
x_grid = LinRange(0,1,nnd)
x_grid_values = collect(x_grid)
signal = zeros(nnd)
clnsig = zeros(nnd)
for i in 1:nnd
    clnsig[i] = simple_step_func(R,S,x_grid_values[i]) # Clean
    signal[i] = clnsig[i] + randn()/100
end

# pp = plot(x_grid,[clnsig,signal],
#           titlefontsize = 11,
#           legend=false,
#           lw=3,
#           )
# display(pp)

# This is the central difference matrix for the term epsilon*u''(x)
L  =  (reg/(stp^2))*spdiagm(-1 =>  -ones(nnd+1),
                             0 => 2*ones(nnd+2),
                             1 =>  -ones(nnd+1))

# This incorporates the homogeneous Newmann boundary conditions into L
L[1,:]     = sparse((1:(nnd+2) .== 1) - (1:(nnd+2) .== 3))
L[nnd+2,:] = sparse((1:(nnd+2) .== (nnd+2)) - (1:(nnd+2) .== nnd))

u_0     = abs.(rand(nnd+2))
sig_obj = SigRes(nl_residual,sig_hess,signal,reg,L)
u_0     = newton_dn(sig_obj,u_0,1e-11,1e-11,10)
plot(u_0[2:(end-1)])

# Compare the results for FD vs. CD
linetypes = [:T :N :DN]
styles = filter((s->begin
                s in Plots.supported_styles()
            end), [:solid, :dash, :dot])
styles = reshape(styles, 1, length(styles))

# The TV regularization is great for denoising but (naturally) affects the
# contrast in the denoised signal. This can be (heuristically!!!) adjusted
# after solving by scaling the solution u, e.g.  3/2 * u gives a decent
# adjustment of contrast.
cntrst = 1.0
# cntrst = 1.5

pp1 = plot(x_grid,[clnsig,signal,cntrst*u_0[2:(nnd+1)]],
          titlefont = font(14, "Helvetica"),
          title= "Noisy vs. Denoised",
          xlabel="\$x\$",
          ylabel="\$u(x)\$",
          titlefontsize = 24,
          label = ["T"  "N"  "DN"],
          line=(1.75, styles),
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=24,
          legend = :bottomleft,
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          legendfont = font(11, "Helvetica")
          )

savefig(pp1,"denoised_signal.pdf")

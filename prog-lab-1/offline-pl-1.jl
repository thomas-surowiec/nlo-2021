cd(dirname(@__FILE__()))    # This ensures the saved output will be in the
                            # same directory as the file itself

using Plots             # This is one of the many plot libraries. In fact,
                        # Plots.jl is a kind of metalibrary. See the docs:
                        # https://docs.juliaplots.org/latest/tutorial/

using PlotThemes        # Provides different plot themes
using Measures          # Use the metric system for plot attributes

using LinearAlgebra     # This is a linear algebra package that uses native
                        # Julia implementations of things like trace,
                        # determinant etc:
                        # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/

#=

Part I: Defining Functions and Approximating Derivatives

=#
################################################################################
# Defining Functions
################################################################################
# Inline function with slightly restricted type definition
bsc_func(x::Real) = x^2 + 1
bsc_func(2)
println("bsc_func(2) = ", bsc_func(2))
@time bsc_func(2)

# Apply to each argument of a vector
@time bsc_func.([1.0,2.0])

# Inline function restricted to Float64 only
bsc_func_typ(x::Float64) = x^2 + 1
bsc_func_typ(2)     # Returns an error since 2 is not a Float64
bsc_func_typ(2.0)   # Returns a value since 2.0 is a FLoat64
println("bsc_func(2) = ", bsc_func(2.0))

@time bsc_func_typ(2.0)

# Apply to each argument of a vector
println("###")
@time bsc_func_typ.([1.0,2.0])
println("###")

# Compare and print to REPL
err = abs(bsc_func_typ(2.0) - bsc_func(2))
println("abs(bsc_func_typ(2.0) - bsc_func(2)) = ", err)

# Inline function without restricted type definition
bsc_func_notyp(x) = x^2 + 1
bsc_func_notyp(2)          # No error msg
bsc_func_notyp(2.0)        # No error msg
bsc_func_notyp.([1.0,2])

@time bsc_func_notyp(2)
@time bsc_func_notyp(2.0)        # No error msg
@time bsc_func_notyp.([1.0,2])

################################################################################
# More Complex Functions
################################################################################
function nwtn_1D(x_0::Real,tol::Real,f::Function,df::Function)
    iter = 0
    x    = copy(x_0)    # If you do not use copy, then the equals sign points
                        # to the value of x_0, rather than copying that value
                        # to the new variable x.
    res = abs(f(x))
    println("--------------")
    println("residual |f(x)| = " , res)
    while res > tol && abs(df(x)) != 0.0
        iter += 1
        x     = x_0 - f(x_0)/df(x_0)
        res   = abs(f(x))
        x_0   = x
        println("residual |f(x)| = " , res)
    end
    println("--------------")
    return x, iter
end
test_func(x) = sin(x::Real)
deriv_tf(x) = cos(x::Real)
nwtn_1D(0.5,1e-8,test_func,deriv_tf)

################################################################################
# Using Callable Objects (Constructors) as Functions
################################################################################
mutable struct NonlinearEq
     F::Function
    dF::Function
end

nleq = NonlinearEq(sin,cos)
nleq.F(2) - sin(2)
nleq.dF(2) - cos(2)

function (f::NonlinearEq)(x::Union{Real,Vector{Real}})
    return f.F(x)
end
nleq(2) - sin(2)

# Test functions for constructing an NLFunc
Ad = rand(10)
A  = Diagonal(Ad)
quad_func(x::Array{Float64,1})  = 0.5*x'*A*x
dquad_func(x::Array{Float64,1}) = A*x

# Construct type NLFunc
nleq_vec = NonlinearEq(quad_func,dquad_func)

test_vec = rand(10)
nleq_vec.F(test_vec)
nleq_vec.dF(test_vec)

mutable struct NLFunc
     Fdata::Union{Matrix{Float64},Diagonal{Float64,Array{Float64,1}}}
         F::Function
        dF::Function
     EvalF::Function
    EvaldF::Function
end

function (f::NLFunc)(x::Array{Float64,1},tog::Int64)
    if tog == 0 # Return a function value
        return f.EvalF(x,f.Fdata)
    else tog == 1# Return a derivative value
        return f.EvaldF(x,f.Fdata)
    end
end

# Test functions for constructing an NLFunc
Ad = rand(10)
A  = Diagonal(Ad)

quad_func(x::Array{Float64,1})  = 0.5*x'*A*x
dquad_func(x::Array{Float64,1}) = A*x

quad_func_A(x::Array{Float64,1},
            A::Union{Matrix{Float64},Diagonal{Float64,Array{Float64,1}}}) = 0.5*x'*A*x
dquad_func_A(x::Array{Float64,1},
             A::Union{Matrix{Float64},Diagonal{Float64,Array{Float64,1}}}) = A*x

quad_func_A(rand(10),A)
dquad_func_A(rand(10),A)

# Construct type NLFunc
nlfunc_A = NLFunc(A,quad_func,dquad_func,quad_func_A,dquad_func_A)

x = rand(10)
nlfunc_A.Fdata
nlfunc_A.F(x)
nlfunc_A.dF(x)
nlfunc_A(x,0)
norm(nlfunc_A(x,0)- nlfunc_A.F(x))
norm(nlfunc_A(x,1)-nlfunc_A.dF(x))

# Change input matrix:
Bd = rand(10)
B  = Diagonal(Bd)
nlfunc_A.Fdata = B
nlfunc_A.F(x)
nlfunc_A.dF(x)
norm(nlfunc_A(x,0)- nlfunc_A.F(x))
norm(nlfunc_A(x,1)-nlfunc_A.dF(x))


################################################################################
# Approximating Derivatives 1D
################################################################################
# This section contains simple functions for approximating the derivative of
# a given function using forward and central differences. As is well-known, the
# central difference formula is more accurate for larger stepsizes. For very
# small step sizes, the error grows significantly for both approaches.

# We will plot the results, compare the two schemes and save the plots as pdfs.

# A function for computing forward differences
function FD(x::Real,h::Real,f::Function)
  return (f(x+h) - f(x))/h
end

# Defines the function f(x) = sin(x)
test_func(x) = sin(x::Real)

# Define the true derivative of test_func
deriv_tf(x) = cos(x::Real)

# This takes the base point x, initial step size h, and tolerance tol, along
# with an arbitrary function f and prints to the absolute error
function loop_error_FD(x::Real,h::Real,tol::Real,
                       f::Function,df::Function)
    println("***")
    err_plt = zeros(20)
    i = 0
    while h > tol
        i += 1
        f_prime    = FD(x,h,f)
        true_error = abs(f_prime - df(x))
        err_plt[i] = true_error
        println(true_error, " h = ", h)
        h = h/10.0
    end
    err_plt = err_plt[1:i]
    return err_plt
    println("***")
end

# Test the forward difference approximation
errors_fd = loop_error_FD(2,1,1e-12,test_func,deriv_tf)

# Plot the results
theme(:wong)            # Sets the plot theme to wong

# Make a grid for the x-axis
h_grid = zeros(length(errors_fd))
for i in 1:length(errors_fd)
    h_grid[i] = 1/10^(i-1)
end
h_grid

p1 = plot(h_grid,errors_fd,
          titlefont = font(18, "Helvetica"),
          title= "Error Plot: Forward Differences",
          xlabel="\$h\$",
          ylabel="\$|d_h f(x)-f'(x)|\$",
          titlefontsize = 24,
          label = false,#["abs_err"],
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=24,
          xaxis=:log,
          yaxis=:log,
          legend=true,
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          # legendfont = font(14, "Helvetica"),
          lw=3)
savefig(p1,"errors_fd.pdf")

# A function for computing central differences
function CD(x::Real,h::Real,f::Function)
    return (f(x+h) - f(x-h))/(2*h)
end

function loop_error_CD(x::Real,h::Real,tol::Real,
                       f::Function,df::Function)
    println("***")
    err_plt = zeros(20)
    i = 0
    while h > tol
        i += 1
        f_prime    = CD(x,h,f)
        true_error = abs(f_prime - df(x))
        err_plt[i] = true_error
        println(true_error, " h = ", h)
        h = h/10.0
    end
    err_plt = err_plt[1:i]
    return err_plt
    println("***")
end

# Test the central difference approximation
errors_cd = loop_error_CD(2,1,1e-12,test_func,deriv_tf)

# Plot the results for just CD
p2 = plot(h_grid,errors_cd,
          titlefont = font(18, "Helvetica"),
          title= "Error Plot: Central Differences",
          xlabel="\$h\$",
          ylabel="\$|d_h f(x)-f'(x)|\$",
          titlefontsize = 24,
          label = false,
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=24,
          xaxis=:log,
          yaxis=:log,
          legend=true,
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          # legendfont = font(14, "Helvetica"),
          lw=3)
savefig(p2,"errors_cd.pdf")

# Compare the results for FD vs. CD
linetypes = [:FD :CD]
styles = filter((s->begin
                s in Plots.supported_styles()
            end), [:dot, :dashdot])
styles = reshape(styles, 1, length(styles))

p3 = plot(h_grid,[errors_fd,errors_cd],
          titlefont = font(18, "Helvetica"),
          title= "Error Plot: Central Differences",
          xlabel="\$h\$",
          ylabel="\$|d_h f(x)-f'(x)|\$",
          label = ["FD" "CD"],
          bottom_margin=10mm,
          left_margin=14mm,
          right_margin=14mm,
          xtickfontsize=10,
          ytickfontsize=10,
          yguidefontsize=18,
          xguidefontsize=20,
          xaxis=:log,
          yaxis=:log,
          legend = :topleft,
          line=(1.0, styles),
          xtickfont = font(12, "Helvetica"),
          ytickfont = font(12, "Helvetica"),
          legendfont = font(8, "Helvetica"),
          lw=3)
savefig(p3,"errors_fd_vs_cd.pdf")

#=

Part II: Putting it all together

=#
mutable struct NwtFunc
        F::Function
       dF::Function
end

function (f::NwtFunc)(x::Float64,tog::Int64)
    if tog == 0 # Return a function value
        return f.F(x)
    else tog == 1# Return a derivative value
        return f.dF(x,1e-6,f.F)
    end
end

function nwtn_1D_a(x_0::Float64,tol::Float64,f::NwtFunc)
    iter = 0
    mxit = 50
    x    = copy(x_0)
    res = abs(f(x,0))
    println("--------------")
    println("residual |f(x)| = " , res)
    while res > tol && abs(f(x,1)) != 0.0 && iter < mxit
        iter += 1
        x     = x_0 - f(x_0,0)/f(x_0,1)
        res   = abs(f(x,0))
        x_0   = x
        println("residual |f(x)| = " , res)
    end
    println("--------------")
    return x, iter
end

# Examples

# Comparison to Newton with true derivative
test_func(x::Real) = sin(x)
inpt_func = NwtFunc(test_func,CD)
nwtn_1D(0.5,1e-8,test_func,deriv_tf)
nwtn_1D_a(0.5,1e-8,inpt_func)

# Example from the lecture notes
test_func(x::Real) = (exp(x) - exp(-x))/(exp(x)+exp(-x))
inpt_func = NwtFunc(test_func,CD)
nwtn_1D_a(1.0,1e-8,inpt_func) #x_0 = 1.0 converges, 1.1 diverges

# Example from the lecture notes for converge rates
test_func(x::Real) = x^4 - 7.0*x^3 + 17.0*x^2 - 17.0*x + 6.0
inpt_func = NwtFunc(test_func,CD)
nwtn_1D_a(1.1,1e-8,inpt_func) #x_0 = 1.1 converges linearly to 1.0
nwtn_1D_a(2.1,1e-8,inpt_func) #x_0 = 2.1 converges quadratically to 2.0
nwtn_1D_a(4.0,1e-8,inpt_func) #x_0 = 2.1 converges quadratically to 3.0

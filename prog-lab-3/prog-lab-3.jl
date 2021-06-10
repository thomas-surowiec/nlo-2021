using Plots; pyplot()
using LinearAlgebra


################################################################################
################################################################################
################################################################################
# Optimization Algorithms
################################################################################
################################################################################
################################################################################

################################################################################
# A simple armijo line search with backtracking strategy
################################################################################
function armijo_bt(x::Array{Float64,1},p::Array{Float64,1},
                   f::Function,df::Function,
                   s::Float64,b::Float64,mu::Float64,maxit::Int64)
    a  = s
    it = 0

    f_0  = f(x)
    df_0 = df(x)
    dftp = dot(df_0,p)

    x_1 = x + a*p
    f_1 = f(x_1)

    armres = f_1 - f_0 - mu*a*dftp
    while armres > 0 && maxit > it
        it += 1
        a  *= b

        x_1 = x + a*p
        f_1 = f(x_1)

        armres = f_1 - f_0 - mu*a*dftp
    end
    return x_1, a
end

function armijo_bt2(x::Array{Float64,1},p::Array{Float64,1},
                   f::Function,df::Function,
                   s::Float64,b::Float64,mu::Float64,maxit::Int64)
    a  = s
    it = 0

    f_0  = f(x)
    df_0 = df(x)
    dftp = dot(df_0,p)

    x_1 = x + a*p
    f_1 = f(x_1)

    armres = f_1 - f_0 - mu*a*dftp
    while armres <= 0 && maxit > it
        it += 1
        a  *= 1/b

        x_1 = x + a*p
        f_1 = f(x_1)

        armres = f_1 - f_0 - mu*a*dftp
        if armres > 0
            a *= b
            x_1 = x + a*p
            return x_1, a
        end
    end

    while armres > 0 && maxit > it
        it += 1
        a  *= b

        x_1 = x + a*p
        f_1 = f(x_1)

        armres = f_1 - f_0 - mu*a*dftp
    end
    return x_1, a
end

################################################################################
# Steepest descent with Armijo (allowing step size increases)
################################################################################
function steepest_descent(x::Array{Float64,1},
                          f::Function,df::Function,
                          atol::Float64,rtol::Float64,mxto::Int64,
                          s::Float64,b::Float64,mu::Float64,mxti::Int64)
    res = norm(df(x))
    r_0 = copy(res)
    it  = 0

    while res > rtol*r_0 + atol && it < mxto
        it += 1
        p   = -df(x)
        # x,s = armijo_bt(x,p,f,df,s,b,mu,mxto)
        x,s = armijo_bt2(x,p,f,df,s,b,mu,mxto)
        res = norm(df(x))

        println("f(x) = ", f(x))
        println("||∇f(x)|| = ", res)
    end

    return x
end

################################################################################
# Newton's method with line search
################################################################################
function  newton_ls(x::Array{Float64,1},
                    f::Function,df::Function,d2::Function,
                    atol::Float64,rtol::Float64,mxto::Int64,
                    s::Float64,b::Float64,mu::Float64,mxti::Int64)

    res = norm(df(x))
    r_0 = copy(res)
    it  = 0

    while res > rtol*r_0 + atol && it < mxto
        it += 1
        H   = d2f(x)
        p   = -H\df(x)
        # x,s = armijo_bt(x,p,f,df,s,b,mu,mxto)
        x,s = armijo_bt2(x,p,f,df,s,b,mu,mxto)
        res = norm(df(x))

        println("f(x) = ", f(x))
        println("||∇f(x)|| = ", res)
    end

    return x
end
################################################################################
# Tests for steepest_descent
################################################################################
test_x = 5*rand(2)
steepest_descent(test_x,rosenbrock,d_rb,1e-4,1e-4,20000,1.0,0.5,0.01,20)
steepest_descent(test_x,beale,d_ble,1e-4,1e-4,5000,1.0,0.5,0.01,20)
steepest_descent(test_x,rastrigin,d_ras,1e-4,1e-4,5000,1.0,0.5,0.01,20)
steepest_descent(test_x,goldstein_price,d_gp,1e-4,1e-4,5000,1.0,0.5,0.01,20)
steepest_descent(test_x,booth,d_bth,1e-4,1e-4,5000,1.0,0.5,0.01,20)

################################################################################
# Test Functions
################################################################################
# The Rosenbrock function. Global Minimizers at (1,1,1...,1)
function rosenbrock(x::Array{Float64,1})
    f = 0.0
    n = length(x)
    for i in 1:(n-1)
        f += 100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2
    end
    return f
end

# Surface plot of Rosenbrock in 2D
x_grid = range(-2,stop=2,length=100)
y_grid = range(-1,stop=3,length=100)
rb_2D(x,y) = rosenbrock([x,y])

my_grad = cgrad(:roma, 1000, categorical = true, scale = :exp)
plot(x_grid,y_grid,rb_2D,st=:surface,c=my_grad,camera=(-30,30))
contour(x_grid,y_grid,rb_2D,levels=1000)

# Approximate gradient of Rosenbrock
d_rb(x::Array{Float64,1}) = CD_grad(x,1e-6,rosenbrock)

# Approximate Hessian of Rosenbrock
d2_rb(x::Array{Float64,1}) = CD_hess(x,1e-6,rosenbrock)
d2_rb(rand(2))
################################################################################
# The Beale function. Global Minimizer (3,0.5)
function beale(x::Array{Float64,1})
    return (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1] + x[1]*x[2]^2)^2 +
            (2.625 - x[1] + x[1]*x[2]^3)^2
end

# Surface plot of Beale
x_grid=range(-4,stop=4,length=100)
y_grid=range(-4,stop=4,length=100)
beale_2D(x,y) = (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2

my_grad = cgrad(:roma, 1000, categorical = true, scale = :exp)
plot(x_grid,y_grid,beale_2D,st=:surface,c=my_grad,camera=(-30,30))
contour(x_grid,y_grid,beale_2D,levels=500)

# Approximate gradient of Beale
d_ble(x::Array{Float64,1}) = CD_grad(x,1e-6,beale)
d_ble(rand(2))

# Approximate Hessian of Beale
d2_ble(x::Array{Float64,1}) = CD_hess(x,1e-6,beale)
d2_ble(rand(2))
################################################################################

################################################################################
# The Rastrigin function. Global Minimizers at (0,0,0...0)
function rastrigin(x::Array{Float64,1})
    f = 0.0
    n = length(x)
    for i in 1:n
        f += x[i]^2 - 10*cos(2*pi*x[i])
    end
    return 10*n + f
end

# Surface plot of Rastrigin
x_grid=range(-5,stop=5,length=100)
y_grid=range(-5,stop=5,length=100)
ras_2D(x,y) = 10*2 + x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y)

my_grad = cgrad(:roma, 1000, categorical = true, scale = :exp)
plot(x_grid,y_grid,ras_2D,st=:surface,c=my_grad,camera=(-30,15))
contour(x_grid,y_grid,ras_2D,levels=1000)

# Approximate gradient of Rastrigin
d_ras(x::Array{Float64,1}) = CD_grad(x,1e-6,rastrigin)
d_ras(rand(10))

# Approximate Hessian of Rastrigin
d2_ras(x::Array{Float64,1}) = CD_hess(x,1e-6,rastrigin)
println(d2_ras(rand(2)))
################################################################################

################################################################################
# The Goldstein-Price function. Global Minimizers at (0,-1)
function goldstein_price_2D(x::Float64,y::Float64)
    a = (1 + (x + y + 1)^2 * (19 - 14*x + 3*x^2 - 14*y + 6*x*y + 3*y^2))
    b = (30 + (2*x - 3*y)^2 *(18 - 32*x + 12*x^2 + 48*y - 36*x*y + 27*y^2))
    return a * b
end

function goldstein_price(x::Array{Float64,1})
    a = (1 + (x[1] + x[2] + 1)^2 * (19 - 14*x[1] + 3*x[1]^2 - 14*x[2] +
            6*x[1]*x[2] + 3*x[2]^2))
    b = (30 + (2*x[1] - 3*x[2])^2 *(18 - 32*x[1] + 12*x[1]^2 + 48*x[2] -
            36*x[1]*x[2] + 27*x[2]^2))
    return a * b
end

# Surface plot of Goldstein-Price
x_grid=range(-2,stop=2,length=100)
y_grid=range(-2,stop=2,length=100)

my_grad = cgrad(:roma, 1000, categorical = true, scale = :exp)
plot(x_grid,y_grid,goldstein_price_2D,st=:surface,c=my_grad,camera=(-30,30))
contour(x_grid,y_grid,goldstein_price_2D,levels=800)

# Approximate gradient of Goldstein-Price
d_gp(x::Array{Float64,1}) = CD_grad(x,1e-6,goldstein_price)
d_gp(rand(2))

# Approximate Hessian of Goldstein-Price
d2_gp(x::Array{Float64,1}) = CD_hess(x,1e-6,goldstein_price)
d2_gp(rand(2))
################################################################################
# The Booth function. Global Minimizers at (1,3)
function booth(x::Array{Float64,1})
    return (x[1] + 2*x[2] - 7)^2 + (2*x[1] + x[2] - 5)^2
end

# Surface plot of Booth
x_grid=range(-5,stop=5,length=100)
y_grid=range(-5,stop=5,length=100)

my_grad = cgrad(:roma, 1000, categorical = true, scale = :exp)
booth2d(x,y) = booth([x,y])
plot(x_grid,y_grid,booth2d,st=:surface,c=my_grad,camera=(-30,30))
contour(x_grid,y_grid,booth2d,levels=800)

# Approximate gradient of Booth
d_bth(x::Array{Float64,1}) = CD_grad(x,1e-6,booth)
d_bth(rand(2))

# Approximate Hessian of Booth
d2_bth(x::Array{Float64,1}) = CD_hess(x,1e-6,booth)
d2_bth(rand(2))

################################################################################
################################################################################
################################################################################
# Approximating Gradients and Hessians using Finite Differences
################################################################################
################################################################################
################################################################################
################################################################################
# A central difference function for finding partial derivatives
function CD(x::Array{Float64},d::Array{Float64},h::Float64,f::Function)
    return (f(x+d) - f(x-d))/(2*h)
end

# A function for approximating the gradient using finite differences
function CD_grad(x::Array{Float64},h::Float64,f::Function)
    n  = length(x)
    it = 1

    gradF = zeros(n)
    while it <= n
        gradF[it] = CD(x,h*(1:n .== it),h,f)
        it += 1
    end

    return gradF
end

function CD_hess(x::Array{Float64,1},h::Float64,f::Function)
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

    return 0.5*(HessF + HessF')
end

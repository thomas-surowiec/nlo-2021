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

    x_1  = x + a*p
    f_1  = f(x_1)

    armres = f_1 - f_0 - mu*a*dftp

    while armres > 0 && maxit > it
        it += 1
        a  *= b

        x_1  = x + a*p
        f_1  = f(x_1)

        armres = f_1 - f_0 - mu*a*dftp
    end
    return x_1, a
end

################################################################################
# A simple armijo line search with backtracking allowing for step lengthing
################################################################################
function armijo_bt2(x::Array{Float64,1},p::Array{Float64,1},
                   f::Function,df::Function,
                   s::Float64,b::Float64,mu::Float64,maxit::Int64)
    a  = s
    it = 0

    f_0  = f(x)
    df_0 = df(x)
    dftp = dot(df_0,p)

    x_1  = x + a*p
    f_1  = f(x_1)

    armres = f_1 - f_0 - mu*a*dftp

    # println("1: ", a)
    while armres <= 0 && maxit > it
        it += 1
        a  *= 1/b

        x_1  = x + a*p
        f_1  = f(x_1)

        armres = f_1 - f_0 - mu*a*dftp

        if armres > 0
            a  *= b
            x_1 = x + a*p
            # println("2: ", a)
            return x_1, a
        end
    end

    while armres > 0 && maxit > it
        it += 1
        a  *= b

        x_1  = x + a*p
        f_1  = f(x_1)

        armres = f_1 - f_0 - mu*a*dftp
    end
    return x_1, a
end
################################################################################

################################################################################
# Steepest descent with Armijo allowing step size increases
################################################################################
function steepest_descent(x::Array{Float64,1},f::Function,df::Function,
                          atol::Float64,rtol::Float64,mxto::Int64,
                          s::Float64,b::Float64,mu::Float64,mxti::Int64)
    res = norm(df(x))
    r_0 = copy(res)
    it  = 0

    f_vec = zeros(mxto)
    r_vec = zeros(mxto)

    f_vec[1] = f(x)
    r_vec[1] = r_0

    # println("###")
    println("f(x) = ", f_vec[1])
    # println("||∇f(x)|| = ", r_vec[1])
    println("iteration = ", it)
    # println("###")

    while res > rtol*r_0 + atol && it < mxto
        it += 1
        p   = -df(x)
        # x,s = armijo_bt(x,p,f,df,s,b,mu,mxti)
        x,s = armijo_bt2(x,p,f,df,s,b,mu,mxti)
        res = norm(df(x))

        f_vec[it] = f(x)
        r_vec[it] = res

        # println("###")
        println("f(x) = ", f_vec[it])
        # println("||∇f(x)|| = ", r_vec[it])
        println("iteration = ", it)
        # println("###")
    end
    f_vec = f_vec[1:it]
    r_vec = r_vec[1:it]

    # p = plot(f_vec)
    # display(p)

    q = plot(r_vec)
    display(q)

    return x
end

################################################################################
# Newton's method with line search
################################################################################
function newton_ls(x::Array{Float64,1},f::Function,df::Function,d2f::Function,
                   atol::Float64,rtol::Float64,mxto::Int64,
                   s::Float64,b::Float64,mu::Float64,mxti::Int64)

    res = norm(df(x))
    r_0 = copy(res)
    it  = 0

    f_vec = zeros(mxto)
    r_vec = zeros(mxto)

    f_vec[1] = f(x)
    r_vec[1] = r_0

    # println("###")
    println("f(x) = ", f_vec[1])
    # println("||∇f(x)|| = ", r_vec[1])
    println("iteration = ", it)
    # println("###")

    while res > rtol*r_0 + atol && it < mxto
        it += 1
        H   = d2f(x)
        # H   = (transpose(H) + H)/2
        p   = -H\df(x)
        # x,s = armijo_bt(x,p,f,df,s,b,mu,mxti)
        x,s = armijo_bt2(x,p,f,df,s,b,mu,mxti)
        res = norm(df(x))

        f_vec[it] = f(x)
        r_vec[it] = res

        # println("###")
        println("f(x) = ", f_vec[it])
        # println("||∇f(x)|| = ", r_vec[it])
        println("iteration = ", it)
        # println("###")
    end
    f_vec = f_vec[1:it]
    r_vec = r_vec[1:it]

    # p = plot(f_vec)
    # display(p)

    q = plot(r_vec)
    display(q)

    return x
end

################################################################################
# A Trust Region Solver
################################################################################
function quad_model(p::Array{Float64,1},g::Array{Float64,1},B::Array{Float64,2})
    phi_p  = dot(g,p)
    Bp     = B*p
    phi_p += 0.5*dot(p,Bp)
    return phi_p
end

mutable struct QuadMod
    m_k::Function
      g::Array{Float64,1}
      B::Array{Float64,2}
end

function (q::QuadMod)(p::Array{Float64,1})
    return q.m_k(p,q.g,q.B)
end

# A simple bisection method to find tau in steihaug_cg
function root_finder(p::Array{Float64,1},d::Array{Float64,1},D::Float64)
    func(tau::Float64) = norm(p + tau*d) - D
    a = 0.0
    b = 0.01
    while func(b) < 0
        b *= 2
    end

    if func(b) == 0
        # print("1")
        return b
    end

    it = 0
    while it < 2000
        c = (a + b)/2
        if abs(func(c)) < 1e-10 || (b-a)/2 < 1e-10
            # print("2", c)
            return c
        end
        if sign(func(c)) == sign(func(a))
            a = c
        else
            b = c
        end
        it += 1
    end
    # print("3")
    return 0.0
end

# p = rand(2)
# d = rand(2)
# D = norm(p+0.1*d)
# root_finder(p,d,D)

# The Steihaug CG method for the Trust Region Subproblem
function steihaug_cg(m_k::QuadMod,Delta::Float64,tol::Float64,maxit::Int64)
    n = length(m_k.g)

    p_i = zeros(n)

    r_i  = -copy(m_k.g)
    rt_i =  copy(r_i) # There is an option to use a preconditioner here.
    d_i  =  copy(rt_i)

    Bd  = (m_k.B)*d_i
    g_i = dot(d_i,Bd)

    for i in 1:maxit
        if g_i <= 0
            tau = root_finder(p_i,d_i,Delta)
            # println("Here", i)
            return p_i + tau*d_i
        else # g_i > 0
            c_i = dot(r_i,rt_i)
            a_i = c_i/g_i

            p_i = p_i + a_i*d_i

            if norm(p_i) >= Delta
                p_i = p_i - a_i*d_i
                tau = root_finder(p_i,d_i,Delta)
                # println("Here", i)
                return p_i + tau*d_i
            end

            r_i = r_i - a_i*Bd

            if norm(r_i)/norm(m_k.g) <= tol
                # println("Here", i)
                return p_i
            end

            rt_i  = copy(r_i) # There is an option to use a preconditioner here.

            b_i = dot(r_i,rt_i)/c_i
            d_i = rt_i + b_i*d_i

            Bd  = (m_k.B)*d_i
            g_i = dot(d_i,Bd)
        end
    end
    return p_i
end


function trust_region_steihaug(x::Array{Float64,1},atol::Float64,mxto::Int64,
                               mu::Float64,lam::Float64,
                               f::Function,df::Function,d2f::Function,
                               D::Float64,Dmax::Float64,cg_tol::Float64,
                               mxti::Int64,m_k::QuadMod)
    m_k.g = df(x)
    m_k.B = d2f(x)

    res  = norm(m_k.g)
    res0 = copy(res)
    it  = 0

    f_vec = zeros(mxto)
    r_vec = zeros(mxto)

    f_vec[1] = f(x)
    r_vec[1] = res

    # println("###")
    # println("f(x) = ", f_vec[1])
    println("||∇f(x)|| = ", r_vec[1])
    # println("iteration = ", it)
    # println("###")

    while res > atol && it < mxto
        p = steihaug_cg(m_k,D,cg_tol,mxti)

        f_old = f(x)
        f_new = f(x + p)
        # println("f_old - f_new = ", f_old - f_new)
        rho_k = (f_old - f_new)/(-m_k(p))
        # println("ρ = ", rho_k)
        if rho_k > lam
            x = x + p
            D *= 2
            D = min(D,Dmax)
            # println("D = ", D)
        else

            while rho_k <= lam
                p     = 0.5*p
                f_new = f(x + p)
                # println("f_old - f_new = ", f_old - f_new)
                rho_k = (f_old - f_new)/(-m_k(p))
                println("ρ = ", rho_k)
            end
            x = x + p
            D *= 0.5
            # println("D = ", D)
        end

        # np = norm(p)
        # if rho_k <= mu
        #     D *= 0.5
        #     D = max(D,0.001*np)
        #     println("np = ", np)
        #     println("D = ", D)
        #     # D  = D + max(0,0.1*np - D) - max(0,D-0.9*np)
        # elseif rho_k > mu && rho_k < lam
        #     D *= 1
        # else
        #     D *= 2
        #     D = min(D,Dmax)
        #     println("np = ", np)
        #     println("D = ", D)
        #     # D  = D + max(0,np - D) - max(0,D-min(10*np,Dmax))
        # end

        m_k.g = df(x)
        m_k.B = d2f(x)
        res = norm(df(x))
        it += 1

        f_vec[it] = f(x)
        r_vec[it] = res

        # println("###")
        println("f(x) = ", f_vec[it])
        println("||∇f(x)|| = ", r_vec[it])
        # println("iteration = ", it)
        # println("###")

        pp = plot(f_vec[1:it])
        display(pp)
    end

    # f_vec = f_vec[1:it]
    # r_vec = r_vec[1:it]
    #
    # p = plot(f_vec)
    # display(p)

    # q = plot(r_vec)
    # display(q)

    return x
end


mutable struct QuadMod
    m_k::Function
      g::Array{Float64,1}
      B::Array{Float64,2}
end

function (q::QuadMod)(p::Array{Float64,1})
    return q.m_k(p,q.g,q.B)
end

qm = QuadMod(quad_model,rand(2),rand(2,2))
test_x = 5*rand(2)

trust_region_steihaug(test_x,1e-8,20,0.25,0.75,rosenbrock,d_rb,d2_rb,
                      10.0,1000.0,1e-8,100,qm)
trust_region_steihaug(test_x,1e-8,50,0.25,0.75,beale,d_ble,d2_ble,
                      10.0,10000.0,1e-8,100,qm)
trust_region_steihaug(test_x,1e-4,30,0.25,0.75,rastrigin,d_ras,d2_ras,
                      10.0,1000.0,1e-8,100,qm)
trust_region_steihaug(test_x,1e-7,30,0.25,0.75,goldstein_price,d_gp,d2_gp,
                      10.0,1000.0,1e-8,100,qm)
trust_region_steihaug(test_x,1e-7,30,0.25,0.75,booth,d_bth,d2_bth,
                      10.0,1000.0,1e-8,100,qm)

trust_region_steihaug(rand(20),1e-4,200,0.25,0.75,rosenbrock,d_rb,d2_rb,
                      10.0,1000.0,1e-8,100,qm)
#
# trust_region_steihaug(rand(20),1e-8,30,0.25,0.75,rastrigin,d_ras,d2_ras,
#                       10.0,1000.0,1e-8,100,qm)
################################################################################

################################################################################
# Tests for steepest_descent
################################################################################
test_x = rand(2)
steepest_descent(test_x,rosenbrock,d_rb,1e-8,1e-8,20000,1.0,0.5,0.01,20)
steepest_descent(test_x,beale,d_ble,1e-6,1e-6,1000,1.0,0.5,0.01,20)
steepest_descent(test_x,rastrigin,d_ras,1e-6,1e-6,200,1.0,0.5,0.01,20)
steepest_descent(test_x,goldstein_price,d_gp,1e-6,1e-6,5000,1.0,0.5,0.01,20)
steepest_descent(test_x,booth,d_bth,1e-6,1e-6,5000,1.0,0.5,0.01,20)

steepest_descent(rand(20),rosenbrock,d_rb,1e-4,1e-4,20000,1.0,0.5,0.01,20)
################################################################################

################################################################################
# Tests for newton_ls
################################################################################
test_x = 10*rand(2)
newton_ls(test_x,rosenbrock,d_rb,d2_rb,1e-6,1e-6,4000,1.0,0.5,0.01,20)
newton_ls(test_x,beale,d_ble,d2_ble,1e-6,1e-8,20,1.0,0.5,0.01,20)
newton_ls(test_x,rastrigin,d_ras,d2_ras,1e-6,1e-8,50,1.0,0.5,0.01,20)
newton_ls(test_x,goldstein_price,d_gp,d2_gp,1e-6,1e-6,50,1.0,0.5,0.01,20)
newton_ls(test_x,booth,d_bth,d2_bth,1e-6,1e-6,50,1.0,0.5,0.01,20)
################################################################################


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

# A function for approximating the Hessians using finite differences
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
################################################################################

################################################################################
################################################################################
################################################################################
# Test functions
################################################################################
################################################################################
################################################################################

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
x_grid=range(-2,stop=2,length=100)
y_grid=range(-1,stop=3,length=100)
rb_2D(x,y) = 100*(y-x^2)^2 + (1-x)^2

my_grad = cgrad(:roma, 1000, categorical = true, scale = :exp)
plot(x_grid,y_grid,rb_2D,st=:surface,c=my_grad,camera=(-30,30))
contour(x_grid,y_grid,rb_2D,levels=1000)

# Approximate gradient of Rosenbrock
d_rb(x::Array{Float64,1}) = CD_grad(x,1e-6,rosenbrock)
d_rb(rand(10))

# Approximate Hessian of Rosenbrock
d2_rb(x::Array{Float64,1}) = CD_hess(x,1e-6,rosenbrock)
d2_rb(rand(2))
################################################################################

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
contour(x_grid,y_grid,beale_2D,levels=1000)

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

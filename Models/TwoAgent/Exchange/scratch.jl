using Bokeh; autoopen(true)

m = BCFL22C()
@time lzbar, lg = simulate_exog(m);
# κ = [0.05, 0.95, -0.1 0.5]
κ = [0.0, 1.0, 0.0]
κ0, κ1, κ2 = κ
ξ = 0.05
deg = 3
sim_data = X[1:capT-1, 2:end]
l♠, κ = main()
fstv = FullState(1.0, 2.0, 3.0, 5.0)
fst = FullState(κ, κ, κ, κ)
asarray(fst)

st = TimeTState([1,2], [3,4])

for (i, x) in enumerate(fst)
    @show i,x
end

Matrix(fst)

plot(grid[:, 1], grid[:, 2], "b*")

plot(LHS)
plot(fsts.l♠)

pf = PolicyFunction{1}(ones(5))
evaluate(pf, fst)

Matrix(fst)

Matrix{Float64}(fst)


plot(l♠)
linear_coefs(m, lzbar, lg, κ, maxiter=5)

capT = length(lzbar)
♠ = ones(capT+1)

# simulate ♠ forward and solve for sa along the way
@inbounds for t=1:capT
    # extract time t state
    ♠t = ♠[t]
    lzt = lzbar[t]

    # update ♠_{t+1}
    ♠p = b0+ b1*♠t + b2*lzt
    ♠[t+1] = ♠p
end
t
showplot(plot(1:capT, ♠[1:capT]))
showplot(plot(1:capT, lzbar))

plot(l♠)

a1 = 1-1e-15
t = 48

showplot(plot(1:length(c1)', [c1 c2], legends=["c1", "c2"]))

using StatsBase

@doc quantile


foobar(m, 0.4, 1.0)


qnwmonomial(2, eye(2), :second)

qnwgh(4, 2)[2]


extrema(exp(lzbar))
1/0.95


♠sim = ♠
zbarsim = exp(lzbar)

showplot(plot(♠sim, zbarsim, "."))
showplot(plot(eds[:,1], eds[:,2], "."))

x = linspace(0, 2pi)
y1 = sin(x)
y2 = cos(x)
y3 = tan(x)

plot(x, [y1 y2 y3], "rs|bo|g*")
showplot()


tol=1e-8
deg = 3  # 5.13 seconds
deg = 5  # 6.48 seconds
κ = κ_old
maxit = 5000
it

x = rand(2, 10)

showplot(plot(l♠))
showplot(plot(LHS))
showplot(plot([c1[1:t-1] c2[1:t-1]], legends=["c1", "c2"]))

#=
omega = 1-0.972; sigma =-2/3;

xi = 1.004;

ratio_phi = 1.9375;

b = @(a) xi/(1+((1-omega)^2/omega^2)^(1/(sigma-1))*(a/(1-a)));

c1 = @(a) ((1-omega)*a^sigma + omega*(xi-b(a))^sigma)^(1/sigma);
c2 = @(a) ((1-omega)*b(a)^sigma + omega*(1-a)^sigmaIs )^(1/sigma);
ratio_implied = @(a) c2(a)^(1-sigma)*omega*(1-a)^(sigma-1)/(c1(a)^(1-sigma)*(1-omega)*a^(sigma-1));

tol=1; alb = eps; aub = 1-eps;

while tol>1e-5

    a = (alb+aub)/2;
    foc = ratio_implied(a) - ratio_phi;
    if foc<=0
        alb = a;
    else
        aub = a;
    end
    tol = abs(foc);
    tol
    pause
end
=#
n_complete(2, 3)

pf = PolicyFunction{3}(rand(10))
@code_warntype evaluate(pf, [1.0, 2.0])

state = [1.0, 2.0]
complete_polynomial([1 3], 3)

complete_polynomial(rand(5, 2), 3)

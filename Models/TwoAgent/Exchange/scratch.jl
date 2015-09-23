using Bokeh

m = BCFL22C()
lzbar, lg = simulate_exog(m)
κ = [0.05, 0.95, -0.1]
κ0, κ1, κ2 = κ
ξ = 0.05

main()
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

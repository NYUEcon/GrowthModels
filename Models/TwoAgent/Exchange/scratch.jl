using Bokeh

m = BCFL22C()
lzbar = simulate_exog(m)
b_old = [0.00, 1.0, 0.0]  # this is the time additive solution (constnat pareto weights)
# b_old = [0.05, 0.95, 0.3]
b0, b1, b2 = b_old
ξ = 0.1

linear_coefs(m, lzbar, b_old)

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

showplot(plot(1:capT, ♠[1:capT]))
showplot(plot(1:capT, lzbar))

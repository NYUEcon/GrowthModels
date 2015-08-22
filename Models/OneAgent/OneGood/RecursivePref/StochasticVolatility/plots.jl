include("main.jl")

using JLD
using PyPlot
using CompEcon
# @load "BFZ_additive_k18-32.jld"
@load "BFZ_additive_k18-32_rho-9.0_alpha-9.0.jld"


evaluate(coefs, k::Vector, x::Vector, v::Vector, order=[0 0 0]) =
    funeval(coefs, m.basis, [k x v], order)

evaluate(coefs, k::Vector, x::Real, v::Real, order=[0 0 0]) =
    funeval(coefs, m.basis, [k fill(x, length(k)) fill(v, length(k))], order)

evaluate(coefs, k::Real, x::Vector, v::Vector, order=[0 0 0]) =
    funeval(coefs, m.basis, [fill(k, length(x)) x fill(v, length(x))], order)

evaluate(coefs, k::Real, x::Real, v::Vector, order=[0 0 0]) =
    funeval(coefs, m.basis, [fill(k, length(v)) fill(x, length(v)) v], order)

function plot(coefs; trans=(x,y)->(x.^(m.agent.ρ-1).*y),
              lab=L"log(J^{\rho-1}J_k)", fn_suffix=1)
    grid, (kgrid, xgrid, vgrid) = nodes(m.basis)
    fig, ax = subplots()

    xbar = mean(xgrid); vbar = mean(vgrid)

    f = evaluate(coefs, kgrid, xbar, vbar)
    f_k = eval_J(kgrid, xbar, vbar, 1)

    y = log(trans(f, f_k))
    x = log(kgrid)

    ax[:plot](x, y)
    ax[:set_xlabel]("log(k)")
    ax[:set_ylabel](lab)

    approx_slope = (y[end] - y[1]) / (x[end] - x[1])
    ax[:set_title]("slope: $approx_slope")


    fig[:savefig]("figures/rho_alpha_m9/plot$(fn_suffix).png")

end


function plot2()
    grid, (kgrid, xgrid, vgrid) = nodes(m.basis)
    fig, ax = subplots()

    xbar = mean(xgrid); vbar = mean(vgrid)

    J = eval_J(kgrid, xbar, vbar)
    J_k = eval_J(kgrid, xbar, vbar)

    y = log(J.^(m.agent.ρ - 1) .* J_k)
    x = log(kgrid)

    ax[:plot](x, y)
    ax[:set_xlabel]("log(k)")
    ax[:set_ylabel](L"log(J^{\rho - 1} J_k)")

    approx_slope = (y[end] - y[1]) / (x[end] - x[1])
    ax[:set_title]("slope: $approx_slope")

    fig[:savefig]("figures/rho_alpham9/plot.png")

end

function main()
    coefs_c = get_coefs(m.basis, m.basis_struc, c)
    plot(coefs, fn_suffix="1")
    plot(coefs, trans=(x,y)->x, lab=L"log(J)", fn_suffix="lJ")
    plot(coefs, trans=(x,y)->y, lab=L"log(J_k)", fn_suffix="lJk")
    plot(coefs_c, trans=(x,y)->x, lab=L"log(c)", fn_suffix="lc")
end
include("main.jl")

using JLD
using PyPlot
using CompEcon

evaluate(coefs, k::Vector, x::Vector, v::Vector, order=[0 0 0]) =
    funeval(coefs, m.basis, [k x v], order)

evaluate(coefs, k::Vector, x::Real, v::Real, order=[0 0 0]) =
    funeval(coefs, m.basis, [k fill(x, length(k)) fill(v, length(k))], order)

evaluate(coefs, k::Real, x::Vector, v::Real, order=[0 0 0]) =
    funeval(coefs, m.basis, [fill(k, length(x)) x fill(v, length(x))], order)

evaluate(coefs, k::Real, x::Real, v::Vector, order=[0 0 0]) =
    funeval(coefs, m.basis, [fill(k, length(v)) fill(x, length(v)) v], order)

add_params!(ax, ρ, α, β) =
    ax[:annotate](s="rho:$(ρ)\nalpha:$(α)\nbeta:$(β)", xy=(0.05, 0.85),
                  xycoords="axes fraction")

function plot(m::X.BFZ, var::Symbol; trans=(x,y,z)->(x.^(m.agent.ρ-1).*y),
              lab=L"log(J^{\rho-1}J_k)", fn_suffix=1)
    j_coefs, j, c, kp, grp_key = X.solve(m, false)

    grid, (kgrid, xgrid, vgrid) = nodes(m.basis)
    fig, ax = subplots()

    k, x, v = var == :k ? (kgrid, mean(xgrid), mean(vgrid)) :
              var == :x ? (mean(kgrid), xgrid, mean(vgrid)) :
              var == :v ? (mean(kgrid), mean(xgrid), vgrid) :
              error("var must be one of `:k`, `:x`, or `:v`")

    j = evaluate(j_coefs, k, x, v, [0 0 0])
    j_k = evaluate(j_coefs, k, x, v, [1 0 0])

    # get c
    c_coefs = get_coefs(m.basis, m.basis_struc, c)
    c = evaluate(c_coefs, k, x, v, [0 0 0])

    y = log(trans(j, j_k, c))

    x = var == :k ? log(kgrid) : var == :x ? xgrid : var == :v ? vgrid :
               error("var must be one of `:k`, `:x`, or `:v`")

    ax[:plot](x, y)
    ax[:set_xlabel](var ==:k ? "log(k)" : "$(var)")
    ax[:set_ylabel](lab)

    approx_slope = (y[end] - y[1]) / (x[end] - x[1])
    ax[:set_title]("slope: $approx_slope")

    add_params!(ax, m.agent.ρ, m.agent.α, m.agent.β)

    fig[:savefig]("figures/model_$(grp_key)/plot_$(var)_$(fn_suffix).png")

end

function plot_kapitalratio(m::X.BFZ)
    j_coefs, j, c, kp, grp_key = X.solve(m, false)

    # Take the solution to the model and get relevant info
    c_coefs, x, v, c_sim, k_sim, y_sim = X.simulate_solution(m, j_coefs, c, T=5000)

    fig, ax = subplots()
    ax[:hist](k_sim[1:end-1] ./ y_sim)
    ax[:set_title]("Capital to Income Ratio")
    add_params!(ax, m.agent.ρ, m.agent.α, m.agent.β)
    fig[:savefig]("figures/model_$(grp_key)/capital_output.png")

end

function main(m::X.BFZ)
    grp_key = X.solve(m, false)[end];
    mkpath("figures/model_$(grp_key)")
    for var in (:k, :x, :v)
        plot(m, var, fn_suffix="1")
        plot(m, var, trans=(x,y,z)->x, lab=L"log(J)", fn_suffix="lJ")
        plot(m, var, trans=(x,y,z)->y, lab=L"log(J_k)", fn_suffix="lJk")
        plot(m, var, trans=(x,y,z)->z, lab=L"log(c)", fn_suffix="lc")
    end
    plot_kapitalratio(m)
end

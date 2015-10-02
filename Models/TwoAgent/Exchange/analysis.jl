# ------------------------------------------------------------------- #
# Analysis of stuff
# ------------------------------------------------------------------- #
using PyPlot

function plot_allocations(m::BCFL22C, allocations::Array{Float64, 2})
    # Create Plot Object
    fig, ax = subplots(3, 1)
    capT = size(allocations, 2)
    tTt = collect(range(1, capT))

    # Plot (a1, b2)
    ax[1][:plot](tTt, allocations[1, :][:], label="a1")
    ax[1][:plot](tTt, allocations[4, :][:], label="b2")
    ax[1][:set_title]("Home Goods")
    ax[1][:legend]()

    # Plot (a2, b1)
    ax[2][:plot](tTt, allocations[2, :][:], label="a2")
    ax[2][:plot](tTt, allocations[3, :][:], label="b1")
    ax[2][:set_title]("Foreign Goods")
    ax[2][:legend]()

    # Plot (c1, c2)
    ax[3][:plot](tTt, allocations[5, :][:], label="c1")
    ax[3][:plot](tTt, allocations[6, :][:], label="c2")
    ax[3][:set_title]("Consumption")
    ax[3][:legend]()

    fig[:tight_layout]()

    return fig, ax
end

"""
Simulate the economy and compute the allocations

NOTE: The `fsts` argument should have been initialized so that `l♠` field has
its first element set to the deisred initial condition. Also the `lzbar`,
`lzbarp`, and `lgp` fields should be set to include the entire simulation path.

The elements `fsts.l♠[2:end]` will be overwritten by this function
"""
function simulate_allocations!(m::BCFL22C, pf::PolicyFunction, fsts::FullState)

    # Allocate Space
    capT = length(fsts) - 1
    allocations = Array(Float64, 6, capT)

    for t=1:capT
        # extract time t state
        fst = fsts[t]
        # update ♠_{t+1}
        l♠p = evaluate(pf, fst)
        fsts.l♠[t+1] = l♠p

        # compute allocation
        alloc = get_allocation(m, fst)
        for (i, x) in enumerate(alloc)
            allocations[i, t] = x
        end

    end

    return allocations, fsts
end

function impulse_response(m::BCFL22C, pf::PolicyFunction, capT::Int)

    # Create a one standard deviation shock for each epsilon
    shock_at_1 = squeeze(eye(1, capT+1), 1)
    zero_shock = zeros(shock_at_1)

    # Simulate Exogenous processes
    lzbar_1, lg_1 = simulate_exog(m, shock_at_1, zero_shock)

    fsts_1 = FullState(zeros(capT+1), lzbar_1[1:capT+1],
                     [lzbar_1[2:capT+1]; NaN], [lg_1[2:capT+1]; NaN])

    lzbar_2, lg_2 = simulate_exog(m, zero_shock, shock_at_1)

    fsts_2 = FullState(zeros(capT+1), lzbar_2[1:capT+1],
                       [lzbar_2[2:capT+1]; NaN], [lg_2[2:capT+1]; NaN])

    # Get allocations
    allocations_1, fsts_1 = simulate_allocations!(m, pf, fsts_1)
    allocations_2, fsts_2 = simulate_allocations!(m, pf, fsts_2)

    # Plot allocations
    fig_1, ax_1 = plot_allocations(m, allocations_1)
    fig_1[:suptitle]("Impulse Response to e1")
    show()

    fig_2, ax_2 = plot_allocations(m, allocations_2)
    fig_2[:suptitle]("Impulse Response to e2")
    show()

    return allocations_1, allocations_2
end

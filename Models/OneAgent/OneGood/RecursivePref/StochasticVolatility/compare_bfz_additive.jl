using JLD
include("main.jl")

# read in solution. Defines m, vf, c, kp, coefs
@load "BFZ_additive.jld"

function get_J_coefs_Jk(m::X.BFZ, vf)
    ρ, α, β = X._unpack(m.agent)
    J = vf.^(ρ) ./ ρ
    J_coefs = X.get_coefs(m.basis, m.basis_struc, J)
    J_k = X.funeval(J_coefs, m.basis, m.grid, [1 0 0])
    J, J_coefs, J_k
end

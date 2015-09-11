include("common.jl")
bcfl =  TwoAgents.BCFL21()

for i in workers()
    remotecall_fetch(i, include, "common.jl")
end

TwoAgents.brutal_solution(bcfl; maxiter=1000)

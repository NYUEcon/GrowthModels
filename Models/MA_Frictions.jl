# ------------------------------------------------------------------- #
# Adjustment Cost
# ------------------------------------------------------------------- #
abstract AbstractAdjustmentCost

immutable AdjCost
    ikbar::Float64
    η::Float64
end

_unpack(ac::AdjCost) = (ac.ikbar, ac.η)

_ac(ac::AdjCost, it, kt) =
    ((ac.ikbar)^(1 - ac.η) * (it/kt)^(ac.η) - (1 - ac.η)*(ac.ikbar)) / ac.η

_dIac(ac::AdjCost, it, kt) =
    ((ac.ikbar)^(1 - ac.η) * it^(ac.η - 1)*kt^(-ac.η))

_dkac(ac::AdjCost, it, kt) =
    -((ac.ikbar)^(1 - ac.η) * it^(ac.η)*kt^(-ac.η-1))

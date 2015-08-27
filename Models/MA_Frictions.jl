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
    ((ac.ikbar)^(1 - ac.η) * (it/kt)^(ac.η) - (1 - ac.η)*(ikbar)) / ac.η

_dIac(ac::AdjCost, it, kt) =
    ((ac.ikbar)^(1 - ac.η) * (it/kt)^(ac.η - 1))

_dkac(ac::AdjCost, it, kt) =
    ((ac.ikbar)^(1 - ac.η) * (kt/it)^(-ac.η - 1.))  # TODO: Double check this later

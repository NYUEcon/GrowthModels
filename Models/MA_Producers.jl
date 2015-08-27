# ------------------------------------------------------------------- #
# CES Producer
# ------------------------------------------------------------------- #
immutable CESProducer <: AbstractProducer
    η::Float64
    ν::Float64
    δ::Float64
end

_production(p::CESProducer, k) = (p.η * k.^p.ν + (1 - p.η)).^(1./p.ν)
f_k(p::CESProducer, k) = _production(p, k).^(1. - p.ν) .* p.η .* k.^(ν - 1)

# ------------------------------------------------------------------- #
# Cobb-Douglas Producer
# ------------------------------------------------------------------- #
immutable CobbDouglassProducer <: AbstractProducer
    α_k::Float64
    α_l::Float64
end

is_crs(i::CobbDouglassProducer) = i.α_k == 1. - i.α_l

# ------------------------------------------------------------------- #
# CES Producer
# ------------------------------------------------------------------- #
immutable CESProducer <: AbstractProducer
    η::Float64
    ν::Float64
end

# call(p::CESProducer, k, l) = (p.η*k.^p.ν + (1-p.η)*l.^(p.ν)).^(1./p.ν)
Base.produce(p::CESProducer, k, l) = (p.η*k.^p.ν + (1-p.η)*l.^(p.ν)).^(1./p.ν)
f_k(p::CESProducer, k, l) = p.η * (produce(p, k, l) ./ k).^(1.-p.ν)

# ------------------------------------------------------------------- #
# Cobb-Douglas Producer
# ------------------------------------------------------------------- #
immutable CobbDouglassProducer <: AbstractProducer
    α_k::Float64
    α_l::Float64
end

is_crs(i::CobbDouglassProducer) = i.α_k == 1. - i.α_l

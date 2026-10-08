using ComradeBase: DomainParams

function plane_coords(N, planes)
    U_vals = range(-10.0e9, 10.0e9; length = N)
    U_final = vec(U_vals' .* ones(N))
    V_final = vec(ones(N)' .* U_vals)
    sz = (length(U_final), map(length, planes)...)
    U = similar(U_final, sz)
    V = similar(V_final, sz)
    for I in CartesianIndices(Base.tail(sz))
        U[:, I] .= U_final .* sum(Tuple(I))
        V[:, I] .= V_final .+ prod(Tuple(I))
    end
    return U, V
end

# Selecting a plane by its Ti and Fr values gives the (Pt,) domain of that plane.
function planes_match(N, planes)
    U, V = plane_coords(N, planes)
    npt = size(U, 1)
    d = ComradeBase.StructuredDomain((ComradeBase.Pt(npt), planes...); U, V)
    img = IntensityMap(zeros(size(d)), d)
    C = true
    for I in CartesianIndices(map(length, planes))
        sel = map((p, i) -> DD.rebuild(p, DD.At(parent(p)[i])), planes, Tuple(I))
        for order in (sel, reverse(sel))
            sub = img[order...]
            g = UnstructuredDomain((U = U[:, I], V = V[:, I]))
            C = C && (domainpoints(axisdims(sub)) == domainpoints(g))
        end
    end
    return C
end

@testset "Test plane selection for visdomain" begin
    ti = Ti(sort(10 * rand(10)))
    fr = Fr(sort(1.0e11 * rand(4)))
    @test planes_match(8, (ti, fr))
    @test planes_match(8, (ti,))
    @test planes_match(8, (fr,))
end

struct FrPower{T} <: DomainParams{T}
    α::T
    ν0::T
end
ComradeBase.paramfield(m::FrPower, p) = (p.Fr / m.ν0)^m.α
ComradeBase.apply_param(base, ::FrPower, f, p) = Base.broadcasted(*, base, f)
ComradeBase.restrict_params(m::FrPower, ix, iy) = m
ComradeBase.stokes(m::FrPower, v) = m

struct Offset{T, A} <: DomainParams{T}
    off::A
end
Offset(off) = Offset{ComradeBase.paramtype(typeof(off)), typeof(off)}(off)
ComradeBase.apply_param(base, m::Offset, _, p) = Base.broadcasted(+, base, m.off)
ComradeBase.restrict_params(m::Offset, ix, iy) = Offset(ComradeBase.restrict_params(m.off, ix, iy))
ComradeBase.stokes(m::Offset, v) = Offset(stokes(m.off, v))

struct Unrestricted{T, A} <: DomainParams{T}
    off::A
end

@testset "MultiDomainParams" begin
    ν0 = 230.0e9
    p = (; Fr = 2ν0)
    base = [1.0 2.0; 3.0 4.0]
    base_orig = copy(base)

    md = MultiDomainParams(base, FrPower(1.0, ν0), Offset(10.0))
    @test md isa DomainParams{Float64}
    @test ComradeBase.paramtype(typeof(md)) === Float64
    out = @inferred build_param(md, p)
    @test out ≈ 2 .* base .+ 10
    @test build_param(MultiDomainParams(base, Offset(10.0), FrPower(1.0, ν0)), p) ≈ 2 .* (base .+ 10)
    @test base == base_orig
    out[1, 1] = -999.0
    @test base == base_orig

    @test @inferred(build_param(MultiDomainParams(5.0, FrPower(1.0, ν0), Offset(1.0)), p)) === 11.0
    @test getparam((; a = md, b = 3.0), :a, p) ≈ 2 .* base .+ 10

    m1, m2 = FrPower(1.0, ν0), Offset(10.0)
    @test MultiDomainParams(MultiDomainParams(base, m1), m2) === MultiDomainParams(base, m1, m2)
    @test_throws "a `MultiDomainParams` cannot be a model in another chain" MultiDomainParams(base, MultiDomainParams(base, m1))
    @test_throws "transforms a base value and has none of its own" build_param(m1, p)
    @test @inferred(build_param((MultiDomainParams(5.0, m2), 2.0), p)) === (15.0, 2.0)
    @test getparam((; s = (MultiDomainParams(5.0, m2), 2.0)), :s, p) === (15.0, 2.0)

    @test startswith(sprint(show, md), "MultiDomainParams(2×2 Matrix{Float64}, ")
end

@testset "MultiDomainParams with polarized values" begin
    ν0 = 230.0e9
    p = (; Fr = 2ν0)
    s = StokesParams(1.0, 0.1, 0.2, 0.05)
    md = MultiDomainParams(s, FrPower(1.0, ν0))
    @test ComradeBase.paramtype(typeof(md)) === StokesParams{Float64}
    @test @inferred(build_param(md, p)) ≈ 2 .* s
    @test ComradeBase.paramtype(typeof(MultiDomainParams(1.0, Offset(s)))) === StokesParams{Float64}
    mdq = stokes(MultiDomainParams(s, Offset(s)), :Q)
    @test build_param(mdq, p) ≈ 0.2
end

@testset "restrict_params" begin
    ν0 = 230.0e9
    p = (; Fr = 2ν0)
    base = rand(8, 8)
    off = rand(8, 8)
    md = MultiDomainParams(base, FrPower(1.5, ν0), Offset(off))
    ix, iy = 2:4, 3:5
    sub = ComradeBase.restrict_params(md, ix, iy)
    @test sub.base == view(base, ix, iy)
    @test sub.models[2].off == view(off, ix, iy)
    @test sub.models[1] === md.models[1]
    @test build_param(sub, p) ≈ build_param(md, p)[ix, iy]
    @test ComradeBase.restrict_params(2.0, ix, iy) === 2.0
    @test_throws "Unrestricted does not define `restrict_params(param, ix, iy)`" ComradeBase.restrict_params(
        MultiDomainParams(base, Unrestricted{Float64, Matrix{Float64}}(off)), ix, iy
    )
end

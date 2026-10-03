using ComradeBase: StructuredDomain, Pt
using Adapt

struct PolTest{T} <: ComradeBase.AbstractModel
    size::T
end

ComradeBase.visanalytic(::Type{<:PolTest}) = ComradeBase.IsAnalytic()
ComradeBase.imanalytic(::Type{<:PolTest}) = ComradeBase.IsAnalytic()
ComradeBase.ispolarized(::Type{<:PolTest}) = ComradeBase.IsPolarized()

function ComradeBase.visibility_point(m::PolTest, p)
    v = exp(-2π^2 * m.size^2 * (p.U^2 + p.V^2))
    return StokesParams(complex(v), complex(v / 10), complex(v * p.U / 5), complex(v / 20))
end

function ComradeBase.intensity_point(m::PolTest, p)
    i = exp(-(p.X^2 + p.Y^2) / (2 * m.size^2))
    return StokesParams(i, i / 10, i * p.X / 5, i / 20)
end

struct DoubleFloats end
Adapt.adapt_storage(::DoubleFloats, x::AbstractArray{<:AbstractFloat}) = 2 .* x

function structured_executors()
    exs = Any[
        Serial(), ThreadsEx(), ThreadsEx(:static), ThreadsEx(:Enzyme), ThreadsEx(:Polyester),
        DynamicScheduler(), StaticScheduler(), SerialScheduler(), CPU(),
    ]
    VERSION ≥ v"1.11" && push!(exs, ThreadsEx(:greedy))
    return exs
end

pointref(f, m, d) = map(p -> f(m, p), domainpoints(d))

asstorage(ref::AbstractArray{<:Number}) = ref
asstorage(ref::AbstractArray{<:StokesParams}) = cat(ntuple(k -> getindex.(ref, k), 4)...; dims = ndims(ref) + 1)

function test_pointmaps(mapfn, mapfn!, pointfn, m, d)
    ref = pointref(pointfn, m, d)
    for ex in structured_executors()
        dex = DD.rebuild(d; executor = ex)
        out = mapfn(m, dex)
        @test out isa IntensityMap
        @test axisdims(out) === dex
        @test collect(baseimage(out)) ≈ asstorage(ref)
        fill!(baseimage(out), zero(eltype(out)))
        mapfn!(out, m)
        @test collect(baseimage(out)) ≈ asstorage(ref)
    end
    return nothing
end

@testset "analytic maps over StructuredDomain" begin
    npt = 11
    ti = [0.0, 1.0, 2.0]
    fr = [230.0e9, 345.0e9]
    U = 0.1 .* randn(npt)
    V = 0.1 .* randn(npt)
    u = 3.0e4 .* randn(npt)
    v = 3.0e4 .* randn(npt)
    U3 = 0.1 .* randn(npt, 3, 2)
    V3 = 0.1 .* randn(npt, 3, 2)
    d1 = StructuredDomain((; U, V))
    d2 = StructuredDomain((Pt(npt), Fr(fr)); u, v)
    d3 = StructuredDomain((Pt(npt), Ti(ti), Fr(fr)); U = U3, V = V3)
    dxy = StructuredDomain((X = randn(npt), Y = randn(npt)))

    @testset "visibilitymap $(nameof(typeof(m))) on $(keys(d))" for m in (GaussTest(), BlobTest(2.0)), d in (d1, d2, d3)
        test_pointmaps(visibilitymap, visibilitymap!, ComradeBase.visibility_point, m, d)
    end

    @testset "(Pt, Fr) converts meters to wavelengths" begin
        vis = visibilitymap(BlobTest(2.0), d2)
        c = ComradeBase.speed_of_light
        @test vis[2, 2] ≈ ComradeBase.visibility_point(BlobTest(2.0), (U = u[2] * fr[2] / c, V = v[2] * fr[2] / c))
    end

    @testset "intensitymap $(nameof(typeof(m))) on (Pt,)" for m in (GaussTest(), BlobTest(2.0), PolTest(1.5))
        test_pointmaps(intensitymap, intensitymap!, ComradeBase.intensity_point, m, dxy)
    end

    @testset "polarized visibilitymap on $(keys(d))" for d in (d1, d2)
        test_pointmaps(visibilitymap, visibilitymap!, ComradeBase.visibility_point, PolTest(1.5), d)
        vp = visibilitymap(PolTest(1.5), d)
        @test vp isa StokesMap
        @test size(baseimage(vp)) == (size(d)..., 4)
    end

    @testset "view coordinates" begin
        Ub = randn(2npt)
        Vb = randn(2npt)
        uv3 = 3.0e4 .* randn(2npt, 4)
        dv1 = StructuredDomain((U = view(Ub, 1:2:(2npt)), V = view(Vb, 2:2:(2npt))))
        dc1 = StructuredDomain((U = Ub[1:2:(2npt)], V = Vb[2:2:(2npt)]))
        dv2 = StructuredDomain((Pt(npt), Fr(fr)); u = view(uv3, 1:npt, 2), v = view(uv3, (npt + 1):(2npt), 3))
        dc2 = StructuredDomain((Pt(npt), Fr(fr)); u = uv3[1:npt, 2], v = uv3[(npt + 1):(2npt), 3])
        for (dv, dc) in ((dv1, dc1), (dv2, dc2)), ex in structured_executors()
            m = BlobTest(2.0)
            @test baseimage(visibilitymap(m, DD.rebuild(dv; executor = ex))) == baseimage(visibilitymap(m, dc))
        end
        dxyv = StructuredDomain((X = view(Ub, 1:npt), Y = view(Vb, 1:npt)))
        dxyc = StructuredDomain((X = Ub[1:npt], Y = Vb[1:npt]))
        @test baseimage(intensitymap(GaussTest(), dxyv)) == baseimage(intensitymap(GaussTest(), dxyc))
    end

    @testset "Adapt converts coordinates and keeps dims" begin
        da = Adapt.adapt(DoubleFloats(), d2)
        @test ComradeBase.coords(da).u == 2 .* u
        @test ComradeBase.coords(da).v == 2 .* v
        @test dims(da) === dims(d2)
    end

    @testset "unknown ThreadsEx scheduler throws" begin
        @test_throws MethodError visibilitymap(GaussTest(), DD.rebuild(d1; executor = ThreadsEx(:nonexistent)))
    end
end

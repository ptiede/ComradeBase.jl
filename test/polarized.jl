using ComradeBase: StokesMap, CoherencyMap, IsPolarized, Pt, StructuredDomain

pointmap_into(dest, f, d, ex) = (ComradeBase._pointmap!(dest, f, d, ex); dest)
readpoint(img, i, j, k) = img[i, j, k]
writepoint!(img, v, i, j, k) = (img[i, j, k] = v; nothing)
stokescomponent(img, k) = stokes(img, k)
times2(img) = img .* 2
slabloss(a) = sum(abs2, stokes(a, :Q)) + sum(stokes(a, :V))
coherencyroundtrip(img, b) = stokesmap(coherencymap(img, b), b)
coherencyroundtrip!(img, b) = stokesmap!(coherencymap!(img, b), b)

function test_convert_loop(f, b, d, src)
    ComradeBase._convert!(f, b, d, src, Serial())
    @test (@allocated ComradeBase._convert!(f, b, d, src, Serial())) == 0
    JET.@test_opt target_modules = (ComradeBase,) ComradeBase._convert!(f, b, d, src, Serial())
    JET.@test_opt target_modules = (ComradeBase,) ComradeBase._convert!(f, b, d, src, ThreadsEx())
    return nothing
end

# A coherency map returned from `@jit` holds a host `ReshapedArray` around the device array.
function test_jit_coherency_storage(c, cref, vis, b)
    @test c isa CoherencyMap
    @test baseimage(c) isa Base.ReshapedArray
    @test Array(baseimage(c)) ≈ baseimage(cref)
    s = @jit stokesmap(c, b)
    @test Array(baseimage(s)) ≈ baseimage(vis)
    r = @jit times2(c)
    @test Array(baseimage(r)) ≈ 2 .* baseimage(cref)
    return nothing
end

@testset "Polarized maps" begin
    g = imagepixels(10.0, 12.0, 6, 5)
    P = rand(6, 5, 4)

    @testset "construction" begin
        img = @inferred IntensityMap(P, g, Stokes())
        @test img isa StokesMap{Float64, 3}
        @test img isa IntensityMap{Float64, 3}
        @test StokesMap <: IntensityMap
        @test StokesMap{Float64, 3} <: IntensityMap{Float64, 3}
        @test eltype(img) === Float64
        @test size(img) == (6, 5, 4)
        @test dims(img) == (dims(g)..., Stokes(DD.NoLookup(Base.OneTo(4))))
        @test @inferred(baseimage(img)) === P
        @test parent(img) === P
        @test DD.data(img) === P
        @test axisdims(img) === g
        @test @inferred(eldims(img)) isa Tuple{Stokes}
        @test length(only(eldims(img))) == 4
        @test IntensityMap(P, g, Stokes(DD.NoLookup(Base.OneTo(4)))) == img

        scalar = @inferred IntensityMap(rand(6, 5), g)
        @test !(scalar isa StokesMap)
        @test @inferred(eldims(scalar)) === ()

        sa = StructArray{StokesParams{Float64}}((rand(6, 5), rand(6, 5), rand(6, 5), rand(6, 5)))
        img2 = @inferred IntensityMap(sa, g)
        @test img2 isa StokesMap{Float64, 3}
        @test baseimage(img2) == cat(sa.I, sa.Q, sa.U, sa.V; dims = 3)
        img3 = @inferred IntensityMap(collect(sa), g)
        @test img3 isa StokesMap{Float64, 3}
        @test baseimage(img3) == baseimage(img2)

        @test_throws "needs size (6, 5, 4)" IntensityMap(rand(6, 5, 3), g, Stokes())
        @test_throws "needs size (6, 5, 4)" IntensityMap(rand(5, 5, 4), g, Stokes())
        @test_throws DimensionMismatch IntensityMap(rand(6, 5), g, Stokes())
        @test_throws "IntensityMap data has size (6, 5, 4), but the RectiGrid has size (6, 5)" IntensityMap(rand(6, 5, 4), g)
    end

    @testset "element access" begin
        Pc = copy(P)
        img = IntensityMap(Pc, g, Stokes())
        @test @inferred(readpoint(img, 2, 3, 2)) === Pc[2, 3, 2]
        @inferred writepoint!(img, 1.5, 1, 2, 3)
        @test Pc[1, 2, 3] == 1.5
        readpoint(img, 1, 1, 1)
        @test (@allocated readpoint(img, 1, 1, 1)) == 0
        writepoint!(img, 2.5, 2, 2, 2)
        @test (@allocated writepoint!(img, 2.5, 2, 2, 2)) == 0
    end

    @testset "Stokes components" begin
        Pc = copy(P)
        img = IntensityMap(Pc, g, Stokes())
        q = @inferred stokes(img, :Q)
        @test q isa IntensityMap{Float64, 2}
        @test !(q isa StokesMap)
        @test axisdims(q) === g
        @test baseimage(q) == Pc[:, :, 2]
        q[3, 3] = -1.0
        @test Pc[3, 3, 2] == -1.0
        for (n, k) in enumerate((:I, :Q, :U, :V))
            c = @inferred stokescomponent(img, k)
            @test typeof(c) === typeof(q)
            @test c == Pc[:, :, n]
        end
        @test_throws "`W` is not a Stokes component; the components are I, Q, U, V" stokes(img, :W)
    end

    @testset "DimensionalData selection and reductions" begin
        Pc = copy(P)
        img = IntensityMap(Pc, g, Stokes())
        s2 = img[Stokes(2)]
        @test s2 isa IntensityMap{Float64, 2}
        @test !(s2 isa StokesMap)
        @test dims(s2) == dims(g)
        @test s2 == stokes(img, :Q)
        v4 = view(img, Stokes(4))
        @test !(v4 isa StokesMap)
        @test parent(baseimage(v4)) === Pc
        v4[1, 1] = 7.0
        @test Pc[1, 1, 4] == 7.0

        sx = img[X = 2:4]
        @test sx isa StokesMap{Float64, 3}
        @test baseimage(sx) == Pc[2:4, :, :]
        @test axisdims(sx).X == g.X[2:4]
        vx = view(img, X = 2:4)
        @test vx isa StokesMap
        @test parent(baseimage(vx)) === Pc
        @test img[X = 2, Y = 3] == Pc[2, 3, :]
        @test img[X = 2, Y = 3] isa DD.DimVector

        r = sum(img; dims = (X, Y))
        @test r isa StokesMap
        @test size(r) == (1, 1, 4)
        @test vec(baseimage(r)) ≈ vec(sum(Pc; dims = (1, 2)))
        dfr = StructuredDomain((Pt(5), Fr([230.0e9, 345.0e9])); u = randn(5), v = randn(5))
        Ps = rand(5, 2, 4)
        simg = IntensityMap(Ps, dfr, Stokes())
        sp = simg[Pt(2:4)]
        @test sp isa StokesMap{Float64, 3}
        @test baseimage(sp) == Ps[2:4, :, :]
        @test ComradeBase.coords(axisdims(sp)).u == ComradeBase.coords(dfr).u[2:4]
        sq = simg[Stokes(3)]
        @test !(sq isa StokesMap)
        @test dims(axisdims(sq)) == dims(dfr)
        @test sq == Ps[:, :, 3]
        @test simg[Fr(1)] isa StokesMap{Float64, 2}
        @test simg[Pt(2)] isa DD.DimArray

        rs = sum(img; dims = Stokes)
        @test size(rs) == (6, 5, 1)
        @test baseimage(rs)[:, :, 1] ≈ dropdims(sum(Pc; dims = 3); dims = 3)
    end

    @testset "similar, copy and broadcasting" begin
        img = IntensityMap(copy(P), g, Stokes())
        s = @inferred similar(img)
        @test typeof(s) === typeof(img)
        @test baseimage(s) !== baseimage(img)
        @test similar(img, Float32) isa StokesMap{Float32, 3}
        c = @inferred copy(img)
        @test typeof(c) === typeof(img)
        @test baseimage(c) == baseimage(img)
        @test baseimage(c) !== baseimage(img)

        r = @inferred times2(img)
        @test typeof(r) === typeof(img)
        @test dims(r) == dims(img)
        @test baseimage(r) ≈ 2 .* P
        @test baseimage(img .+ img) ≈ 2 .* P
        @test abs.(img) isa StokesMap
        sc = IntensityMap(rand(6, 5), g)
        @test baseimage(stokes(img, :I) .* sc) ≈ P[:, :, 1] .* parent(sc)
        dest = similar(img)
        dest .= img .* 3
        @test baseimage(dest) ≈ 3 .* P
    end

    @testset "flux, centroid and second moment" begin
        img = IntensityMap(copy(P), g, Stokes())
        @test flux(img) isa StokesParams{Float64}
        @test flux(img) ≈ StokesParams(ntuple(k -> sum(P[:, :, k]), 4)...)
        @test centroid(img) == centroid(stokes(img, :I))
        @test second_moment(img) == second_moment(stokes(img, :I))
        g4 = imagepixels(10.0, 12.0, 6, 5; mdims = (Ti([0.0, 1.0]), Fr([230.0e9, 345.0e9, 690.0e9])))
        P4 = rand(6, 5, 2, 3, 4)
        img4 = IntensityMap(P4, g4, Stokes())
        f4 = flux(img4)
        @test f4.Q[1, 1, 2, 3] ≈ sum(P4[:, :, 2, 3, 2])
        @test centroid(img4) == centroid(stokes(img4, :I))
    end

    @testset "allocation" begin
        dpt = UnstructuredDomain((; U = 0.1 .* randn(7), V = 0.1 .* randn(7)))
        dfr = StructuredDomain((Pt(7), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(7), v = 3.0e4 .* randn(7))
        m = PolTest(1.5)
        vr = @inferred ComradeBase.allocate_vismap(m, g)
        @test vr isa StokesMap{ComplexF64, 3}
        @test size(baseimage(vr)) == (6, 5, 4)
        @test axisdims(vr) === g
        ir = @inferred ComradeBase.allocate_imgmap(m, g)
        @test ir isa StokesMap{Float64, 3}
        vs = @inferred ComradeBase.allocate_vismap(m, dpt)
        @test vs isa StokesMap{ComplexF64, 2}
        @test size(baseimage(vs)) == (7, 4)
        vf = @inferred ComradeBase.allocate_vismap(m, dfr)
        @test vf isa StokesMap{ComplexF64, 3}
        @test size(vf) == (7, 2, 4)
        @test @inferred(ComradeBase.allocate_imgmap(IsPolarized(), GaussTest(), dpt)) isa StokesMap{Float64, 2}
        @test @inferred(ComradeBase.allocate_map(Array{Float32}, g)) isa IntensityMap{Float32, 2}
    end

    @testset "point maps are type stable" begin
        dpt = UnstructuredDomain((; U = 0.1 .* randn(7), V = 0.1 .* randn(7)))
        m = PolTest(1.5)
        f = Base.Fix1(ComradeBase.visibility_point, m)
        dest = ComradeBase.allocate_vismap(m, dpt)
        @test @inferred(pointmap_into(dest, f, dpt, Serial())) === dest
        ref = map(p -> ComradeBase.visibility_point(m, p), domainpoints(dpt))
        @test baseimage(dest) ≈ asstorage(ref)
        pointmap_into(dest, f, dpt, Serial())
        @test (@allocated pointmap_into(dest, f, dpt, Serial())) == 0
        JET.@test_opt target_modules = (ComradeBase,) pointmap_into(dest, f, dpt, Serial())
        JET.@test_opt target_modules = (ComradeBase,) visibilitymap(m, dpt)
        dptf = StructuredDomain(
            (Pt(7), Ti([0.0, 1.0]), Fr([230.0e9, 345.0e9, 690.0e9]));
            u = 3.0e4 .* randn(7, 2), v = 3.0e4 .* randn(7, 2)
        )
        for mv in (m, GaussTest())
            vis = visibilitymap(mv, dptf)
            visibilitymap!(vis, mv)
            @test (@allocated visibilitymap!(vis, mv)) == 0
        end
        @test_throws "cannot fill the trailing dims of size ()" ComradeBase._setpoint!(zeros(ComplexF64, 7), CartesianIndex(1), f(first(domainpoints(dpt))))
        @test_throws "does not start with the axes" ComradeBase._pointindices(zeros(ComplexF64, 6, 4), domainpoints(dpt))
    end

    @testset "analytic polarized maps on every executor" begin
        m = PolTest(1.5)
        dpt = UnstructuredDomain((; U = 0.1 .* randn(9), V = 0.1 .* randn(9)))
        dfr = StructuredDomain((Pt(9), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(9), v = 3.0e4 .* randn(9))
        for d in (dpt, dfr)
            test_pointmaps(visibilitymap, visibilitymap!, ComradeBase.visibility_point, m, d)
        end
        dx, dy = pixelsizes(g)
        ref = asstorage(map(p -> ComradeBase.intensity_point(m, p), domainpoints(g))) .* dx .* dy
        for ex in (Serial(), ThreadsEx(), ThreadsEx(:static), DynamicScheduler(), StaticScheduler(), CPU())
            gex = imagepixels(10.0, 12.0, 6, 5; executor = ex)
            img = intensitymap(m, gex)
            @test img isa StokesMap{Float64, 3}
            @test baseimage(img) ≈ ref
        end
    end
end

@testset "Coherency maps" begin
    g = imagepixels(10.0, 12.0, 6, 5)
    img = IntensityMap(rand(6, 5, 4), g, Stokes())
    dfr = StructuredDomain((Pt(5), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(5), v = 3.0e4 .* randn(5))
    vis = visibilitymap(PolTest(1.5), dfr)
    bases = (CirBasis(), LinBasis())

    @testset "construction and dims" begin
        S = rand(6, 5, 2, 2)
        c = @inferred IntensityMap(S, g, Fa(), Fb())
        @test c isa CoherencyMap{Float64, 4}
        @test CoherencyMap <: IntensityMap
        @test !(c isa StokesMap)
        @test baseimage(c) === S
        @test axisdims(c) === g
        feeds = (Fa(DD.NoLookup(Base.OneTo(2))), Fb(DD.NoLookup(Base.OneTo(2))))
        @test dims(c) == (dims(g)..., feeds...)
        @test @inferred(eldims(c)) isa Tuple{Fa, Fb}
        @test_throws "with trailing dims (:Fa, :Fb) needs size (6, 5, 2, 2)" IntensityMap(rand(6, 5, 2), g, Fa(), Fb())
        @test_throws "with trailing dims (:Fa, :Fb) needs size (6, 5, 2, 2)" IntensityMap(rand(6, 5, 4), g, Fa(), Fb())

        for x in (img, vis), b in bases
            cx = @inferred coherencymap(x, b)
            @test cx isa CoherencyMap{complex(eltype(x)), ndims(x) + 1}
            @test axisdims(cx) === axisdims(x)
            @test dims(cx) == (dims(axisdims(x))..., feeds...)
            @test @inferred(eldims(cx)) isa Tuple{Fa, Fb}
        end
    end

    @testset "round trip and agreement with PolarizedTypes" begin
        for x in (img, vis), b in bases
            c = coherencymap(x, b)
            r = @inferred stokesmap(c, b)
            @test r isa StokesMap{complex(eltype(x)), ndims(x)}
            @test axisdims(r) === axisdims(x)
            @test baseimage(r) ≈ baseimage(x)
            P = baseimage(x)
            C = baseimage(c)
            for I in CartesianIndices(baseimage(stokes(x, :I)))
                s = StokesParams(P[I, 1], P[I, 2], P[I, 3], P[I, 4])
                @test C[I, :, :] ≈ CoherencyMatrix(s, b)
            end
        end
    end

    @testset "components" begin
        c = coherencymap(vis, CirBasis())
        e12 = @inferred coherency(c, 1, 2)
        @test e12 isa IntensityMap{ComplexF64, 2}
        @test !(e12 isa CoherencyMap)
        @test axisdims(e12) === dfr
        @test e12 == baseimage(c)[:, :, 1, 2]
        @test e12 ≈ stokes(vis, :Q) .+ im .* stokes(vis, :U)
        e12[1, 1] = 0
        @test baseimage(c)[1, 1, 1, 2] == 0
        @test typeof(coherency(c, 2, 1)) === typeof(e12)
        coherency(c, 2, 1)
        @test (@allocated coherency(c, 2, 1)) == 0
        JET.@test_opt target_modules = (ComradeBase,) coherency(c, 2, 1)
        q = @inferred stokescomponent(vis, :Q)
        @test axisdims(q) === dfr
        stokescomponent(vis, :Q)
        @test (@allocated stokescomponent(vis, :Q)) == 0
        JET.@test_opt target_modules = (ComradeBase,) stokescomponent(vis, :Q)
        @test axisdims(view(vis, Stokes(2))) === dfr
        @test axisdims(img[Stokes = 3]) === g
    end

    @testset "unsupported basis" begin
        c = coherencymap(img, CirBasis())
        msg = "the supported bases are CirBasis() and LinBasis()"
        @test_throws msg coherencymap(img, (CirBasis(), LinBasis()))
        @test_throws msg coherencymap(img, :circular)
        @test_throws msg stokesmap(c, (CirBasis(), CirBasis()))
    end

    @testset "slicing and rebuild" begin
        c = coherencymap(img, CirBasis())
        @test @inferred(ComradeBase._splitdims(dims(c))) == (dims(g), eldims(c))
        @test @inferred(ComradeBase._splitdims(dims(c)[[1, 2, 4]])) == ((), dims(c)[[1, 2, 4]])
        @test @inferred(ComradeBase._splitdims(dims(c)[1:3])) == ((), dims(c)[1:3])
        @test @inferred(ComradeBase._splitdims((dims(g)..., Fb(1:2)))) == ((), (dims(g)..., Fb(1:2)))
        @test @inferred(ComradeBase._splitdims(dims(c)[3:4])) == ((), dims(c)[3:4])

        e21 = c[Fa = 2, Fb = 1]
        @test e21 isa IntensityMap{ComplexF64, 2}
        @test dims(e21) == dims(g)
        @test e21 == coherency(c, 2, 1)
        @test c[Fa = 1] isa DD.DimArray
        @test !(c[Fa = 1] isa IntensityMap)
        @test c[Fb = 2] isa DD.DimArray
        @test !(c[Fb = 2] isa IntensityMap)
        @test c[Fb = 2] == baseimage(c)[:, :, :, 2]
        sx = c[X = 2:4]
        @test sx isa CoherencyMap{ComplexF64, 4}
        @test baseimage(sx) == baseimage(c)[2:4, :, :, :]
        @test c[X = 2, Y = 3] isa DD.DimMatrix

        cs = coherencymap(vis, LinBasis())
        sp = cs[Pt(2:4)]
        @test sp isa CoherencyMap{ComplexF64, 4}
        @test ComradeBase.coords(axisdims(sp)).u == ComradeBase.coords(dfr).u[2:4]
        @test cs[Fa = 1] isa DD.DimArray
        @test !(cs[Fa = 1] isa IntensityMap)
        @test cs[Fb = 1] isa DD.DimArray
        @test cs[Pt(2)] isa DD.DimArray
        @test cs[Fr(1)] isa CoherencyMap{ComplexF64, 3}
        @test cs[Fa = 1, Fb = 1] == coherency(cs, 1, 1)
    end

    @testset "similar, copy and broadcasting" begin
        c = coherencymap(img, CirBasis())
        s = @inferred similar(c)
        @test typeof(s) === typeof(c)
        @test baseimage(s) !== baseimage(c)
        cc = @inferred copy(c)
        @test typeof(cc) === typeof(c)
        @test baseimage(cc) == baseimage(c)
        r = @inferred times2(c)
        @test typeof(r) === typeof(c)
        @test baseimage(r) ≈ 2 .* baseimage(c)
        @test conj.(c) isa CoherencyMap
    end

    @testset "image reductions fail fast" begin
        c = coherencymap(img, CirBasis())
        @test_throws "`flux` is not defined for a map with trailing dims (:Fa, :Fb)" flux(c)
        @test_throws "`second_moment` is not defined for a map with trailing dims (:Fa, :Fb)" second_moment(c)
        creal = IntensityMap(rand(6, 5, 2, 2), g, Fa(), Fb())
        @test_throws "`centroid` is not defined for a map with trailing dims (:Fa, :Fb)" centroid(creal)
        @test_throws "a Stokes I component is not defined" ComradeBase._stokesI(creal)
    end

    @testset "in-place conversions" begin
        for b in bases
            v = copy(vis)
            S = baseimage(v)
            c = @inferred coherencymap!(v, b)
            @test c isa CoherencyMap{ComplexF64, 4}
            @test axisdims(c) === dfr
            @test baseimage(c) ≈ baseimage(coherencymap(vis, b))
            @test pointer(baseimage(c)) == pointer(S)
            S[2] = 7
            @test baseimage(c)[2] == 7
            c = coherencymap!(copy(vis), b)
            Sc = baseimage(c)
            s = @inferred stokesmap!(c, b)
            @test s isa StokesMap{ComplexF64, 3}
            @test axisdims(s) === dfr
            @test baseimage(s) ≈ baseimage(vis)
            @test pointer(baseimage(s)) == pointer(Sc)
            @test baseimage(coherencyroundtrip!(copy(vis), b)) ≈ baseimage(vis)
            @test baseimage(stokesmap!(coherencymap(vis, b), b)) ≈ baseimage(vis)
        end

        for b in bases
            v = copy(vis)
            coherencymap!(v, b)
            @test (@allocated stokesmap!(coherencymap!(v, b), b)) <= 512
            src = ComradeBase._stokesslabs(v)
            dest = ComradeBase._stokesslabs(similar(v))
            test_convert_loop(ComradeBase._coherencypoint, b, src, src)
            test_convert_loop(ComradeBase._coherencypoint, b, dest, src)
            test_convert_loop(ComradeBase._stokespoint, b, src, src)
            test_convert_loop(ComradeBase._stokespoint, b, dest, src)
            @test all(s -> s isa SubArray && Base.IndexStyle(s) isa IndexLinear, src)
        end
        JET.@test_opt target_modules = (ComradeBase,) coherencymap!(copy(vis), CirBasis())
        JET.@test_opt target_modules = (ComradeBase,) stokesmap!(coherencymap(vis, LinBasis()), LinBasis())

        executors = (
            Serial(), ThreadsEx(), ThreadsEx(:static), ThreadsEx(:Polyester), ThreadsEx(:Enzyme),
            DynamicScheduler(), StaticScheduler(), CPU(),
        )
        for ex in executors, b in bases
            vex = IntensityMap(copy(baseimage(vis)), DD.rebuild(dfr; executor = ex), Stokes())
            cex = coherencymap(vex, b)
            @test baseimage(cex) ≈ baseimage(coherencymap(vis, b))
            @test baseimage(stokesmap(cex, b)) ≈ baseimage(vis)
            c = coherencymap!(vex, b)
            @test baseimage(c) ≈ baseimage(coherencymap(vis, b))
            @test baseimage(stokesmap!(c, b)) ≈ baseimage(vis)
        end

        kaalloc(n) = begin
            x = IntensityMap(rand(ComplexF64, n, 4), UnstructuredDomain((; U = randn(n), V = randn(n))), Stokes())
            src = ComradeBase._stokesslabs(x)
            dest = ComradeBase._stokesslabs(similar(x))
            ComradeBase._convert!(ComradeBase._coherencypoint, CirBasis(), dest, src, CPU())
            @allocated ComradeBase._convert!(ComradeBase._coherencypoint, CirBasis(), dest, src, CPU())
        end
        @test kaalloc(10) == kaalloc(10_000)

        img32 = IntensityMap(rand(Float32, 6, 5, 4), g, Stokes())
        for b in bases
            c32 = coherencymap(img32, b)
            @test eltype(c32) === ComplexF32
            s32 = stokesmap(c32, b)
            @test eltype(s32) === ComplexF32
            @test eltype(stokesmap!(coherencymap!(s32, b), b)) === ComplexF32
        end

        @test_throws "`coherencymap!` needs complex storage, but the map has element type Float64; use `coherencymap` instead" coherencymap!(copy(img), CirBasis())
        creal = IntensityMap(rand(6, 5, 2, 2), g, Fa(), Fb())
        @test_throws "`stokesmap!` needs complex storage, but the map has element type Float64; use `stokesmap` instead" stokesmap!(creal, CirBasis())
        msg = "the supported bases are CirBasis() and LinBasis()"
        @test_throws msg coherencymap!(copy(vis), :circular)
        @test_throws msg stokesmap!(coherencymap(vis, CirBasis()), :circular)
    end

    @testset "type stability" begin
        JET.@test_opt target_modules = (ComradeBase,) coherencymap(img, CirBasis())
        JET.@test_opt target_modules = (ComradeBase,) coherencymap(vis, LinBasis())
        c = coherencymap(vis, CirBasis())
        JET.@test_opt target_modules = (ComradeBase,) stokesmap(c, CirBasis())
        JET.@test_opt target_modules = (ComradeBase,) stokesmap(c, LinBasis())
        @test @inferred(coherencyroundtrip(img, LinBasis())) isa StokesMap{ComplexF64, 3}
    end
end

@testset "Polarized maps under Reactant" begin
    m = PolTest(1.5)
    mr = Reactant.to_rarray(m; track_numbers = Number)
    g = imagepixels(10.0, 12.0, 8, 6)
    gr = @jit identity(g)

    ip = @jit intensitymap(mr, gr)
    @test ip isa StokesMap
    @test Array(baseimage(ip)) ≈ baseimage(intensitymap(m, g))
    test_clean_hlo(repr(@code_hlo intensitymap(mr, gr)))

    guv = RectiGrid((U(range(-0.2, 0.2; length = 6)), V(range(-0.2, 0.2; length = 5))))
    @test Array(baseimage(@jit visibilitymap(mr, Reactant.to_rarray(guv)))) ≈ baseimage(visibilitymap(m, guv))
    test_clean_hlo(repr(@code_hlo visibilitymap(mr, Reactant.to_rarray(guv))))

    dfr = StructuredDomain((Pt(6), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(6), v = 3.0e4 .* randn(6))
    dfrr = Reactant.to_rarray(dfr)
    vp = @jit visibilitymap(mr, dfrr)
    @test vp isa StokesMap
    @test Array(baseimage(vp)) ≈ baseimage(visibilitymap(m, dfr))
    test_clean_hlo(repr(@code_hlo visibilitymap(mr, dfrr)))

    img = IntensityMap(rand(8, 6, 4), g, Stokes())
    rimg = Reactant.to_rarray(img)
    r = @jit times2(rimg)
    @test r isa StokesMap
    @test dims(r) == dims(img)
    @test Array(baseimage(r)) ≈ 2 .* baseimage(img)
    @test Float64(@jit slabloss(rimg)) ≈ slabloss(img)

    for b in (CirBasis(), LinBasis())
        rc = @jit coherencymap(rimg, b)
        @test dims(rc) == dims(coherencymap(img, b))
        test_jit_coherency_storage(rc, coherencymap(img, b), img, b)
        rs = @jit stokesmap(rc, b)
        @test rs isa StokesMap
        @test !(baseimage(rs) isa Base.ReshapedArray)
        test_clean_hlo(repr(@code_hlo coherencymap(rimg, b)))
        test_clean_hlo(repr(@code_hlo stokesmap(rc, b)))

        img32 = IntensityMap(rand(Float32, 8, 6, 4), g, Stokes())
        rc32 = @jit coherencymap(Reactant.to_rarray(img32), b)
        @test eltype(baseimage(rc32)) === ComplexF32
        @test Array(baseimage(rc32)) ≈ baseimage(coherencymap(img32, b))
        @test eltype(baseimage(@jit stokesmap(rc32, b))) === ComplexF32
        @test !occursin("f64", repr(@code_hlo coherencymap(Reactant.to_rarray(img32), b)))

        cv = visibilitymap(m, dfr)
        rcv = @jit coherencymap!(Reactant.to_rarray(cv), b)
        test_jit_coherency_storage(rcv, coherencymap(cv, b), cv, b)
        test_clean_hlo(repr(@code_hlo coherencymap!(Reactant.to_rarray(cv), b)))
        dpt = UnstructuredDomain((; U = 0.1 .* randn(9), V = 0.1 .* randn(9)))
        vpt = visibilitymap(m, dpt)
        rpt = Reactant.to_rarray(vpt)
        @test Array(baseimage(@jit coherencymap(rpt, b))) ≈ baseimage(coherencymap(vpt, b))
        @test Array(baseimage(@jit stokesmap(Reactant.to_rarray(coherencymap(vpt, b)), b))) ≈ baseimage(vpt)
        @test Array(baseimage(@jit coherencymap!(Reactant.to_rarray(vpt), b))) ≈ baseimage(coherencymap(vpt, b))
        test_clean_hlo(repr(@code_hlo coherencymap(rpt, b)))
        test_clean_hlo(repr(@code_hlo stokesmap(Reactant.to_rarray(coherencymap(vpt, b)), b)))
        test_clean_hlo(repr(@code_hlo coherencymap!(Reactant.to_rarray(vpt), b)))
        @test_throws "`stokesmap!` does not run under Reactant" @jit stokesmap!(Reactant.to_rarray(coherencymap(cv, b)), b)
    end
end

@testset "Sharding polarized maps" begin
    ndev = length(Reactant.devices())
    if ndev == 1
        @warn "Polarized sharding tests skipped: start Julia with XLA_FLAGS=--xla_force_host_platform_device_count=4"
    else
        mesh = Reactant.Sharding.Mesh(reshape(Reactant.devices(), ndev), (:d,))
        loss(P, g) = slabloss(IntensityMap(P, g, Stokes()))
        gradient(P, g) = Enzyme.gradient(Enzyme.Reverse, loss, P, Enzyme.Const(g))[1]
        function expected_gradient(P)
            gr = zero(P)
            selectdim(gr, ndims(P), 2) .= 2 .* selectdim(P, ndims(P), 2)
            selectdim(gr, ndims(P), 4) .= 1
            return gr
        end

        @testset "along X" begin
            nx = 2ndev
            g = imagepixels(10.0, 10.0, nx, 4)
            P = rand(nx, 4, 4)
            img = IntensityMap(P, g, Stokes())
            simg = shard(img, ShardLayout(mesh; X = :d))
            @test simg isa StokesMap
            @test axisdims(simg) === axisdims(img)
            @test stored_blocks(baseimage(simg), 1) == split_blocks(nx, ndev)
            @test stored_blocks(baseimage(simg), 3) == [1:4]
            @test Array(baseimage(@jit times2(simg))) ≈ 2 .* P
            @test Float64(@jit slabloss(simg)) ≈ slabloss(img)
            gsh = Array(@jit gradient(baseimage(simg), g))
            gun = Array(@jit gradient(Reactant.to_rarray(P), g))
            @test gsh ≈ gun
            @test gun ≈ expected_gradient(P)
        end

        @testset "along Fr of a 4-d map" begin
            nf = 2ndev
            g = imagepixels(10.0, 10.0, 6, 5; mdims = (Ti([0.0, 1.0]), Fr(range(86.0e9, 345.0e9; length = nf))))
            P = rand(6, 5, 2, nf, 4)
            img = IntensityMap(P, g, Stokes())
            simg = shard(img, ShardLayout(mesh; Fr = :d))
            @test stored_blocks(baseimage(simg), 4) == split_blocks(nf, ndev)
            @test stored_blocks(baseimage(simg), 5) == [1:4]
            @test stored_blocks(baseimage(simg), 3) == [1:2]
            @test Array(baseimage(@jit times2(simg))) ≈ 2 .* P
            @test_throws "available dimensions are (:X, :Y, :Ti, :Fr, :Stokes)" shard(img, ShardLayout(mesh; Pol = :d))

            gsh = Array(@jit gradient(baseimage(simg), g))
            gun = Array(@jit gradient(Reactant.to_rarray(P), g))
            @test all(isfinite, gsh)
            @test gsh ≈ gun
            @test gun ≈ expected_gradient(P)
        end

        @testset "along Stokes" begin
            img = IntensityMap(rand(6, 5, 4), imagepixels(10.0, 10.0, 6, 5), Stokes())
            simg = shard(img, ShardLayout(mesh; Stokes = :d))
            @test stored_blocks(baseimage(simg), 3) == split_blocks(4, ndev)
            @test Array(baseimage(@jit times2(simg))) ≈ 2 .* baseimage(img)
        end

        @testset "coherency of a (Pt, Fr) map along Fr" begin
            nf = 2ndev
            d = StructuredDomain(
                (Pt(6), Fr(range(86.0e9, 345.0e9; length = nf)));
                u = 3.0e4 .* randn(6), v = 3.0e4 .* randn(6)
            )
            vis = visibilitymap(PolTest(1.5), d)
            svis = shard(vis, ShardLayout(mesh; Fr = :d))
            @test stored_blocks(baseimage(svis), 2) == split_blocks(nf, ndev)
            c = coherencymap(vis, CirBasis())
            test_clean_hlo(repr(@code_hlo coherencymap(svis, CirBasis())))
            test_clean_hlo(repr(@code_hlo coherencymap!(shard(vis, ShardLayout(mesh; Fr = :d)), CirBasis())))
            test_clean_hlo(repr(@code_hlo stokesmap(shard(c, ShardLayout(mesh; Fr = :d)), CirBasis())))
            rt = @jit coherencyroundtrip(svis, CirBasis())
            @test stored_blocks(baseimage(rt), 2) == split_blocks(nf, ndev)
            @test stored_blocks(baseimage(rt), 3) == [1:4]
            @test Array(baseimage(rt)) ≈ baseimage(vis)
            for convert in (coherencymap, coherencymap!), sharded in (false, true)
                x = sharded ? shard(vis, ShardLayout(mesh; Fr = :d)) : Reactant.to_rarray(vis)
                sc = @jit convert(x, CirBasis())
                if sharded
                    @test sc isa CoherencyMap
                    @test Array(baseimage(sc)) ≈ baseimage(c)
                    @test stored_blocks(parent(baseimage(sc)), 2) == split_blocks(nf, ndev)
                    @test stored_blocks(parent(baseimage(sc)), 3) == [1:4]
                    # The returned domain coordinates are replicated arrays that come back
                    # without a sharding; passing them through a second `@jit` next to sharded
                    # arrays throws Reactant's `TODO(#2234)` error.
                    @test_broken (@jit(stokesmap(sc, CirBasis())); true)
                else
                    test_jit_coherency_storage(sc, c, vis, CirBasis())
                end
            end
            shc = shard(c, ShardLayout(mesh; Fr = :d))
            @test shc isa CoherencyMap
            @test stored_blocks(baseimage(shc), 2) == split_blocks(nf, ndev)
            @test stored_blocks(baseimage(shc), 3) == [1:2]
            @test_throws "available dimensions are (:Pt, :Fr, :Fa, :Fb)" shard(c, ShardLayout(mesh; Stokes = :d))
        end
    end
end

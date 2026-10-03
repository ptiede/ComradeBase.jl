using ComradeBase: StokesMap, IsPolarized, Pt, StructuredDomain

pointmap_into(dest, f, d, ex) = (ComradeBase._pointmap!(dest, f, d, ex); dest)
readpoint(img, i, j, k) = img[i, j, k]
writepoint!(img, v, i, j, k) = (img[i, j, k] = v; nothing)
stokescomponent(img, k) = stokes(img, k)
times2(img) = img .* 2
slabloss(a) = sum(abs2, stokes(a, :Q)) + sum(stokes(a, :V))

@testset "Polarized maps" begin
    g = imagepixels(10.0, 12.0, 6, 5)
    P = rand(6, 5, 4)

    @testset "construction" begin
        img = @inferred IntensityMap(P, g, Stokes())
        @test img isa StokesMap{Float64, 3}
        @test img isa IntensityMap{Float64, 3}
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
        dest = baseimage(ComradeBase.allocate_vismap(m, dpt))
        @test @inferred(pointmap_into(dest, f, dpt, Serial())) === dest
        ref = map(p -> ComradeBase.visibility_point(m, p), domainpoints(dpt))
        @test dest ≈ asstorage(ref)
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
        @test_throws "cannot fill the trailing dims of size ()" pointmap_into(zeros(ComplexF64, 7), f, dpt, Serial())
        @test_throws "does not start with the axes" pointmap_into(zeros(ComplexF64, 6, 4), f, dpt, Serial())
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

@testset "Polarized maps under Reactant" begin
    m = PolTest(1.5)
    mr = Reactant.to_rarray(m; track_numbers = Number)
    g = imagepixels(10.0, 12.0, 8, 6)
    gr = @jit identity(g)

    ip = @jit intensitymap(mr, gr)
    @test ip isa StokesMap
    @test Array(baseimage(ip)) ≈ baseimage(intensitymap(m, g))
    @test !occursin("stablehlo.while", repr(@code_hlo intensitymap(mr, gr)))

    guv = RectiGrid((U(range(-0.2, 0.2; length = 6)), V(range(-0.2, 0.2; length = 5))))
    @test Array(baseimage(@jit visibilitymap(mr, Reactant.to_rarray(guv)))) ≈ baseimage(visibilitymap(m, guv))

    dfr = StructuredDomain((Pt(6), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(6), v = 3.0e4 .* randn(6))
    dfrr = Reactant.to_rarray(dfr)
    vp = @jit visibilitymap(mr, dfrr)
    @test vp isa StokesMap
    @test Array(baseimage(vp)) ≈ baseimage(visibilitymap(m, dfr))
    @test !occursin("stablehlo.while", repr(@code_hlo visibilitymap(mr, dfrr)))

    img = IntensityMap(rand(8, 6, 4), g, Stokes())
    rimg = Reactant.to_rarray(img)
    r = @jit times2(rimg)
    @test r isa StokesMap
    @test dims(r) == dims(img)
    @test Array(baseimage(r)) ≈ 2 .* baseimage(img)
    @test Float64(@jit slabloss(rimg)) ≈ slabloss(img)
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
    end
end

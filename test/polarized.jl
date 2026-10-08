using ComradeBase: StokesMap, CoherencyMap, IsPolarized, Pt, StructuredDomain

pointmap_into(dest, f, d, ex) = (ComradeBase._pointmap!(dest, f, d, ex); dest)
readpoint(img, i, j) = img[i, j]
writepoint!(img, v, i, j) = (img[i, j] = v; nothing)
stokescomponent(img, k) = stokes(img, k)
times2(img) = img .* 2
slabloss(a) = sum(abs2, stokes(a, :Q)) + sum(stokes(a, :V))
coherencyroundtrip(img, b) = stokesmap(coherencymap(img, b), b)
stokesview(P, g) = IntensityMap(FieldDimArray{StokesParams}(P), g)
dense(img) = parent(baseimage(img))
fluxQ(img) = flux(img).Q
fluxQmap(img) = baseimage(flux(img)).Q
stokesloss(P, g) = slabloss(stokesview(P, g))
coherencyloss(P, g) = sum(abs2, coherency(coherencymap(stokesview(P, g), CirBasis()), 1, 2))

@testset "Polarized maps" begin
    g = spatialgrid(10.0, 12.0, 6, 5)
    P = rand(6, 5, 4)
    sa = StructArray{StokesParams{Float64}}((rand(6, 5), rand(6, 5), rand(6, 5), rand(6, 5)))
    arr = collect(sa)

    @testset "construction" begin
        v = FieldDimArray{StokesParams}(P)
        img = @inferred IntensityMap(v, g)
        @test img isa StokesMap{Float64, 2}
        @test img isa IntensityMap{StokesParams{Float64}, 2}
        @test StokesMap <: IntensityMap
        @test size(img) == (6, 5)
        @test dims(img) == dims(g)
        @test axisdims(img) === g
        @test @inferred(baseimage(img)) === v
        @test parent(baseimage(img)) === P

        for data in (sa, arr)
            x = @inferred IntensityMap(data, g)
            @test x isa StokesMap{Float64, 2}
            @test baseimage(x) === data
        end
        @test !(IntensityMap(rand(6, 5), g) isa StokesMap)

        @test_throws "IntensityMap data has size (5, 5), but the RectiGrid has size (6, 5)" stokesview(rand(5, 5, 4), g)
        @test_throws "IntensityMap data has size (6, 5, 4), but the RectiGrid has size (6, 5)" IntensityMap(rand(6, 5, 4), g)
    end

    @testset "element access" begin
        Pc = copy(P)
        img = stokesview(Pc, g)
        @test @inferred(readpoint(img, 2, 3)) === StokesParams(Pc[2, 3, :]...)
        s = StokesParams(1.0, 2.0, 3.0, 4.0)
        @inferred writepoint!(img, s, 1, 2)
        @test Pc[1, 2, :] == [1.0, 2.0, 3.0, 4.0]
        readpoint(img, 1, 1)
        @test (@allocated readpoint(img, 1, 1)) == 0
        writepoint!(img, s, 2, 2)
        @test (@allocated writepoint!(img, s, 2, 2)) == 0
    end

    @testset "Stokes components" begin
        Pc = copy(P)
        img = stokesview(Pc, g)
        q = @inferred stokes(img, :Q)
        @test q isa IntensityMap{Float64, 2}
        @test !(q isa StokesMap)
        @test axisdims(q) === g
        @test q == Pc[:, :, 2]
        q[3, 3] = -1.0
        @test Pc[3, 3, 2] == -1.0
        for (n, k) in enumerate((:I, :Q, :U, :V))
            c = @inferred stokescomponent(img, k)
            @test typeof(c) === typeof(q)
            @test c == Pc[:, :, n]
        end
        stokescomponent(img, :U)
        @test (@allocated stokescomponent(img, :U)) == 0
        JET.@test_opt target_modules = (ComradeBase,) stokescomponent(img, :U)
        @test_throws "has no component named :W" stokes(img, :W)

        sc = copy(sa)
        simg = IntensityMap(sc, g)
        sq = @inferred stokescomponent(simg, :Q)
        @test baseimage(sq) === sc.Q
        @test stokes(IntensityMap(arr, g), :U) == sa.U
    end

    @testset "DimensionalData selection and reductions" begin
        Pc = copy(P)
        img = stokesview(Pc, g)
        sx = img[X = 2:4]
        @test sx isa StokesMap{Float64, 2}
        @test dense(sx) == Pc[2:4, :, :]
        @test axisdims(sx).X == g.X[2:4]
        vx = view(img, X = 2:4)
        @test vx isa StokesMap
        @test parent(dense(vx)) === Pc
        @test img[X = 2, Y = 3] === StokesParams(Pc[2, 3, :]...)

        r = sum(img; dims = (X, Y))
        @test r isa StokesMap
        @test size(r) == (1, 1)
        @test only(r) ≈ StokesParams(vec(sum(Pc; dims = (1, 2)))...)
        @test sum(img) ≈ only(r)

        dfr = StructuredDomain((Pt(5), Fr([230.0e9, 345.0e9])); u = randn(5), v = randn(5))
        Ps = rand(5, 2, 4)
        simg = stokesview(Ps, dfr)
        sp = simg[Pt(2:4)]
        @test sp isa StokesMap{Float64, 2}
        @test dense(sp) == Ps[2:4, :, :]
        @test ComradeBase.coords(axisdims(sp)).u == ComradeBase.coords(dfr).u[2:4]
        @test simg[Fr(1)] isa StokesMap{Float64, 1}
        @test simg[Pt(2)] isa DD.DimArray
    end

    @testset "similar, copy and broadcasting" begin
        img = stokesview(copy(P), g)
        s = @inferred similar(img)
        @test typeof(s) === typeof(img)
        @test dense(s) !== dense(img)
        @test similar(img, StokesParams{Float32}) isa StokesMap{Float32, 2}
        c = @inferred copy(img)
        @test typeof(c) === typeof(img)
        @test dense(c) == dense(img)
        @test dense(c) !== dense(img)

        r = @inferred times2(img)
        @test typeof(r) === typeof(img)
        @test dims(r) == dims(img)
        @test dense(r) ≈ 2 .* P
        @test dense(img .+ img) ≈ 2 .* P
        lp = (x -> x.Q + x.U).(img)
        @test lp isa IntensityMap{Float64, 2}
        @test lp ≈ P[:, :, 2] .+ P[:, :, 3]
        sc = IntensityMap(rand(6, 5), g)
        @test baseimage(stokes(img, :I) .* sc) ≈ P[:, :, 1] .* parent(sc)
        dest = similar(img)
        dest .= img .* 3
        @test dense(dest) ≈ 3 .* P
    end

    @testset "flux, centroid and second moment" begin
        img = stokesview(copy(P), g)
        @test @inferred(flux(img)) isa StokesParams{Float64}
        @test flux(img) ≈ StokesParams(ntuple(k -> sum(P[:, :, k]), 4)...)
        @test flux(IntensityMap(sa, g)) ≈ sum(sa)
        @test centroid(img) == centroid(stokes(img, :I))
        @test second_moment(img) == second_moment(stokes(img, :I))
        g4 = spatialgrid(10.0, 12.0, 6, 5) ⊗ Ti([0.0, 1.0]) ⊗ Fr([230.0e9, 345.0e9, 690.0e9])
        P4 = rand(6, 5, 2, 3, 4)
        img4 = stokesview(P4, g4)
        f4 = flux(img4)
        @test f4 isa StokesMap{Float64, 4}
        @test baseimage(f4) isa StructArray
        @test size(f4) == (1, 1, 2, 3)
        @test f4[1, 1, 2, 3].Q ≈ sum(P4[:, :, 2, 3, 2])
        @test centroid(img4) == centroid(stokes(img4, :I))
    end

    @testset "Enzyme on the CPU" begin
        Pc = copy(P)
        expected = zero(Pc)
        expected[:, :, 2] .= 2 .* Pc[:, :, 2]
        expected[:, :, 4] .= 1
        @test Enzyme.gradient(Enzyme.Reverse, stokesloss, Pc, Enzyme.Const(g))[1] ≈ expected
        expected[:, :, 3] .= 2 .* Pc[:, :, 3]
        expected[:, :, 4] .= 0
        @test Enzyme.gradient(Enzyme.Reverse, coherencyloss, Pc, Enzyme.Const(g))[1] ≈ expected
    end

    @testset "allocation" begin
        dpt = UnstructuredDomain((; U = 0.1 .* randn(7), V = 0.1 .* randn(7)))
        dfr = StructuredDomain((Pt(7), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(7), v = 3.0e4 .* randn(7))
        m = PolTest(1.5)
        vr = @inferred ComradeBase.allocate_vismap(m, g)
        @test vr isa StokesMap{ComplexF64, 2}
        @test baseimage(vr) isa FieldDimArray{StokesParams{ComplexF64}, 2, Array{ComplexF64, 3}}
        @test size(dense(vr)) == (6, 5, 4)
        @test axisdims(vr) === g
        @test @inferred(ComradeBase.allocate_imgmap(m, g)) isa StokesMap{Float64, 2}
        vs = @inferred ComradeBase.allocate_vismap(m, dpt)
        @test vs isa StokesMap{ComplexF64, 1}
        @test size(dense(vs)) == (7, 4)
        @test @inferred(ComradeBase.allocate_vismap(m, dfr)) isa StokesMap{ComplexF64, 2}
        @test @inferred(ComradeBase.allocate_imgmap(IsPolarized(), GaussTest(), dpt)) isa StokesMap{Float64, 1}
        @test @inferred(ComradeBase.allocate_map(Array{Float32}, g)) isa IntensityMap{Float32, 2}
        @test @inferred(ComradeBase.allocate_map(Array{StokesParams{Float32}}, g)) isa StokesMap{Float32, 2}
    end

    @testset "point maps are type stable" begin
        dpt = UnstructuredDomain((; U = 0.1 .* randn(7), V = 0.1 .* randn(7)))
        m = PolTest(1.5)
        f = Base.Fix1(ComradeBase.visibility_point, m)
        dest = ComradeBase.allocate_vismap(m, dpt)
        @test @inferred(pointmap_into(dest, f, dpt, Serial())) === dest
        @test collect(baseimage(dest)) ≈ map(p -> ComradeBase.visibility_point(m, p), domainpoints(dpt))
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
        @test_throws "does not have the axes" ComradeBase._pointindices(zeros(ComplexF64, 6), domainpoints(dpt))
    end

    @testset "analytic polarized maps on every executor" begin
        m = PolTest(1.5)
        dpt = UnstructuredDomain((; U = 0.1 .* randn(9), V = 0.1 .* randn(9)))
        dfr = StructuredDomain((Pt(9), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(9), v = 3.0e4 .* randn(9))
        for d in (dpt, dfr)
            test_pointmaps(visibilitymap, visibilitymap!, ComradeBase.visibility_point, m, d)
        end
        ref = map(p -> ComradeBase.intensity_point(m, p), domainpoints(g)) .* prod(pixelsizes(g))
        for ex in (Serial(), ThreadsEx(), ThreadsEx(:static), DynamicScheduler(), StaticScheduler(), CPU())
            gex = spatialgrid(10.0, 12.0, 6, 5; executor = ex)
            img = intensitymap(m, gex)
            @test img isa StokesMap{Float64, 2}
            @test collect(baseimage(img)) ≈ ref
            for data in (similar(sa), similar(arr))
                dest = IntensityMap(data, gex)
                intensitymap!(dest, m)
                @test collect(baseimage(dest)) ≈ ref
            end
        end
    end
end

@testset "Coherency maps" begin
    g = spatialgrid(10.0, 12.0, 6, 5)
    img = stokesview(rand(6, 5, 4), g)
    dfr = StructuredDomain((Pt(5), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(5), v = 3.0e4 .* randn(5))
    vis = visibilitymap(PolTest(1.5), dfr)
    bases = (CirBasis(), LinBasis())

    @testset "construction" begin
        @test CoherencyMap <: IntensityMap
        for x in (img, vis), b in bases
            c = @inferred coherencymap(x, b)
            @test c isa CoherencyMap{ComplexF64, 2}
            @test !(c isa StokesMap)
            @test baseimage(c) isa FieldDimArray
            @test size(dense(c)) == (size(x)..., 2, 2)
            @test axisdims(c) === axisdims(x)
            @test dims(c) == dims(axisdims(x))
        end
        S = rand(ComplexF64, 6, 5, 2, 2)
        c = IntensityMap(FieldDimArray{SMatrix{2, 2}}(S), g)
        @test c isa CoherencyMap{ComplexF64, 2}
        @test dense(c) === S
    end

    @testset "round trip and agreement with PolarizedTypes" begin
        for x in (img, vis), b in bases
            c = coherencymap(x, b)
            r = @inferred stokesmap(c, b)
            @test r isa StokesMap{ComplexF64, 2}
            @test axisdims(r) === axisdims(x)
            @test dense(r) ≈ dense(x)
            for I in CartesianIndices(x)
                @test c[I] ≈ CoherencyMatrix(x[I], b)
                @test r[I] ≈ StokesParams(CoherencyMatrix(c[I]..., b))
            end
        end
    end

    @testset "other containers" begin
        sa = StructArray{StokesParams{Float64}}((rand(6, 5), rand(6, 5), rand(6, 5), rand(6, 5)))
        for b in bases
            ref = coherencymap(stokesview(cat(sa.I, sa.Q, sa.U, sa.V; dims = 3), g), b)
            cs = coherencymap(IntensityMap(sa, g), b)
            @test cs isa CoherencyMap{ComplexF64, 2}
            @test baseimage(cs) isa StructArray
            @test cs ≈ ref
            e = coherency(cs, 2, 1)
            @test parent(e) === StructArrays.component(baseimage(cs), 2)
            ca = coherencymap(IntensityMap(collect(sa), g), b)
            @test baseimage(ca) isa Array
            @test ca ≈ ref
            @test coherency(ca, 1, 2) == coherency(ref, 1, 2)
            @test stokesmap(cs, b) ≈ sa
        end
    end

    @testset "components" begin
        c = coherencymap(vis, CirBasis())
        e12 = @inferred coherency(c, 1, 2)
        @test e12 isa IntensityMap{ComplexF64, 2}
        @test !(e12 isa CoherencyMap)
        @test axisdims(e12) === dfr
        @test e12 == dense(c)[:, :, 1, 2]
        @test e12 ≈ stokes(vis, :Q) .+ im .* stokes(vis, :U)
        e12[1, 1] = 0
        @test dense(c)[1, 1, 1, 2] == 0
        @test typeof(coherency(c, 2, 1)) === typeof(e12)
        coherency(c, 2, 1)
        @test (@allocated coherency(c, 2, 1)) == 0
        JET.@test_opt target_modules = (ComradeBase,) coherency(c, 2, 1)
        @test_throws "feed indices must be 1 or 2; got (3, 1)" coherency(c, 3, 1)
    end

    @testset "unsupported basis" begin
        c = coherencymap(img, CirBasis())
        msg = "the supported bases are CirBasis() and LinBasis()"
        @test_throws msg coherencymap(img, (CirBasis(), LinBasis()))
        @test_throws msg coherencymap(img, :circular)
        @test_throws msg stokesmap(c, (CirBasis(), CirBasis()))
    end

    @testset "slicing" begin
        c = coherencymap(img, CirBasis())
        sx = c[X = 2:4]
        @test sx isa CoherencyMap{ComplexF64, 2}
        @test dense(sx) == dense(c)[2:4, :, :, :]
        @test c[X = 2, Y = 3] === c[2, 3]
        @test c[X = 2, Y = 3] isa SMatrix{2, 2, ComplexF64}

        cs = coherencymap(vis, LinBasis())
        sp = cs[Pt(2:4)]
        @test sp isa CoherencyMap{ComplexF64, 2}
        @test ComradeBase.coords(axisdims(sp)).u == ComradeBase.coords(dfr).u[2:4]
        @test cs[Fr(1)] isa CoherencyMap{ComplexF64, 1}
        @test cs[Pt(2)] isa DD.DimArray
    end

    @testset "similar, copy and broadcasting" begin
        c = coherencymap(img, CirBasis())
        s = @inferred similar(c)
        @test typeof(s) === typeof(c)
        @test dense(s) !== dense(c)
        cc = @inferred copy(c)
        @test typeof(cc) === typeof(c)
        @test dense(cc) == dense(c)
        r = @inferred times2(c)
        @test typeof(r) === typeof(c)
        @test dense(r) ≈ 2 .* dense(c)
        @test conj.(c) isa CoherencyMap
    end

    @testset "executors and precision" begin
        for ex in (Serial(), ThreadsEx(), CPU())
            vex = IntensityMap(copy(baseimage(vis)), DD.rebuild(dfr; executor = ex))
            @test dense(coherencyroundtrip(vex, LinBasis())) ≈ dense(vis)
        end
        img32 = stokesview(rand(Float32, 6, 5, 4), g)
        for b in bases
            c32 = coherencymap(img32, b)
            @test c32 isa CoherencyMap{ComplexF32, 2}
            @test stokesmap(c32, b) isa StokesMap{ComplexF32, 2}
        end
    end

    @testset "type stability" begin
        JET.@test_opt target_modules = (ComradeBase,) coherencymap(img, CirBasis())
        JET.@test_opt target_modules = (ComradeBase,) coherencymap(vis, LinBasis())
        c = coherencymap(vis, CirBasis())
        JET.@test_opt target_modules = (ComradeBase,) stokesmap(c, CirBasis())
        JET.@test_opt target_modules = (ComradeBase,) stokesmap(c, LinBasis())
        @test @inferred(coherencyroundtrip(img, LinBasis())) isa StokesMap{ComplexF64, 2}
    end
end

@testset "Polarized maps under Reactant" begin
    m = PolTest(1.5)
    mr = Reactant.to_rarray(m; track_numbers = Number)
    g = spatialgrid(10.0, 12.0, 8, 6)
    gr = @jit identity(g)

    ip = @jit intensitymap(mr, gr)
    @test ip isa StokesMap
    @test Array(dense(ip)) ≈ dense(intensitymap(m, g))
    test_clean_hlo(repr(@code_hlo intensitymap(mr, gr)))

    guv = RectiGrid((U(range(-0.2, 0.2; length = 6)), V(range(-0.2, 0.2; length = 5))))
    @test Array(dense(@jit visibilitymap(mr, Reactant.to_rarray(guv)))) ≈ dense(visibilitymap(m, guv))
    test_clean_hlo(repr(@code_hlo visibilitymap(mr, Reactant.to_rarray(guv))))

    dfr = StructuredDomain((Pt(6), Fr([230.0e9, 345.0e9])); u = 3.0e4 .* randn(6), v = 3.0e4 .* randn(6))
    dfrr = Reactant.to_rarray(dfr)
    vp = @jit visibilitymap(mr, dfrr)
    @test vp isa StokesMap
    @test Array(dense(vp)) ≈ dense(visibilitymap(m, dfr))
    test_clean_hlo(repr(@code_hlo visibilitymap(mr, dfrr)))

    img = stokesview(rand(8, 6, 4), g)
    rimg = Reactant.to_rarray(img)
    @test rimg isa StokesMap{Float64, 2}
    r = @jit times2(rimg)
    @test r isa StokesMap
    @test dims(r) == dims(img)
    @test Array(dense(r)) ≈ 2 .* dense(img)
    @test Float64(@jit slabloss(rimg)) ≈ slabloss(img)
    @test Float64(@jit fluxQ(rimg)) ≈ flux(img).Q
    g4 = spatialgrid(10.0, 12.0, 8, 6) ⊗ Fr([230.0e9, 345.0e9])
    img4 = stokesview(rand(8, 6, 2, 4), g4)
    @test vec(Array(@jit fluxQmap(Reactant.to_rarray(img4)))) ≈ vec(baseimage(flux(img4)).Q)
    test_clean_hlo(repr(@code_hlo times2(rimg)))

    sa = StructArray{StokesParams{Float64}}((rand(8, 6), rand(8, 6), rand(8, 6), rand(8, 6)))
    rs = @jit times2(Reactant.to_rarray(IntensityMap(sa, g)))
    @test baseimage(rs) isa StructArray
    @test Array(baseimage(rs).Q) ≈ 2 .* sa.Q

    for b in (CirBasis(), LinBasis())
        rc = @jit coherencymap(rimg, b)
        @test rc isa CoherencyMap
        @test size(dense(rc)) == (8, 6, 2, 2)
        @test Array(dense(rc)) ≈ dense(coherencymap(img, b))
        @test Array(dense(@jit stokesmap(rc, b))) ≈ dense(img)
        test_clean_hlo(repr(@code_hlo coherencymap(rimg, b)))
        test_clean_hlo(repr(@code_hlo stokesmap(rc, b)))

        img32 = stokesview(rand(Float32, 8, 6, 4), g)
        rc32 = @jit coherencymap(Reactant.to_rarray(img32), b)
        @test rc32 isa CoherencyMap{ComplexF32}
        @test Array(dense(rc32)) ≈ dense(coherencymap(img32, b))
        @test !occursin("f64", repr(@code_hlo coherencymap(Reactant.to_rarray(img32), b)))

        dpt = UnstructuredDomain((; U = 0.1 .* randn(9), V = 0.1 .* randn(9)))
        vpt = visibilitymap(m, dpt)
        rpt = Reactant.to_rarray(vpt)
        @test Array(dense(@jit coherencyroundtrip(rpt, b))) ≈ dense(vpt)
        test_clean_hlo(repr(@code_hlo coherencyroundtrip(rpt, b)))
    end
end

@testset "Sharding polarized maps" begin
    ndev = length(Reactant.devices())
    if ndev == 1
        @warn "Polarized sharding tests skipped: start Julia with XLA_FLAGS=--xla_force_host_platform_device_count=4"
    else
        mesh = Reactant.Sharding.Mesh(reshape(Reactant.devices(), ndev), (:d,))
        loss(P, g) = slabloss(stokesview(P, g))
        gradient(P, g) = Enzyme.gradient(Enzyme.Reverse, loss, P, Enzyme.Const(g))[1]
        function expected_gradient(P)
            gr = zero(P)
            selectdim(gr, ndims(P), 2) .= 2 .* selectdim(P, ndims(P), 2)
            selectdim(gr, ndims(P), 4) .= 1
            return gr
        end

        @testset "along X" begin
            nx = 2ndev
            g = spatialgrid(10.0, 10.0, nx, 4)
            P = rand(nx, 4, 4)
            img = stokesview(P, g)
            simg = shard(img, ShardLayout(mesh; X = :d))
            @test simg isa StokesMap
            @test baseimage(simg) isa FieldDimArray
            @test axisdims(simg) === axisdims(img)
            @test stored_blocks(dense(simg), 1) == split_blocks(nx, ndev)
            @test stored_blocks(dense(simg), 3) == [1:4]
            r = @jit times2(simg)
            @test Array(dense(r)) ≈ 2 .* P
            @test stored_blocks(dense(r), 1) == split_blocks(nx, ndev)
            @test Float64(@jit slabloss(simg)) ≈ slabloss(img)
            gsh = @jit gradient(dense(simg), g)
            gun = Array(@jit gradient(Reactant.to_rarray(P), g))
            @test Array(gsh) ≈ gun
            @test stored_blocks(gsh, 1) == split_blocks(nx, ndev)
            @test gun ≈ expected_gradient(P)
            @test_throws "available dimensions are (:X, :Y)" shard(img, ShardLayout(mesh; Stokes = :d))
        end

        @testset "along Fr of a 4-d map" begin
            nf = 2ndev
            g = spatialgrid(10.0, 10.0, 6, 5) ⊗ Ti([0.0, 1.0]) ⊗ Fr(range(86.0e9, 345.0e9; length = nf))
            P = rand(6, 5, 2, nf, 4)
            img = stokesview(P, g)
            simg = shard(img, ShardLayout(mesh; Fr = :d))
            @test stored_blocks(dense(simg), 4) == split_blocks(nf, ndev)
            @test stored_blocks(dense(simg), 5) == [1:4]
            @test stored_blocks(dense(simg), 3) == [1:2]
            @test Array(dense(@jit times2(simg))) ≈ 2 .* P

            gsh = Array(@jit gradient(dense(simg), g))
            gun = Array(@jit gradient(Reactant.to_rarray(P), g))
            @test all(isfinite, gsh)
            @test gsh ≈ gun
            @test gun ≈ expected_gradient(P)
        end

        @testset "coherency of a (Pt, Fr) map along Fr" begin
            nf = 2ndev
            d = StructuredDomain(
                (Pt(6), Fr(range(86.0e9, 345.0e9; length = nf)));
                u = 3.0e4 .* randn(6), v = 3.0e4 .* randn(6)
            )
            vis = visibilitymap(PolTest(1.5), d)
            svis = shard(vis, ShardLayout(mesh; Fr = :d))
            @test stored_blocks(dense(svis), 2) == split_blocks(nf, ndev)
            c = coherencymap(vis, CirBasis())
            test_clean_hlo(repr(@code_hlo coherencymap(svis, CirBasis())))
            test_clean_hlo(repr(@code_hlo stokesmap(shard(c, ShardLayout(mesh; Fr = :d)), CirBasis())))
            rt = @jit coherencyroundtrip(svis, CirBasis())
            @test stored_blocks(dense(rt), 2) == split_blocks(nf, ndev)
            @test stored_blocks(dense(rt), 3) == [1:4]
            @test Array(dense(rt)) ≈ dense(vis)
            sc = @jit coherencymap(svis, CirBasis())
            @test sc isa CoherencyMap
            @test Array(dense(sc)) ≈ dense(c)
            @test stored_blocks(dense(sc), 2) == split_blocks(nf, ndev)
            @test stored_blocks(dense(sc), 3) == [1:2]
            # The returned domain coordinates are replicated arrays that come back without a
            # sharding; passing them through a second `@jit` next to sharded arrays throws
            # Reactant's `TODO(#2234)` error.
            @test_broken (@jit(stokesmap(sc, CirBasis())); true)
            shc = shard(c, ShardLayout(mesh; Fr = :d))
            @test shc isa CoherencyMap
            @test stored_blocks(dense(shc), 2) == split_blocks(nf, ndev)
            @test stored_blocks(dense(shc), 3) == [1:2]
            @test_throws "available dimensions are (:Pt, :Fr)" shard(c, ShardLayout(mesh; Fa = :d))
        end
    end
end

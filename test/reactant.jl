using Reactant
Reactant.set_default_backend("cpu")

# The optimized module of a map computation is plain elementwise work over full arrays.
function test_clean_hlo(hlo)
    for op in ("stablehlo.while", "stablehlo.scatter", "enzyme.batch", "dynamic_slice", "dynamic_update_slice")
        @test !occursin(op, hlo)
    end
    return nothing
end

@testset "Reactant" begin
    x = rand(54, 32)
    r = Reactant.to_rarray(x)

    @test @jit(ComradeBase.rgetindex(r, 10)) ≈ x[10]

    @test ComradeBase.rgetindex(x, 1:10) ≈ x[1:10]
    @test ComradeBase.rgetindex(x, :, 1) ≈ x[:, 1]
    @test ComradeBase.rgetindex(x, 1) ≈ x[1]

    @test @jit(ComradeBase.rgetindex(r, 1:10)) ≈ x[1:10]
    @test @jit(ComradeBase.rgetindex(r, :, 1)) ≈ x[:, 1]
    @test @jit(ComradeBase.rgetindex(r, 1:2, 1:2)) ≈ x[1:2, 1:2]

    @jit(ComradeBase.rsetindex!(r, 3.14, 20))
    @test @allowscalar ComradeBase.rgetindex(r, 20) ≈ 3.14
    @jit(ComradeBase.setindex!(r, ones(10, 10), 1:10, 1:10))
    @test ComradeBase.rgetindex(r, 1:10, 1:10) ≈ ones(10, 10)

    g = spatialgrid(10.0, 10.0, 8, 8)
    go = @jit(identity(g))
    @test executor(go) isa ComradeBase.ReactantEx

    m1 = BlobTest(4.0)
    m2 = @jit BlobTest(ConcreteRNumber(m1.size))

    guv = UnstructuredDomain((; U = 0.2 * randn(64), V = 0.2 * randn(64)))
    guvr = Reactant.to_rarray(guv)

    @test baseimage(@jit(intensitymap(m2, go))) ≈ baseimage(intensitymap(m1, g))
    @test baseimage(@jit(visibilitymap(m2, guvr))) ≈ baseimage(visibilitymap(m1, guv))
    test_clean_hlo(repr(@code_hlo intensitymap(m2, go)))
    test_clean_hlo(repr(@code_hlo visibilitymap(m2, guvr)))

    for (extra, posang) in (((Fr([230.0e9, 345.0e9]),), 0.0), ((), 0.3), ((Fr([230.0e9, 345.0e9]),), 0.3))
        gf = gridproduct(spatialgrid(10.0, 10.0, 8, 6; posang), extra...)
        @test Array(baseimage(@jit(intensitymap(m2, @jit(identity(gf)))))) ≈ baseimage(intensitymap(m1, gf))
        guvf = RectiGrid((U(range(-0.2, 0.2; length = 8)), V(range(-0.2, 0.2; length = 6)), extra...); posang)
        @test Array(baseimage(@jit(visibilitymap(m2, @jit(identity(guvf)))))) ≈ baseimage(visibilitymap(m1, guvf))
        test_clean_hlo(repr(@code_hlo intensitymap(m2, @jit(identity(gf)))))
        test_clean_hlo(repr(@code_hlo visibilitymap(m2, @jit(identity(guvf)))))
    end

    g32 = RectiGrid(
        (X(range(-1.0f0, 1.0f0; length = 4)), Y(range(-2.0f0, 2.0f0; length = 3)), Fr([230.0e9, 345.0e9]));
        posang = 0.3f0
    )
    g32r = @jit(identity(g32))
    img32 = @jit(intensitymap(PointSum(), g32r))
    @test eltype(baseimage(img32)) === Float32
    @test Array(baseimage(img32)) ≈ baseimage(intensitymap(PointSum(), g32))
    @test !occursin("f64", repr(@code_hlo intensitymap(PointSum(), g32r)))

    img1 = intensitymap(m1, g)
    img2 = @jit(intensitymap(m2, go))

    # circ shift the image so we get an actual centroid
    img1 = circshift(img1, (2, 3))
    img2 = Reactant.to_rarray(img1) #circshift stackoverflows with Reactant TODO: fix this

    c1 = centroid(img1)
    c2 = @jit(centroid(img2))
    @test c1[1] ≈ c2[1]
    @test c1[2] ≈ c2[2]

    @testset "StructuredDomain" begin
        npt = 6
        ti = [0.0, 1.0, 2.0]
        fr = [230.0e9, 345.0e9]
        U = randn(Float32, npt, 3, 2)
        V = randn(Float32, npt, 3, 2)
        d = ComradeBase.StructuredDomain((ComradeBase.Node(npt), Ti(ti), Fr(fr)); U, V)
        dr = Reactant.to_rarray(d)
        @test ComradeBase.coords(dr).U isa Reactant.ConcreteRArray{Float32, 3}
        @test dims(dr) === dims(d)
        @test executor(dr) isa ComradeBase.ReactantEx
        @test @jit(sum(ComradeBase.coords(dr).U)) ≈ sum(U)
        @test @jit(sum(ComradeBase.coords(d).U)) ≈ sum(U)

        dout = @jit(identity(dr))
        @test dout isa ComradeBase.StructuredDomain
        @test dims(dout) == dims(d)
        @test parent(parent(dims(dout, Fr))) isa Vector{Float64}
        @test Array(ComradeBase.coords(dout).V) ≈ V

        dm = ComradeBase.StructuredDomain(
            (ComradeBase.Node(npt), Fr(fr)); u = 1.0e6 .* randn(npt), v = 1.0e6 .* randn(npt)
        )
        sumU(d) = sum(d.U)
        @test @jit(sumU(Reactant.to_rarray(dm))) ≈ sum(dm.U)

        function vissize(m, d)
            vis = ComradeBase.allocate_vismap(m, d)
            return size(baseimage(vis)), baseimage(vis) isa Reactant.TracedRArray{ComplexF32, 3}
        end
        @test @jit(vissize(m1, dr)) == ((npt, 3, 2), true)
        function visfill(m, d)
            vis = ComradeBase.allocate_vismap(m, d)
            baseimage(vis) .= 1
            return baseimage(vis)
        end
        filled = @jit(visfill(m1, dr))
        @test size(filled) == size(d)
        @test all(==(1), Array(filled))
        function polsize(m, d)
            vis = ComradeBase.allocate_vismap(ComradeBase.IsPolarized(), m, d)
            return size(baseimage(stokes(vis, :Q)))
        end
        @test @jit(polsize(m1, dr)) == (npt, 3, 2)

        @testset "analytic maps under @jit" begin
            mp = PolTest(1.5)
            mpr = @jit PolTest(ConcreteRNumber(mp.size))
            d2 = ComradeBase.StructuredDomain(
                (ComradeBase.Node(npt), Fr(fr)); u = 3.0e4 .* randn(npt), v = 3.0e4 .* randn(npt)
            )
            d3 = ComradeBase.StructuredDomain(
                (ComradeBase.Node(npt), Ti(ti), Fr(fr)); U = 0.1 .* randn(npt, 3, 2), V = 0.1 .* randn(npt, 3, 2)
            )
            for d in (d2, d3)
                dr = Reactant.to_rarray(d)
                @test Array(baseimage(@jit(visibilitymap(m2, dr)))) ≈ baseimage(visibilitymap(m1, d))
                vp = @jit(visibilitymap(mpr, dr))
                @test vp isa StokesMap
                @test Array(parent(baseimage(vp))) ≈ parent(baseimage(visibilitymap(mp, d)))
                test_clean_hlo(repr(@code_hlo visibilitymap(m2, dr)))
                test_clean_hlo(repr(@code_hlo visibilitymap(mpr, dr)))
            end
            dpt = UnstructuredDomain((; U = 0.2 .* randn(npt), V = 0.2 .* randn(npt)))
            vpt = @jit(visibilitymap(mpr, Reactant.to_rarray(dpt)))
            @test Array(parent(baseimage(vpt))) ≈ parent(baseimage(visibilitymap(mp, dpt)))
            test_clean_hlo(repr(@code_hlo visibilitymap(mpr, Reactant.to_rarray(dpt))))
            dxy = UnstructuredDomain((X = randn(npt), Y = randn(npt)))
            @test Array(baseimage(@jit(intensitymap(m2, Reactant.to_rarray(dxy))))) ≈ baseimage(intensitymap(m1, dxy))
            test_clean_hlo(repr(@code_hlo intensitymap(m2, Reactant.to_rarray(dxy))))
        end
    end
end

# Index ranges along `dim` held by each device, deduplicated and sorted.
function stored_blocks(a, dim)
    slices = a.sharding.device_to_array_slices
    return sort!(unique(map(s -> s[dim], slices)); by = first)
end
# Blocks of `n` elements split over `k` devices, with the last block padded when needed.
function split_blocks(n, k)
    b = cld(n, k)
    return [((i - 1) * b + 1):(i * b) for i in 1:k]
end

@testset "Sharding" begin
    @testset "ShardLayout construction" begin
        l = ShardLayout(:mesh; Ti = :t, Fr = :f)
        @test l.mesh === :mesh
        @test l.axes == (Ti = :t, Fr = :f)
        @test ShardLayout(:mesh; X = (:a, :b)).axes == (X = (:a, :b),)
        @test_throws "requires at least one dimension" ShardLayout(:mesh)
        @test_throws "the value for dimension `X` must be a `Symbol` or a tuple of `Symbol`s" ShardLayout(:mesh; X = "a")
        @test_throws "the value for dimension `Fr` must be" ShardLayout(:mesh; Fr = (:a, 1))
        @test_throws "the value for dimension `Fr` must be" ShardLayout(:mesh; Fr = ())
    end

    @testset "ReactantEx" begin
        @test Base.issingletontype(typeof(ReactantEx()))
        dvis = UnstructuredDomain((; U = randn(4), V = randn(4)))
        @test executor(dvis) isa Serial
        @test executor(Reactant.to_rarray(dvis)) === ReactantEx()
        @test executor(@jit(identity(spatialgrid(10.0, 10.0, 8, 8)))) === ReactantEx()
    end

    @info "Reactant runtime: $(Reactant.XLA.REACTANT_XLA_RUNTIME)"
    @test Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"

    ndev = length(Reactant.devices())
    img0 = IntensityMap(rand(2ndev, 4), spatialgrid(10.0, 10.0, 2ndev, 4))
    mesh1 = Reactant.Sharding.Mesh(reshape(Reactant.devices()[1:1], 1), (:d,))
    @test_throws "single-device meshes are not supported by Reactant" shard(img0, ShardLayout(mesh1; X = :d))
    @test_throws "single-device meshes are not supported by Reactant" shard(img0, Reactant.Sharding.DimsSharding(mesh1, (1,), (:d,)))
    @test_throws "must be a `Reactant.Sharding.Mesh`" shard(img0, ShardLayout(:notamesh; X = :d))

    if ndev == 1
        @warn "Multi-device sharding tests skipped: start Julia with XLA_FLAGS=--xla_force_host_platform_device_count=4"
    else
        mesh = Reactant.Sharding.Mesh(reshape(Reactant.devices(), ndev), (:d,))
        mesh2 = Reactant.Sharding.Mesh(reshape(Reactant.devices(), ndev, 1), (:a, :b))

        @testset "Validation" begin
            @test_throws "dimension `Fr` is not a dimension of the IntensityMap; available dimensions are (:X, :Y)" shard(img0, ShardLayout(mesh; Fr = :d))
            @test_throws "mesh axis `q` (for dimension `X`) is not in the mesh; available mesh axes are (:d,)" shard(img0, ShardLayout(mesh; X = :q))
            @test_throws "mesh axis `c` (for dimension `X`) is not in the mesh" shard(img0, ShardLayout(mesh2; X = (:a, :c)))
        end

        @testset "Partition evidence" begin
            x = rand(2ndev, 3)
            split = Reactant.to_rarray(x; sharding = Reactant.Sharding.DimsSharding(mesh, (1,), (:d,)))
            replicated = Reactant.to_rarray(x; sharding = Reactant.Sharding.Replicated(mesh))
            @test stored_blocks(split, 1) == split_blocks(2ndev, ndev)
            @test stored_blocks(replicated, 1) == [1:(2ndev)]
            @test Reactant.Sharding.is_sharded(replicated)
        end

        @testset "IntensityMap along X" begin
            nx = 2ndev
            img = IntensityMap(rand(nx, 6), spatialgrid(10.0, 10.0, nx, 6))
            simg = shard(img, ShardLayout(mesh; X = :d))
            @test axisdims(simg) === axisdims(img)
            f(a) = baseimage(a) .* 2 .+ sum(baseimage(a))
            @test Array(@jit(f(simg))) ≈ f(img)
            @test stored_blocks(baseimage(simg), 1) == split_blocks(nx, ndev)
            @test stored_blocks(baseimage(simg), 2) == [1:6]
            simg2 = @jit((a -> a .* 2)(simg))
            @test simg2 isa IntensityMap
            @test Array(baseimage(simg2)) ≈ 2 .* baseimage(img)
        end

        @testset "Non-divisible dimension is padded" begin
            nx = ndev + 1
            img = IntensityMap(rand(nx, 4), spatialgrid(10.0, 10.0, nx, 4))
            simg = shard(img, ShardLayout(mesh; X = :d))
            @test stored_blocks(baseimage(simg), 1) == split_blocks(nx, ndev)
            @test last(last(stored_blocks(baseimage(simg), 1))) > nx
            f(a) = baseimage(a) .* 2 .+ sum(baseimage(a))
            @test Array(@jit(f(simg))) ≈ f(img)
        end

        @testset "IntensityMap along Fr" begin
            nf = 2ndev
            x = X(range(-10.0, 10.0; length = 6))
            y = Y(range(-10.0, 10.0; length = 6))
            g = RectiGrid((x, y, Ti([0.0, 0.5, 0.8]), Fr(range(86.0e9, 345.0e9; length = nf))))
            img = IntensityMap(rand(6, 6, 3, nf), g)
            f(a) = sum(baseimage(a); dims = (1, 2))
            for layout in (ShardLayout(mesh; Fr = :d), ShardLayout(mesh2; Fr = :a, Ti = :b), ShardLayout(mesh2; Fr = (:a, :b)))
                simg = shard(img, layout)
                @test Array(@jit(f(simg))) ≈ f(img)
                @test stored_blocks(baseimage(simg), 4) == split_blocks(nf, ndev)
                @test stored_blocks(baseimage(simg), 3) == [1:3]
            end
        end

        @testset "Polarized IntensityMap along X" begin
            nx = 2ndev
            img = IntensityMap(FieldDimArray{StokesParams}(rand(nx, 4, 4)), spatialgrid(10.0, 10.0, nx, 4))
            simg = shard(img, ShardLayout(mesh; X = :d))
            f(a) = sum(abs2, baseimage(stokes(a, :Q))) + sum(baseimage(stokes(a, :V)))
            @test Float64(@jit(f(simg))) ≈ f(img)
            @test stored_blocks(parent(baseimage(simg)), 1) == split_blocks(nx, ndev)
            @test stored_blocks(parent(baseimage(simg)), 3) == [1:4]
        end

        @testset "Raw sharding of a (Node,) StructuredDomain" begin
            nvis = 16ndev
            U = 0.2 .* randn(nvis)
            V = 0.2 .* randn(nvis)
            dvis = UnstructuredDomain((; U, V))
            m = BlobTest(4.0)
            mr = Reactant.to_rarray(m)
            sdvis = shard(dvis, Reactant.Sharding.DimsSharding(mesh, (1,), (:d,)))
            @test executor(sdvis) === ReactantEx()
            vis = @jit(visibilitymap(mr, sdvis))
            @test Array(baseimage(vis)) ≈ baseimage(visibilitymap(m, dvis))
            @test stored_blocks(ComradeBase.coords(sdvis).U, 1) == split_blocks(nvis, ndev)
            @test stored_blocks(baseimage(vis), 1) == split_blocks(nvis, ndev)
            test_clean_hlo(repr(@code_hlo visibilitymap(mr, sdvis)))
            mp = PolTest(1.5)
            mpr = @jit PolTest(ConcreteRNumber(mp.size))
            vp = @jit(visibilitymap(mpr, sdvis))
            @test Array(parent(baseimage(vp))) ≈ parent(baseimage(visibilitymap(mp, dvis)))
            @test stored_blocks(parent(baseimage(vp)), 1) == split_blocks(nvis, ndev)
            @test stored_blocks(parent(baseimage(vp)), 2) == [1:4]
            test_clean_hlo(repr(@code_hlo visibilitymap(mpr, sdvis)))
        end

        npt = 10
        nf = 2ndev
        frs = Fr(range(200.0e9, 300.0e9; length = nf))
        dvf = StructuredDomain((Node(npt), frs); u = 1.0e5 .* randn(npt, nf), v = 1.0e5 .* randn(npt, nf), Ti = rand(npt))

        @testset "StructuredDomain along Fr" begin
            sd = shard(dvf, ShardLayout(mesh; Fr = :d))
            @test dims(sd) === dims(dvf)
            @test executor(sd) === ReactantEx(ShardLayout(mesh; Fr = :d))
            sdata = shard(IntensityMap(randn(npt, nf), dvf), @jit(identity(sd)))
            @test stored_blocks(baseimage(sdata), 2) == split_blocks(nf, ndev)
            cs = ComradeBase.coords(sd)
            @test Array(cs.u) == ComradeBase.coords(dvf).u
            @test stored_blocks(cs.u, 2) == split_blocks(nf, ndev)
            @test stored_blocks(cs.u, 1) == [1:npt]
            @test stored_blocks(cs.Ti, 1) == [1:npt]
            m = BlobTest(4.0)
            vis = @jit(visibilitymap(m, sd))
            @test Array(baseimage(vis)) ≈ baseimage(visibilitymap(m, dvf))
            @test stored_blocks(baseimage(vis), 2) == split_blocks(nf, ndev)
            @test_throws "dimension `Ti` is not a dimension of the StructuredDomain; available dimensions are (:Node, :Fr)" shard(dvf, ShardLayout(mesh; Ti = :d))
        end

        @testset "StructuredDomain along Node" begin
            dpt = UnstructuredDomain((; U = 0.2 .* randn(4ndev), V = 0.2 .* randn(4ndev)))
            sd = shard(dpt, ShardLayout(mesh; Node = :d))
            @test stored_blocks(ComradeBase.coords(sd).U, 1) == split_blocks(4ndev, ndev)
            m = BlobTest(4.0)
            @test Array(baseimage(@jit(visibilitymap(m, sd)))) ≈ baseimage(visibilitymap(m, dpt))
        end

        @testset "IntensityMap over a StructuredDomain" begin
            vis = visibilitymap(BlobTest(4.0), dvf)
            svis = shard(vis, ShardLayout(mesh; Fr = :d))
            @test stored_blocks(baseimage(svis), 2) == split_blocks(nf, ndev)
            @test stored_blocks(ComradeBase.coords(axisdims(svis)).v, 2) == split_blocks(nf, ndev)
            @test Array(baseimage(svis)) == baseimage(vis)

            c = IntensityMap(FieldDimArray{SMatrix{2, 2}}(rand(ComplexF64, npt, nf, 2, 2)), dvf)
            sc = shard(c, ShardLayout(mesh; Fr = :d))
            @test sc isa CoherencyMap
            @test stored_blocks(parent(baseimage(sc)), 2) == split_blocks(nf, ndev)
            @test stored_blocks(parent(baseimage(sc)), 3) == [1:2]
            @test stored_blocks(parent(baseimage(sc)), 4) == [1:2]
            @test stored_blocks(ComradeBase.coords(axisdims(sc)).u, 2) == split_blocks(nf, ndev)
            f(c) = sum(abs2, baseimage(coherency(c, 1, 2)))
            @test Float64(@jit(f(sc))) ≈ f(c)
        end

        if ndev == 4
            @testset "Image stack on a 2×2 (Ti, Fr) mesh" begin
                mesh22 = Reactant.Sharding.Mesh(reshape(Reactant.devices(), 2, 2), (:t, :f))
                g = RectiGrid((X(range(-10.0, 10.0; length = 6)), Y(range(-10.0, 10.0; length = 6)), Ti([0.0, 0.5, 0.8, 1.2]), Fr([86.0e9, 230.0e9])))
                img = IntensityMap(rand(6, 6, 4, 2), g)
                simg = shard(img, ShardLayout(mesh22; Ti = :t, Fr = :f))
                @test axisdims(simg) === g
                @test stored_blocks(baseimage(simg), 3) == split_blocks(4, 2)
                @test stored_blocks(baseimage(simg), 4) == split_blocks(2, 2)
                @test stored_blocks(baseimage(simg), 1) == [1:6]
                f(a) = sum(baseimage(a); dims = (1, 2))
                @test Array(@jit(f(simg))) ≈ f(img)
            end
        end

        @testset "Collections and dual domains" begin
            dvf2 = StructuredDomain((Node(npt + 3), frs); U = randn(npt + 3, nf), V = randn(npt + 3, nf))
            layout = ShardLayout(mesh; Fr = :d)
            st = shard((dvf, dvf2), layout)
            @test st isa Tuple{<:StructuredDomain, <:StructuredDomain}
            @test stored_blocks(ComradeBase.coords(st[2]).U, 2) == split_blocks(nf, ndev)
            snt = shard((; a = dvf, b = dvf2), layout)
            @test keys(snt) == (:a, :b)
            @test stored_blocks(ComradeBase.coords(snt.a).u, 2) == split_blocks(nf, ndev)
            @test_throws "dimension `Fr` is not a dimension of the StructuredDomain" shard((dvf, UnstructuredDomain((; U = randn(3), V = randn(3)))), layout)

            gim = RectiGrid((X(range(-10.0, 10.0; length = 6)), Y(range(-10.0, 10.0; length = 6)), Ti([0.0, 0.5, 0.8, 1.2]), frs))
            dd = DualDomain(gim, dvf)
            sdd = shard(dd, ShardLayout(mesh; Fr = :d))
            @test sdd isa DualDomain
            @test imgdomain(sdd) === gim
            @test dims(visdomain(sdd)) === dims(dvf)
            @test stored_blocks(ComradeBase.coords(visdomain(sdd)).u, 2) == split_blocks(nf, ndev)
            @test !haskey(ComradeBase.coords(visdomain(sdd)), :valid)
            sdata = shard(IntensityMap(randn(npt, nf), dvf), visdomain(sdd))
            @test stored_blocks(baseimage(sdata), 2) == split_blocks(nf, ndev)
            @test stored_blocks(baseimage(sdata), 1) == [1:npt]
            @test_throws "dimension `Pol` is not a dimension of the DualDomain" shard(dd, ShardLayout(mesh; Pol = :d))
        end

        @testset "Dual domain blocked along Ti" begin
            nt = 2ndev
            ti = frames(Ti, collect(0.0:(nt - 1)), collect(0.0:(nt - 1)) .+ 0.9)
            occupancy = [k == 1 ? 3 : 1 for k in 1:nt]
            t = reduce(vcat, [fill(k - 0.5, occupancy[k]) for k in 1:nt])
            n = length(t)
            vis = StructuredDomain((; U = 0.2 .* randn(n), V = 0.2 .* randn(n), Ti = t))
            g = spatialgrid(10.0, 10.0, 6, 6) ⊗ ti
            sdd = shard(DualDomain(g, vis), ShardLayout(mesh; Ti = :d))
            sv = visdomain(sdd)
            Mb = 4
            slots = parent(DD.lookup(first(dims(sv))))
            @test length(slots) == Mb * ndev
            @test slots[1:Mb] == [1, 2, 3, 4]
            @test filter(!=(0), slots) == 1:n
            @test stored_blocks(ComradeBase.coords(sv).U, 1) == split_blocks(Mb * ndev, ndev)
            @test stored_blocks(ComradeBase.coords(sv).valid, 1) == split_blocks(Mb * ndev, ndev)
            @test Array(ComradeBase.coords(sv).valid) == (slots .!= 0)

            data = IntensityMap(randn(ComplexF64, n), vis)
            sdata = shard(data, sv)
            @test axisdims(sdata) === sv
            @test stored_blocks(baseimage(sdata), 1) == split_blocks(Mb * ndev, ndev)
            real = findall(!=(0), slots)
            @test Array(baseimage(sdata))[real] == baseimage(data)
            w = shard(IntensityMap(ones(n), vis), sv; pad = 0.0)
            @test Array(baseimage(w)) == (slots .!= 0)

            m = BlobTest(4.0)
            svis = @jit(visibilitymap(m, sv))
            @test stored_blocks(baseimage(svis), 1) == split_blocks(Mb * ndev, ndev)
            @test Array(baseimage(svis))[real] ≈ baseimage(visibilitymap(m, vis))
            hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings visibilitymap(m, sv))
            for op in ("all-reduce", "all-gather", "collective-permute", "all-to-all")
                @test !occursin(op, hlo)
            end

            @test_throws "sort the data by `Ti`" shard(DualDomain(g, StructuredDomain((; U = randn(n), V = randn(n), Ti = reverse(t)))), ShardLayout(mesh; Ti = :d))
            @test_throws "cannot also split `Node`" shard(DualDomain(g, vis), ShardLayout(mesh2; Ti = :a, Node = :b))
            @test_throws "the domain is not sharded" shard(data, vis)
            @test_throws "the domain is not sharded" shard(data, Reactant.to_rarray(vis))

            sdr = visdomain(shard(DualDomain(g, Reactant.to_rarray(vis)), ShardLayout(mesh; Ti = :d)))
            @test parent(DD.lookup(first(dims(sdr)))) == slots
            @test Array(ComradeBase.coords(sdr).U) == Array(ComradeBase.coords(sv).U)
            @test_throws "`Node` lookup contains 0" shard(DualDomain(g, sv), ShardLayout(mesh; Ti = :d))
            @test_throws "use a mesh axis twice" shard(IntensityMap(rand(8, 8), spatialgrid(10.0, 10.0, 8, 8)), ShardLayout(mesh; X = :d, Y = :d))
        end

        if ndev == 4
            @testset "Dual domains on a 2×2 mesh" begin
                mesh22 = Reactant.Sharding.Mesh(reshape(Reactant.devices(), 2, 2), (:t, :f))
                ti = frames(Ti, [0.0, 1.0, 2.0, 3.0], [0.9, 1.9, 2.9, 3.9])
                t = [0.1, 0.2, 0.3, 2.1, 2.2, 3.5]
                n = length(t)
                fr = Fr([86.0e9, 230.0e9])
                devblocks(a) = a.sharding.device_to_array_slices

                vis = StructuredDomain((Node(n), fr); u = 1.0e5 .* randn(n), v = 1.0e5 .* randn(n), Ti = t)
                g = spatialgrid(10.0, 10.0, 6, 6) ⊗ ti ⊗ fr
                layout = ShardLayout(mesh22; Ti = :t, Fr = :f)
                sv = visdomain(shard(DualDomain(g, vis), layout))
                @test size(sv) == (6, 2)
                data = shard(IntensityMap(randn(n, 2), vis), sv)
                img = shard(IntensityMap(rand(6, 6, 4, 2), g), layout)
                for (ib, db) in zip(devblocks(baseimage(img)), devblocks(baseimage(data)))
                    @test cld(last(ib[3]), 2) == cld(last(db[1]), 3)
                    @test ib[4] == db[2]
                end
                m = BlobTest(4.0)
                real = findall(!=(0), DD.lookup(dims(sv), Node))
                @test Array(baseimage(@jit(visibilitymap(m, sv))))[real, :] ≈ baseimage(visibilitymap(m, vis))
                @test_throws "use a mesh axis twice" shard(DualDomain(spatialgrid(10.0, 10.0, 6, 6) ⊗ ti, vis), ShardLayout(mesh22; Ti = :t, Fr = :t))
                dfr = shard(IntensityMap(randn(n, 2), vis), shard(vis, ShardLayout(mesh22; Fr = :f)))
                @test stored_blocks(baseimage(dfr), 2) == [1:1, 2:2]
                @test stored_blocks(baseimage(dfr), 1) == [1:n]

                f = [86.0e9, 230.0e9, 230.0e9, 86.0e9, 86.0e9, 230.0e9]
                vf = StructuredDomain((; U = randn(n), V = randn(n), Ti = t, Fr = f))
                svf = visdomain(shard(DualDomain(g, vf), layout))
                @test parent(DD.lookup(first(dims(svf)))) == [1, 0, 2, 3, 4, 5, 6, 0]
                dataf = shard(IntensityMap(randn(n), vf), svf)
                for (ib, db) in zip(devblocks(baseimage(img)), devblocks(baseimage(dataf)))
                    @test 2 * (cld(last(ib[3]), 2) - 1) + last(ib[4]) == cld(last(db[1]), 2)
                end
            end
        end

        @testset "Masked gather from a Ti-sharded stack" begin
            nt = 2ndev
            G = randn(ComplexF64, 16, 16, nt)
            n = 200
            idx = hcat(rand(1:16, n), rand(1:16, n), rand(1:nt, n))
            Gs = Reactant.to_rarray(G; sharding = Reactant.Sharding.DimsSharding(mesh, (3,), (:d,)))
            idxr = Reactant.to_rarray(idx; sharding = Reactant.Sharding.Replicated(mesh))
            gather(G, idx) = Reactant.Ops.gather_getindex(G, idx)
            @test Array(@jit(gather(Gs, idxr))) ≈ [G[idx[p, 1], idx[p, 2], idx[p, 3]] for p in 1:n]
            hlo = repr(@code_xla shardy_passes = :to_mhlo_shardings gather(Gs, idxr))
            @test !occursin("all-gather", hlo)
            @test occursin("all-reduce", hlo)

            loss(G, idx) = sum(abs2, gather(G, idx))
            function grad!(dG, G, idx)
                Enzyme.autodiff(Reverse, loss, Active, Duplicated(G, dG), Const(idx))
                return dG
            end
            dGs = Reactant.to_rarray(zero(G); sharding = Reactant.Sharding.DimsSharding(mesh, (3,), (:d,)))
            dG = @jit(grad!(dGs, Gs, idxr))
            host = zero(G)
            for p in 1:n
                host[idx[p, 1], idx[p, 2], idx[p, 3]] += 2G[idx[p, 1], idx[p, 2], idx[p, 3]]
            end
            @test Array(dG) ≈ host
            @test stored_blocks(dG, 3) == split_blocks(nt, ndev)
            @test !occursin("all-gather", repr(@code_xla shardy_passes = :to_mhlo_shardings grad!(dGs, Gs, idxr)))
        end

        @testset "Raw sharding of an IntensityMap keeps the grid on the host" begin
            simg = shard(img0, Reactant.Sharding.DimsSharding(mesh, (1,), (:d,)))
            @test axisdims(simg) === axisdims(img0)
            @test Array(baseimage(simg)) == baseimage(img0)
            @test stored_blocks(baseimage(simg), 1) == split_blocks(2ndev, ndev)
        end
    end
end

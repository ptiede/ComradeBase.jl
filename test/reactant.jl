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

    g = imagepixels(10.0, 10.0, 8, 8)
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

    for (mdims, posang) in (((Fr([230.0e9, 345.0e9]),), 0.0), ((), 0.3), ((Fr([230.0e9, 345.0e9]),), 0.3))
        gf = imagepixels(10.0, 10.0, 8, 6; mdims, posang)
        @test Array(baseimage(@jit(intensitymap(m2, @jit(identity(gf)))))) ≈ baseimage(intensitymap(m1, gf))
        guvf = RectiGrid((U(range(-0.2, 0.2; length = 8)), V(range(-0.2, 0.2; length = 6)), mdims...); posang)
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
        d = ComradeBase.StructuredDomain((ComradeBase.Pt(npt), Ti(ti), Fr(fr)); U, V)
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
            (ComradeBase.Pt(npt), Fr(fr)); u = 1.0e6 .* randn(npt), v = 1.0e6 .* randn(npt)
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
                (ComradeBase.Pt(npt), Fr(fr)); u = 3.0e4 .* randn(npt), v = 3.0e4 .* randn(npt)
            )
            d3 = ComradeBase.StructuredDomain(
                (ComradeBase.Pt(npt), Ti(ti), Fr(fr)); U = 0.1 .* randn(npt, 3, 2), V = 0.1 .* randn(npt, 3, 2)
            )
            for d in (d2, d3)
                dr = Reactant.to_rarray(d)
                @test Array(baseimage(@jit(visibilitymap(m2, dr)))) ≈ baseimage(visibilitymap(m1, d))
                vp = @jit(visibilitymap(mpr, dr))
                @test vp isa StokesMap
                @test Array(baseimage(vp)) ≈ baseimage(visibilitymap(mp, d))
                test_clean_hlo(repr(@code_hlo visibilitymap(m2, dr)))
                test_clean_hlo(repr(@code_hlo visibilitymap(mpr, dr)))
            end
            dpt = UnstructuredDomain((; U = 0.2 .* randn(npt), V = 0.2 .* randn(npt)))
            vpt = @jit(visibilitymap(mpr, Reactant.to_rarray(dpt)))
            @test Array(baseimage(vpt)) ≈ baseimage(visibilitymap(mp, dpt))
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
        @test Base.issingletontype(ReactantEx)
        dvis = UnstructuredDomain((; U = randn(4), V = randn(4)))
        @test executor(dvis) isa Serial
        @test executor(Reactant.to_rarray(dvis)) === ReactantEx()
        @test executor(@jit(identity(imagepixels(10.0, 10.0, 8, 8)))) === ReactantEx()
    end

    @info "Reactant runtime: $(Reactant.XLA.REACTANT_XLA_RUNTIME)"
    @test Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"

    ndev = length(Reactant.devices())
    img0 = IntensityMap(rand(2ndev, 4), imagepixels(10.0, 10.0, 2ndev, 4))
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
            @test_throws "dimension `Fr` is not a dimension of the image; available dimensions are (:X, :Y)" shard(img0, ShardLayout(mesh; Fr = :d))
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
            img = IntensityMap(rand(nx, 6), imagepixels(10.0, 10.0, nx, 6))
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
            img = IntensityMap(rand(nx, 4), imagepixels(10.0, 10.0, nx, 4))
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
            sa = StructArray{StokesParams{Float64}}((I = rand(nx, 4), Q = rand(nx, 4), U = rand(nx, 4), V = rand(nx, 4)))
            img = IntensityMap(sa, imagepixels(10.0, 10.0, nx, 4))
            simg = shard(img, ShardLayout(mesh; X = :d))
            f(a) = sum(abs2, baseimage(stokes(a, :Q))) + sum(baseimage(stokes(a, :V)))
            @test Float64(@jit(f(simg))) ≈ f(img)
            @test stored_blocks(baseimage(simg), 1) == split_blocks(nx, ndev)
            @test stored_blocks(baseimage(simg), 3) == [1:4]
        end

        @testset "Raw sharding of a (Pt,) StructuredDomain" begin
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
            @test Array(baseimage(vp)) ≈ baseimage(visibilitymap(mp, dvis))
            @test stored_blocks(baseimage(vp), 1) == split_blocks(nvis, ndev)
            @test stored_blocks(baseimage(vp), 2) == [1:4]
            test_clean_hlo(repr(@code_hlo visibilitymap(mpr, sdvis)))
        end

        @testset "Raw sharding of an IntensityMap keeps the grid on the host" begin
            simg = shard(img0, Reactant.Sharding.DimsSharding(mesh, (1,), (:d,)))
            @test axisdims(simg) === axisdims(img0)
            @test Array(baseimage(simg)) == baseimage(img0)
            @test stored_blocks(baseimage(simg), 1) == split_blocks(2ndev, ndev)
        end
    end
end

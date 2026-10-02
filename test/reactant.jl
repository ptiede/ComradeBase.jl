using Reactant
Reactant.set_default_backend("cpu")

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

    img1 = intensitymap(m1, g)
    img2 = @jit(intensitymap(m2, go))

    # circ shift the image so we get an actual centroid
    img1 = circshift(img1, (2, 3))
    img2 = Reactant.to_rarray(img1) #circshift stackoverflows with Reactant TODO: fix this

    c1 = centroid(img1)
    c2 = @jit(centroid(img2))
    @test c1[1] ≈ c2[1]
    @test c1[2] ≈ c2[2]

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
            f(a) = sum(abs2, baseimage(a).Q) + sum(baseimage(a).V)
            @test Float64(@jit(f(simg))) ≈ f(img)
            for c in StructArrays.components(baseimage(simg))
                @test stored_blocks(c, 1) == split_blocks(nx, ndev)
            end
        end

        @testset "Raw sharding of an UnstructuredDomain" begin
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
            @test stored_blocks(domainpoints(sdvis).U, 1) == split_blocks(nvis, ndev)
        end

        @testset "Raw sharding of an IntensityMap keeps the grid on the host" begin
            simg = shard(img0, Reactant.Sharding.DimsSharding(mesh, (1,), (:d,)))
            @test axisdims(simg) === axisdims(img0)
            @test Array(baseimage(simg)) == baseimage(img0)
            @test stored_blocks(baseimage(simg), 1) == split_blocks(2ndev, ndev)
        end
    end
end

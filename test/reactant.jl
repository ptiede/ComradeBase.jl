using Reactant

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

@testset "Sharding" begin
    # --- Construction (backend-agnostic, no devices needed) ---
    @test ReactantEx() === ReactantEx(nothing)
    @test ComradeBase.sharding(ReactantEx()) === nothing
    spec = ComradeBase.sharding(ReactantEx(:fakemesh; X = :dx, Y = :dy))
    @test spec isa ShardSpec
    @test spec.mesh === :fakemesh
    @test spec.axes == (X = :dx, Y = :dy)
    @test ComradeBase.sharding(shard_image(:m)).axes == (X = :dx, Y = :dy)
    @test ComradeBase.sharding(shard_frequency(:m)).axes == (Fr = :dev,)
    @test ComradeBase.sharding(shard_time(:m)).axes == (Ti = :dev,)

    # --- regroup ---
    Frs = [3.0, 1.0, 2.0, 1.0, 3.0, 2.0]
    dvis = UnstructuredDomain((; U = randn(6), V = randn(6), Ti = zeros(6), Fr = Frs))
    rdom, perm = regroup(dvis, :Fr)
    @test issorted(domainpoints(rdom).Fr)
    @test domainpoints(rdom).U ≈ domainpoints(dvis).U[perm]
    @test domainpoints(dvis).Fr[perm] ≈ domainpoints(rdom).Fr
    @test domainpoints(rdom).U[invperm(perm)] ≈ domainpoints(dvis).U  # round trip

    # The executor carries the (unsharded) spec without changing the result, and the unsharded
    # ReactantEx path is unaffected.
    @test ComradeBase.sharding(executor(imagepixels(10.0, 10.0, 8, 8))) === nothing
    mesh_like = :placeholder_mesh
    gspec = imagepixels(10.0, 10.0, 8, 8; executor = ReactantEx(mesh_like; X = :dx))
    @test ComradeBase.sharding(executor(gspec)) isa ShardSpec

    # --- End-to-end: declare (Fr), regroup, to_sharded, evaluate -> scalar ---
    # Sharding is layout-only: the result must equal the serial reference. The `is_sharded`
    # assertion only has teeth with >1 device, but the path runs (replicated) on a single device.
    ndev = length(Reactant.devices())
    mesh = ComradeBase.shardmesh(ndev; names = (:dev,))
    Nper = 16
    U = 0.2 .* randn(ndev * Nper)
    V = 0.2 .* randn(ndev * Nper)
    Fr = repeat(Float64.(1:ndev); inner = Nper)
    Ti = zeros(ndev * Nper)

    m = BlobTest(4.0)
    mr = Reactant.to_rarray(m)
    f(mod, dom) = sum(abs2, baseimage(visibilitymap(mod, dom)))
    ref = f(m, UnstructuredDomain((; U, V, Ti, Fr)))   # order-independent reduction

    dvis = UnstructuredDomain((; U, V, Ti, Fr); executor = ReactantEx(mesh; Fr = :dev))
    rdvis, perm = regroup(dvis, :Fr)
    rds = to_sharded(rdvis)
    @test issorted(domainpoints(rdvis).Fr)
    @test Float64(@jit(f(mr, rds))) ≈ ref
    ndev > 1 && @test Reactant.Sharding.is_sharded(domainpoints(rds).U)

    # --- Extensibility: tuple-valued axes, callable escape hatch, multi-key regroup ---
    @test ComradeBase.sharding(ReactantEx(:m; X = (:a, :b))).axes == (X = (:a, :b),)
    hatch = x -> :anything
    @test ComradeBase.sharding(ReactantEx(hatch)) === hatch

    # multi-key (lexicographic) regroup, Fr major then Ti minor
    Frm = [2.0, 1, 2, 1, 2, 1, 2, 1]
    Tim = [1.0, 1, 2, 2, 1, 1, 2, 2]
    dml = UnstructuredDomain((; U = collect(1.0:8), V = zeros(8), Ti = Tim, Fr = Frm))
    rml, pml = regroup(dml, :Fr, :Ti)
    @test issorted(collect(zip(domainpoints(rml).Fr, domainpoints(rml).Ti)))
    @test domainpoints(rml).U ≈ domainpoints(dml).U[pml]

    # Escape hatch end-to-end: a function returning a raw Reactant sharding
    sh = Reactant.Sharding.DimsSharding(mesh, (1,), (:dev,))
    dh = UnstructuredDomain((; U, V, Ti, Fr); executor = ReactantEx(_ -> sh))
    rdh = to_sharded(dh)
    @test Float64(@jit(f(mr, rdh))) ≈ ref
    ndev > 1 && @test Reactant.Sharding.is_sharded(domainpoints(rdh).U)

    # Image raster X/Y sharding end-to-end
    gimg = imagepixels(10.0, 10.0, 8, 8; executor = ReactantEx(mesh; X = :dev))
    img = IntensityMap(rand(8, 8), gimg)
    imgref = sum(abs2, baseimage(img))
    imgs = to_sharded(img)
    @test Float64(@jit((a -> sum(abs2, baseimage(a)))(imgs))) ≈ imgref
    ndev > 1 && @test Reactant.Sharding.is_sharded(baseimage(imgs))
end

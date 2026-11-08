using ComradeBase: StructuredDomain, Node, shapedcoords, coordspans

broadcast_points(sc) = broadcast((x...) -> NamedTuple{keys(sc)}(x), values(sc)...)
function points_approx(a, b)
    size(a) == size(b) || return false
    return all(zip(a, b)) do (x, y)
        keys(x) == keys(y) && all(map(isapprox, values(x), values(y)))
    end
end
function allocated_getindex(p, i)
    p[i]
    return @allocated p[i]
end

@testset "StructuredDomain" begin
    npt, nti, nfr = 7, 3, 2
    ti = [0.0, 1.0, 2.0]
    fr = [230.0e9, 345.0e9]

    @testset "(Node,)" begin
        nt = (U = randn(npt), V = randn(npt), Ti = rand(npt), Fr = fill(230.0e9, npt))
        d = StructuredDomain(nt)
        @test size(d) == (npt,)
        @test keys(d) == (:Node,)
        @test ndims(d) == 1
        @test length(d) == npt
        @test axes(d) == (Base.OneTo(npt),)
        @test keys(named_dims(d)) == (:Node,)
        @test coordspans(d) == (U = (:Node,), V = (:Node,), Ti = (:Node,), Fr = (:Node,))
        @test d.U === nt.U
        @test propertynames(d) == (:U, :V, :Ti, :Fr)
        @test_throws "no property `W`" d.W
        @test executor(d) == Serial()
        @test header(d) == ComradeBase.NoHeader()

        p = domainpoints(d)
        @test size(p) == (npt,)
        @test eltype(p) == NamedTuple{(:U, :V, :Ti, :Fr), NTuple{4, Float64}}
        for i in eachindex(nt.U)
            @test p[i] == map(x -> x[i], nt)
        end
        @test allocated_getindex(p, 3) == 0
        sc = shapedcoords(d)
        @test broadcast_points(sc) == collect(p)

        d2 = StructuredDomain((Node(npt),); U = nt.U, V = nt.V)
        @test domainpoints(d2) == domainpoints(StructuredDomain((; U = nt.U, V = nt.V)))
        @test DD.rebuild(d; executor = ThreadsEx()) isa StructuredDomain
        @test executor(DD.rebuild(d; executor = ThreadsEx())) == ThreadsEx()

        s = sprint(show, MIME"text/plain"(), d)
        @test occursin("StructuredDomain", s)
        @test occursin("spanning (:Node,)", s)
        @test occursin("StructuredDomain with dims (:Node,)", summary(d))
    end

    @testset "(Node, Fr) with coordinates spanning Node" begin
        U = randn(npt)
        V = randn(npt)
        u = 1.0e6 .* randn(npt)
        d = StructuredDomain((Node(npt), Fr(fr)); U, V, u)
        @test size(d) == (npt, nfr)
        @test keys(d) == (:Node, :Fr)
        @test coordspans(d) == (U = (:Node,), V = (:Node,), u = (:Node,))
        @test propertynames(d) == (:U, :V, :u, :Fr)
        @test d.U === U
        @test d.u === u
        @test d.Fr == fr
        p = domainpoints(d)
        for I in ((1, 1), (3, 2), (npt, 1))
            @test p[I...].U == U[I[1]]
            @test p[I...].u == u[I[1]]
            @test p[I...].Fr == fr[I[2]]
        end
        @test keys(p[1]) == (:U, :V, :u, :Fr)
        sc = shapedcoords(d)
        @test size(sc.U) == (npt, 1)
        @test size(sc.Fr) == (1, nfr)
        @test broadcast_points(sc) == collect(p)
        @test sprint(show, MIME"text/plain"(), d) isa String
    end

    @testset "U, V, W with an Fr coordinate" begin
        fobs = fr' .+ 1.0e6 .* (1:npt)
        U, V, W = randn(npt, nfr), randn(npt, nfr), randn(npt, nfr)
        d = StructuredDomain((Node(npt), Fr(1:nfr)); U, V, W, Fr = fobs)
        @test coordspans(d) == (U = (:Node, :Fr), V = (:Node, :Fr), W = (:Node, :Fr), Fr = (:Node, :Fr))
        @test eltype(d) === Float64
        @test domainpoints(d)[3, 2] == (U = U[3, 2], V = V[3, 2], W = W[3, 2], Fr = fobs[3, 2])
        @test broadcast_points(shapedcoords(d)) == collect(domainpoints(d))

        f1 = rand(fr, npt)
        dp = StructuredDomain((; U = U[:, 1], V = V[:, 1], Fr = f1))
        @test domainpoints(dp)[3].Fr == f1[3]
    end

    @testset "(Node, Ti, Fr)" begin
        u = randn(npt, nti)
        v = randn(npt, nti)
        tobs = ti' .+ 0.01 .* rand(npt, nti)
        a1 = rand(1:5, npt)
        d = StructuredDomain((Node(npt), Ti(ti), Fr(fr)); U = u, V = v, Ti = tobs, antenna1 = a1)
        @test size(d) == (npt, nti, nfr)
        @test keys(d) == (:Node, :Ti, :Fr)
        @test keys(named_dims(d)) == (:Node, :Ti, :Fr)
        @test coordspans(d) ==
            (U = (:Node, :Ti), V = (:Node, :Ti), Ti = (:Node, :Ti), antenna1 = (:Node,))
        @test d.Ti === tobs
        @test d.antenna1 === a1
        @test d.U === u
        p = domainpoints(d)
        @test size(p) == size(d)
        @test (@inferred p[2, 3, 2]) isa NamedTuple
        @test keys(p[1]) == (:U, :V, :Ti, :antenna1, :Fr)
        for I in ((1, 1, 1), (2, 3, 2), (npt, 2, 1))
            i, t, f = I
            @test p[I...].U == u[i, t]
            @test p[I...].V == v[i, t]
            @test p[I...].Ti == tobs[i, t]
            @test p[I...].Fr == fr[f]
            @test p[I...].antenna1 == a1[i]
        end
        sc = shapedcoords(d)
        @test size(sc.antenna1) == (npt, 1, 1)
        @test size(sc.Ti) == (npt, nti, 1)
        @test points_approx(broadcast_points(sc), collect(p))

        dl = StructuredDomain((Node(npt), Ti(ti), Fr(fr)); U = randn(npt, nti, nfr), V = randn(npt, nti, nfr))
        @test domainpoints(dl)[2, 3, 1].Ti == ti[3]

        de = StructuredDomain(
            (Node(nti), Ti(ti), Fr(fr));
            U = randn(nti, nti) => (:Node, :Ti), V = randn(nti, nti) => (:Node, :Ti),
            valid = trues(nti) => :Node
        )
        @test coordspans(de) == (U = (:Node, :Ti), V = (:Node, :Ti), valid = (:Node,))
        @test domainpoints(de)[1, 2, 2].valid
        s = sprint(show, MIME"text/plain"(), d)
        @test occursin("spanning (:Node, :Ti)", s)
    end

    @testset "validation" begin
        u = randn(npt)
        @test_throws "first dim of a StructuredDomain must be `Node`" StructuredDomain((Fr(fr), Node(npt)); U = randn(nfr, npt))
        @test_throws "must be DimensionalData dimensions" StructuredDomain((Node(npt), fr); U = u)
        @test_throws "needs dims starting with `Node`" StructuredDomain(())
        @test_throws "needs a vector lookup" StructuredDomain((Node(npt), Fr(2.0)); U = u)
        @test_throws "dim names must be distinct" StructuredDomain((Node(npt), Fr(fr), Fr(fr)); U = u)
        @test_throws DimensionMismatch StructuredDomain((Node(npt),); U = randn(npt + 1))
        @test_throws "matches no ordered subset" StructuredDomain((Node(npt),); U = randn(npt + 1))
        @test_throws "is ambiguous" StructuredDomain((Node(2), Fr(fr)); U = randn(2))
        @test_throws "`U = U => (:Node,)`" StructuredDomain((Node(2), Fr(fr)); U = randn(2))
        @test_throws "not a dim of this domain" StructuredDomain((Node(npt),); U = u => (:Ti,))
        @test_throws "in the domain's dim order" StructuredDomain((Node(npt), Ti(ti)); U = randn(nti, npt) => (:Ti, :Node))
        @test_throws "in the domain's dim order" StructuredDomain((Node(npt), Ti(ti)); U = randn(npt, npt) => (:Node, :Node))
        @test_throws DimensionMismatch StructuredDomain((Node(npt), Ti(ti)); U = randn(npt, 2) => (:Node, :Ti))
        @test_throws "the dims (:Node, :Ti) it spans have axes" StructuredDomain((Node(npt), Ti(ti)); U = randn(npt, 2) => (:Node, :Ti))
        @test_throws "must be a dim name or a tuple of dim names" StructuredDomain((Node(npt),); U = u => 1)
        @test_throws "must be an array or `array => (dim names...)`" StructuredDomain((Node(npt),); U = 1.0)
        @test_throws "`valid` must have a Bool element type" StructuredDomain((Node(npt),); U = u, valid = ones(npt))
        @test_throws "needs at least one coordinate" StructuredDomain(NamedTuple())
        d = StructuredDomain((Node(npt), Ti(ti)); U = randn(npt, nti))
        @test_throws DimensionMismatch DD.rebuild(d; dims = (Node(npt + 1), Ti(ti)))
    end

    @testset "generic indexing: view coords" begin
        ubig = 1.0e6 .* randn(npt + 2, nti + 1)
        vbig = 1.0e6 .* randn(npt + 2, nti + 1)
        uv = view(ubig, 2:(npt + 1), 1:nti)
        vv = view(vbig, 2:(npt + 1), 1:nti)
        dims = (Node(npt), Ti(ti), Fr(fr))
        dview = StructuredDomain(dims; U = uv, V = vv)
        dcopy = StructuredDomain(dims; U = copy(uv), V = copy(vv))
        @test collect(domainpoints(dview)) == collect(domainpoints(dcopy))
        @test dview.U == dcopy.U
        scv = shapedcoords(dview)
        @test broadcast_points(scv) == collect(domainpoints(dcopy))

        p1 = view(randn(2npt), 1:2:(2npt))
        dp = StructuredDomain((; U = p1, V = p1))
        @test collect(domainpoints(dp)) == collect(domainpoints(StructuredDomain((; U = collect(p1), V = collect(p1)))))
    end
end

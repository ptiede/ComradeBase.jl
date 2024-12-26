function domain4d(N, Nt, Nf)
    U_vals = range(-10e9, 10e9; length=N)
    U_vals = U_vals' .* ones(N)
    V_vals = range(-10e9, 10e9; length=N)
    V_vals = V_vals' .* ones(N)
    U_vals = U_vals'

    # Flatten the U and V grids
    U_final = vec(U_vals)
    V_final = vec(V_vals)

    ti = sort(10 * rand(Nt))
    fr = sort(1e11 * rand(Nf))

    # Repeat U and V to match Ti dimensions
    U_repeated = repeat(U_final; outer=(length(ti)))
    V_repeated = repeat(V_final; outer=(length(ti)))
    Ti_repeated = repeat(ti; inner=(Int(length(U_final))))

    # Repeat U and V and Ti to match Fr dimensions
    U_repeated = repeat(U_repeated; outer=(length(fr)))
    V_repeated = repeat(V_repeated; outer=(length(fr)))
    Ti_repeated = repeat(Ti_repeated; outer=(length(fr)))
    Fr_repeated = repeat(fr; inner=(Int(length(U_repeated) / length(fr))))
    visdomain = UnstructuredDomain((; U=U_repeated, V=V_repeated, Ti=Ti_repeated,
                                    Fr=Fr_repeated))

    C1 = true
    for ti_point in ti
        for fr_point in fr
            f = visdomain[Ti=ti_point, Fr=fr_point]
            g = UnstructuredDomain((; U=U_final, V=V_final,
                                    Ti=vcat(fill(ti_point, length(U_final))),
                                    Fr=vcat(fill(fr_point, length(U_final)))))
            C1 = C1 && (domainpoints(f) == domainpoints(g))
        end
    end

    #Switch Ti and Fr order
    C2 = true
    for ti_point in ti
        for fr_point in fr
            f = visdomain[Fr=fr_point, Ti=ti_point]
            g = UnstructuredDomain((; U=U_final, V=V_final,
                                    Ti=vcat(fill(ti_point, length(U_final))),
                                    Fr=vcat(fill(fr_point, length(U_final)))))
            C2 = C2 && (domainpoints(f) == domainpoints(g))
        end
    end

    return C1, C2
end

function domain3df(N, Nf)
    U_vals = range(-10e9, 10e9; length=N)
    U_vals = U_vals' .* ones(N)
    V_vals = range(-10e9, 10e9; length=N)
    V_vals = V_vals' .* ones(N)
    U_vals = U_vals'

    # Flatten the U and V grids
    U_final = vec(U_vals)
    V_final = vec(V_vals)

    fr = sort(1e11 * rand(Nf))

    # Repeat U and V to match Fr dimensions
    U_repeated = repeat(U_final; outer=(length(fr)))
    V_repeated = repeat(V_final; outer=(length(fr)))
    Fr_repeated = repeat(fr; inner=(Int(length(U_final))))

    visdomain = UnstructuredDomain((; U=U_repeated, V=V_repeated, Fr=Fr_repeated))

    C = true
    for fr_point in fr
        f = visdomain[Fr=fr_point]
        g = UnstructuredDomain((; U=U_final, V=V_final,
                                Fr=vcat(fill(fr_point, length(U_final)))))
        C = C && (domainpoints(f) == domainpoints(g))
    end

    return C
end

function domain3dt(N, Nt)
    U_vals = range(-10e9, 10e9; length=N)
    U_vals = U_vals' .* ones(N)
    V_vals = range(-10e9, 10e9; length=N)
    V_vals = V_vals' .* ones(N)
    U_vals = U_vals'

    # Flatten the U and V grids
    U_final = vec(U_vals)
    V_final = vec(V_vals)

    ti = sort(10 * rand(Nt))

    # Repeat U and V to match Ti dimensions
    U_repeated = repeat(U_final; outer=(length(ti)))
    V_repeated = repeat(V_final; outer=(length(ti)))
    Ti_repeated = repeat(ti; inner=(Int(length(U_final))))

    visdomain = UnstructuredDomain((; U=U_repeated, V=V_repeated, Ti=Ti_repeated))

    C = true
    for ti_point in ti
        f = visdomain[Ti=ti_point]
        g = UnstructuredDomain((; U=U_final, V=V_final,
                                Ti=vcat(fill(ti_point, length(U_final)))))
        C = C && (domainpoints(f) == domainpoints(g))
    end

    return C
end

@testset "Test getindex for visdomain" begin
    C1, C2 = domain4d(64, 10, 4)
    @test C1
    @test C2
    C3 = domain3dt(64, 10)
    @test C3
    C4 = domain3df(64, 4)
    @test C4
end


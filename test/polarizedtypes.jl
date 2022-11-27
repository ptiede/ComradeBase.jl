@testset "Polarized type test" begin

    @testset "Basis transform" begin
        @test basis_transform(PolBasis{XPol,YPol}()=>PolBasis{RPol,LPol}())*basis_transform(PolBasis{RPol,LPol}()=>PolBasis{XPol,YPol}()) ≈ [1.0 0.0;0.0 1.0]
        @test basis_transform(PolBasis{RPol,LPol}()=>PolBasis{XPol,YPol}())*basis_transform(PolBasis{XPol,YPol}()=>PolBasis{RPol,LPol}()) ≈ [1.0 0.0;0.0 1.0]

        for (e1, e2) in [(RPol, LPol), (LPol, RPol),
                         (XPol, YPol), (XPol, YPol),
                         ]
            @test basis_transform(PolBasis{e1,e2}()=>PolBasis{e1,e2}()) ≈ [1.0 0.0; 0.0 1.0]
        end
    end

    @testset "Non-orthogonal" begin
        @test_throws AssertionError basis_transform(PolBasis{XPol,YPol}(), PolBasis{RPol,XPol}())
        @test_throws AssertionError basis_transform(PolBasis{XPol,YPol}(), PolBasis{RPol,YPol}())
        @test_throws AssertionError basis_transform(PolBasis{XPol,YPol}(), PolBasis{XPol,RPol}())
        @test_throws AssertionError basis_transform(PolBasis{XPol,YPol}(), PolBasis{XPol,LPol}())
    end

    @testset "Missing feeds" begin
        for E in (XPol,YPol,RPol,LPol)
            c1 = basis_components(E(), PolBasis{E, Missing}())
            @test c1[1] ≈ 1.0
            @test c1[2] isa Missing

            c2 = basis_components(E(), PolBasis{Missing, E}())
            @test c2[2] ≈ 1.0
            @test c2[1] isa Missing

            # Test that the compiler realized the Union
            @inferred basis_components(E(), PolBasis{Missing, E}())
            @inferred basis_components(E(), PolBasis{E, Missing}())
            @inferred basis_transform(PolBasis{E, Missing}()=>CirBasis())
            @inferred basis_transform(PolBasis{E, Missing}()=>LinBasis())
            @inferred basis_transform(CirBasis()=>PolBasis{E, Missing}())
            @inferred basis_transform(LinBasis()=>PolBasis{E, Missing}())

        end
    end


    @testset "Simple stokes test" begin
        sQ = StokesParams(1.0, 0.5, 0.0, 0.0)
        sU = StokesParams(1.0, 0.0, 0.5, 0.0)
        sV = StokesParams(1.0, 0.0, 0.0, 0.5)

        @test CoherencyMatrix(sQ, LinBasis()) ≈ inv(2)*([1.5 0.0; 0.0 0.5])
        @test CoherencyMatrix(sQ, CirBasis()) ≈ inv(2)*([1.0 0.5; 0.5 1.0])

        @test CoherencyMatrix(sU, LinBasis()) ≈ inv(2)*([1.0 0.5; 0.5 1.0])
        @test CoherencyMatrix(sU, CirBasis()) ≈ inv(2)*([1.0 0.5im; -0.5im 1.0])

        @test CoherencyMatrix(sV, LinBasis()) ≈ inv(2)*([1.0 0.5im; -0.5im 1.0])
        @test CoherencyMatrix(sV, CirBasis()) ≈ inv(2)*([1.5 0.0; 0.0 0.5])
    end

    @testset "Simple Coherency test" begin
        cRR = CoherencyMatrix(0.5, 0.0, 0.0, 0.0, CirBasis())
        cLR = CoherencyMatrix(0.0, 0.5, 0.0, 0.0, CirBasis())
        cRL = CoherencyMatrix(0.0, 0.0, 0.5, 0.0, CirBasis())
        cLL = CoherencyMatrix(0.0, 0.0, 0.0, 0.5, CirBasis())

        @test StokesParams(cRR) ≈ [0.5, 0.0, 0.0, 0.5]
        @test StokesParams(cLR) ≈ [0.0, 0.5, 0.5im, 0.0]
        @test StokesParams(cRL) ≈ [0.0, 0.5, -0.5im, 0.0]
        @test StokesParams(cLL) ≈ [0.5, 0.0, 0.0, -0.5]


        cXX = CoherencyMatrix(0.5, 0.0, 0.0, 0.0, LinBasis())
        cYX = CoherencyMatrix(0.0, 0.5, 0.0, 0.0, LinBasis())
        cXY = CoherencyMatrix(0.0, 0.0, 0.5, 0.0, LinBasis())
        cYY = CoherencyMatrix(0.0, 0.0, 0.0, 0.5, LinBasis())

        @test StokesParams(cXX) ≈ [0.5, 0.5, 0.0, 0.0]
        @test StokesParams(cYX) ≈ [0.0, 0.0, 0.5, 0.5im]
        @test StokesParams(cXY) ≈ [0.0, 0.0, 0.5, -0.5im]
        @test StokesParams(cYY) ≈ [0.5, -0.5, 0.0, 0.0]

    end

    @testset "Conversions back and forward" begin
        s = StokesParams(1.0 .+ 0.0im, 0.2 + 0.2im, 0.2 - 0.2im, 0.1+0.05im)

        s ≈ StokesParams(CoherencyMatrix(s, CirBasis()))
        s ≈ StokesParams(CoherencyMatrix(s, LinBasis()))
        s ≈ StokesParams(CoherencyMatrix(s, CirBasis(), LinBasis()))
        s ≈ StokesParams(CoherencyMatrix(s, LinBasis(), CirBasis()))
        s ≈ StokesParams(CoherencyMatrix(s, PolBasis{YPol,XPol}(), PolBasis{LPol,RPol}()))
    end

    @testset "Conversion Consistency" begin
        s = StokesParams(1.0 .+ 0.0im, 0.2 + 0.2im, 0.2 - 0.2im, 0.1+0.05im)
        c_lin1 = CoherencyMatrix(s, LinBasis())
        c_lin2 = CoherencyMatrix(s, PolBasis{XPol,YPol}())
        c_lin3 = CoherencyMatrix(s, PolBasis{XPol,YPol}(), PolBasis{XPol,YPol}())

        @test c_lin1 ≈ c_lin2 ≈ c_lin3

        c_cir1 = CoherencyMatrix(s, CirBasis())
        c_cir2 = CoherencyMatrix(s, PolBasis{RPol,LPol}())
        c_cir3 = CoherencyMatrix(s, PolBasis{RPol,LPol}(), PolBasis{RPol,LPol}())

        @test c_cir1 ≈ c_cir2 ≈ c_cir3

        t1 = basis_transform(LinBasis()=>CirBasis())
        t2 = basis_transform(CirBasis()=>LinBasis())

        @test t2*c_cir1*t1 ≈ c_lin1
        @test t1*c_lin1*t2 ≈ c_cir1

        # Test the mixed basis
        @test c_cir1*t1 ≈ t1*c_lin1
        @test c_lin1*t2 ≈ t2*c_cir1

    end

    @testset "Performance test" begin
        s = StokesParams(1.0 .+ 0.0im, 0.2 + 0.2im, 0.2 - 0.2im, 0.1+0.05im)
        @test_opt StokesParams(CoherencyMatrix(s, LinBasis()))
        @test_opt StokesParams(CoherencyMatrix(s, CirBasis()))
        @test_opt StokesParams(CoherencyMatrix(s, LinBasis(), CirBasis()))
        @test_opt StokesParams(CoherencyMatrix(s, LinBasis(), CirBasis(), LinBasis()))
    end

end

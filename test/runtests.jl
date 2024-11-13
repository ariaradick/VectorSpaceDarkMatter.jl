using VectorSpaceDarkMatter
using Test
VSDM = VectorSpaceDarkMatter

wv_g = Wavelet(960.0*VSDM.km_s)
g1 = GaussianF(0.4, VSDM.cart_to_sph([0.0, 0.0, -230.0*VSDM.km_s]), 
               220.0*VSDM.km_s)
pfg_test = ProjectF(g1, (2^10-1, 10), wv_g)

gg = GaussianF(1.0, [200.0*VSDM.km_s,0.0,0.0], 100.0*VSDM.km_s)
pfgg = ProjectF(gg, (2^10-1,10), Wavelet(960.0*VSDM.km_s))

ff = GaussianF(1.0, [2.0*VSDM.qBohr,0.0,0.0], 3.0*VSDM.qBohr)
pfff = ProjectF(ff, (2^10-1,10), Wavelet(10*VSDM.qBohr))

mm = VSDM.ModelDMSM(0, 100e6, 511e3, 4.03)
rate_test = 2.243e-22 # approx. numerical integral result

@testset "VectorSpaceDarkMatter.jl" begin
    @test isapprox(f2_norm(g1), f2_norm(pfg_test); rtol=1e-4)
    @test isapprox(ProjectF(x -> VSDM.haar_sph_value(0) *
                    VSDM.ylm_real(0,0,0,0), (0,0), Wavelet()).fnlm[1],
                    1.0)
    @test isapprox(rate(mm, pfgg, pfff), rate_test; rtol=1e-3)
end

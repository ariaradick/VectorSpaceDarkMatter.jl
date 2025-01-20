module VectorSpaceDarkMatter

using Measurements
import LinearAlgebra: dot, *, tr, LowerTriangular, lmul!
import DelimitedFiles: readdlm, writedlm
import SpecialFunctions: gamma, besselix
import PolyLog: reli2
# import HypergeometricFunctions._₂F₁ as hyp2f1
import HCubature: hcubature
using MCIntegration
using Quaternionic
using SphericalFunctions

export Wavelet, Tophat, f_uSph, GaussianF, f2_norm, ProjectF,
       ProjectedF, FCoeffs, update!, writeFnlm, readFnlm, rate,
       ModelDMSM, partial_rate, readK, writeK, PartialRate

include("units.jl")
include("utils.jl")
include("wigner.jl")
include("haar.jl")
include("basis.jl")
include("gaussians.jl")
include("projection.jl")
include("analytic_mcalI.jl")
include("matrixcalc.jl")
include("ratecalc.jl")

end

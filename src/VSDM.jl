module VSDM

import HCubature: hcubature
using MCIntegration
using Measurements
import LinearAlgebra: dot
import DelimitedFiles: readdlm, writedlm
import SpecialFunctions: gamma, besselix
import PolyLog: reli2
import HypergeometricFunctions._₂F₁ as hyp2f1

export Wavelet, Tophat, f_uSph, GaussianF, norm_energy, ProjectF,
       ProjectedF, FCoeffs, update!, writeFnlm, readFnlm

include("units.jl")
include("utils.jl")
include("haar.jl")
include("basis.jl")
include("gaussians.jl")
include("projection.jl")
include("analytic_mcalI.jl")
include("matrixcalc.jl")

end

using Documenter
using VSDM

makedocs(
    sitename = "VSDM",
    format = Documenter.HTML()#,
    #modules = [VSDM]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

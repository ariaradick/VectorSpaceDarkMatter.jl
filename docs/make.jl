using Documenter
using VectorSpaceDarkMatter

makedocs(
    sitename = "VectorSpaceDarkMatter",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "Reference" => "ref/reference.md",
    ]
    #modules = [VectorSpaceDarkMatter]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/ariaradick/VectorSpaceDarkMatter.jl.git"
)

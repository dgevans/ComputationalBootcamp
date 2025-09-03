using NBInclude

# get files in current director that end with .ipynb
for file in readdir(".")
    if endswith(file, ".ipynb")
        #remove .ipynb from file name
        file = replace(file, ".ipynb" => "")
        #create new file with same name but with .jl extension
        nbexport(file * ".jl", file * ".ipynb", markdown=false)
    end
end
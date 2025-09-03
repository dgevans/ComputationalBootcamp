#!/usr/bin/env julia

function list_ipynb_files(target_directory::AbstractString; absolute::Bool=false)
    entries = readdir(target_directory; join=false)
    ipynb_files = filter(name -> endswith(lowercase(name), ".ipynb") && isfile(joinpath(target_directory, name)), entries)
    sort!(ipynb_files; by=lowercase)
    for file_name in ipynb_files
        full_path = joinpath(target_directory, file_name)
        println(absolute ? abspath(full_path) : file_name)
    end
end

function main()
    # Defaults
    target_directory = pwd()
    absolute = false

    # Simple arg parsing: any non-flag is treated as the directory path
    for arg in ARGS
        if arg == "--absolute" || arg == "-a"
            absolute = true
        elseif startswith(arg, "-")
            @warn "Unknown flag ignored" arg
        else
            target_directory = abspath(arg)
        end
    end

    list_ipynb_files(target_directory; absolute=absolute)
end

main()



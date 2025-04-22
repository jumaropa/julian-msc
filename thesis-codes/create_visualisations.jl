include("functions.jl")

gridsizes = [(50, 7), (100, 14), (150, 21), (200,28)]

for order in 1:3
    for grid in gridsizes
        #use desired function from functions.jl
        lag_PE(grid[1], grid[2], order, write=true)
    end
end

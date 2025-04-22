using JLD2
include("functions.jl")

order = 3
@load "refsolution_rav_PE_$(order).jld2" uh_ref ph_ref U_ref P_ref dΩ_ref

gridsizes = [(50, 7), (100, 14), (150, 21), (200, 28)]
eul2s = Float64[]
eulinfs = Float64[]
epl2s = Float64[]
eplinfs = Float64[]
hs = Float64[]
for n in gridsizes
    n_x, n_y = n
    h = L_x/n_x
    uh, ph, U, P, dΩ = rav_PE(n_x, n_y, order)

    iuh = Interpolable(uh)
    iuh = interpolate_everywhere(iuh, U_ref)
    eu = uh_ref - iuh
    
    eul2   = sqrt(sum( ∫( eu⊙eu )*dΩ_ref ))
    eulinf = maximum( broadcast(abs, eu.args[1].free_values - eu.args[2].free_values ))
    
    iph = Interpolable(ph)
    iph = interpolate_everywhere(iph, P_ref)
    ep = ph_ref - iph
    
    epl2   = sqrt(sum( ∫( ep⋅ep )*dΩ_ref ))
    eplinf = maximum( broadcast(abs, ep.args[1].free_values - ep.args[2].free_values ))
    
    push!(eul2s,eul2)
    push!(eulinfs,eulinf)
    push!(epl2s,epl2)
    push!(eplinfs,eplinf)
    push!(hs,h)
    
    writevtk(get_triangulation(eu),"errors-rav-PE-$(order)-$(n_x)-$(n_y)", cellfields=["uh_ref" => uh_ref, "ph_ref" => ph_ref, "eu" => eu, "ep" => ep])
end

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  (linreg[2])
end

println("")

println("u L2 slope: ", slope(hs,eul2s))
println("u Linf slope: ", slope(hs,eulinfs))
println("p L2 slope: ", slope(hs,epl2s))
println("p Linf slope: ", slope(hs,eplinfs))
println("")
println("u L2 errors: ", eul2s)
println("u Linf errors: ", eulinfs)
println("p L2 errors: ", epl2s)
println("p Linf errors: ", eplinfs)

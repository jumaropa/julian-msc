using JLD2
include("functions.jl")

order = 1
@load "refsolution_rav_NSD_$(order).jld2" uhf_ref phf_ref uhp_ref php_ref Uf_ref Pf_ref Up_ref Pp_ref dΩ_f_ref dΩ_p_ref

gridsizes = [(50, 7), (100, 14), (150, 21), (200, 28)]
eufl2s = Float64[]
euflinfs = Float64[]
epfl2s = Float64[]
epflinfs = Float64[]
eupl2s = Float64[]
euplinfs = Float64[]
eppl2s = Float64[]
epplinfs = Float64[]

eutotl2s = Float64[]
eptotl2s = Float64[]
eutotlinfs = Float64[]
eptotlinfs = Float64[]

hs = Float64[]
for n in gridsizes
    n_x, n_y = n
    h = L_x/n_x
    uhf, phf, uhp, php, Uf, Pf, Up, Pp, dΩ_f, dΩ_p = rav_NSD(n_x, n_y, order)
    
    iuhf = Interpolable(uhf)
    iuhf = interpolate_everywhere(iuhf, Uf_ref)
    euf = uhf_ref - iuhf
    
    eufl2   = sqrt(sum( ∫( euf⊙euf )*dΩ_f_ref ))
    euflinf = maximum( broadcast(abs, euf.args[1].free_values - euf.args[2].free_values ))

    iuhp = Interpolable(uhp)
    iuhp = interpolate_everywhere(iuhp, Up_ref)
    eup = uhp_ref - iuhp
    
    eupl2   = sqrt(sum( ∫( eup⊙eup )*dΩ_p_ref ))
    euplinf = maximum( broadcast(abs, eup.args[1].free_values - eup.args[2].free_values ))
    
    iphf = Interpolable(phf)
    iphf = interpolate_everywhere(iphf, Pf_ref)
    epf = phf_ref - iphf
    
    epfl2   = sqrt(sum( ∫( epf⋅epf )*dΩ_f_ref ))
    epflinf = maximum( broadcast(abs, epf.args[1].free_values - epf.args[2].free_values ))
    
    iphp = Interpolable(php)
    iphp = interpolate_everywhere(iphp, Pp_ref)
    epp = php_ref - iphp
    
    eppl2   = sqrt(sum( ∫( epp⋅epp )*dΩ_p_ref ))
    epplinf = maximum( broadcast(abs, epp.args[1].free_values - epp.args[2].free_values ))
    
    eutotl2 = sqrt(sum( ∫( euf⊙euf )*dΩ_f_ref ) + sum( ∫( eup⊙eup )*dΩ_p_ref ))
    eptotl2 = sqrt(sum( ∫( epf⋅epf )*dΩ_f_ref ) + sum( ∫( epp⋅epp )*dΩ_p_ref ))
    
    eutotlinf = maximum([euflinf,euplinf])
    eptotlinf = maximum([epflinf,epplinf])
    
    push!(eufl2s,eufl2)
    push!(euflinfs,euflinf)
    push!(epfl2s,epfl2)
    push!(epflinfs,epflinf)
    push!(eupl2s,eupl2)
    push!(euplinfs,euplinf)
    push!(eppl2s,eppl2)
    push!(epplinfs,epplinf)
    
    push!(eutotl2s,eutotl2)
    push!(eptotl2s,eptotl2)
    push!(eutotlinfs,eutotlinf)
    push!(eptotlinfs,eptotlinf)
    
    push!(hs,h)
end

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  (linreg[2])
end

println("")
println("FLUID")
println("uf L2 slope: ", slope(hs,eufl2s))
println("uf Linf slope: ", slope(hs,euflinfs))
println("pf L2 slope: ", slope(hs,epfl2s))
println("pf Linf slope: ", slope(hs,epflinfs))
println("")
println("uf L2 errors: ", eufl2s)
println("uf Linf errors: ", euflinfs)
println("pf L2 errors: ", epfl2s)
println("pf Linf errors: ", epflinfs)
println("")
println("POROUS")
println("up L2 slope: ", slope(hs,eupl2s))
println("up Linf slope: ", slope(hs,euplinfs))
println("pp L2 slope: ", slope(hs,eppl2s))
println("pp Linf slope: ", slope(hs,epplinfs))
println("")
println("up L2 errors: ", eupl2s)
println("up Linf errors: ", euplinfs)
println("pp L2 errors: ", eppl2s)
println("pp Linf errors: ", epplinfs)
println("")
println("TOTAL")
println("u total L2 slope: ", slope(hs,eutotl2s))
println("u total Linf slope: ", slope(hs,eutotlinfs))
println("p total L2 slope: ", slope(hs,eptotl2s))
println("p total Linf slope: ", slope(hs,eptotlinfs))
println("")
println("u total L2 errors: ", eutotl2s)
println("u total Linf errors: ", eutotlinfs)
println("p total L2 errors: ", eptotl2s)
println("p total Linf errors: ", eptotlinfs)


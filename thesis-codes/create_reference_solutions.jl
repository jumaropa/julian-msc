using JLD2
include("functions.jl")

#change gridsizes as desired

for order in 1:3
    uh_ref, ph_ref, U_ref, P_ref, dΩ_ref = lag_PE(1000, 140, order)
    @save "refsolution_lag_PE_$(order).jld2" uh_ref ph_ref U_ref P_ref dΩ_ref
end

for order in 1:3
    uhf_ref, phf_ref, uhp_ref, php_ref, Uf_ref, Pf_ref, Up_ref, Pp_ref, dΩ_f_ref, dΩ_p_ref = lag_NSD(1000, 140, order)
    @save "refsolution_lag_NSD_$(order).jld2" uhf_ref phf_ref uhp_ref php_ref Uf_ref Pf_ref Up_ref Pp_ref dΩ_f_ref dΩ_p_ref
end

for order in 1:3
    uhf_ref, phf_ref, uhp_ref, php_ref, Uf_ref, Pf_ref, Up_ref, Pp_ref, dΩ_f_ref, dΩ_p_ref = lag_NSF(1000, 140, order)
    @save "refsolution_lag_NSF_$(order).jld2" uhf_ref phf_ref uhp_ref php_ref Uf_ref Pf_ref Up_ref Pp_ref dΩ_f_ref dΩ_p_ref
end

for order in 1:3
    uh_ref, ph_ref, U_ref, P_ref, dΩ_ref = rav_PE(1000, 140, order)
    @save "refsolution_rav_PE_$(order).jld2" uh_ref ph_ref U_ref P_ref dΩ_ref
end

for order in 1:3
    uhf_ref, phf_ref, uhp_ref, php_ref, Uf_ref, Pf_ref, Up_ref, Pp_ref, dΩ_f_ref, dΩ_p_ref = lag_NSD(1000, 140, order)
    @save "refsolution_rav_NSD_$(order).jld2" uhf_ref phf_ref uhp_ref php_ref Uf_ref Pf_ref Up_ref Pp_ref dΩ_f_ref dΩ_p_ref
end

for order in 1:3
    uhf_ref, phf_ref, uhp_ref, php_ref, Uf_ref, Pf_ref, Up_ref, Pp_ref, dΩ_f_ref, dΩ_p_ref = rav_NSF(1000, 140, order)
    @save "refsolution_rav_NSF_$(order).jld2" uhf_ref phf_ref uhp_ref php_ref Uf_ref Pf_ref Up_ref Pp_ref dΩ_f_ref dΩ_p_ref
end

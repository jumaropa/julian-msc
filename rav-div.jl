module Divergence
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues

Ω = CartesianDiscreteModel((0,1,0,1), (5,5))
labels_Ω = get_face_labeling(Ω)
Ωₕ = Triangulation(Ω)

reffe_rt  = ReferenceFE(raviart_thomas, Float64, 1)
reffe_lag = ReferenceFE(lagrangian, Float64, 1)
#reffe_rt = RaviartThomasRefFE(Float64, QUAD, 1)
#reffe_lag = LagrangianRefFE(Float64, QUAD, 1)

V = TestFESpace(Ω,
                reffe_rt,
                labels=labels_Ω,
                conformity=:Hdiv)
                
Q = TestFESpace(Ω,
                reffe_lag,
                labels=labels_Ω,
                conformity=:L2)

f((x,y)) = VectorValue(y,x)
div_f = divergence(f)
idiv_f = interpolate_everywhere(div_f, Q)

f_int = interpolate_everywhere(f, V)
div_f_int = divergence(f_int)
idiv_f_int = interpolate_everywhere(div_f_int, Q)

writevtk(Ωₕ,"rav-div-results",cellfields=["f_int"=>f_int, "idiv_f"=>idiv_f, "div_f_int"=>div_f_int])

end

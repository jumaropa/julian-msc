module twodomain
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues

## Parameters (see Cimolin (2013))
ρ = 1.184
μ = 1.855e-5
K = 3.71e-7
C_F = 0.5
α_BJ = 1.0

#characteristic values
L = 1e-3        #charac. length
U_char = 1e-1   #charac. velocity

Re = ρ*L*U_char/μ   #Reynolds number

Gr_f = ρ*C_F*U_char*sqrt(K)/μ
Gr_n = ρ*K*U_char/(μ*L)
Gr_c = α_BJ*L/sqrt(K)

## Domain
L_x = 50     #length of whole domain
L_y_p = 3    #height of porous medium
L_y_f = 4    #height of fluid region

domain = (0,L_x,-L_y_p,L_y_f)
n_x = 150
n_y = 21
partition = (n_x,n_y)
println(partition)
model_Ω = CartesianDiscreteModel(domain,partition) # create a square grid with the desired partition

labels_Ω = get_face_labeling(model_Ω)            # get the labels from the model

add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
add_tag_from_tags!(labels_Ω,"left_corner",[1])
add_tag_from_tags!(labels_Ω,"right_corner",[2])

#check if element (with coordinates coords) is in the porous medium
function is_p(coords)
    av = sum(coords)/length(coords) #average of the coords
    -L_y_p < av[2] ≤ 0
end

# checks for elements in left or right boundary in the porous medium
function is_p_boundary(list, pos) 
    truth_list = []
    for sublist in list
        all_zero = all(t -> t[1] ≈ pos, sublist)
        push!(truth_list, all_zero * is_p(sublist))
    end
    return truth_list
end

entity_tag_left_p = num_entities(labels_Ω) + 1     # add a new tag for the left porous boundary (δ_1 in the paper)
entity_tag_right_p = num_entities(labels_Ω) + 2    # add a new tag for the right porous boundary (δ_3 in the paper)
entity_tag_p = num_entities(labels_Ω) + 3          # add a new tag for (interior) porous region (Ω_p)

# this for-loop finds all the vertices and edges in δ_1 and δ_3
# and the faces in Ω_p and assigns the new tags to them
for d in 0:2
    face_coords = get_cell_coordinates(Grid(ReferenceFE{d}, model_Ω))
    left_p_boundary  = findall(is_p_boundary(face_coords, 0.0))
    for i in left_p_boundary
        labels_Ω.d_to_dface_to_entity[d+1][i] = entity_tag_left_p
    end
    right_p_boundary  = findall(is_p_boundary(face_coords, L_x))
    for i in right_p_boundary
        labels_Ω.d_to_dface_to_entity[d+1][i] = entity_tag_right_p
    end
    
    p_region = findall([lazy_map(is_p, face_coords)[i] for i in 1:length(face_coords)])
    for i in p_region
        if labels_Ω.d_to_dface_to_entity[d+1][i] == 9 #only change the tag for the interior faces
            labels_Ω.d_to_dface_to_entity[d+1][i] = entity_tag_p
        end
    end
end
add_tag!(labels_Ω,"left_p",[entity_tag_left_p])
add_tag!(labels_Ω,"right_p",[entity_tag_right_p])
add_tag!(labels_Ω,"porous",[entity_tag_p])

add_tag_from_tags!(labels_Ω,"porous_b",["porous", "left_p", "right_p", "bottom", "left_corner", "right_corner"]) #porous region + boundaries
add_tag_from_tags!(labels_Ω,"fluid_b", ["interior", "inlet", "top", "outlet"])

###Porous and fluid domains
Ω = Interior(model_Ω)
Ω_p = Interior(model_Ω,tags="porous_b")
Ω_f = Interior(model_Ω,tags="fluid_b")

###Interface
Γ_fp = InterfaceTriangulation(Ω_f,Ω_p)
n_Γfp = get_normal_vector(Γ_fp)

## Boundary Conditions
u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)  #inlet velocity profile
u_0 = VectorValue(0.0,0.0)

## FE Spaces
k = 2

reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
reffeₚ = ReferenceFE(lagrangian,Float64,k-1)

##test spaces
# fluid velocity
Vf = TestFESpace(
  Ω_f,
  reffeᵤ,
  conformity=:H1,
  dirichlet_tags=["left_p", "inlet", "top"],
  dirichlet_masks=[(true, false), (true, true), (true, true)])

# porous velocity
Vp = TestFESpace(
  Ω_p,
  reffeᵤ,
  conformity=:H1,
  dirichlet_tags=["left_p", "bottom", "left_corner"],
  dirichlet_masks=[(true, false), (false, true), (true, true)])

# fluid pressure
Qf = TestFESpace(
  Ω_f,
  reffeₚ,
  conformity=:C0)

# porous pressure
Qp = TestFESpace(
  Ω_p,
  reffeₚ,
  conformity=:C0,
  dirichlet_tags=["right_p", "right_corner"])

##trial spaces
Uf = TrialFESpace(Vf,[u_0, u_pois, u_0])
Up = TrialFESpace(Vp,[u_0, u_0, u_0])
Pf = TrialFESpace(Qf)
Pp = TrialFESpace(Qp, [0.0, 0.0])

Y = MultiFieldFESpace([Vf,Qf,Vp,Qp]) #test
X = MultiFieldFESpace([Uf,Pf,Up,Pp]) #trial

## Measures
degree = 2*k
dΩ_f = Measure(Ω_f, degree) #fluid domain
dΩ_p = Measure(Ω_p, degree) #porous domain

idegree = 2*k
dΓ_fp = Measure(Γ_fp,idegree) #interface

#auxiliary functions
unorm(u) = (u⋅u).^(1/2) + 1e-12
conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

## Weak Form ##
#fluid
af((uf,pf),(vf,qf)) = ∫( 1/Re*∇(uf)⊙∇(vf) - pf*(∇⋅vf) + qf*(∇⋅uf) )dΩ_f
c((uf,pf),(vf,qf)) = ∫( vf⊙(conv∘(uf,∇(uf))) )dΩ_f
dc(uf,duf,vf) = ∫( vf⊙(dconv∘(duf,∇(duf),uf,∇(uf))) )dΩ_f

#porous
ap((up,pp),(vp,qp)) = ∫( -up⋅∇(qp) )dΩ_p +
                      ∫( up⋅vp + Gr_n*∇(pp)⋅vp )dΩ_p
apb((up,pp),(vp,qp)) = ∫( vp⊙(Gr_f*((up⋅up).^(1/2))*up) )dΩ_p
dapb(up,dup,vp) = ∫(vp⊙(Gr_f*((up⋅up).^(1/2))*dup))dΩ_p + ∫(vp⊙(Gr_f*((up⋅dup)/(unorm(up))*up)))dΩ_p

#interface
apf((uf,pf,up,pp),(vf,qf,vp,qp)) = ∫( Gr_c/Re*(uf.plus-(uf.plus⋅n_Γfp.plus)*n_Γfp.plus)⋅(vf.plus-(vf.plus⋅n_Γfp.plus)*n_Γfp.plus) )dΓ_fp - 
                                   ∫( (uf.plus⋅n_Γfp.plus)*qp.minus )dΓ_fp +
                                   ∫( (vf.plus⋅n_Γfp.plus)*pp.minus )dΓ_fp

l((vf,qf)) = ∫(0*qf)dΩ_f

a((uf,pf,up,pp),(vf,qf,vp,qp)) = af((uf,pf),(vf,qf)) + ap((up,pp),(vp,qp)) + apf((uf,pf,up,pp),(vf,qf,vp,qp))

res((uf,pf,up,pp),(vf,qf,vp,qp)) = a((uf,pf,up,pp),(vf,qf,vp,qp)) + c((uf,pf),(vf,qf)) + apb((up,pp),(vp,qp))
jac((uf,pf,up,pp),(duf,dpf,dup,dpp),(vf,qf,vp,qp)) = a((duf,dpf,dup,dpp),(vf,qf,vp,qp)) + dc(uf,duf,vf) + dapb(up,dup,vp)

op = FEOperator(res,jac,X,Y)

using LineSearches: BackTracking
nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=50)

## solve
uhf, phf, uhp, php = solve(nls, op)
writevtk(Ω_f,"results-lag-NSF-f", cellfields=["uhf" => uhf, "phf" => phf])
writevtk(Ω_p,"results-lag-NSF-p", cellfields=["uhp" => uhp, "php" => php])

end

module channel_flow_test
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues

## Parameters
ρ = 1.184
μ = 1.855
K = 3.71e-7
C_F = 0.5

L = 1e-3
U_char = 1e-1

Re = ρ*L*U_char/μ

Gr_v(x) = (x[2]<0)*μ*L/(ρ*U_char*K)
Gr_i(x) = (x[2]<0)*C_F*L/sqrt(K)

## Domain
L_x = 50e-3     #length of whole domain
L_y_p = 3e-3    #height of porous medium
L_y_f = 4e-3    #height of fluid region

domain = (0,L_x,-L_y_p,L_y_f)
partition = (50,20)
model_Ω = CartesianDiscreteModel(domain,partition) # create a square grid with the desired partition

labels_Ω = get_face_labeling(model_Ω) 
add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
add_tag_from_tags!(labels_Ω,"bottom",[1,2,5])    # assign the label "bottom" to the entity 1,2 and 5 (bottom corners and bottom side)
add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
add_tag_from_tags!(labels_Ω,"water",[9])

## Boundary Conditions
u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)

#check if element (with coordinates coords) is in the porous medium
function is_p(coords)
    av = sum(coords)/length(coords) #average of the coords
    -L_y_p < av[2] < 0 # I choose not to include the bottom boundary here
end

function is_p_boundary(list, pos) # this only checks for left or right boundary in p
    truth_list = []
    for sublist in list
        all_zero = all(t -> t[1] ≈ pos, sublist)
        push!(truth_list, all_zero * is_p(sublist))
    end
    return truth_list
end

entity_tag_left_p = num_entities(labels_Ω) + 1 # add a new tag for the left porous boundary (δ_1 in the paper)

# this for-loop finds all the vertices and edges in δ_1 and assigns the new tag to them
for d in 0:1
    face_coords = get_cell_coordinates(Grid(ReferenceFE{d}, model_Ω))
    left_p_boundary  = findall(is_p_boundary(face_coords, 0.0))

    for i in left_p_boundary
        labels_Ω.d_to_dface_to_entity[d+1][i] = entity_tag_left_p
    end
end
add_tag!(labels_Ω,"left_p",[entity_tag_left_p])

writevtk(model_Ω, "model")

## Spaces
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

# strongly assign BCs to γ_1, γ_2, δ_1, and δ_2
V = TestFESpace(model_Ω,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet", "top", "left_p", "bottom"], 
                dirichlet_masks=[(true, true), (true, true), (true, false), (false, true)])
u0= VectorValue(0,0)
U = TrialFESpace(V, [u_pois, u0, u0, u0])

reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
Q = TestFESpace(model_Ω,reffeₚ,conformity=:H1)
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

degree = 2
Ω = Triangulation(model_Ω)
dΩ = Measure(Ω,degree)

### Bilinear Forms
#m(t,u,v) = ∫( ∂t(u)⋅v )*dΩ
a((u,p),(v,q)) = ∫(1/Re * ∇(u)⊙∇(v) - p*(∇⋅v) + q*(∇⋅u))dΩ
l((v,q)) = ∫(0*q)dΩ

#create the operator
op = AffineFEOperator(a,l,X,Y)

#solve the resulting system
uh, ph = solve(op)

#write to vtk for visualisation
writevtk(Ω, "results", cellfields=["uh"=>uh, "ph"=>ph])

end




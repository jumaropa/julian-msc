module channel_flow_test
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues

## Parameters (see Cimolin (2013))
ρ = 1.184
μ = 1.855e-5
K = 3.71e-7
C_F = 0.5

#characteristic values
L = 1e-3        #charac. length
U_char = 1e-1   #charac. velocity

Re = ρ*L*U_char/μ   #Reynolds number

Gr_v = μ*L/(ρ*U_char*K)
Gr_i = C_F*L/sqrt(K)

## Domain
L_x = 50     #length of whole domain
L_y_p = 3    #height of porous medium
L_y_f = 4    #height of fluid region

domain = (0,L_x,-L_y_p,L_y_f)
partition = (50,20)
model_Ω = CartesianDiscreteModel(domain,partition) # create a square grid with the desired partition

labels_Ω = get_face_labeling(model_Ω)            # get the labels from the model
# By default, the tags/labels are assigned as follows:
# tag - label - description
# 1 - "tag_1" - lower left corner
# 2 - "tag_2" - lower right corner
# 3 - "tag_3" - upper left corner
# 4 - "tag_4" - upper right corner
# 5 - "tag_5" - bottom side
# 6 - "tag_6" - top side
# 7 - "tag_7" - left side
# 8 - "tag_8" - right side
# 9 - "interior" - interior (all the other points)
# [1,2,3,4,5,6,7,8] - "boundary" - boundary (all points not in interior)

add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
add_tag_from_tags!(labels_Ω,"bottom",[1,2,5])    # assign the label "bottom" to the entity 1,2 and 5 (bottom corners and bottom side)
add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
add_tag_from_tags!(labels_Ω,"water",[9])

## Boundary Conditions
u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)  #inlet velocity profile

#check if element (with coordinates coords) is in the porous medium
function is_p(coords)
    av = sum(coords)/length(coords) #average of the coords
    -L_y_p < av[2] < 0
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

add_tag_from_tags!(labels_Ω,"porous_b",["porous", "left_p", "right_p", "bottom"]) #porous region + boundaries
add_tag_from_tags!(labels_Ω,"left_p_b",["left_p","tag_1"])                        #left porous boundary + lower left corner
add_tag_from_tags!(labels_Ω,"right_p_b",["right_p","tag_2"])                      #right porous boundary + lower right corner

writevtk(model_Ω, "model")

## Spaces
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

# velocity test and trial space
# strongly assign BCs to γ_1, γ_2, δ_1, and δ_2
V = TestFESpace(model_Ω,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet", "top", "left_p_b", "bottom"], 
                dirichlet_masks=[(true, true), (true, true), (true, false), (false, true)])
u0= VectorValue(0,0)
U = TrialFESpace(V, [u_pois, u0, u0, u0])

#pressure test and trial space
reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
Q = TestFESpace(model_Ω,reffeₚ,conformity=:H1)
P = TrialFESpace(Q)

#multi-field spaces
X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

degree = 2

#entire domain
Ω = Triangulation(model_Ω)
dΩ = Measure(Ω,degree)

#porous domain
Ω_p = Triangulation(model_Ω, tags="porous_b")
dΩ_p = Measure(Ω_p,degree)

### Bilinear Forms
#m(t,u,v) = ∫( ∂t(u)⋅v )*dΩ
a((u,p),(v,q)) = ∫(1/Re * ∇(u)⊙∇(v) - p*(∇⋅v) + q*(∇⋅u))dΩ      #Stokes
b((u,p),(v,q)) = ∫(v⊙(Gr_v*u) + v⊙(Gr_i*((u⋅u).^(1/2))*u))dΩ_p  #Darcy-Forchheimer
c(u,v) = ∫(v⋅((∇(u)')⋅u))dΩ                                      #convective term
l((v,q)) = ∫(0*q)dΩ

#residual
res((u,p),(v,q)) = a((u,p),(v,q)) + b((u,p),(v,q)) + c(u,v) - l((v,q))

#create the operator
op = FEOperator(res,X,Y)

#solve the resulting system
uh, ph = solve(op)

#write to vtk for visualisation
writevtk(Ω, "results", cellfields=["uh"=>uh, "ph"=>ph])

end




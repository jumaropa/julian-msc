module ChannelRavStatic
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
n_x = 50
n_y = 7
partition = (n_x,n_y)
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

add_tag_from_tags!(labels_Ω,"top",[6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
add_tag_from_tags!(labels_Ω,"left_corner",[1])
add_tag_from_tags!(labels_Ω,"right_corner",[2])

## Boundary Conditions
u_pois((x,y), t::Real) = VectorValue(y*(L_y_f-y),0)  #inlet velocity profile
u_pois(t::Real) = x -> u_pois(x, t::Real)

## Initial Condition
u₀((x,y), t::Real) = u_pois((x,y), t)*(y>0) + VectorValue(0,0)*(y<=0)
#u₀((x,y), t::Real) = VectorValue(0,0)
u₀(t::Real) = x -> u₀(x, t::Real)

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

## FEM
order = 1
reffeᵤ = ReferenceFE(raviart_thomas,Float64,order)
reffeₚ = ReferenceFE(lagrangian,Float64,order; space=:P)

V = TestFESpace(model_Ω,
                reffeᵤ,
                labels=labels_Ω,
                conformity=:Hdiv,
                dirichlet_tags=["inlet", "top", "left_p", "bottom", "left_corner"])
                
Q = TestFESpace(model_Ω,
                reffeₚ,
                conformity=:L2,
                constraint=:zeromean)

Y = MultiFieldFESpace([V,Q])

u0(x, t::Real)= VectorValue(0,0)
u0(t::Real) = x -> u0(x, t)
U = TransientTrialFESpace(V, [u_pois, u0, u0, u0, u0])
P = TrialFESpace(Q)
X = TransientMultiFieldFESpace([U,P])

degree = 2
Ωₕ = Triangulation(model_Ω)
dΩ = Measure(Ωₕ,degree)

Ω_p = Triangulation(model_Ω, tags="porous_b")
dΩ_p = Measure(Ω_p,degree)
Ω_f = Interior(model_Ω,tags="fluid_b")
dΩ_f = Measure(Ω_f,degree)


#g(x) = x[1] == 0 && x[2] > 0 ? u_pois(x) : u0
#g(x) = u_pois(x)

Γ = BoundaryTriangulation(model_Ω, tags=["inlet", "top"])
Λ = Skeleton(model_Ω)
n_Γ = get_normal_vector(Γ)
n_Λ = get_normal_vector(Λ)
dΓ = Measure(Γ,degree)
dΛ = Measure(Λ,degree)
const γd = 500.0    # Nitsche coefficient
const h = 1/n_x

f = VectorValue(0.0,0.0)

### From Gridap tutorial 8
conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

unorm(u) = (u⋅u).^(1/2) + 1e-12
   
m(t,u,v) = ∫( ∂t(u)⋅v )dΩ
a((u,p),(v,q)) = ∫( 1/Re*∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ +
                 #∫( (γd/h)*v⋅(u-u_pois) - (∇(u)⋅n_Γ)⋅v - (∇(v)⋅n_Γ)⋅(u-u_pois) )dΓ +
                 #∫( (γd/h)*jump(v)⋅jump(u) - (mean(∇(u))⋅n_Λ.⁺)⋅jump(v) - (mean(∇(v))⋅n_Λ.⁺)⋅jump(u) )dΛ +
                 ∫(v⊙(Gr_v*u))dΩ_p
b(u,v) =         ∫(v⊙(Gr_i*((u⋅u).^(1/2))*u))dΩ_p
db(u,du,v) = ∫(v⊙(Gr_i*((u⋅u).^(1/2))*du))dΩ_p + ∫(v⊙(Gr_i*((u⋅du)/(unorm(u))*u)))dΩ_p
   
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ_f
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ_f

l((v,q)) = ∫( 0*q )dΩ 

res(t,(u,p),(v,q)) = m(t,u,v) + a((u,p),(v,q)) + b(u,v) + c(u,v)
jac(t,(u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + db(u,du,v) + dc(u,du,v)
jac_t(t,u,dtu,v) = ∫( v⋅dtu )dΩ

#create the operator
op = TransientFEOperator(res,jac,jac_t,X,Y)

Δt = 10.0
θ = 1.0
ode_solver = ThetaMethod(LUSolver(),Δt,θ)

#solve the resulting system
x₀ = interpolate_everywhere([u₀(0.0),0.0],X(0.0))
t₀ = 0.0
T = 100.0
xₕₜ = solve(ode_solver,op,x₀,t₀,T)

createpvd("channel_flow_transient_solution") do pvd
  for ((uh,ph),t) in xₕₜ
    println(t)
    pvd[t] = createvtk(Ωₕ,"results-rav-PE-trans_$t"*".vtu",cellfields=["uh"=>uh, "ph"=>ph], order=1)
  end
end

end

module StokesRav
using Gridap
using Gridap.ReferenceFEs
# using GridapEmbedded

n = 100
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6])
add_tag_from_tags!(labels,"diri0",[5,7,8])

order = 1
reffeᵤ = ReferenceFE(raviart_thomas,Float64,order)
reffeₚ = ReferenceFE(lagrangian,Float64,order; space=:P)

V = TestFESpace(model, reffeᵤ, labels=labels, conformity=:Hdiv, dirichlet_tags=["diri0","diri1"])
Q = TestFESpace(model, reffeₚ, conformity=:L2, constraint=:zeromean)

Y = MultiFieldFESpace([V,Q])

u0 = VectorValue(0,0)
u1 = VectorValue(1,0)
U = TrialFESpace(V, [u0, u1])
P = TrialFESpace(Q)
X = MultiFieldFESpace([U,P])

degree = 2
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

u_D(x) = x[2] == 1 ? u1 : u0

Γ = BoundaryTriangulation(model)
Λ = Skeleton(model)
n_Γ = get_normal_vector(Γ)
n_Λ = get_normal_vector(Λ)
dΓ = Measure(Γ,degree)
dΛ = Measure(Λ,degree)
const γd = 50.0    # Nitsche coefficient
const h = 1/n

f = VectorValue(1.0,0.0)
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ +
                 ∫( (γd/h)*v⋅u - (∇(u)⋅n_Γ)⋅v - (∇(v)⋅n_Γ)⋅u )dΓ +
                 ∫( (γd/h)*jump(v)⋅jump(u) - (mean(∇(u))⋅n_Λ.⁺)⋅jump(v) - (mean(∇(v))⋅n_Λ.⁺)⋅jump(u) )dΛ
l((v,q)) = ∫( v⋅f )dΩ +
           ∫( (γd/h)*v⋅u_D - (∇(v)⋅n_Γ)⋅u_D )dΓ

op = AffineFEOperator(a,l,X,Y)

uh, ph = solve(op)

writevtk(Ωₕ,"stokes-rav-results",order=2,cellfields=["uh"=>uh,"ph"=>ph])
end

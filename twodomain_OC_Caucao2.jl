module twodomain_Caucao2023
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues
using LineSearches: BackTracking

function main(n_x,n_y)

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
  # n_x = 50
  # n_y = 7
  partition = (n_x,n_y)
  println(partition)
  model_Ω = CartesianDiscreteModel(domain,partition) # create a square grid with the desired partition

  labels_Ω = get_face_labeling(model_Ω)            # get the labels from the model

  add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
  add_tag_from_tags!(labels_Ω,"bottom",[5])        # assign the label "bottom" to the entity 5 (bottom side)
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
  writevtk(model_Ω,"model_two_domain")

  ###Porous and fluid domains
  Ω = Interior(model_Ω)
  Ω_p = Interior(model_Ω,tags="porous_b")
  Ω_f = Interior(model_Ω,tags="fluid_b")

  ###Boundary
  Γ_f = Boundary(model_Ω, tags=["inlet", "top"])
  Γ_f_out = Boundary(model_Ω, tags="outlet")
  Γ_p_out = Boundary(model_Ω, tags="right_p")

  ###Interface
  Γfp = Interface(Ω_f,Ω_p)
  #Λ = Skeleton(Ω)

  ## Boundary Conditions
  u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)  #inlet velocity profile
  u_0 = VectorValue(0.0,0.0)

  ## FE Spaces
  k = 2
  reffe_fu = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_pu = ReferenceFE(raviart_thomas,Float64,k)
  reffeₚ = ReferenceFE(lagrangian,Float64,k-1)

  ##test spaces
  Vf = TestFESpace(Ω_f,reffe_fu,conformity=:H1,dirichlet_tags=["inlet", "top"])
  Vp = TestFESpace(Ω_p,reffe_pu,conformity=:Hdiv,dirichlet_tags=["left_p", "bottom", "left_corner"])
  Q = TestFESpace(Ω,reffeₚ,conformity=:L2, dirichlet_tags="right_p")
  Ξ = TestFESpace(Γfp, reffeₚ, conformity=:H1)

  ##trial spaces
  Uf = TrialFESpace(Vf,[u_pois, u_0])
  Up = TrialFESpace(Vp,[u_0, u_0, u_0])
  P = TrialFESpace(Q, 0.0)
  Λ = TrialFESpace(Ξ)

  Y = MultiFieldFESpace([Vf,Vp,Q,Ξ]) #test
  X = MultiFieldFESpace([Uf,Up,P,Λ]) #trial

  ## Measures
  degree = 2*k
  dΩ = Measure(Ω, degree) #fluid domain

  bdegree = 2*k
  dΓ_f_out = Measure(Γ_f_out, bdegree) #fluid right boundary
  dΓ_p_out = Measure(Γ_p_out, bdegree) #porous right boundary

  idegree = 2*k
  dΓfp  = Measure(Γfp, idegree) #interface

  n_Γ_f_out = get_normal_vector(Γ_f_out)
  n_Γ_p_out = get_normal_vector(Γ_p_out)
  n_Γfp = get_normal_vector(Γfp)

  ## Properties
  #μ = 1.0
  Kf = 1.0e6
  Kd = K

#⁺⁻
  ## Weak Form ##
  a((uf,up),(vf,vp)) = ∫( (1/Re*∇(uf)⊙∇(vf)) + up⋅vp )dΩ
  ap((uf,up,λ),(vf,vp,ξ)) = ∫( (vf.⁺⋅n_Γfp.⁺ + Gr_n*vp.⁻⋅n_Γfp.⁻)*λ )dΓfp +
                            ∫( (uf.⁺⋅n_Γfp.⁺ + up.⁻⋅n_Γfp.⁻)*ξ )dΓfp #+
                            #∫( Gr_c/Re*(uf.⁺-(uf.⁺⋅n_Γfp.⁺)*n_Γfp.⁺)⋅(vf.⁺-(vf.⁺⋅n_Γfp.⁺)*n_Γfp.⁺) )dΓ_fp  # tangential IC

  b((uf,up,p),(vf,vp,q)) = ∫( q*(∇⋅up) + q*(∇⋅uf) )dΩ -
                           ∫( Gr_n*p*(∇⋅vp) + p*(∇⋅vf) )dΩ
  
  res((uf,up,p,λ),(vf,vp,q,ξ)) = a((uf,up),(vf,vp)) + b((uf,up,p),(vf,vp,q)) + ap((uf,up,λ),(vf,vp,ξ))
  jac((uf,up,p,λ),(duf,dup,dp,dλ),(vf,vp,q,ξ)) = a((duf,dup),(vf,vp)) + b((duf,dup,dp),(vf,vp,q)) + ap((duf,dup,dλ),(vf,vp,ξ))
  
  op = FEOperator(res,jac,X,Y)
  nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=10)

  ## solve
  uhf, uhp, ph, λh = solve(nls, op)
  #writevtk(Ω,"twodomain-Caucao2023", cellfields=["uhf" => uhf, "uhp" => uhp, "ph" => ph])
  writevtk(Ω_f,"twodomain-Caucao2023-uf", cellfields=["uhf" => uhf])
  writevtk(Ω_p,"twodomain-Caucao2023-up", cellfields=["uhp" => uhp])
  writevtk(Ω,"twodomain-Caucao2023-p", cellfields=["ph" => ph])

end

main(2,7)
main(100,21)

end

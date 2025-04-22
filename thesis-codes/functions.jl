using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.CellData
using Gridap.Visualization

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

#PE parameters
Gr_v = μ*L/(ρ*U_char*K)
Gr_i = C_F*L/sqrt(K)

#NSD/NSF parameters
Gr_f = ρ*C_F*U_char*sqrt(K)/μ
Gr_n = ρ*K*U_char/(μ*L)
Gr_c = α_BJ*L/sqrt(K)

## Domain
L_x = 50     #length of whole domain
L_y_p = 3    #height of porous medium
L_y_f = 4    #height of fluid region

###FUNCTIONS

##Auxiliary functions
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

##Model functions
function lag_PE(n_x, n_y, order; write=false)
    println("LAG - PE")
    println("order: ", order)
    partition = (n_x,n_y)
    println(partition)
    domain = (0,L_x,-L_y_p,L_y_f)
    model_Ω = CartesianDiscreteModel(domain,partition) # create a grid with the desired partition

    labels_Ω = get_face_labeling(model_Ω)            # get the labels from the model

    add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
    add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
    add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
    add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
    add_tag_from_tags!(labels_Ω,"left_corner",[1])
    add_tag_from_tags!(labels_Ω,"right_corner",[2])

    ## Boundary Conditions
    u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)  #inlet velocity profile
    u0 = VectorValue(0,0)

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

    ## Spaces
    reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

    #test spaces
    V = TestFESpace(model_Ω,
                    reffeᵤ,
                    conformity=:H1,
                    dirichlet_tags=["inlet", "top", "left_p", "bottom", "left_corner"], 
                    dirichlet_masks=[(true, true), (true, true), (true, false), (false, true), (true, true)])
                    
    Q = TestFESpace(model_Ω,
                    reffeₚ,
                    conformity=:L2)
    
    #trial spaces
    U = TrialFESpace(V, [u_pois, u0, u0, u0, u0])
    P = TrialFESpace(Q)

    #multi-field spaces
    X = MultiFieldFESpace([U,P])
    Y = MultiFieldFESpace([V,Q])

    degree = 2*order

    #entire domain
    Ω = Triangulation(model_Ω)
    dΩ = Measure(Ω,degree)

    #subdomains
    Ω_p = Triangulation(model_Ω, tags="porous_b")
    dΩ_p = Measure(Ω_p,degree)
    Ω_f = Triangulation(model_Ω,tags="fluid_b")
    dΩ_f = Measure(Ω_f,degree)

    #auxiliary functions
    conv(u,∇u) = (∇u')⋅u
    dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
    unorm(u) = (u⋅u).^(1/2) + 1e-12

    ### Bilinear Forms
    a((u,p),(v,q)) = ∫(1/Re * ∇(u)⊙∇(v) - p*(∇⋅v) + q*(∇⋅u))dΩ +     #Stokes
                     ∫( v⊙(Gr_v*u) )dΩ_p  #Darcy
    b(u,v) =         ∫( v⊙(Gr_i*((u⋅u).^(1/2))*u) )dΩ_p
    db(u,du,v) = ∫(v⊙(Gr_i*((u⋅u).^(1/2))*du))dΩ_p + ∫(v⊙(Gr_i*((u⋅du)/(unorm(u))*u)))dΩ_p

    c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ   #convective term
    dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

    l((v,q)) = ∫(0*q)dΩ

    #residual
    res((u,p),(v,q)) = a((u,p),(v,q)) + b(u,v) + c(u,v)
    jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + db(u,du,v) + dc(u,du,v)

    op = FEOperator(res,jac,X,Y)
    nls = NLSolver(show_trace=true, method=:newton, iterations=10)

    xh = solve(nls,op)
    uh, ph = xh

    div_uh = divergence(uh)
    println(sqrt(sum(∫( div_uh*div_uh )dΩ)))
    
    if write
        writevtk(Ω, "results-lag-PE-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uh"=>div_uh, "lag-uh$(n_x)"=>uh, "lag-ph$(n_x)"=>ph])
    end
    
    (uh, ph, U, P, dΩ)
end

function lag_NSD(n_x, n_y, order; write=false)
    println("LAG - NSD")
    println("order: ", order)
    partition = (n_x,n_y)
    println(partition)
    domain = (0,L_x,-L_y_p,L_y_f)
    model_Ω = CartesianDiscreteModel(domain,partition) # create a square grid with the desired partition

    labels_Ω = get_face_labeling(model_Ω)            # get the labels from the model

    add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
    add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
    add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
    add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
    add_tag_from_tags!(labels_Ω,"left_corner",[1])
    add_tag_from_tags!(labels_Ω,"right_corner",[2])

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

    reffe_uf = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_pf = ReferenceFE(lagrangian,Float64,order-1)
    reffe_up = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_pp = ReferenceFE(lagrangian,Float64,order)

    ##test spaces
    # fluid velocity
    Vf = TestFESpace(
      Ω_f,
      reffe_uf,
      conformity=:H1,
      dirichlet_tags=["left_p", "inlet", "top"],
      dirichlet_masks=[(true, false), (true, true), (true, true)])

    # porous velocity
    Vp = TestFESpace(
      Ω_p,
      reffe_up,
      conformity=:H1)

    # fluid pressure
    Qf = TestFESpace(
      Ω_f,
      reffe_pf,
      conformity=:L2)

    # porous pressure
    Qp = TestFESpace(
      Ω_p,
      reffe_pp,
      conformity=:H1,
      dirichlet_tags=["right_p", "right_corner"])

    ##trial spaces
    Uf = TrialFESpace(Vf,[u_0, u_pois, u_0])
    Up = TrialFESpace(Vp)
    Pf = TrialFESpace(Qf)
    Pp = TrialFESpace(Qp, [0.0, 0.0])

    Y = MultiFieldFESpace([Vf,Qf,Vp,Qp]) #test
    X = MultiFieldFESpace([Uf,Pf,Up,Pp]) #trial

    ## Measures
    degree = 2*order
    dΩ_f = Measure(Ω_f, degree) #fluid domain
    dΩ_p = Measure(Ω_p, degree) #porous domain

    idegree = 2*order
    dΓ_fp = Measure(Γ_fp,idegree) #interface

    #auxiliary functions
    conv(u,∇u) = (∇u')⋅u
    dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

    ## Weak Form ##
    #fluid
    af((uf,pf),(vf,qf)) = ∫( 1/Re*∇(uf)⊙∇(vf) - pf*(∇⋅vf) + qf*(∇⋅uf) )dΩ_f
    c((uf,pf),(vf,qf)) = ∫( vf⊙(conv∘(uf,∇(uf))) )dΩ_f
    dc(uf,duf,vf) = ∫( vf⊙(dconv∘(duf,∇(duf),uf,∇(uf))) )dΩ_f
    
    #porous
    ap((up,pp),(vp,qp)) = ∫( Gr_n*∇(pp)⋅∇(qp) )dΩ_p +
                          ∫( up⋅vp + Gr_n*∇(pp)⋅vp )dΩ_p

    #interface
    apf((uf,pf,up,pp),(vf,qf,vp,qp)) = ∫( Gr_c/Re*(uf.plus-(uf.plus⋅n_Γfp.plus)*n_Γfp.plus)⋅(vf.plus-(vf.plus⋅n_Γfp.plus)*n_Γfp.plus) )dΓ_fp - 
                                       ∫( (uf.plus⋅n_Γfp.plus)*qp.minus )dΓ_fp +
                                       ∫( (vf.plus⋅n_Γfp.plus)*pp.minus )dΓ_fp

    l((vf,qf)) = ∫(0*qf)dΩ_f

    a((uf,pf,up,pp),(vf,qf,vp,qp)) = af((uf,pf),(vf,qf)) + ap((up,pp),(vp,qp)) + apf((uf,pf,up,pp),(vf,qf,vp,qp))

    res((uf,pf,up,pp),(vf,qf,vp,qp)) = a((uf,pf,up,pp),(vf,qf,vp,qp)) + c((uf,pf),(vf,qf))
    jac((uf,pf,up,pp),(duf,dpf,dup,dpp),(vf,qf,vp,qp)) = a((duf,dpf,dup,dpp),(vf,qf,vp,qp)) + dc(uf,duf,vf)

    op = FEOperator(res,jac,X,Y)

    nls = NLSolver(show_trace=true, method=:newton, iterations=10)

    ## solve
    uhf, phf, uhp, php = solve(nls, op)
    div_uhf = divergence(uhf)
    println("div uhf: ", sqrt(sum(∫( div_uhf*div_uhf )dΩ_f)))
    div_uhp = divergence(uhp)
    println("div uhp: ", sqrt(sum(∫( div_uhp*div_uhp )dΩ_p)))
    println("total: ", sqrt(sum(∫( div_uhf*div_uhf )dΩ_f + ∫( div_uhp*div_uhp )dΩ_p)))
    
    if write
        writevtk(Ω_f, "results-lag-NSDf-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhf"=>div_uhf, "lag-uhf$(n_x)"=>uhf, "lag-phf$(n_x)"=>phf])
        writevtk(Ω_p, "results-lag-NSDp-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhp"=>div_uhp, "lag-uhp$(n_x)"=>uhp, "lag-php$(n_x)"=>php])
    end
        
    (uhf, phf, uhp, php, Uf, Pf, Up, Pp, dΩ_f, dΩ_p)
end

function lag_NSF(n_x, n_y, order; write=false)
    println("LAG - NSF")
    println("order: ", order)
    partition = (n_x,n_y)
    println(partition)
    domain = (0,L_x,-L_y_p,L_y_f)
    model_Ω = CartesianDiscreteModel(domain,partition) # create a square grid with the desired partition

    labels_Ω = get_face_labeling(model_Ω)            # get the labels from the model

    add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
    add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
    add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
    add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
    add_tag_from_tags!(labels_Ω,"left_corner",[1])
    add_tag_from_tags!(labels_Ω,"right_corner",[2])

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

    reffe_uf = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_pf = ReferenceFE(lagrangian,Float64,order-1)
    reffe_up = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_pp = ReferenceFE(lagrangian,Float64,order)

    ##test spaces
    # fluid velocity
    Vf = TestFESpace(
      Ω_f,
      reffe_uf,
      conformity=:H1,
      dirichlet_tags=["left_p", "inlet", "top"],
      dirichlet_masks=[(true, false), (true, true), (true, true)])

    # porous velocity
    Vp = TestFESpace(
      Ω_p,
      reffe_up,
      conformity=:H1)

    # fluid pressure
    Qf = TestFESpace(
      Ω_f,
      reffe_pf,
      conformity=:L2)

    # porous pressure
    Qp = TestFESpace(
      Ω_p,
      reffe_pp,
      conformity=:H1,
      dirichlet_tags=["right_p", "right_corner"])

    ##trial spaces
    Uf = TrialFESpace(Vf,[u_0, u_pois, u_0])
    Up = TrialFESpace(Vp)
    Pf = TrialFESpace(Qf)
    Pp = TrialFESpace(Qp, [0.0, 0.0])

    Y = MultiFieldFESpace([Vf,Qf,Vp,Qp]) #test
    X = MultiFieldFESpace([Uf,Pf,Up,Pp]) #trial

    ## Measures
    degree = 2*order
    dΩ_f = Measure(Ω_f, degree) #fluid domain
    dΩ_p = Measure(Ω_p, degree) #porous domain

    idegree = 2*order
    dΓ_fp = Measure(Γ_fp,idegree) #interface

    #auxiliary functions
    conv(u,∇u) = (∇u')⋅u
    dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
    unorm(u) = (u⋅u).^(1/2) + 1e-12

    ## Weak Form ##
    #fluid
    af((uf,pf),(vf,qf)) = ∫( 1/Re*∇(uf)⊙∇(vf) - pf*(∇⋅vf) + qf*(∇⋅uf) )dΩ_f
    c((uf,pf),(vf,qf)) = ∫( vf⊙(conv∘(uf,∇(uf))) )dΩ_f
    dc(uf,duf,vf) = ∫( vf⊙(dconv∘(duf,∇(duf),uf,∇(uf))) )dΩ_f
    
    #porous
    ap((up,pp),(vp,qp)) = ∫( up⋅vp + Gr_n*∇(pp)⋅vp - ∇(qp)⋅up )dΩ_p
    apb((up,pp),(vp,qp)) = ∫( vp⊙(Gr_f*((up⋅up).^(1/2))*up) )dΩ_p
    dapb(up,dup,vp) = ∫(vp⊙(Gr_f*((up⋅up).^(1/2))*dup))dΩ_p + ∫(vp⊙(Gr_f*((up⋅dup)/(unorm(up))*up)))dΩ_p

    #interface
    apf((uf,pf,up,pp),(vf,qf,vp,qp)) = ∫( Gr_c/Re*(uf.plus-(uf.plus⋅n_Γfp.plus)*n_Γfp.plus)⋅(vf.plus-(vf.plus⋅n_Γfp.plus)*n_Γfp.plus) )dΓ_fp - 
                                       ∫( (uf.plus⋅n_Γfp.plus)*qp.minus )dΓ_fp +
                                       ∫( (vf.plus⋅n_Γfp.plus)*pp.minus )dΓ_fp

    l((vf,qf)) = ∫(0*qf)dΩ_f

    a((uf,pf,up,pp),(vf,qf,vp,qp)) = af((uf,pf),(vf,qf)) + ap((up,pp),(vp,qp)) + apf((uf,pf,up,pp),(vf,qf,vp,qp))

    res((uf,pf,up,pp),(vf,qf,vp,qp)) = a((uf,pf,up,pp),(vf,qf,vp,qp)) + apb((up,pp),(vp,qp)) + c((uf,pf),(vf,qf))
    jac((uf,pf,up,pp),(duf,dpf,dup,dpp),(vf,qf,vp,qp)) = a((duf,dpf,dup,dpp),(vf,qf,vp,qp)) + dapb(up,dup,vp) + dc(uf,duf,vf)

    op = FEOperator(res,jac,X,Y)

    nls = NLSolver(show_trace=true, method=:newton, iterations=10)

    ## solve
    uhf, phf, uhp, php = solve(nls, op)
    div_uhf = divergence(uhf)
    println("div uhf: ", sqrt(sum(∫( div_uhf*div_uhf )dΩ_f)))
    div_uhp = divergence(uhp)
    println("div uhp: ", sqrt(sum(∫( div_uhp*div_uhp )dΩ_p)))
    println("total: ", sqrt(sum(∫( div_uhf*div_uhf )dΩ_f + ∫( div_uhp*div_uhp )dΩ_p)))
    
    if write
        writevtk(Ω_f, "results-lag-NSFf-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhf"=>div_uhf, "lag-uhf$(n_x)"=>uhf, "lag-phf$(n_x)"=>phf])
        writevtk(Ω_p, "results-lag-NSFp-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhp"=>div_uhp, "lag-uhp$(n_x)"=>uhp, "lag-php$(n_x)"=>php])
    end
        
    (uhf, phf, uhp, php, Uf, Pf, Up, Pp, dΩ_f, dΩ_p)
end

function rav_PE(n_x, n_y, order; write=false)
    println("RAV - PE")
    println("order: ", order)
    partition = (n_x,n_y)
    println(partition)
    domain = (0,L_x,-L_y_p,L_y_f)
    model = CartesianDiscreteModel(domain,partition)
    labels_Ω = get_face_labeling(model)

    add_tag_from_tags!(labels_Ω,"top",[6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
    add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
    add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
    add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
    add_tag_from_tags!(labels_Ω,"left_corner",[1])
    add_tag_from_tags!(labels_Ω,"right_corner",[2])

    entity_tag_left_p = num_entities(labels_Ω) + 1     # add a new tag for the left porous boundary (δ_1 in the paper)
    entity_tag_right_p = num_entities(labels_Ω) + 2    # add a new tag for the right porous boundary (δ_3 in the paper)
    entity_tag_p = num_entities(labels_Ω) + 3          # add a new tag for (interior) porous region (Ω_p)

    # this for-loop finds all the vertices and edges in δ_1 and δ_3
    # and the faces in Ω_p and assigns the new tags to them
    for d in 0:2
        face_coords = get_cell_coordinates(Grid(ReferenceFE{d}, model))
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

    reffeᵤ = ReferenceFE(raviart_thomas,Float64,order-1)
    reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

    V = FESpace(model,
                reffeᵤ,
                conformity=:HDiv,
                dirichlet_tags=["inlet", "top", "left_p", "bottom", "left_corner"])

    Q = FESpace(model,
                reffeₚ,
                conformity=:L2)

    uD((x,y)) = VectorValue(0.0,0.0)
    u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)
    U = TrialFESpace(V,[u_pois,uD,uD,uD,uD])
    P = TrialFESpace(Q)

    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])

    Ω = Triangulation(model)
    degree = max(3,2*order+1)
    dΩ = Measure(Ω,degree)

    Ω_p = Triangulation(model, tags="porous_b")
    dΩ_p = Measure(Ω_p,degree)
    Ω_f = Interior(model,tags="fluid_b")
    dΩ_f = Measure(Ω_f,degree)

    px = get_physical_coordinate(Ω)

    Γ = BoundaryTriangulation(model, tags=["inlet", "top"])
    Λ = Skeleton(model)
    n_Γ = get_normal_vector(Γ)
    n_Λ = get_normal_vector(Λ)
    dΓ = Measure(Γ,degree)
    dΛ = Measure(Λ,degree)
    γd = 100   # Nitsche coefficient
    h = 1/n_x

    g(x) = u_pois(x)
    conv(u,∇u) = (∇u')⋅u
    dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
    unorm(u) = (u⋅u).^(1/2) + 1e-12

    a((u,p), (v,q)) = ∫(1/Re*∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u))dΩ + 
                      ∫(v⊙(Gr_v*u))dΩ_p +
                      ∫( (γd/h)*v⋅(u-g) - (∇(u)⋅n_Γ)⋅v - (∇(v)⋅n_Γ)⋅(u-g) )dΓ +
                      ∫( (γd/h)*jump(v)⋅jump(u) - (mean(∇(u))⋅n_Λ.⁺)⋅jump(v) - (mean(∇(v))⋅n_Λ.⁺)⋅jump(u) )dΛ
    b(u,v) =          ∫( v⊙(Gr_i*((u⋅u).^(1/2))*u) )dΩ_p
    db(u,du,v) = ∫(v⊙(Gr_i*((u⋅u).^(1/2))*du))dΩ_p + ∫(v⊙(Gr_i*((u⋅du)/(unorm(u))*u)))dΩ_p

    c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ   #convective term
    dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

    l((v,q)) = ∫( 0*q )dΩ

    res((u,p),(v,q)) = a((u,p),(v,q)) + b(u,v) + c(u,v)
    jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + db(u,du,v) + dc(u,du,v)

    op = FEOperator(res,jac,X,Y)
    tol = order >= 3 ? 1e-4 : 1e-8
    nls = NLSolver(show_trace=true, method=:newton, iterations=10, ftol=tol)

    xh = solve(nls, op)
    uh, ph = xh

    div_uh = divergence(uh)
    l2_norm = sqrt(sum(∫( div_uh*div_uh )dΩ))
    println(l2_norm)
    
    if write
        writevtk(Ω, "results-rav-PE-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uh"=>div_uh, "rav-uh$(n_x)"=>uh, "rav-ph$(n_x)"=>ph])
    end
    
    (uh, ph, U, P, dΩ)
end

function rav_NSD(n_x, n_y, order; write=false)
    println("RAV - NSD")
    println("order: ", order)
    partition = (n_x,n_y)
    println(partition)
    domain = (0,L_x,-L_y_p,L_y_f)
    model = CartesianDiscreteModel(domain,partition)
    labels_Ω = get_face_labeling(model)

    add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
    add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
    add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
    add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
    add_tag_from_tags!(labels_Ω,"left_corner",[1])
    add_tag_from_tags!(labels_Ω,"right_corner",[2])

    entity_tag_left_p = num_entities(labels_Ω) + 1     # add a new tag for the left porous boundary (δ_1 in the paper)
    entity_tag_right_p = num_entities(labels_Ω) + 2    # add a new tag for the right porous boundary (δ_3 in the paper)
    entity_tag_p = num_entities(labels_Ω) + 3          # add a new tag for (interior) porous region (Ω_p)

    # this for-loop finds all the vertices and edges in δ_1 and δ_3
    # and the faces in Ω_p and assigns the new tags to them
    for d in 0:2
        face_coords = get_cell_coordinates(Grid(ReferenceFE{d}, model))
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

    Ω_p = Triangulation(model, tags="porous_b")
    Ω_f = Triangulation(model,tags="fluid_b")

    reffe_uf = ReferenceFE(raviart_thomas,Float64,order-1)
    reffe_pf = ReferenceFE(lagrangian,Float64,order-1)
    reffe_up = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_pp = ReferenceFE(lagrangian,Float64,order)

    ##test spaces
    # fluid velocity
    Vf = TestFESpace(
      Ω_f,
      reffe_uf,
      conformity=:Hdiv,
      dirichlet_tags=["inlet", "top"])

    # porous velocity
    Vp = TestFESpace(
      Ω_p,
      reffe_up,
      conformity=:H1)

    # fluid pressure
    Qf = TestFESpace(
      Ω_f,
      reffe_pf,
      conformity=:L2)

    # porous pressure
    Qp = TestFESpace(
      Ω_p,
      reffe_pp,
      conformity=:H1,
      dirichlet_tags=["right_p"])

    uD((x,y)) = VectorValue(0.0,0.0)
    u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)

    ##trial spaces
    Uf = TrialFESpace(Vf, [u_pois, uD])
    Up = TrialFESpace(Vp)
    Pf = TrialFESpace(Qf)
    Pp = TrialFESpace(Qp, [0.0])

    Y = MultiFieldFESpace([Vf,Qf,Vp,Qp]) #test
    X = MultiFieldFESpace([Uf,Pf,Up,Pp]) #trial
    Ω = Triangulation(model)

    degree = max(3,2*order+1)

    dΩ_p = Measure(Ω_p,degree)
    dΩ_f = Measure(Ω_f,degree)

    dΩ = Measure(Ω,degree)

    ###Interface
    Γ_fp = InterfaceTriangulation(Ω_f,Ω_p)
    n_Γfp = get_normal_vector(Γ_fp)

    idegree = 2*order+1
    dΓ_fp = Measure(Γ_fp,idegree) #interface

    Γ_f = BoundaryTriangulation(model, tags=["inlet", "top"])
    Λ_f = Skeleton(Ω_f)
    n_Λ_f = get_normal_vector(Λ_f)
    dΛ_f = Measure(Λ_f,degree)
    Λ_p = Skeleton(Ω_p)
    n_Λ_p = get_normal_vector(Λ_p)
    dΛ_p = Measure(Λ_p,degree)
    bdegree = 2*order+1
    dΓ_f = Measure(Γ_f, bdegree) #fluid boundary
    n_Γ_f = get_normal_vector(Γ_f)

    γd = 100.0    # Nitsche coefficient
    h = 1/n_x

    g(x) = u_pois(x)

    conv(u,∇u) = (∇u')⋅u
    dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

    c((uf,pf),(vf,qf)) = ∫( vf⊙(conv∘(uf,∇(uf))) )dΩ_f
    dc(uf,duf,vf) = ∫( vf⊙(dconv∘(duf,∇(duf),uf,∇(uf))) )dΩ_f
    
    #fluid
    af((uf,pf),(vf,qf)) = ∫( 1/Re*∇(vf)⊙∇(uf) - pf*(∇⋅vf) + qf*(∇⋅uf) )dΩ_f + 
                          ∫( (γd/h)*vf⋅(uf-g) - (∇(uf)⋅n_Γ_f)⋅vf - (∇(vf)⋅n_Γ_f)⋅(uf-g) )dΓ_f +
                          ∫( (γd/h)*jump(vf)⋅jump(uf) - (mean(∇(uf))⋅n_Λ_f.⁺)⋅jump(vf) - (mean(∇(vf))⋅n_Λ_f.⁺)⋅jump(uf) )dΛ_f
    
    #porous
    ap((up,pp),(vp,qp)) = ∫( Gr_n*∇(pp)⋅∇(qp) )dΩ_p +
                          ∫( up⋅vp + Gr_n*∇(pp)⋅vp )dΩ_p
    
    #interface
    apf((uf,pf,up,pp),(vf,qf,vp,qp)) = ∫( Gr_c/Re*(uf.plus-(uf.plus⋅n_Γfp.plus)*n_Γfp.plus)⋅(vf.plus-(vf.plus⋅n_Γfp.plus)*n_Γfp.plus) )dΓ_fp - 
                                 ∫( (uf.plus⋅n_Γfp.plus)*qp.minus )dΓ_fp +
                                 ∫( (vf.plus⋅n_Γfp.plus)*pp.minus )dΓ_fp
                                       
    a((uf,pf,up,pp),(vf,qf,vp,qp)) = af((uf,pf),(vf,qf)) + ap((up,pp),(vp,qp)) + apf((uf,pf,up,pp),(vf,qf,vp,qp))

    l((vf,qf)) = ∫(0*qf)dΩ_f

    res((uf,pf,up,pp),(vf,qf,vp,qp)) = a((uf,pf,up,pp),(vf,qf,vp,qp)) + c((uf,pf),(vf,qf))
    jac((uf,pf,up,pp),(duf,dpf,dup,dpp),(vf,qf,vp,qp)) = a((duf,dpf,dup,dpp),(vf,qf,vp,qp)) + dc(uf,duf,vf)

    op = FEOperator(res,jac,X,Y)
    
    tol = order >= 3 ? 1e-4 : 1e-8
    nls = NLSolver(show_trace=true, method=:newton, iterations=10, ftol=tol)

    uhf, phf, uhp, php = solve(nls, op)

    div_uhf = divergence(uhf)
    div_uhp = divergence(uhp)
    l2_norm_uhf = sqrt(sum(∫( div_uhf*div_uhf )dΩ_f))
    l2_norm_uhp = sqrt(sum(∫( div_uhp*div_uhp )dΩ_p))
    println("uhf div: ", l2_norm_uhf)
    println("uhp div: ", l2_norm_uhp)
    println("total: ", sqrt(sum(∫( div_uhf*div_uhf )dΩ_f + ∫( div_uhp*div_uhp )dΩ_p)))
    
    if write
        writevtk(Ω_f, "results-rav-NSDf-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhf"=>div_uhf, "rav-uhf$(n_x)"=>uhf, "rav-phf$(n_x)"=>phf])
        writevtk(Ω_p, "results-rav-NSDp-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhp"=>div_uhp, "rav-uhp$(n_x)"=>uhp, "rav-php$(n_x)"=>php])
    end
    
    (uhf, phf, uhp, php, Uf, Pf, Up, Pp, dΩ_f, dΩ_p) 
end

function rav_NSF(n_x, n_y, order; write=false)
    println("RAV - NSF")
    println("order: ", order)
    partition = (n_x,n_y)
    println(partition)
    domain = (0,L_x,-L_y_p,L_y_f)
    model = CartesianDiscreteModel(domain,partition)
    labels_Ω = get_face_labeling(model)

    add_tag_from_tags!(labels_Ω,"top",[3,4,6])       # assign the label "top" to the entity 3,4 and 6 (top corners and top side)
    add_tag_from_tags!(labels_Ω,"bottom",[5])    # assign the label "bottom" to the entity 5 (bottom side)
    add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
    add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
    add_tag_from_tags!(labels_Ω,"left_corner",[1])
    add_tag_from_tags!(labels_Ω,"right_corner",[2])

    entity_tag_left_p = num_entities(labels_Ω) + 1     # add a new tag for the left porous boundary (δ_1 in the paper)
    entity_tag_right_p = num_entities(labels_Ω) + 2    # add a new tag for the right porous boundary (δ_3 in the paper)
    entity_tag_p = num_entities(labels_Ω) + 3          # add a new tag for (interior) porous region (Ω_p)

    # this for-loop finds all the vertices and edges in δ_1 and δ_3
    # and the faces in Ω_p and assigns the new tags to them
    for d in 0:2
        face_coords = get_cell_coordinates(Grid(ReferenceFE{d}, model))
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

    Ω_p = Triangulation(model, tags="porous_b")
    Ω_f = Triangulation(model,tags="fluid_b")

    reffe_uf = ReferenceFE(raviart_thomas,Float64,order-1)
    reffe_pf = ReferenceFE(lagrangian,Float64,order-1)
    reffe_up = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_pp = ReferenceFE(lagrangian,Float64,order)

    ##test spaces
    # fluid velocity
    Vf = TestFESpace(
      Ω_f,
      reffe_uf,
      conformity=:Hdiv,
      dirichlet_tags=["inlet", "top"])

    # porous velocity
    Vp = TestFESpace(
      Ω_p,
      reffe_up,
      conformity=:H1)

    # fluid pressure
    Qf = TestFESpace(
      Ω_f,
      reffe_pf,
      conformity=:L2)

    # porous pressure
    Qp = TestFESpace(
      Ω_p,
      reffe_pp,
      conformity=:H1,
      dirichlet_tags=["right_p"])

    uD((x,y)) = VectorValue(0.0,0.0)
    u_pois((x,y)) = VectorValue(y*(L_y_f-y),0)

    ##trial spaces
    Uf = TrialFESpace(Vf, [u_pois, uD])
    Up = TrialFESpace(Vp)
    Pf = TrialFESpace(Qf)
    Pp = TrialFESpace(Qp, [0.0])

    Y = MultiFieldFESpace([Vf,Qf,Vp,Qp]) #test
    X = MultiFieldFESpace([Uf,Pf,Up,Pp]) #trial
    Ω = Triangulation(model)

    degree = max(3,2*order+1)

    dΩ_p = Measure(Ω_p,degree)
    dΩ_f = Measure(Ω_f,degree)

    dΩ = Measure(Ω,degree)

    ###Interface
    Γ_fp = InterfaceTriangulation(Ω_f,Ω_p)
    n_Γfp = get_normal_vector(Γ_fp)

    idegree = 2*order+1
    dΓ_fp = Measure(Γ_fp,idegree) #interface

    Γ_f = BoundaryTriangulation(model, tags=["inlet", "top"])
    Λ_f = Skeleton(Ω_f)
    n_Λ_f = get_normal_vector(Λ_f)
    dΛ_f = Measure(Λ_f,degree)
    Λ_p = Skeleton(Ω_p)
    n_Λ_p = get_normal_vector(Λ_p)
    dΛ_p = Measure(Λ_p,degree)
    bdegree = 2*order+1
    dΓ_f = Measure(Γ_f, bdegree) #fluid boundary
    n_Γ_f = get_normal_vector(Γ_f)

    γd = 100.0    # Nitsche coefficient
    h = 1/n_x

    #auxiliayr functions
    g(x) = u_pois(x)
    conv(u,∇u) = (∇u')⋅u
    dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
    unorm(u) = (u⋅u).^(1/2) + 1e-12
    
    #fluid
    af((uf,pf),(vf,qf)) = ∫( 1/Re*∇(vf)⊙∇(uf) - pf*(∇⋅vf) + qf*(∇⋅uf) )dΩ_f + 
                          ∫( (γd/h)*vf⋅(uf-g) - (∇(uf)⋅n_Γ_f)⋅vf - (∇(vf)⋅n_Γ_f)⋅(uf-g) )dΓ_f +
                          ∫( (γd/h)*jump(vf)⋅jump(uf) - (mean(∇(uf))⋅n_Λ_f.⁺)⋅jump(vf) - (mean(∇(vf))⋅n_Λ_f.⁺)⋅jump(uf) )dΛ_f
    
    c((uf,pf),(vf,qf)) = ∫( vf⊙(conv∘(uf,∇(uf))) )dΩ_f
    dc(uf,duf,vf) = ∫( vf⊙(dconv∘(duf,∇(duf),uf,∇(uf))) )dΩ_f
    
    #porous
    ap((up,pp),(vp,qp)) = ∫( up⋅vp + Gr_n*∇(pp)⋅vp - ∇(qp)⋅up )dΩ_p
    apb((up,pp),(vp,qp)) = ∫( vp⊙(Gr_f*((up⋅up).^(1/2))*up) )dΩ_p
    dapb(up,dup,vp) = ∫(vp⊙(Gr_f*((up⋅up).^(1/2))*dup))dΩ_p + ∫(vp⊙(Gr_f*((up⋅dup)/(unorm(up))*up)))dΩ_p

    #interface
    apf((uf,pf,up,pp),(vf,qf,vp,qp)) = ∫( Gr_c/Re*(uf.plus-(uf.plus⋅n_Γfp.plus)*n_Γfp.plus)⋅(vf.plus-(vf.plus⋅n_Γfp.plus)*n_Γfp.plus) )dΓ_fp - 
                                       ∫( (uf.plus⋅n_Γfp.plus)*qp.minus )dΓ_fp +
                                       ∫( (vf.plus⋅n_Γfp.plus)*pp.minus )dΓ_fp

    l((vf,qf)) = ∫(0*qf)dΩ_f

    a((uf,pf,up,pp),(vf,qf,vp,qp)) = af((uf,pf),(vf,qf)) + ap((up,pp),(vp,qp)) + apf((uf,pf,up,pp),(vf,qf,vp,qp))

    res((uf,pf,up,pp),(vf,qf,vp,qp)) = a((uf,pf,up,pp),(vf,qf,vp,qp)) + apb((up,pp),(vp,qp)) + c((uf,pf),(vf,qf))
    jac((uf,pf,up,pp),(duf,dpf,dup,dpp),(vf,qf,vp,qp)) = a((duf,dpf,dup,dpp),(vf,qf,vp,qp)) + dapb(up,dup,vp) + dc(uf,duf,vf)

    op = FEOperator(res,jac,X,Y)
    
    tol = order >= 3 ? 1e-4 : 1e-8
    nls = NLSolver(show_trace=true, method=:newton, iterations=10, ftol=tol)

    uhf, phf, uhp, php = solve(nls, op)

    div_uhf = divergence(uhf)
    div_uhp = divergence(uhp)
    l2_norm_uhf = sqrt(sum(∫( div_uhf*div_uhf )dΩ_f))
    l2_norm_uhp = sqrt(sum(∫( div_uhp*div_uhp )dΩ_p))
    println("uhf div: ", l2_norm_uhf)
    println("uhp div: ", l2_norm_uhp)
    println("total: ", sqrt(sum(∫( div_uhf*div_uhf )dΩ_f + ∫( div_uhp*div_uhp )dΩ_p)))
    
    if write
        writevtk(Ω_f, "results-rav-NSFf-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhf"=>div_uhf, "rav-uhf$(n_x)"=>uhf, "rav-phf$(n_x)"=>phf])
        writevtk(Ω_p, "results-rav-NSFp-$(order)-$(n_x)-$(n_y)", order=order, cellfields=["div_uhp"=>div_uhp, "rav-uhp$(n_x)"=>uhp, "rav-php$(n_x)"=>php])
    end
    
    (uhf, phf, uhp, php, Uf, Pf, Up, Pp, dΩ_f, dΩ_p) 
end

#include "nonlin_solver.hpp"
    
NonlinSolver::NonlinSolver(Network &nw, int NF, std::vector<Perturb> &pert, std::vector<Measure> &meas) {
    
    NNDOF = DIM * nw.NN;
    NADOF = nw.enable_affine ? DIM*DIM : 0;
    
    NDOF = NNDOF + NADOF;
    NFGDOF = nw.NGDOF;
        
    this->nw = nw;
    this->NF = NF;
    this->pert = pert;
    this->meas = meas;
    
    K = XVec::Ones(nw.NE);
    
    amp = 1.0;
    
    NM.resize(NF);
    for(int t = 0; t < NF; t++) {
        NM[t] = meas[t].NOstrain + meas[t].NOstress;
    }
    
    setupGlobalConMat();
}

void NonlinSolver::setIntStrengths(std::vector<double> &K) {
    vectorToEigen(K, this->K);
}

void NonlinSolver::setAmplitude(double amp) {
    this->amp = amp;
}

void NonlinSolver::solveDOF(std::vector<std::vector<double> > &disp, std::vector<std::vector<double> > &strain_tensor) {
    std::vector<std::vector<double> > u;
    
    solveAll(u);
    
    disp.resize(NF);
    strain_tensor.resize(NF);
    for(int t = 0; t < NF; t++) {
        disp[t].reserve(NNDOF);
        disp[t].assign(u[t].begin(), u[t].begin()+NNDOF);
        
        strain_tensor[t].reserve(NADOF);
        strain_tensor[t].assign(u[t].begin()+NNDOF, u[t].begin()+NNDOF+NADOF);
    }
    
    
}

void NonlinSolver::solveAll(std::vector<std::vector<double> > &u) {
    
    if((int)(u.size()) < NF) {
        u.resize(NF);
    }
    
    for(int t = 0; t < NF; t++) {
        
        int NC = 1;

        XVec u0_vec = XVec::Zero(NDOF);
        if((int)(u[t].size()) == NDOF) {
            
            XVecMap vmap(u[t].data(), NNDOF);

            u0_vec.segment(0, NNDOF) = vmap;


            if(nw.enable_affine) {
                XVecMap mmap(u[t].data()+NNDOF, NADOF);
                u0_vec.segment(NNDOF, NNDOF+NADOF) = mmap;
            }
        }

        alglib::real_1d_array u0;
        u0.setcontent(NDOF, u0_vec.data());

        // Algorithm state
        alglib::minnlcstate state;
        // Create algorithm state
        alglib::minnlccreate(NDOF, u0, state);


        // Penalty parameter
        double rho = 1e2;
        // Number of iterations to update lagrange multiplier
        alglib::ae_int_t outerits = 10;
        // Set augmented lagrangian parameters
        alglib::minnlcsetalgoaul(state, rho, outerits);


        // Tolerance on gradient, change in function and change in x
        double epsg = 10*MEPS;
        double epsf = 10*MEPS;
        double epsx = 10*MEPS;
        // Max iterations
        alglib::ae_int_t maxits = 0;
        // Set stopping conditions
        alglib::minnlcsetcond(state, epsg, epsf, epsx, maxits);
        
        // Scale of variables
        XVec s_vec = XVec::Ones(NDOF);
        alglib::real_1d_array s;
        s.setcontent(NDOF, s_vec.data());
       
        // Set scale of variables
        alglib::minnlcsetscale(state, s);

        // Set number of nonlinear equality and inequality constraints
        alglib::minnlcsetnlc(state, NC, 0);

        // alglib::minnlcsetgradientcheck(state, 1e-2);
        
        NonlinOptParams params;
        params.t = t;
        params.solver = this;
        
        // Perform optimization
        alglib::minnlcoptimize(state, NonlinSolver::FuncJacWrapper, NULL, &params);
        
        // Result 
        alglib::real_1d_array u_fin;
        // Optimization report
        alglib::minnlcreport rep;
        // Retrieve optimization results
        alglib::minnlcresults(state, u_fin, rep);
        
        u[t].assign(u_fin.getcontent(), u_fin.getcontent()+u_fin.length());
        
        std::cout << "Termination Type: ";

        switch(rep.terminationtype) {
            case -8:
                std::cout << "Internal integrity control detected  infinite  or  NAN  values  in function/gradient. Abnormal termination signalled." << std::endl;
                break;
            case -7:
                std::cout << "Gradient verification failed." << std::endl;
                break;
            case 1:
                std::cout << "Relative function improvement is no more than EpsF." << std::endl;
                break;
            case 2:
                std::cout << "Relative step is no more than EpsX." << std::endl;
                break;
            case 4:
                std::cout << "Gradient norm is no more than EpsG." << std::endl;
                break;
            case 5:
                std::cout << "MaxIts steps was taken." << std::endl;
                break;
            case 7:
                std::cout << "Stopping conditions are too stringent, further improvement is impossible, X contains best point found so far." << std::endl;
                break;
            default:
                std::cout << "Unkown error code: " << rep.terminationtype << std::endl;
        }
    }
}

void NonlinSolver::FuncJacWrapper(const alglib::real_1d_array &x, alglib::real_1d_array &func, 
                                   alglib::real_2d_array &jac, void *ptr) {
    
    
    NonlinOptParams *params = static_cast<NonlinOptParams*>(ptr);
	int t = params->t;
	NonlinSolver *solver = params->solver;
    
    std::vector<double> u;
    u.assign(x.getcontent(), x.getcontent()+x.length());
    double energy;
    solver->solveEnergy(t, u, energy);
    std::vector<double> grad;
    solver->solveGrad(t, u, grad);
    double con;
    solver->solveCon(t, u, con);
    std::vector<double> con_grad;
    solver->solveConGrad(t, u, con_grad);
    
    
    
    func[0] = energy;
    func[1] = con;
    
    // std::cout << energy << "\t" << con << std::endl;
    
    
    XVecMap grad_map(jac[0], solver->NDOF);
    XVecMap _grad_map(grad.data(), solver->NDOF);
    
    grad_map = _grad_map;
    
    XVecMap con_grad_map(jac[1], solver->NDOF);
    XVecMap _con_grad_map(con_grad.data(), solver->NDOF);
    
    con_grad_map = _con_grad_map;
}

void NonlinSolver::solveMeas(std::vector<std::vector<double> > &m) {
        
    
    
    std::vector<std::vector<double> > u;
    solveAll(u);
    
    m.resize(NF);
    for(int t = 0; t < NF; t++) {
        
        XVecMap vmap(u[t].data(), NNDOF);

        XVec disp = vmap;

        DMat strain_tensor;
        DMat def_tensor;

        if(nw.enable_affine) {
            DMatMap mmap(u[t].data()+NNDOF);
            strain_tensor = mmap; 

            def_tensor = DMat::Identity() + strain_tensor;
        }

        XVec _m = XVec::Zero(NM[t]);
        for(int c = 0; c < meas[t].NOstrain; c++) {
            DVec X0ij = meas[t].ostrain_vec.segment<DIM>(DIM*c);

            DVec Xij = X0ij + disp.segment<DIM>(DIM * meas[t].ostrain_nodesj(c)) - disp.segment<DIM>(DIM * meas[t].ostrain_nodesi(c));

            if(nw.enable_affine) {
                Xij = def_tensor*Xij;
            }

            double X0norm = X0ij.norm();
            double Xnorm = Xij.norm();

            double eps = Xnorm / X0norm - 1.0;

            _m(c) = eps;
        }

//         if(meas[t].measure_affine_strain) {

//             _m.segment(NNDOF, 
            
//             con += (strain_tensor - meas[t].strain_tensor).squaredNorm();

//         }
        
        eigenToVector(_m, m[t]);
    }
    
}

void NonlinSolver::solveEnergy(int t, std::vector<double> &u, double &energy) {
            
    XVecMap vmap(u.data(), NNDOF);

    XVec disp = vmap;

    DMat strain_tensor;
    DMat def_tensor;

    if(nw.enable_affine) {
        DMatMap mmap(u.data()+NNDOF);
        strain_tensor = mmap; 

        def_tensor = DMat::Identity() + strain_tensor;
    }

    double E = 0;
    for(int e = 0; e < nw.NE; e++) {

        DVec Xij = nw.bvecij.segment<DIM>(DIM*e) 
            + disp.segment<DIM>(DIM * nw.edgej(e)) - disp.segment<DIM>(DIM * nw.edgei(e));

        if(nw.enable_affine) {
            Xij = def_tensor*Xij;
        }

        double Xnorm = Xij.norm();

        double k = K(e);

        E += 0.5 * k * pow((Xnorm - nw.eq_length(e)), 2.0);
    }

    energy = E;
        
    
}

void NonlinSolver::solveGrad(int t, std::vector<double> &u, std::vector<double> &grad) {            
        
    XVecMap vmap(u.data(), NNDOF);

    XVec disp = vmap;

    DMat strain_tensor;
    DMat def_tensor;

    if(nw.enable_affine) {
        DMatMap mmap(u.data()+NNDOF);
        strain_tensor = mmap; 

        def_tensor = DMat::Identity() + strain_tensor;
    }

    XVec g = XVec::Zero(NDOF);
    for(int e = 0; e < nw.NE; e++) {

        DVec Xij = nw.bvecij.segment<DIM>(DIM*e) 
            + disp.segment<DIM>(DIM * nw.edgej(e)) - disp.segment<DIM>(DIM * nw.edgei(e));

        DVec XRef;
        if(nw.enable_affine) {
            XRef = Xij;
            Xij = def_tensor*Xij;
        }

        double Xnorm = Xij.norm();

        double k = K(e);

        DVec dEdX = k * (Xnorm - nw.eq_length(e)) * Xij / Xnorm;

        if(nw.enable_affine) {
            DVec dEdu = dEdX.transpose() * def_tensor;

            g.segment<DIM>(DIM * nw.edgej(e)) += dEdu;
            g.segment<DIM>(DIM * nw.edgei(e)) -= dEdu;
        } else {
            g.segment<DIM>(DIM * nw.edgej(e)) += dEdX;
            g.segment<DIM>(DIM * nw.edgei(e)) -= dEdX;
        }

        if(nw.enable_affine) {

            DMat dEdF = dEdX * XRef.transpose();

            for(int d = 0; d < DIM; d++) {
                g.segment<DIM>(NNDOF+DIM*d) += dEdF.row(d);
            }

        }
    }

    eigenToVector(g, grad);        
                
    
}

void NonlinSolver::solveCon(int t, std::vector<double> &u, double &con) {
    
    XVecMap vmap(u.data(), NNDOF);

    XVec disp = vmap;

    DMat strain_tensor;
    DMat def_tensor;

    if(nw.enable_affine) {
        DMatMap mmap(u.data()+NNDOF);
        strain_tensor = mmap; 

        def_tensor = DMat::Identity() + strain_tensor;
    }
    
    con = 0.0;
    for(int c = 0; c < pert[t].NIstrain; c++) {
        DVec X0ij = pert[t].istrain_vec.segment<DIM>(DIM*c);
        
        DVec Xij = X0ij + disp.segment<DIM>(DIM * pert[t].istrain_nodesj(c)) - disp.segment<DIM>(DIM * pert[t].istrain_nodesi(c));
                
        if(nw.enable_affine) {
            Xij = def_tensor*Xij;
        }
        
        double X0norm = X0ij.norm();
        double Xnorm = Xij.norm();
        
        double eps = Xnorm / X0norm - 1.0;
        
        con += pow(eps - amp * pert[t].istrain(c), 2.0);
    }
    
    if(pert[t].apply_affine_strain) {
        con += (strain_tensor - amp * pert[t].strain_tensor).squaredNorm();
    }

    con += (G.transpose() * disp).squaredNorm();
        
}

void NonlinSolver::solveConGrad(int t, std::vector<double> &u, std::vector<double> &grad) {
    
    XVecMap vmap(u.data(), NNDOF);

    XVec disp = vmap;

    DMat strain_tensor;
    DMat def_tensor;

    if(nw.enable_affine) {
        DMatMap mmap(u.data()+NNDOF);
        strain_tensor = mmap; 

        def_tensor = DMat::Identity() + strain_tensor;
    }
    
    XVec g = XVec::Zero(NDOF);
    
    for(int c = 0; c < pert[t].NIstrain; c++) {
        
        DVec X0ij = pert[t].istrain_vec.segment<DIM>(DIM*c);
        
        DVec Xij = X0ij + disp.segment<DIM>(DIM * pert[t].istrain_nodesj(c)) - disp.segment<DIM>(DIM * pert[t].istrain_nodesi(c));
                
        DVec XRef = DVec::Zero();
        if(pert[t].apply_affine_strain) {
            XRef = Xij;
            Xij = def_tensor*Xij;
        }
        
        double X0norm = X0ij.norm();
        double Xnorm = Xij.norm();
        
        double eps = Xnorm / X0norm - 1.0;
                
        DVec dCdX = 2.0 * (eps - amp * pert[t].istrain(c)) * (1.0 / X0norm) * (Xij / Xnorm);
        
        if(nw.enable_affine) {
            DVec dCdu = dCdX.transpose() * def_tensor;

            g.segment<DIM>(DIM * pert[t].istrain_nodesj(c)) += dCdu;
            g.segment<DIM>(DIM * pert[t].istrain_nodesi(c)) -= dCdu;
        } else {
            g.segment<DIM>(DIM * pert[t].istrain_nodesj(c)) += dCdX;
            g.segment<DIM>(DIM * pert[t].istrain_nodesi(c)) -= dCdX;
        }

        if(nw.enable_affine) {

            DMat dCdF = dCdX * XRef.transpose();

            for(int d = 0; d < DIM; d++) {
                g.segment<DIM>(NNDOF+DIM*d) += dCdF.row(d);
            }

        }
        
    }
    
    if(pert[t].apply_affine_strain) {
                
        DMat dCdF = 2 * (strain_tensor - amp * pert[t].strain_tensor);

        for(int d = 0; d < DIM; d++) {
            g.segment<DIM>(NNDOF+DIM*d) += dCdF.row(d);
        }
        
    }
    
    g.segment(0, NNDOF) += 2.0 * G * G.transpose() * disp;
        
    eigenToVector(g, grad);
    
}

void NonlinSolver::setupGlobalConMat() {
    
    G = XMat::Zero(NNDOF, NFGDOF);        

    // Fix translational dofs
    for(int d = 0; d < DIM; d++) {
        G.block(0, d, NNDOF, 1) = XVec::Constant(NNDOF, -1);
    }
    
    // Fix rotational dofs
    if(NFGDOF <= DIM) {
        return;
    }
                
    // center of mass
    DVec COM = DVec::Zero();
    for(int i = 0; i < nw.NN; i++) {
        COM += nw.node_pos.segment<DIM>(DIM*i);

    }
    COM /= nw.NN;


    XMat rot_axes = XMat::Zero(DIM*(DIM-1)/2 , 3);

    if(DIM == 2) {
        rot_axes << 0, 0, 1;
    } else if(DIM == 3) {
        rot_axes << 1, 0, 0,
                    0, 1, 0,
                    0, 0, 1;
    }

    for(int d = 0; d < DIM*(DIM-1)/2; d++) {
        XVec global_rot = XVec::Zero(NNDOF);

        for(int i = 0; i < nw.NN; i++) {
            Vec3d pos = Vec3d::Zero();
            if(DIM == 2) {
                pos.segment<2>(0) = nw.node_pos.segment<2>(2*i) - COM.segment<2>(0);
            } else if(DIM  == 3) {
                pos = nw.node_pos.segment<3>(3*i) - COM.segment<3>(0);
            }

            Vec3d axis = rot_axes.row(d);

            Vec3d rot_dir = axis.cross(pos);

            if(DIM == 2) {
                global_rot.segment<2>(2*i) = rot_dir.segment<2>(0);
            } else if(DIM  == 3) {
                global_rot.segment<3>(3*i) = rot_dir;
            }               
        }

        global_rot.normalize();

        for(int i = 0; i < NNDOF; i++) {
            G.block(0, DIM+d, NNDOF, 1) = -global_rot;
        }

    }
}

#include "lin_solver.hpp"
    
LinSolver::LinSolver(Network &nw, int NF, std::vector<Perturb> &pert, std::vector<Measure> &meas) {
        
    int p = 0;
    for(int m = 0; m < DIM; m++) {
        for(int n = m; n < DIM; n++) {
            if(m == n) {
                sm_index(m, n) = p;
            } else {
                sm_index(m, n) = p;
                sm_index(n, m) = p;
            }
            
            p++;
        }
    }
    
    NNDOF = DIM * nw.NN;
    NADOF = nw.enable_affine ? DIM*(DIM+1)/2 : 0;
    NFGDOF = nw.NGDOF;
    
    NDOF = NNDOF + NADOF + NFGDOF;
        
    this->nw = nw;
    this->NF = NF;
    this->pert = pert;
    this->meas = meas;
    
    K = XVec::Ones(nw.NE);
    
    need_H = true;
    need_Hinv = true;
    need_HinvPert = true;
    need_HinvM = true;
        
    setupEqMat();
    setupGlobalConMat();
    setupPertMat();
    setupMeasMat();
}

void LinSolver::setIntStrengths(std::vector<double> &K) {
    
    vectorToEigen(K, this->K);
    
    need_H = true;
    need_Hinv = true;
    need_HinvPert = true;
    need_HinvM = true;
}


void LinSolver::calcHessian() {
    
    if(!need_H) {
        return;
    }
    
    
    SMat I(NDOF, NDOF);
    std::vector<Trip > id_trip_list;
    for(int k = 0; k < NNDOF; k++) {
        // id_trip_list.push_back(Trip(k, k, 1e-6));
    }
    I.setFromTriplets(id_trip_list.begin(), id_trip_list.end());
    
    H = Q * K.asDiagonal() * Q.transpose() + G + I;
          
//     std::vector<double> evals;
//     getEigenvals(evals, false);
    
//     std::cout << evals[0] << "\t" << evals[1] << "\t" << evals[2] << "\t" << evals[3] << std::endl;
    
    // Perform LU decomposition
    solver.compute(H);
    if (solver.info() != Eigen::Success) {
        // decomposition failed
                        
        std::cout << "LU decomposition of Hessian failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    need_H = false;
}

void LinSolver::calcInvHessian() {
    
    if(!need_Hinv) {
        return;
    }
    
    calcHessian();
                
    // Perform LU decomposition
    Hinv = solver.solve(XMat(XMat::Identity(H.rows(), H.cols())));
    // Hinv = XMat(H).inverse();
            
    if (solver.info() != Eigen::Success) {
        std::cout << "Solving H^{-1} failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    need_Hinv = false;
}

void LinSolver::calcPert(bool use_full_inverse) {
    
    if(!need_HinvPert) {
        return;
    }
    
    if(use_full_inverse) {
        calcInvHessian();
    } else {
        calcHessian();
    }
    
    
    HinvC1.resize(NF);
    Hinvf.resize(NF);
    
    for(int t = 0; t < NF; t++) {
    
        //  H^{-1}C_1
        if(C1[t].nonZeros() > 0) {
            if(use_full_inverse) {
                HinvC1[t] = Hinv * C1[t];
            } else {
                HinvC1[t] = solver.solve(XMat(C1[t]));
                if (solver.info() != Eigen::Success) {
                    std::cout << "Solving H^{-1}C_1 failed." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
                
        }

        //  H^{-1}f
        if(f[t].nonZeros() > 0) {
            if(use_full_inverse) {
                Hinvf[t] = Hinv * f[t];
            } else {
                Hinvf[t] = solver.solve(XVec(f[t]));
                if (solver.info() != Eigen::Success) {
                    std::cout << "Solving H^{-1}f failed." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
    
    need_HinvPert = false;
    
}

void LinSolver::calcMeas(bool use_full_inverse) {
    
    if(!need_HinvM) {
        return;
    }
    
    if(use_full_inverse) {
        calcInvHessian();
    } else {
        calcHessian();
    }
    
    HinvM.resize(NF);
    
    for(int t = 0; t < NF; t++) {
    
        //  H^{-1}M
        if(M[t].nonZeros() > 0) {
            if(use_full_inverse) {
                HinvM[t] = Hinv * M[t];
            } else {
                HinvM[t] = solver.solve(XMat(M[t]));
                if (solver.info() != Eigen::Success) {
                    std::cout << "Solving H^{-1}M failed." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }  

    need_HinvM = false;
}

void LinSolver::isolveU(std::vector<XVec > &u) {
    
    SMat BH;
    
    BH.resize(NDOF+NC[0], NDOF+NC[0]);
    SMat H = Q * K.asDiagonal() * Q.transpose() + G;
    
    std::vector<Trip > BH_trip_list; 
    for (int k = 0; k < H.outerSize(); ++k) {
        for (SMat::InnerIterator it(H, k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    
    for (int k = 0; k < C1[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(C1[0], k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), NDOF + it.col(), -it.value()));
            BH_trip_list.push_back(Trip(NDOF + it.col(), it.row(), -it.value()));
        }
    }
    
    BH.setFromTriplets(BH_trip_list.begin(), BH_trip_list.end());
    
    Bsolver.compute(BH);
    
    
    std::vector<Trip > Bf_trip_list; 
    SMat Bf;
    Bf.resize(NDOF+NC[0], 1);
    for (int k = 0; k < f[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(f[0], k); it; ++it) {
            Bf_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    for(int k = 0; k < NC[0]; k++) {
        Bf_trip_list.push_back(Trip(NDOF+k, 0, C0[0](k)));
    }
    Bf.setFromTriplets(Bf_trip_list.begin(), Bf_trip_list.end());
    
    XMat BHinvf = Bsolver.solve(XMat(Bf));
    if (Bsolver.info() != Eigen::Success) {
        std::cout << "Solving BH^{-1}f failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    u.resize(NF);
    u[0] = BHinvf.block(0, 0, NDOF, 1);
   

}

// void LinSolver::isolveU(std::vector<XVec > &u) {
    
//     bool use_full_inverse = (NC_tot > NDOF);
    
//     calcPert(use_full_inverse);
   
//     u.resize(NF);
    
//     for(int t = 0; t < NF; t++) {
    
//         if(C1[t].nonZeros() > 0) {
//             XMat A = (C1[t].transpose() * HinvC1[t]).inverse();

//             if(f[t].nonZeros() > 0) {
//                 u[t] = Hinvf[t] - HinvC1[t] * A * (C1[t].transpose() * Hinvf[t] + C0[t]);
//             } else {
//                 u[t] = - HinvC1[t] * A * C0[t];
//             }
//         } else if(f[t].nonZeros() > 0) {
//             u[t] = Hinvf[t];
//         } else {
//             std::cout << "Func Error" << std::endl;
//             exit(EXIT_SUCCESS);
//         }            
//     }
// }

void LinSolver::isolveLambda(std::vector<XVec > &lambda) {
    
    bool use_full_inverse = (NC_tot > NDOF);
    
    calcPert(use_full_inverse);
   
    lambda.resize(NF);
    
    for(int t = 0; t < NF; t++) {
    
        if(C1[t].nonZeros() > 0) {
            XMat A = (C1[t].transpose() * HinvC1[t]).inverse();

            if(f[t].nonZeros() > 0) {
                lambda[t] = - A * (C1[t].transpose() * Hinvf[t] + C0[t]);
            } else {
                lambda[t] = - A * C0[t];
            }
        } else if(f[t].nonZeros() > 0) {
            lambda[t] = XVec::Zero(NC[t]);
        } else {
            std::cout << "Func Error" << std::endl;
            exit(EXIT_SUCCESS);
        }            
    }
}


void LinSolver::isolveM(XVec &meas) {
    std::vector<XVec > u;
    isolveU(u);
    
    meas = XVec::Zero(NM_tot);
    for(int t = 0; t < NF; t++) {
        
        meas.segment(meas_index[t], NM[t]) = M[t].transpose() * u[t];
        
    }
    
}

void LinSolver::isolveMGrad(XVec &meas, std::vector<XVec > &grad) {

    bool use_full_inverse = (NC_tot > NDOF) || (NM_tot > NDOF);
    
    calcPert(use_full_inverse);
    calcMeas(use_full_inverse);
    
    meas = XVec::Zero(NM_tot);
    grad.resize(NM_tot);
    
    for(int t = 0; t < NF; t++) {
        
        XVec u;
    
        XMat Hinv11M;
        if(C1[t].nonZeros() > 0) {
            XMat A = (C1[t].transpose() * HinvC1[t]).inverse();
            Hinv11M = HinvM[t] - HinvC1[t] * A * C1[t].transpose() * HinvM[t];

            if(f[t].nonZeros() > 0) {
                u = (Hinvf[t] - HinvC1[t] * A * (C1[t].transpose() * Hinvf[t] + C0[t]));
            } else {
                u = - HinvC1[t] * A * C0[t];
            }
        } else if(f[t].nonZeros() > 0) {
            Hinv11M = HinvM[t];
            u = Hinvf[t];

        } else {
            std::cout << "Grad Error" << std::endl;
            exit(EXIT_SUCCESS);
        }   
        
        meas.segment(meas_index[t], NM[t]) = M[t].transpose() * u;

        XMat grad_mat = XMat::Zero(NM[t], nw.NE);
        for(int b = 0; b < nw.NE; b++) {
            double qu = (Q.col(b).transpose() * u)(0);
            
            grad_mat.col(b) = - Hinv11M.transpose() * Q.col(b) * qu;
        }
            
        for(int im = 0; im < NM[t]; im++) {
            grad[meas_index[t] + im] = grad_mat.row(im);
        }      
        
        
    } 
    
}

void LinSolver::isolveMHess(XVec &meas, std::vector<XVec > &grad, std::vector<XMat > &hess) {
    bool use_full_inverse = true;
        
    calcInvHessian();
    calcPert(use_full_inverse);
    calcMeas(use_full_inverse);
        
    meas = XVec::Zero(NM_tot);
    grad.resize(NM_tot);
    hess.resize(NM_tot, XMat::Zero(nw.NE, nw.NE));
       
    
    for(int t = 0; t < NF; t++) {
        
        XVec u;
    
        XMat Hinv11;
        if(C1[t].nonZeros() > 0) {
            
            XMat A = (C1[t].transpose() * HinvC1[t]).inverse();
            Hinv11 = Hinv - HinvC1[t] * A * HinvC1[t].transpose();
                        
            if(f[t].nonZeros() > 0) {
                u = (Hinvf[t] - HinvC1[t] * A * (C1[t].transpose() * Hinvf[t] + C0[t]));
            } else {                
                u = - HinvC1[t] * A * C0[t];
            }
        } else if(f[t].nonZeros() > 0) {
            Hinv11 = Hinv;
            u = Hinvf[t];

        } else {
            std::cout << "Grad Error" << std::endl;
            exit(EXIT_SUCCESS);
        } 
                
        meas.segment(meas_index[t], NM[t]) = M[t].transpose() * u;

        XMat grad_mat = XMat::Zero(nw.NE, nw.NE);
        XMat mgrad_mat = XMat::Zero(NM[t], nw.NE);
        for(int b = 0; b < nw.NE; b++) {
            double qu = (Q.col(b).transpose() * u)(0);
            
            grad_mat.col(b) = -Hinv11 * Q.col(b) * qu;
            mgrad_mat.col(b) = M[t].transpose() * grad_mat.col(b);
        }
        
        for(int im = 0; im < NM[t]; im++) {
            grad[meas_index[t] + im] = mgrad_mat.row(im);
        }        
                
        XVec hess_el = XVec::Zero(NM[t]);
        XMat Hinv11M =  M[t].transpose() * Hinv11;
        for(int bi = 0; bi < nw.NE; bi++) {
            for(int bj = bi; bj < nw.NE; bj++) {
                
                double qidudkj = (Q.col(bi).transpose() * grad_mat.col(bj))(0);
                double qjdudki = (Q.col(bj).transpose() * grad_mat.col(bi))(0);
                
                hess_el = -Hinv11M * (Q.col(bi) * qidudkj + Q.col(bj) * qjdudki);
                
                for(int im = 0; im < NM[t]; im++) {
                    hess[meas_index[t] + im](bi, bj) += hess_el(im);
                    hess[meas_index[t] + im](bj, bi) += hess_el(im);
                    
                }
            }
        }
                    
    } 
}



void LinSolver::solveDOF(std::vector<std::vector<double> > &disp, std::vector<std::vector<double> > &strain_tensor) {
    std::vector<std::vector<double> > u;
    
    solveU(u);
    
    disp.resize(NF);
    strain_tensor.resize(NF);
    for(int t = 0; t < NF; t++) {
        disp[t].reserve(NNDOF);
        disp[t].assign(u[t].begin(), u[t].begin()+NNDOF);
        
        strain_tensor[t].reserve(NADOF);
        strain_tensor[t].assign(u[t].begin()+NNDOF, u[t].begin()+NNDOF+NADOF);
    }
    
    
}


void LinSolver::prepareUpdateList(std::vector<Update> &up_list) {
        
    this->up_list = up_list;
    
    dBHinv = XMat::Zero(NDOF+NC[0], NDOF+NC[0]);
    
    // Initialize bordered Hessian
    BH.resize(NDOF+NC[0], NDOF+NC[0]);
    SMat H = Q * K.asDiagonal() * Q.transpose() + G;
    
    std::vector<Trip > BH_trip_list; 
    for (int k = 0; k < H.outerSize(); ++k) {
        for (SMat::InnerIterator it(H, k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    
    for (int k = 0; k < C1[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(C1[0], k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), NDOF + it.col(), -it.value()));
            BH_trip_list.push_back(Trip(NDOF + it.col(), it.row(), -it.value()));
        }
    }
    
    for (int k = 0; k < NNDOF; k++) {
        // BH_trip_list.push_back(Trip(k, k, 1e-6));
    }
    
    BH.setFromTriplets(BH_trip_list.begin(), BH_trip_list.end());
    
    Bsolver.compute(BH);
    
    BHinv = Bsolver.solve(XMat(XMat::Identity(BH.rows(), BH.cols())));
    
    // Initialize perturbation
    std::vector<Trip > Bf_trip_list; 
    Bf.resize(NDOF+NC[0], 1);
    for (int k = 0; k < f[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(f[0], k); it; ++it) {
            Bf_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    for(int k = 0; k < NC[0]; k++) {
        Bf_trip_list.push_back(Trip(NDOF+k, 0, C0[0](k)));
    }
    Bf.setFromTriplets(Bf_trip_list.begin(), Bf_trip_list.end());
    
    BHinvf = Bsolver.solve(XMat(Bf));
    if (Bsolver.info() != Eigen::Success) {
        std::cout << "Solving BH^{-1}f failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    // Initialize measurements
    std::vector<Trip > BM_trip_list; 
    BM.resize(NDOF+NC[0], NM[0]);
    for (int k = 0; k < M[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(M[0], k); it; ++it) {
            BM_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    BM.setFromTriplets(BM_trip_list.begin(), BM_trip_list.end());
    
    BHinvM = Bsolver.solve(XMat(BM));
    if (Bsolver.info() != Eigen::Success) {
        std::cout << "Solving BH^{-1}M failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    XMat tmp = BH * BHinvM;
    tmp -= BM;
    double error = tmp.cwiseAbs().maxCoeff();
    std::cout << "Max Meas Error: " << error << std::endl;
    
    
    
    SM_updates.resize(up_list.size());
    
    for(std::vector<int>::size_type i = 0; i < up_list.size(); i++) {
                
        // Setup Hessian perturbation
        SMat U(NDOF, up_list[i].NSM); 

        // Changes in bond stretch moduli
        for(int s = 0; s < up_list[i].NSM; s++) {
            int b = up_list[i].sm_bonds(s);

            U.col(s) = Q.col(b);
        }

        //Calculate Hessian perturbation

        // H^{-1}U
        
        SMat BU(NDOF+NC[0], up_list[i].NSM);
        
        std::vector<Trip > BU_trip_list; 
        for (int k = 0; k < U.outerSize(); ++k) {
            for (SMat::InnerIterator it(U, k); it; ++it) {
                BU_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
            }
        }
        BU.setFromTriplets(BU_trip_list.begin(), BU_trip_list.end());
        
        XMat BHinvU = Bsolver.solve(XMat(BU)); 
        if (Bsolver.info() != Eigen::Success) {
            std::cout << "Solving BH^{-1}U failed." << std::endl;
        }
       
        SM_updates[i] = BU.transpose() * BHinvU;
    }
    
    /////////////////////////////////////
    
//     dHinv = XMat::Zero(NDOF, NDOF);
    
//     calcMeas(false);
//     calcPert(false);
    
//     SM_updates.resize(up_list.size());
    
//     for(std::vector<int>::size_type i = 0; i < up_list.size(); i++) {
//         // Setup Hessian perturbation
//         SMat U(NDOF, up_list[i].NSM); 

//         // Changes in bond stretch moduli
//         for(int s = 0; s < up_list[i].NSM; s++) {
//             int b = up_list[i].sm_bonds(s);

//             U.col(s) = Q.col(b);
//         }

//         //Calculate Hessian perturbation

//         // H^{-1}U
//         XMat HinvU;
//         HinvU = solver.solve(XMat(U)); 
//         if (solver.info() != Eigen::Success) {
//             std::cout << "Solving H^{-1}U failed." << std::endl;
//         }
       
//         SM_updates[i] = U.transpose() * HinvU;
//     }
        
}

double LinSolver::solveMeasUpdate(int i, std::vector<std::vector<double> > &meas) {

    // Setup Hessian perturbation
    SMat U(NDOF, up_list[i].NSM); 
    XVec dK(up_list[i].NSM);

    XVec K = this->K;
    
    // Changes in bond stretch moduli
    for(int s = 0; s < up_list[i].NSM; s++) {
        int b = up_list[i].sm_bonds(s);

        U.col(s) = Q.col(b);

        double k1 = K(b);
        double k2 = up_list[i].stretch_mod(s);

        dK(s) = k2 - k1;
        
        K(b) = k2;

    }
    
    
    SMat BU(NDOF+NC[0], up_list[i].NSM);
        
    std::vector<Trip > BU_trip_list; 
    for (int k = 0; k < U.outerSize(); ++k) {
        for (SMat::InnerIterator it(U, k); it; ++it) {
            BU_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    BU.setFromTriplets(BU_trip_list.begin(), BU_trip_list.end());

    XMat A = dK.asDiagonal() * SM_updates[i] + XMat::Identity(up_list[i].NSM, up_list[i].NSM);

    double det = fabs(A.determinant());
        
    if(det < 1e-4) {
        // std::cout << "Hessian update creates zero mode..." << std::endl;
        // std::cout << "|det(A)|: " << det << " < " << 1e-3 << std::endl;
        return -1.0;
    }
    
    meas.resize(NF);
    for(int t = 0; t < NF; t++ ) {

        XVec m = BM.transpose() * BHinvf - BHinvM.transpose() * BU * dK.asDiagonal() * A.inverse() * BU.transpose() * BHinvf;
                
        eigenToVector(m, meas[t]);
    }
    
    
    ///////////////////////////////
    
//     // Setup Hessian perturbation
//     SMat U(NDOF, up_list[i].NSM); 
//     XVec dK(up_list[i].NSM);

//     // Changes in bond stretch moduli
//     for(int s = 0; s < up_list[i].NSM; s++) {
//         int b = up_list[i].sm_bonds(s);

//         U.col(s) = Q.col(b);

//         double k1 = K(b);
//         double k2 = up_list[i].stretch_mod(s);

//         dK(s) = k2 - k1;

//     }

//     XMat A = dK.asDiagonal() * SM_updates[i] + XMat::Identity(up_list[i].NSM, up_list[i].NSM);

//     double det = fabs(A.determinant());
        
//     if(det < 1e-4) {
//         // std::cout << "Hessian update creates zero mode..." << std::endl;
//         // std::cout << "|det(A)|: " << det << " < " << 1e-10 << std::endl;
//         return -1;
//     }
    
//     meas.resize(NF);
//     for(int t = 0; t < NF; t++ ) {
//         // Calculate updated inverse Hessian terms
//         XMat delta = dK.asDiagonal() * A.inverse() * U.transpose() * HinvC1[t];

//         XMat MTHinvC1 = M[t].transpose() * HinvC1[t] - HinvM[t].transpose() * U * delta; 
//         XMat C1THinvC1 = C1[t].transpose() * HinvC1[t] - HinvC1[t].transpose() * U * delta; 


//         XVec m = -MTHinvC1 * C1THinvC1.inverse() * C0[t];
        
//         eigenToVector(m, meas[t]);
//     }
    
    
    
   /////////////////////////////////////////////// 
    
    
    
//     SMat H = Q * K.asDiagonal() * Q.transpose() + G;
        
//     std::vector<Trip > BH_trip_list; 
//     for (int k = 0; k < H.outerSize(); ++k) {
//         for (SMat::InnerIterator it(H, k); it; ++it) {
//             BH_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
//         }
//     }
    
//     for (int k = 0; k < C1[0].outerSize(); ++k) {
//         for (SMat::InnerIterator it(C1[0], k); it; ++it) {
//             BH_trip_list.push_back(Trip(it.row(), NDOF + it.col(), -it.value()));
//             BH_trip_list.push_back(Trip(NDOF + it.col(), it.row(), -it.value()));
//         }
//     }
    
        
//     for (int k = 0; k < NNDOF; k++) {
//         // BH_trip_list.push_back(Trip(k, k, 1e-6));
//     }
    
//     SMat BH;
//     BH.resize(NDOF + NC[0], NDOF + NC[0]);
//     BH.setFromTriplets(BH_trip_list.begin(), BH_trip_list.end());
    
    
//     XMat tmp = BH * (BHinvf - ((BHinv+dBHinv) * BU) * dK.asDiagonal() * A.inverse() * (BU.transpose() * BHinvf));
                
//     tmp -= Bf;
//     double error = tmp.cwiseAbs().maxCoeff() / XMat(Bf).cwiseAbs().maxCoeff();
//     // std::cout << "Max Response Error: " << error << std::endl;

//     XMat BHinvU = BHinv * BU;
        
//     double condition =  XMat(BH).cwiseAbs().rowwise().sum().maxCoeff() * (BHinv + dBHinv - BHinvU * dK.asDiagonal() * A.inverse() * BHinvU.transpose()).cwiseAbs().rowwise().sum().maxCoeff();
    
    double condition = 1.0;
    
    return condition;
    
    
}

void LinSolver::replaceUpdates(std::vector<int> &replace_index, std::vector<Update > &replace_update) {

    for(int i = 0; i < int(replace_index.size()); i++) {
        up_list[replace_index[i]] = replace_update[i];
    }
    
}

double LinSolver::setUpdate(int i, std::vector<std::vector<double> > &meas) {
        
    // Setup Hessian perturbation
    SMat U(NDOF, up_list[i].NSM); 
    XVec dK(up_list[i].NSM);
    
    // Changes in bond stretch moduli
    for(int s = 0; s < up_list[i].NSM; s++) {
        int b = up_list[i].sm_bonds(s);

        U.col(s) = Q.col(b);

        double k1 = K(b);
        double k2 = up_list[i].stretch_mod(s);

        dK(s) = k2 - k1;
        
        // std::cout << b << "\t" << k1 << "\t" << k2 << std::endl;

    }

    // Need to somehow update the matrix used to Bsolver here
    // A memory inefficient way is to store dHinv
    //Don't necessarily need to recalculate HinvU if you just calculate it in the beginning
    // H^{-1}U
    SMat BU(NDOF+NC[0], up_list[i].NSM);
        
    std::vector<Trip > BU_trip_list; 
    for (int k = 0; k < U.outerSize(); ++k) {
        for (SMat::InnerIterator it(U, k); it; ++it) {
            BU_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    BU.setFromTriplets(BU_trip_list.begin(), BU_trip_list.end());
    
    XMat BHinvU = Bsolver.solve(XMat(BU)); 
    if (Bsolver.info() != Eigen::Success) {
        std::cout << "Solving BH^{-1}U failed." << std::endl;
    }
        
    BHinvU += dBHinv * BU;
    
        
    XMat A = dK.asDiagonal() * SM_updates[i] + XMat::Identity(up_list[i].NSM, up_list[i].NSM);

    double det = fabs(A.determinant());
    std::cout << "|det(A)|: " << det << std::endl;
    if(det < 1e-4) {
        // std::cout << "Hessian update creates zero mode..." << std::endl;
        // std::cout << "|det(A)|: " << det << " < " << 1e-10 << std::endl;
        return false;
    }
    
    // Calculate updated inverse Hessian terms
    XMat delta = -BHinvU * dK.asDiagonal() * A.inverse() * BHinvU.transpose();
    dBHinv += delta;
    
    std::cout << "Delta Max: " << delta.cwiseAbs().maxCoeff() << std::endl;

    for(int t = 0; t < NF; t++) {
        BHinvf += delta * Bf;
        BHinvM += delta * BM;
    }

    // Calculate updated Sherman-Morrison updates
    for(std::vector<int>::size_type j = 0; j < up_list.size(); j++) {

        // Setup Hessian perturbation
        SMat V(NDOF, up_list[j].NSM); 

        // Changes in bond stretch moduli
        for(int s = 0; s < up_list[j].NSM; s++) {
            int b = up_list[j].sm_bonds(s);

            V.col(s) = Q.col(b);
        }
        
        SMat BV(NDOF+NC[0], up_list[j].NSM);
        
        std::vector<Trip > BV_trip_list; 
        for (int k = 0; k < V.outerSize(); ++k) {
            for (SMat::InnerIterator it(V, k); it; ++it) {
                BV_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
            }
        }
        BV.setFromTriplets(BV_trip_list.begin(), BV_trip_list.end());

        //Calculate Hessian perturbation

        SM_updates[j] += BV.transpose() * delta * BV;   
        
    }

    // Update interactions
    for(int s = 0; s < up_list[i].NSM; s++) {
        int b = up_list[i].sm_bonds(s);
        K(b) = up_list[i].stretch_mod(s);
    }
    
    SMat H = Q * K.asDiagonal() * Q.transpose() + G;
        
    std::vector<Trip > BH_trip_list; 
    for (int k = 0; k < H.outerSize(); ++k) {
        for (SMat::InnerIterator it(H, k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    
    for (int k = 0; k < C1[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(C1[0], k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), NDOF + it.col(), -it.value()));
            BH_trip_list.push_back(Trip(NDOF + it.col(), it.row(), -it.value()));
        }
    }
    
        
    for (int k = 0; k < NNDOF; k++) {
        // BH_trip_list.push_back(Trip(k, k, 1e-6));
    }
    
    SMat BH;
    BH.resize(NDOF + NC[0], NDOF + NC[0]);
    BH.setFromTriplets(BH_trip_list.begin(), BH_trip_list.end());
         
    double error = ((BHinv + dBHinv) * BH - XMat::Identity(BH.rows(), BH.cols())).cwiseAbs().maxCoeff();
    std::cout << "Max Bordered Hessian Error: " << error << std::endl;
    
    XMat tmp = BH * BHinvf;
    tmp -= Bf;
    error = tmp.cwiseAbs().maxCoeff();
    std::cout << "Max Force Error: " << error << std::endl;
    
    tmp = BH * BHinvM;
    tmp -= BM;
    error = tmp.cwiseAbs().maxCoeff() / XMat(BM).maxCoeff();
    std::cout << "Max Meas Error: " << error << std::endl;
    
    double condition =  XMat(BH).cwiseAbs().rowwise().sum().maxCoeff() * (BHinv + dBHinv).cwiseAbs().rowwise().sum().maxCoeff();
    
    std::cout << "Condition Number: " << condition << std::endl;
    
    // if(error > 0.0) {
    //     std::cout << "Resetting..." << std::endl;
    //     prepareUpdateList(up_list);
    // }
    
    
    meas.resize(NF);
    for(int t = 0; t < NF; t++ ) {

        XVec m = BM.transpose() * BHinvf;
                
        eigenToVector(m, meas[t]);
    }
    
    
    
    
    
//     // Setup Hessian perturbation
//     SMat U(NDOF, up_list[i].NSM); 
//     XVec dK(up_list[i].NSM);
    
//     // Changes in bond stretch moduli
//     for(int s = 0; s < up_list[i].NSM; s++) {
//         int b = up_list[i].sm_bonds(s);

//         U.col(s) = Q.col(b);

//         double k1 = K(b);
//         double k2 = up_list[i].stretch_mod(s);

//         dK(s) = k2 - k1;
        
//         // std::cout << b << "\t" << k1 << "\t" << k2 << std::endl;

//     }

//     // Need to somehow update the matrix used to solver here
//     // A memory inefficient way is to store dHinv
//     //Don't necessarily need to recalculate HinvU if you just calculate it in the beginning
//     // H^{-1}U
//     XMat HinvU;
//     HinvU = solver.solve(XMat(U)); 
//     if (solver.info() != Eigen::Success) {
//         std::cout << "Solving H^{-1}U failed." << std::endl;
//         return false;
//     }
    
//     HinvU += dHinv * U;
    
//     XMat A = dK.asDiagonal() * SM_updates[i] + XMat::Identity(up_list[i].NSM, up_list[i].NSM);
    
//     double det = fabs(A.determinant());
//     // std::cout << "|det(A)|: " << det << std::endl;
//     if(det < SQRTMEPS) {
//         // std::cout << "Hessian update creates zero mode..." << std::endl;
//         // std::cout << "|det(A)|: " << det << " < " << 1e-10 << std::endl;
//         return false;
//     }
    
//     // Calculate updated inverse Hessian terms
//     XMat delta = -HinvU * dK.asDiagonal() * A.inverse() * HinvU.transpose();
//     dHinv += delta;
    
//     for(int t = 0; t < NF; t++) {
//         HinvC1[t] += delta * C1[t];
//         HinvM[t] += delta * M[t];
//     }

//     // Calculate updated Sherman-Morrison updates
//     for(std::vector<int>::size_type j = 0; j < up_list.size(); j++) {
//         // Setup Hessian perturbation
//         SMat V(NDOF, up_list[j].NSM); 

//         // Changes in bond stretch moduli
//         for(int s = 0; s < up_list[j].NSM; s++) {
//             int b = up_list[j].sm_bonds(s);

//             V.col(s) = Q.col(b);
//         }

//         //Calculate Hessian perturbation

//         SM_updates[j] += V.transpose() * delta * V;
        
//     }
    
//     // Update interactions
//     for(int s = 0; s < up_list[i].NSM; s++) {
//         int b = up_list[i].sm_bonds(s);
//         K(b) = up_list[i].stretch_mod(s);
//     }
    
//     SMat H = Q * K.asDiagonal() * Q.transpose() + G;
    
//     calcInvHessian();
         
//     double error = ((Hinv + dHinv) * H - XMat::Identity(H.rows(), H.cols())).cwiseAbs().maxCoeff();
//     std::cout << "Max Hessian Error: " << error << std::endl;
    
//     XMat tmp = H * Hinvf;
//     tmp -= f[0];
//     error = tmp.cwiseAbs().maxCoeff();
//     std::cout << "Max Force Error: " << error << std::endl;
    
//     tmp = H * HinvM;
//     tmp -= M[0];
//     error = tmp.cwiseAbs().maxCoeff() / XMat(BM).maxCoeff();
//     std::cout << "Max Meas Error: " << error << std::endl;
    
//     double condition =  XMat(H).cwiseAbs().rowwise().sum().maxCoeff() * (Hinv + dHinv).cwiseAbs().rowwise().sum().maxCoeff();
    
//     std::cout << "Condition Number: " << condition << std::endl;
    
//     if(error > 1e-8) {
//         std::cout << "Resetting..." << std::endl;
//         prepareUpdateList(up_list);
//     }
    
    
//     meas.resize(NF);
//     for(int t = 0; t < NF; t++ ) {
//         // Calculate updated inverse Hessian terms
//         XMat delta = dK.asDiagonal() * A.inverse() * U.transpose() * HinvC1[t];

//         XMat MTHinvC1 = M[t].transpose() * HinvC1[t] - HinvM[t].transpose() * U * delta; 
//         XMat C1THinvC1 = C1[t].transpose() * HinvC1[t] - HinvC1[t].transpose() * U * delta; 


//         XVec m = -MTHinvC1 * C1THinvC1.inverse() * C0[t];
        
//         eigenToVector(m, meas[t]);
//     }
    
    
    
    return error;
    
}


// void LinSolver::prepareUpdateList(std::vector<Update> &up_list) {
        
//     this->up_list = up_list;
    
//     dHinv = XMat::Zero(NDOF, NDOF);
    
//     calcMeas(false);
//     calcPert(false);
    
//     SM_updates.resize(up_list.size());
    
//     for(std::vector<int>::size_type i = 0; i < up_list.size(); i++) {
//         // Setup Hessian perturbation
//         SMat U(NDOF, up_list[i].NSM); 

//         // Changes in bond stretch moduli
//         for(int s = 0; s < up_list[i].NSM; s++) {
//             int b = up_list[i].sm_bonds(s);

//             U.col(s) = Q.col(b);
//         }

//         //Calculate Hessian perturbation

//         // H^{-1}U
//         XMat HinvU;
//         HinvU = solver.solve(XMat(U)); 
//         if (solver.info() != Eigen::Success) {
//             std::cout << "Solving H^{-1}U failed." << std::endl;
//         }
       
//         SM_updates[i] = U.transpose() * HinvU;
//     }
        
// }

// bool LinSolver::solveMeasUpdate(int i, std::vector<std::vector<double> > &meas) {

//     // Setup Hessian perturbation
//     SMat U(NDOF, up_list[i].NSM); 
//     XVec dK(up_list[i].NSM);

//     // Changes in bond stretch moduli
//     for(int s = 0; s < up_list[i].NSM; s++) {
//         int b = up_list[i].sm_bonds(s);

//         U.col(s) = Q.col(b);

//         double k1 = K(b);
//         double k2 = up_list[i].stretch_mod(s);

//         dK(s) = k2 - k1;

//     }

//     XMat A = dK.asDiagonal() * SM_updates[i] + XMat::Identity(up_list[i].NSM, up_list[i].NSM);

//     double det = fabs(A.determinant());
        
//     if(det < 1e-6) {
//         // std::cout << "Hessian update creates zero mode..." << std::endl;
//         // std::cout << "|det(A)|: " << det << " < " << 1e-10 << std::endl;
//         return false;
//     }
    
//     meas.resize(NF);
//     for(int t = 0; t < NF; t++ ) {
//         // Calculate updated inverse Hessian terms
//         XMat delta = dK.asDiagonal() * A.inverse() * U.transpose() * HinvC1[t];

//         XMat MTHinvC1 = M[t].transpose() * HinvC1[t] - HinvM[t].transpose() * U * delta; 
//         XMat C1THinvC1 = C1[t].transpose() * HinvC1[t] - HinvC1[t].transpose() * U * delta; 


//         XVec m = -MTHinvC1 * C1THinvC1.inverse() * C0[t];
        
//         eigenToVector(m, meas[t]);
//     }

//     return true;
    
    
// }

// void LinSolver::replaceUpdates(std::vector<int> &replace_index, std::vector<Update > &replace_update) {

//     for(int i = 0; i < int(replace_index.size()); i++) {
//         up_list[replace_index[i]] = replace_update[i];
//     }
    
// }

// bool LinSolver::setUpdate(int i) {
    
//     // Setup Hessian perturbation
//     SMat U(NDOF, up_list[i].NSM); 
//     XVec dK(up_list[i].NSM);
    
//     // Changes in bond stretch moduli
//     for(int s = 0; s < up_list[i].NSM; s++) {
//         int b = up_list[i].sm_bonds(s);

//         U.col(s) = Q.col(b);

//         double k1 = K(b);
//         double k2 = up_list[i].stretch_mod(s);

//         dK(s) = k2 - k1;
        
//         // std::cout << b << "\t" << k1 << "\t" << k2 << std::endl;

//     }

//     // Need to somehow update the matrix used to solver here
//     // A memory inefficient way is to store dHinv
//     //Don't necessarily need to recalculate HinvU if you just calculate it in the beginning
//     // H^{-1}U
//     XMat HinvU;
//     HinvU = solver.solve(XMat(U)); 
//     if (solver.info() != Eigen::Success) {
//         std::cout << "Solving H^{-1}U failed." << std::endl;
//         return false;
//     }
    
//     HinvU += dHinv * U;
    
//     XMat A = dK.asDiagonal() * SM_updates[i] + XMat::Identity(up_list[i].NSM, up_list[i].NSM);
    
//     double det = fabs(A.determinant());
//     // std::cout << "|det(A)|: " << det << std::endl;
//     if(det < 1e-4) {
//         // std::cout << "Hessian update creates zero mode..." << std::endl;
//         // std::cout << "|det(A)|: " << det << " < " << 1e-10 << std::endl;
//         return false;
//     }
    
//     // Calculate updated inverse Hessian terms
//     XMat delta = -HinvU * dK.asDiagonal() * A.inverse() * HinvU.transpose();
//     dHinv += delta;
    
//     for(int t = 0; t < NF; t++) {
//         HinvC1[t] += delta * C1[t];
//         HinvM[t] += delta * M[t];
//     }

//     // Calculate updated Sherman-Morrison updates
//     for(std::vector<int>::size_type j = 0; j < up_list.size(); j++) {
//         // Setup Hessian perturbation
//         SMat V(NDOF, up_list[j].NSM); 

//         // Changes in bond stretch moduli
//         for(int s = 0; s < up_list[j].NSM; s++) {
//             int b = up_list[j].sm_bonds(s);

//             V.col(s) = Q.col(b);
//         }

//         //Calculate Hessian perturbation

//         SM_updates[j] += V.transpose() * delta * V;
        
//     }
    
//     // Update interactions
//     for(int s = 0; s < up_list[i].NSM; s++) {
//         int b = up_list[i].sm_bonds(s);
//         K(b) = up_list[i].stretch_mod(s);
//     }
    
//     SMat H = Q * K.asDiagonal() * Q.transpose() + G;
    
//     calcInvHessian();
    
//     std::cout << "Max Error: " << ((Hinv + dHinv) * H - XMat::Identity(H.rows(), H.cols())).maxCoeff() << std::endl;
    
//     return true;
    
// }


double LinSolver::getConditionNum() {
    
    SMat H = Q * K.asDiagonal() * Q.transpose() + G;
        
    std::vector<Trip > BH_trip_list; 
    for (int k = 0; k < H.outerSize(); ++k) {
        for (SMat::InnerIterator it(H, k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    
    for (int k = 0; k < C1[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(C1[0], k); it; ++it) {
            BH_trip_list.push_back(Trip(it.row(), NDOF + it.col(), -it.value()));
            BH_trip_list.push_back(Trip(NDOF + it.col(), it.row(), -it.value()));
        }
    }
    
        
    // for (int k = 0; k < NNDOF; k++) {
    //     // BH_trip_list.push_back(Trip(k, k, 1e-6));
    // }
    
    SMat BH;
    BH.resize(NDOF + NC[0], NDOF + NC[0]);
    BH.setFromTriplets(BH_trip_list.begin(), BH_trip_list.end());
    
    Bsolver.compute(BH);
    
    BHinv = Bsolver.solve(XMat(XMat::Identity(BH.rows(), BH.cols())));
    
    double condition =  XMat(BH).cwiseAbs().rowwise().sum().maxCoeff() * BHinv.cwiseAbs().rowwise().sum().maxCoeff();
    
    return condition;
}

void LinSolver::setupEqMat() {
          

    std::vector<Trip> Q_trip_list;
    // Harmonic spring interactions
    for(int e = 0; e < nw.NE; e++) {
                
        int ei =  nw.edgei(e);
        int ej =  nw.edgej(e);
        
        DVec Xij =  nw.bvecij.segment<DIM>(DIM*e);
        DVec Xhatij = Xij.normalized();
        
        for(int m = 0; m < DIM; m++) {
            Q_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m)));
            Q_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m)));
        }
        
        if(nw.enable_affine) {
            
            SMVec XhatX = SMVec::Zero();
            for(int m = 0; m < DIM; m++) {
                for(int n = 0; n < DIM; n++) {
                        XhatX(sm_index(m, n)) += Xhatij(m)*Xij(n);
                }
            }
            
            for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                Q_trip_list.push_back(Trip(NNDOF+m, e, XhatX(m)));
            }

        }
    } 
    
    Q.resize(NDOF, nw.NE);
    Q.setFromTriplets(Q_trip_list.begin(), Q_trip_list.end());
    Q.prune(0.0);
    
}

void LinSolver::setupGlobalConMat() {
    
    G.resize(NDOF, NDOF);        
    std::vector<Trip> G_trip_list;

    // Fix translational dofs
    for(int d = 0; d < DIM; d++) {
       for(int i = d; i < NNDOF; i+=DIM) {               
           G_trip_list.push_back(Trip(i, NDOF-NFGDOF+d, -1.0));
           G_trip_list.push_back(Trip(NDOF-NFGDOF+d, i, -1.0));
       }
    }
    
    // Fix rotational dofs
    if(NFGDOF > DIM) {
                
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
                G_trip_list.push_back(Trip(i, NDOF-NFGDOF+DIM+d, -global_rot(i)));
                G_trip_list.push_back(Trip(NDOF-NFGDOF+DIM+d, i, -global_rot(i)));
            }
            
        }
        
    }

    G.setFromTriplets(G_trip_list.begin(), G_trip_list.end());
}

void LinSolver::setupPertMat() {
        
    NC.resize(NF);
    C0.resize(NF);
    C1.resize(NF);
    f.resize(NF);
    NC_tot = 0;
    
    for(int t = 0; t < NF; t++) {
        NC[t] = pert[t].NIstrain + DIM*pert[t].NFix;
        if(pert[t].apply_affine_strain) {
            NC[t] += DIM*(DIM+1)/2;
        }
        NC_tot += NC[t];
    
        // Apply input strains and affine deformations as constraints in an bordered Hessian

        C0[t] = XVec::Zero(NC[t]);
        C1[t].resize(NDOF, NC[t]);
        std::vector<Trip> C1_trip_list;
        // Input strain constraints
        for(int e = 0; e < pert[t].NIstrain; e++) {
            int ei =  pert[t].istrain_nodesi(e);
            int ej =  pert[t].istrain_nodesj(e);

            DVec Xij = pert[t].istrain_vec.segment<DIM>(DIM*e);
            double l0 = Xij.norm();
            DVec Xhatij = Xij.normalized();

            C0[t](e) = -pert[t].istrain(e);

            for(int m = 0; m<DIM; m++) {
                // C1_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m) / l0));
                // C1_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m) / l0));
                
                C1_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m)));
                C1_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m)));
            }


            if(nw.enable_affine) {

                SMVec XhatXhat = SMVec::Zero();
                for(int m = 0; m < DIM; m++) {
                    for(int n = 0; n < DIM; n++) {
                            XhatXhat(sm_index(m, n)) += Xhatij(m)*Xhatij(n);
                    }
                }

                for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                    C1_trip_list.push_back(Trip(NNDOF+m, e, XhatXhat(m)));
                }
            }
        }
        
        // Affine deformation constraints
        if(pert[t].apply_affine_strain) {
            SMVec Gamma = SMVec::Zero();

            // Does there need to be a 1/2 for nondiagonal terms?
            for(int m = 0; m < DIM; m++) {
                for(int n = m; n < DIM; n++) {
                    Gamma(sm_index(m, n)) = pert[t].strain_tensor(m, n);                    
                }
            }

            C0[t].segment<DIM*(DIM+1)/2>(pert[t].NIstrain) = -Gamma;


            for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                C1_trip_list.push_back(Trip(NNDOF+m, pert[t].NIstrain+m, 1.0));
            }             
        }
        
        
        
        // Fixed nodes
        for(int n = 0; n < pert[t].NFix; n++) {
            int offset = pert[t].NIstrain;
            if(pert[t].apply_affine_strain) {
                offset += DIM*(DIM+1)/2;
            }
            
            int ni = pert[t].fixed_nodes[n];
            
            for(int d = 0; d < DIM; d++) {
                C1_trip_list.push_back(Trip(DIM*ni+d, offset+DIM*n+d, 1.0));
            }
        }

        C1[t].setFromTriplets(C1_trip_list.begin(), C1_trip_list.end());


        //Apply local input stress and affine input stress as forces
        f[t].resize(NDOF, 1);

        for(int e = 0; e < pert[t].NIstress; e++) {


            int ei = pert[t].istress_bonds(e);      

            f[t] += Q.col(ei) * pert[t].istress(e) / nw.eq_length(ei);
        }

        if(pert[t].apply_affine_stress) {

            for(int m = 0; m < DIM; m++) {
                for(int n = 0; n < DIM; n++) {
                    f[t].coeffRef(NNDOF+sm_index(m, n), 0) += pert[t].stress_tensor(m, n);
                }
            }

        }
    }

}

void LinSolver::setupMeasMat() {

    NM.resize(NF);
    M.resize(NF);
    NM_tot = 0;
    
    for(int t = 0; t < NF; t++) {
    
        NM[t] = meas[t].NOstrain + meas[t].NOstress;
        NM_tot += NM[t];
        int NMA = 0;
        if(meas[t].measure_affine_strain) {
            NMA = DIM*(DIM+1)/2;
            NM[t] += NMA;
        }

        M[t].resize(NDOF, NM[t]);
        std::vector<Trip> M_trip_list;

        // Output strain responses
        for(int e = 0; e < meas[t].NOstrain; e++) {
            int ei =  meas[t].ostrain_nodesi(e);
            int ej =  meas[t].ostrain_nodesj(e);

            DVec Xij = meas[t].ostrain_vec.segment<DIM>(DIM*e);
            double l0 = Xij.norm();
            DVec Xhatij = Xij.normalized();

            for(int m = 0; m<DIM; m++) {
                M_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m)));
                M_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m)));
                
                // M_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m) / l0));
                // M_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m) / l0));
            }


            if(nw.enable_affine) {

                SMVec XhatXhat = SMVec::Zero();
                for(int m = 0; m < DIM; m++) {
                    for(int n = 0; n < DIM; n++) {
                        XhatXhat(sm_index(m, n)) += Xhatij(m)*Xhatij(n);
                    }
                }

                for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                    M_trip_list.push_back(Trip(NNDOF+m, e, XhatXhat(m)));
                }
            }
        }
        // Affine deformation responses
        if(meas[t].measure_affine_strain) {           
            for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                M_trip_list.push_back(Trip(NNDOF+m, meas[t].NOstrain+m, 1.0));
            }             
        }


        // Output stress responses
        for(int e = 0; e < meas[t].NOstress; e++) {
            int ei =  nw.edgei(meas[t].ostress_bonds(e));
            int ej =  nw.edgej(meas[t].ostress_bonds(e));

            DVec Xij = nw.node_pos.segment<DIM>(DIM*ej)-nw.node_pos.segment<DIM>(DIM*ei);
            DVec Xhatij = Xij.normalized();

            for(int m = 0; m<DIM; m++) {
                M_trip_list.push_back(Trip(DIM*ei+m, meas[t].NOstrain + NMA + e, -Xhatij(m)));
                M_trip_list.push_back(Trip(DIM*ej+m, meas[t].NOstrain + NMA + e, Xhatij(m)));
            }


            if(nw.enable_affine) {

                SMVec XhatXhat = SMVec::Zero();
                for(int m = 0; m < DIM; m++) {
                    for(int n = 0; n < DIM; n++) {
                            XhatXhat(sm_index(m, n)) += Xhatij(m)*Xhatij(n);
                    }
                }

                for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                    M_trip_list.push_back(Trip(NNDOF+m, meas[t].NOstrain + NMA + e, XhatXhat(m)));
                }
            }
        }

        M[t].setFromTriplets(M_trip_list.begin(), M_trip_list.end());
    }
    
    meas_index.resize(NF, 0);
    for(int t = 1; t < NF; t++) {
        meas_index[t] = meas_index[t-1] + NM[t];
    }
    
    // std::cout << M[0].transpose() << std::endl;
        
}

void LinSolver::solveFeasability(AbstractObjFunc &obj_func, std::vector<bool> &feasible, 
                                 std::vector<std::vector<double> > &u, std::vector<std::vector<double> > &con_err) {
        
    std::vector<double> _obj_C;
    std::vector<int> _obj_CT;
    obj_func.getConstraints(_obj_C, _obj_CT);
    
    XVec obj_C;
    vectorToEigen(_obj_C, obj_C);
    
    XiVec obj_CT;
    vectorToEigen(_obj_CT, obj_CT);
    
    u.resize(NF);
    feasible.assign(NF, false);
    con_err.resize(NF);
    
    int C_index = 0;
    for(int t = 0; t < NF; t++) {
        
        XVec u0_vec = XVec::Zero(NNDOF+NADOF);

        alglib::real_1d_array u0;
        u0.setcontent(NNDOF+NADOF, u0_vec.data());

        // Algorithm state
        alglib::minbleicstate state;
        // Create algorithm state
        alglib::minbleiccreate(NNDOF+NADOF, u0, state);


        // Tolerance on gradient, change in function and change in x
        double epsg = 10 * MEPS;
        double epsf = 10 * MEPS;
        double epsx = 10 * MEPS;
        // Max iterations
        alglib::ae_int_t maxits = 0;
        // Set stopping conditions
        alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);
        
        // Scale of variables
        XVec s_vec = XVec::Ones(NNDOF+NADOF);
        alglib::real_1d_array s;
        s.setcontent(NNDOF+NADOF, s_vec.data());
       
        // Set scale of variables
        alglib::minbleicsetscale(state, s);

        
        XMat C_mat = XMat::Zero(NC[t]+NFGDOF+NM[t], NNDOF+NADOF+1);
        
        C_mat.block(0, 0, NFGDOF, NNDOF+NADOF) = G.block(NNDOF+NADOF, 0, NFGDOF, NNDOF+NADOF);
        C_mat.block(NFGDOF, 0, NC[t], NNDOF+NADOF) = C1[t].transpose();
        C_mat.block(NFGDOF, NNDOF+NADOF, NC[t], 1) = -C0[t];
        C_mat.block(NC[t]+NFGDOF, 0, NM[t], NNDOF+NADOF) = M[t].transpose();
        C_mat.block(NC[t]+NFGDOF, NNDOF+NADOF, NM[t], 1) = obj_C.segment(C_index, NM[t]);
                        
        alglib::real_2d_array C;
        C.setlength(NFGDOF+NC[t]+NM[t], NNDOF+NADOF+1);
        
        for(int i = 0; i < C_mat.rows(); i++) {
            XVecMap C_map(C[i], NNDOF+NADOF+1);
            C_map = C_mat.row(i);
        }
        
        XiVec CT_vec = XiVec::Zero(NFGDOF+NC[t]+NM[t]);
        CT_vec.segment(NC[t]+NFGDOF, NM[t]) = obj_CT.segment(C_index, NM[t]);
        
        alglib::integer_1d_array CT;
        CT.setlength(NFGDOF+NC[t]+NM[t]);
        for(int i = 0; i < NFGDOF+NC[t]+NM[t]; i++) {
            CT[i] = CT_vec(i);
            
        }
        // CT.setcontent(NNDOF+NADOF+1, CT_vec.data());
        C_index += NM[t];
        
        
        // Set equality and inequality constraints
        alglib::minbleicsetlc(state, C, CT);

        alglib::minbleicsetgradientcheck(state, 1e-2);
        
        LinOptParams params;
        params.t = t;
        params.solver = this;
        
        // Perform optimization
        alglib::minbleicoptimize(state, LinSolver::FeasibilityFuncJacWrapper, NULL, &params);
        
        // Result 
        alglib::real_1d_array u_fin;
        // Optimization report
        alglib::minbleicreport rep;
        // Retrieve optimization results
        alglib::minbleicresults(state, u_fin, rep);
        
        u[t].assign(u_fin.getcontent(), u_fin.getcontent()+u_fin.length());
        
        std::cout << "Termination Type: ";
        
        switch(rep.terminationtype) {
            case -8:
                std::cout << "Internal integrity control detected  infinite  or  NAN  values  in function/gradient. Abnormal termination signalled." << std::endl;
                break;
            case -7:
                std::cout << "Gradient verification failed." << std::endl;
                break;
            case -3:
                std::cout << "Inconsistent constraints. Feasible point is either nonexistent or too hard to find. Try to restart optimizer with better initial approximation." << std::endl;
                break;
            case 1:
                std::cout << "Relative function improvement is no more than EpsF." << std::endl;
                feasible[t] = true;
                break;
            case 2:
                std::cout << "Relative step is no more than EpsX." << std::endl;
                feasible[t] = true;
                break;
            case 4:
                std::cout << "Gradient norm is no more than EpsG." << std::endl;
                feasible[t] = true;
                break;
            case 5:
                std::cout << "MaxIts steps was taken." << std::endl;
                feasible[t] = true;
                break;
            case 7:
                std::cout << "Stopping conditions are too stringent, further improvement is impossible, X contains best point found so far." << std::endl;
                feasible[t] = true;
                break;
            case 8:
                std::cout << "Terminated by user who called minbleicrequesttermination(). X contains point which was \"current accepted\"  when termination request was submitted." << std::endl;
                feasible[t] = true;
                break;
            default:
                std::cout << "Unkown error code: " << rep.terminationtype << std::endl;
        }
        
        XVecMap u_fin_map(u_fin.getcontent(), NNDOF+NADOF);
        XVec c_err = C_mat.block(0, 0, NFGDOF+NC[t]+NM[t], NNDOF+NADOF) * u_fin_map - C_mat.col(NNDOF+NADOF);
        for(int i = 0; i < NFGDOF+NC[t]+NM[t]; i++) {
            if((CT[i] == 1 && c_err(i) > 0.0) || (CT[i] == -1 && c_err(i) < 0.0)) {
                c_err(i) = 0.0;
            }  
                
        }
        
        
        eigenToVector(c_err, con_err[t]);
        
        // std::cout << "Constraint Values: " << C_mat.block(0, 0, NFGDOF+NC[t]+NM[t], NNDOF+NADOF) * u_fin_map << std::endl;
    }
}


void LinSolver::FeasibilityFuncJacWrapper(const alglib::real_1d_array &x, double &func, 
                                   alglib::real_1d_array &grad, void *ptr) {
        
    LinOptParams *params = static_cast<LinOptParams*>(ptr);
	int t = params->t;
	LinSolver *solver = params->solver;
    
    std::vector<double> u;
    u.assign(x.getcontent(), x.getcontent()+x.length());
    double energy;
    solver->solveEnergy(t, u, energy);
    std::vector<double> g;
    solver->solveGrad(t, u, g);
    
    func = energy;
    
    // std::cout << energy << std::endl;    
    
    XVecMap grad_map(g.data(), solver->NNDOF+solver->NADOF);
    XVecMap _grad_map(grad.getcontent(), solver->NNDOF+solver->NADOF);
    
    _grad_map = grad_map;
        
}


void LinSolver::solveEnergy(int t, std::vector<double> &u, double &energy) {
            
    XVecMap u_map(u.data(), NNDOF+NADOF);

    energy = 0.5 * u_map.transpose() * Q.block(0, 0, NNDOF+NADOF, nw.NE) * K.asDiagonal() * 
        Q.block(0, 0, NNDOF+NADOF, nw.NE).transpose() * u_map;    
    
}

void LinSolver::solveGrad(int t, std::vector<double> &u, std::vector<double> &grad) {            
        
    XVecMap u_map(u.data(), NNDOF+NADOF);

    XVec g = Q.block(0, 0, NNDOF+NADOF, nw.NE) * K.asDiagonal() * 
        Q.block(0, 0, NNDOF+NADOF, nw.NE).transpose() * u_map;

    eigenToVector(g, grad);                 
    
}

void LinSolver::getAugHessian(std::vector<std::vector<std::vector<double> > > &AH) {
    
    if(need_H) {
        H = Q * K.asDiagonal() * Q.transpose() + G;
    }
    
    AH.resize(NF);
    for(int t = 0; t < NF; t++) {
    
        XMat AH_mat = XMat::Zero(NDOF+NC[t], NDOF+NC[t]);
        AH_mat.block(0,0,NDOF, NDOF) = H;
        AH_mat.block(NDOF, 0, NC[t], NDOF) = -C1[t].transpose();
        AH_mat.block(0, NDOF, NDOF, NC[t]) = -C1[t];
        
        eigenMatToVector(AH_mat, AH[t]);
        
                
    }
    
}

void LinSolver::getAugMeasMat(std::vector<std::vector<std::vector<double> > > &AM) {
    
    AM.resize(NF);
    for(int t = 0; t < NF; t++) {
        XMat M_mat = XMat::Zero(NDOF+NC[t], NM[t]);
        M_mat.block(0, 0, NDOF, NM[t]) = M[t];
        eigenMatToVector(M_mat, AM[t]);   
                
    }
    
}

void LinSolver::getAugForce(std::vector<std::vector<double> > &Af) {
    
    Af.resize(NF);
    for(int t = 0; t < NF; t++) {
        XVec f_vec = XVec::Zero(NDOF+NC[t]);
        f_vec.segment(0, NDOF) = f[t];
        f_vec.segment(NDOF, NC[t]) = C0[t];
        
        
        eigenToVector(f_vec, Af[t]);   
                
    }
    
}

void LinSolver::getEigenvals(std::vector<double> &evals, bool bordered) {
    
    SMat H = Q * K.asDiagonal() * Q.transpose() + G;

    if(bordered) {
    
        std::vector<Trip > BH_trip_list; 
        for (int k = 0; k < H.outerSize(); ++k) {
            for (SMat::InnerIterator it(H, k); it; ++it) {
                BH_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
            }
        }

        for (int k = 0; k < C1[0].outerSize(); ++k) {
            for (SMat::InnerIterator it(C1[0], k); it; ++it) {
                BH_trip_list.push_back(Trip(it.row(), NDOF + it.col(), -it.value()));
                BH_trip_list.push_back(Trip(NDOF + it.col(), it.row(), -it.value()));
            }
        }


        for (int k = 0; k < NNDOF; k++) {
            // BH_trip_list.push_back(Trip(k, k, 1e-6));
        }

        SMat BH;
        BH.resize(NDOF + NC[0], NDOF + NC[0]);
        BH.setFromTriplets(BH_trip_list.begin(), BH_trip_list.end());


        eigen_solver.compute(XMat(BH));
    } else {
        eigen_solver.compute(XMat(H));
        
    }
        
    if (eigen_solver.info() != Eigen::Success) {
        std::cout << "Calculating eigen decomposition failed" << std::endl;
        return;
    }
    
//     std::cout << "one" << std::endl;
//     std::cout << eigen_solver.eigenvectors().col(2) << std::endl;;
    
//     std::cout << "two" << std::endl;
//     std::cout << eigen_solver.eigenvectors().col(3) << std::endl;;
    
//     std::cout << "three" << std::endl;
//     std::cout << eigen_solver.eigenvectors().col(4) << std::endl;;
    
    evals.resize(NDOF+NC[0]);
    evals.assign(eigen_solver.eigenvalues().data(), eigen_solver.eigenvalues().data()+NDOF+NC[0]);

}

void LinSolver::initEdgeCalc(std::vector<std::vector<double> > &meas) {
        
    NEr = nw.NE;
    NSSS = NEr - NNDOF + NFGDOF;
    NSCS = NEr - NSSS;
    
    r2f.resize(NEr);
    f2r.resize(NEr);
    for(int b = 0; b < NEr; b++) {
        r2f[b] = b;
        f2r[b] = b;
    }
    
    setupEdgeCalcMats();
    calcEdgeBasis();    
    calcEdgeResponse(meas);
    
}

// Setup Kr, Qr and Mr here
void LinSolver::setupEdgeCalcMats() {
    
    Kr = XVec::Zero(NEr);
    for(int b = 0; b < NEr; b++) {
        Kr(b) = K(r2f[b]);
    }
    
    Qr.resize(NNDOF, NEr);
    std::vector<Trip > Qr_trip_list;
    for (int k = 0; k < Q.outerSize(); ++k) {
        for (SMat::InnerIterator it(Q, k); it; ++it) {
            if(it.row() < NNDOF && f2r[it.col()] != -1) {
                Qr_trip_list.push_back(Trip(it.row(), f2r[it.col()], it.value()));
            }
        }
    }
    Qr.setFromTriplets(Qr_trip_list.begin(), Qr_trip_list.end());
    
    Mr.resize(NEr, NM[0]);
    std::vector<Trip > Mr_trip_list;
    
    
    for(int b = 0; b < NM[0]; b++) {
                
        Mr_trip_list.push_back(Trip(f2r[meas[0].ostrain_bonds(b)], b, 1.0 / nw.eq_length(meas[0].ostrain_bonds(b)) 
                                   / 1.0 / sqrt(Kr(f2r[meas[0].ostrain_bonds(b)]))));
    }
    Mr.setFromTriplets(Mr_trip_list.begin(), Mr_trip_list.end());
        
}

void LinSolver::calcEdgeBasis() {
    SMat QQ = (Qr * Kr.array().sqrt().matrix().asDiagonal()).transpose() 
        * (Qr * Kr.array().sqrt().matrix().asDiagonal());
    
    edge_basis_solver.compute(XMat(QQ));
    
    SSS_basis = XMat::Zero(NEr, NSSS);
    for(int b = 0; b < NSSS; b++) {
        // std::cout << "SSS " << b << "\t" << edge_basis_solver.eigenvalues()(b) << std::endl;
        
        SSS_basis.col(b) = edge_basis_solver.eigenvectors().col(b);
    }
    
    SCS_basis = XMat::Zero(NEr, NSCS);
    for(int b = 0; b < NSCS; b++) {
        // std::cout << "SCS " << b << "\t" << edge_basis_solver.eigenvalues()(NSSS+b) << std::endl;
        
        SCS_basis.col(b) = edge_basis_solver.eigenvectors().col(NSSS + b);
    }  
    
    
    // std::cout << SSS_basis.transpose() * SCS_basis << std::endl;
    
}

void LinSolver::calcEdgeResponse(std::vector<std::vector<double> > &meas) {
        
    int s = f2r[pert[0].istrain_bonds(0)];
    
    Cs = SCS_basis * SCS_basis.row(s).transpose();
            
    double l0 = nw.eq_length(pert[0].istrain_bonds(0));
    Mext = Mr.transpose() * Cs;
    Mext *= pert[0].istrain(0) * l0 * sqrt(Kr(s)) / Cs(s);
        
    meas.resize(NF);
    for(int t = 0; t < NF; t++ ) {                
        eigenToVector(Mext, meas[t]);
    }    
    
    
    // for(int b = 0; b < nw.NE; b++) {
    //     std::cout << b << "\t" << Cs(b) / Cs(s) * pert[0].istrain(0) * l0 / nw.eq_length(b) << std::endl;
    // }
}

double LinSolver::solveEdgeUpdate(int i, std::vector<std::vector<double> > &meas) {
    
    int b = f2r[i];
    
    if(b == -1) {
        return -1.0;
    }
    
    double S2 = SSS_basis.row(b) * SSS_basis.row(b).transpose();
        
    if(S2 < SQRTMEPS) {
        return -1.0;
    }
    
    XVec Ci = SCS_basis * SCS_basis.row(b).transpose();
           
    int s = f2r[pert[0].istrain_bonds(0)];
       
    double l0 = nw.eq_length(pert[0].istrain_bonds(0));
    
    XVec Mext_tmp = XMat(Mr.transpose() * Cs);
    Mext_tmp += XMat(Mr.transpose() * Ci) * Ci(s) / S2;
    Mext_tmp *= pert[0].istrain(0) * l0 *sqrt(Kr(s)) / (Cs(s) + Ci(s) * Ci(s) / S2);
        
        
    meas.resize(NF);
    for(int t = 0; t < NF; t++ ) {                
        eigenToVector(Mext_tmp, meas[t]);
    }
    
    return 1.0;
    
}

double LinSolver::removeEdge(int irem, std::vector<std::vector<double> > &meas) {
    int b = f2r[irem];
    
    if(b == -1) {
        return -1.0;
    }
    
    // Check this
    XVec Si = SSS_basis * SSS_basis.row(b).transpose();
          
    for(int j = 0; j < NSSS; j++) {
                
        double proj = (SSS_basis.col(j).transpose() * Si)(0) / Si(b);
                
        SSS_basis.col(j) -= proj * Si;
        
    }
         
    //Check this
    SSS_basis.block(b, 0, NEr-1-b, NSSS) = SSS_basis.block(b+1, 0, NEr-1-b ,NSSS);
    SSS_basis.col(NSSS-2) += SSS_basis.col(NSSS-1);
    SSS_basis.conservativeResize(NEr-1, NSSS-1);
        
    // Check this
    qr_solver.compute(SSS_basis);
    SSS_basis = qr_solver.householderQ() * XMat::Identity(NEr-1, NSSS-1);
    
    SCS_basis.block(b, 0, NEr-1-b, NSCS) = SCS_basis.block(b+1, 0, NEr-1-b ,NSCS);
    SCS_basis.conservativeResize(NEr-1, NSCS);
    
    qr_solver.compute(SCS_basis);
    SCS_basis = qr_solver.householderQ() * XMat::Identity(NEr-1, NSCS);
    
    NEr -= 1;
    NSSS -= 1;
            
    // update index maps
    f2r[irem] = -1;
    for(int j = irem+1; j < nw.NE; j++) {
        if(f2r[j] > 0) {
            f2r[j] -= 1;
        }
    }
    
    r2f.erase(r2f.begin() + b);
    
    setupEdgeCalcMats();
    calcEdgeResponse(meas);
    
    double SC_error = (SSS_basis.transpose() * SCS_basis).cwiseAbs().maxCoeff();
    double SS_error = (SSS_basis.transpose() * SSS_basis - XMat::Identity(NSSS, NSSS)).cwiseAbs().maxCoeff();
    double CC_error = (SCS_basis.transpose() * SCS_basis - XMat::Identity(NSCS, NSCS)).cwiseAbs().maxCoeff();
    
    double error = SC_error;
    if(SS_error > error) {
        error = SS_error;
    }
    if(CC_error > error) {
        error = CC_error;
    }
    
    return error;
    
}

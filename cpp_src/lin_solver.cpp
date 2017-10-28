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
    
    setupEqMat();
    setupGlobalConMat();
    setupPertMat();
    setupMeasMat();
    setupHessian(H);
    
    bordered = false;
}

void LinSolver::setIntStrengths(std::vector<double> &K) {
    vectorToEigen(K, this->K);
    setupHessian(H);
}


// void LinSolver::calcHessian() {
    
//     if(!need_H) {
//         return;
//     }
    
    
//     SMat I(NDOF, NDOF);
//     std::vector<Trip > id_trip_list;
//     for(int k = 0; k < NNDOF; k++) {
//         // id_trip_list.push_back(Trip(k, k, 1e-6));
//     }
//     I.setFromTriplets(id_trip_list.begin(), id_trip_list.end());
    
//     H = Q * K.asDiagonal() * Q.transpose() + G + I;
          
// //     std::vector<double> evals;
// //     getEigenvals(evals, false);
    
// //     std::cout << evals[0] << "\t" << evals[1] << "\t" << evals[2] << "\t" << evals[3] << std::endl;
    
//     // Perform LU decomposition
//     solver.compute(H);
//     if (solver.info() != Eigen::Success) {
//         // decomposition failed
                        
//         std::cout << "LU decomposition of Hessian failed." << std::endl;
//         std::exit(EXIT_FAILURE);
//     }
    
//     need_H = false;
// }

// void LinSolver::calcInvHessian() {
    
//     if(!need_Hinv) {
//         return;
//     }
    
//     calcHessian();
                
//     // Perform LU decomposition
//     Hinv = solver.solve(XMat(XMat::Identity(H.rows(), H.cols())));
//     // Hinv = XMat(H).inverse();
            
//     if (solver.info() != Eigen::Success) {
//         std::cout << "Solving H^{-1} failed." << std::endl;
//         std::exit(EXIT_FAILURE);
//     }
    
//     need_Hinv = false;
// }

// void LinSolver::calcPert(bool use_full_inverse) {
    
//     if(!need_HinvPert) {
//         return;
//     }
    
//     if(use_full_inverse) {
//         calcInvHessian();
//     } else {
//         calcHessian();
//     }
    
    
//     HinvC1.resize(NF);
//     Hinvf.resize(NF);
    
//     for(int t = 0; t < NF; t++) {
    
//         //  H^{-1}C_1
//         if(C1[t].nonZeros() > 0) {
//             if(use_full_inverse) {
//                 HinvC1[t] = Hinv * C1[t];
//             } else {
//                 HinvC1[t] = solver.solve(XMat(C1[t]));
//                 if (solver.info() != Eigen::Success) {
//                     std::cout << "Solving H^{-1}C_1 failed." << std::endl;
//                     std::exit(EXIT_FAILURE);
//                 }
//             }
                
//         }

//         //  H^{-1}f
//         if(f[t].nonZeros() > 0) {
//             if(use_full_inverse) {
//                 Hinvf[t] = Hinv * f[t];
//             } else {
//                 Hinvf[t] = solver.solve(XVec(f[t]));
//                 if (solver.info() != Eigen::Success) {
//                     std::cout << "Solving H^{-1}f failed." << std::endl;
//                     std::exit(EXIT_FAILURE);
//                 }
//             }
//         }
//     }
    
//     need_HinvPert = false;
    
// }

// void LinSolver::calcMeas(bool use_full_inverse) {
    
//     if(!need_HinvM) {
//         return;
//     }
    
//     if(use_full_inverse) {
//         calcInvHessian();
//     } else {
//         calcHessian();
//     }
    
//     HinvM.resize(NF);
    
//     for(int t = 0; t < NF; t++) {
    
//         //  H^{-1}M
//         if(M[t].nonZeros() > 0) {
//             if(use_full_inverse) {
//                 HinvM[t] = Hinv * M[t];
//             } else {
//                 HinvM[t] = solver.solve(XMat(M[t]));
//                 if (solver.info() != Eigen::Success) {
//                     std::cout << "Solving H^{-1}M failed." << std::endl;
//                     std::exit(EXIT_FAILURE);
//                 }
//             }
//         }
//     }  

//     need_HinvM = false;
// }

void LinSolver::isolveU(std::vector<XVec > &u) {
    
    setupBorderedHessian(H);
    extendBorderedSystem();
        
    solver.compute(H);
        
    u.resize(NF);
    Hinvf.resize(NF);
    HinvC1.resize(NF);
    for(int t = 0; t < NF; t++ ) {
        XVec u_tmp;
        if(NF == 1 || C1[t].nonZeros() == 0) {
            
            Hinvf[t] = solver.solve(XMat(f[t]));
            if (solver.info() != Eigen::Success) {
                std::cout << "Solving H^{-1}f failed." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            u_tmp = Hinvf[t];
        } else {
            
            HinvC1[t] = solver.solve(XMat(C1[t]));
            if (solver.info() != Eigen::Success) {
                std::cout << "Solving H^{-1}C_1 failed." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            
            XMat CHiC = C1[t].transpose() * HinvC1[t];

            if(f[t].nonZeros() > 0) {
                Hinvf[t] = solver.solve(XMat(f[t]));
                if (solver.info() != Eigen::Success) {
                    std::cout << "Solving H^{-1}f failed." << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                
                u_tmp = (Hinvf[t] - HinvC1[t] * CHiC * (C1[t].transpose() * Hinvf[t] + C0[t]));
            } else {
                u_tmp = - HinvC1[t] * CHiC * C0[t];
            }
        } 
        
        u[t] = u_tmp.segment(0, NDOF);
    }

}

void LinSolver::isolveLambda(std::vector<XVec > &lambda) {
    
//     bool use_full_inverse = (NC_tot > NDOF);
    
//     calcPert(use_full_inverse);
   
//     lambda.resize(NF);
    
//     for(int t = 0; t < NF; t++) {
    
//         if(C1[t].nonZeros() > 0) {
//             XMat A = (C1[t].transpose() * HinvC1[t]).inverse();

//             if(f[t].nonZeros() > 0) {
//                 lambda[t] = - A * (C1[t].transpose() * Hinvf[t] + C0[t]);
//             } else {
//                 lambda[t] = - A * C0[t];
//             }
//         } else if(f[t].nonZeros() > 0) {
//             lambda[t] = XVec::Zero(NC[t]);
//         } else {
//             std::cout << "Func Error" << std::endl;
//             exit(EXIT_SUCCESS);
//         }            
//     }
}


void LinSolver::isolveM(XVec &meas) {
    std::vector<XVec > u;
        
    isolveU(u);
            
    meas = XVec::Zero(NM_tot);
    for(int t = 0; t < NF; t++) {
        
        meas.segment(meas_index[t], NM[t]) = (M2K[t] * (K.asDiagonal() * M2K[t].transpose()) + IM[t])  * M[t].transpose() * u[t];
        
    }
        
}

void LinSolver::isolveMGrad(XVec &meas, std::vector<XVec > &grad) {

//     bool use_full_inverse = (NC_tot > NDOF) || (NM_tot > NDOF);
    
//     calcPert(use_full_inverse);
//     calcMeas(use_full_inverse);
    
//     meas = XVec::Zero(NM_tot);
//     grad.resize(NM_tot);
    
//     for(int t = 0; t < NF; t++) {
        
//         XVec u;
    
//         XMat Hinv11M;
//         if(C1[t].nonZeros() > 0) {
//             XMat A = (C1[t].transpose() * HinvC1[t]).inverse();
//             Hinv11M = HinvM[t] - HinvC1[t] * A * C1[t].transpose() * HinvM[t];

//             if(f[t].nonZeros() > 0) {
//                 u = (Hinvf[t] - HinvC1[t] * A * (C1[t].transpose() * Hinvf[t] + C0[t]));
//             } else {
//                 u = - HinvC1[t] * A * C0[t];
//             }
//         } else if(f[t].nonZeros() > 0) {
//             Hinv11M = HinvM[t];
//             u = Hinvf[t];

//         } else {
//             std::cout << "Grad Error" << std::endl;
//             exit(EXIT_SUCCESS);
//         }   
        
//         meas.segment(meas_index[t], NM[t]) = M[t].transpose() * u;

//         XMat grad_mat = XMat::Zero(NM[t], nw.NE);
//         for(int b = 0; b < nw.NE; b++) {
//             double qu = (Q.col(b).transpose() * u)(0);
            
//             grad_mat.col(b) = - Hinv11M.transpose() * Q.col(b) * qu;
//         }
            
//         for(int im = 0; im < NM[t]; im++) {
//             grad[meas_index[t] + im] = grad_mat.row(im);
//         }      
        
        
//     } 
    
}

void LinSolver::isolveMHess(XVec &meas, std::vector<XVec > &grad, std::vector<XMat > &hess) {
//     bool use_full_inverse = true;
        
//     calcInvHessian();
//     calcPert(use_full_inverse);
//     calcMeas(use_full_inverse);
        
//     meas = XVec::Zero(NM_tot);
//     grad.resize(NM_tot);
//     hess.resize(NM_tot, XMat::Zero(nw.NE, nw.NE));
       
    
//     for(int t = 0; t < NF; t++) {
        
//         XVec u;
    
//         XMat Hinv11;
//         if(C1[t].nonZeros() > 0) {
            
//             XMat A = (C1[t].transpose() * HinvC1[t]).inverse();
//             Hinv11 = Hinv - HinvC1[t] * A * HinvC1[t].transpose();
                        
//             if(f[t].nonZeros() > 0) {
//                 u = (Hinvf[t] - HinvC1[t] * A * (C1[t].transpose() * Hinvf[t] + C0[t]));
//             } else {                
//                 u = - HinvC1[t] * A * C0[t];
//             }
//         } else if(f[t].nonZeros() > 0) {
//             Hinv11 = Hinv;
//             u = Hinvf[t];

//         } else {
//             std::cout << "Grad Error" << std::endl;
//             exit(EXIT_SUCCESS);
//         } 
                
//         meas.segment(meas_index[t], NM[t]) = M[t].transpose() * u;

//         XMat grad_mat = XMat::Zero(nw.NE, nw.NE);
//         XMat mgrad_mat = XMat::Zero(NM[t], nw.NE);
//         for(int b = 0; b < nw.NE; b++) {
//             double qu = (Q.col(b).transpose() * u)(0);
            
//             grad_mat.col(b) = -Hinv11 * Q.col(b) * qu;
//             mgrad_mat.col(b) = M[t].transpose() * grad_mat.col(b);
//         }
        
//         for(int im = 0; im < NM[t]; im++) {
//             grad[meas_index[t] + im] = mgrad_mat.row(im);
//         }        
                
//         XVec hess_el = XVec::Zero(NM[t]);
//         XMat Hinv11M =  M[t].transpose() * Hinv11;
//         for(int bi = 0; bi < nw.NE; bi++) {
//             for(int bj = bi; bj < nw.NE; bj++) {
                
//                 double qidudkj = (Q.col(bi).transpose() * grad_mat.col(bj))(0);
//                 double qjdudki = (Q.col(bj).transpose() * grad_mat.col(bi))(0);
                
//                 hess_el = -Hinv11M * (Q.col(bi) * qidudkj + Q.col(bj) * qjdudki);
                
//                 for(int im = 0; im < NM[t]; im++) {
//                     hess[meas_index[t] + im](bi, bj) += hess_el(im);
//                     hess[meas_index[t] + im](bj, bi) += hess_el(im);
                    
//                 }
//             }
//         }
                    
//     } 
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
    
    setupHessian(H);
    setupBorderedHessian(H);
    extendBorderedSystem();
        
    dHinv = XMat::Zero(NDOF, NDOF);

    //Apply SMat to make a new reference so H can change without causing seg faults
    solver.compute(H);

    Hinv = solver.solve(XMat(XMat::Identity(H.rows(), H.cols())));

    
    
    Hinvf.resize(NF);
    HinvC1.resize(NF);
    HinvM.resize(NF);
    for(int t = 0; t < NF; t++) {
        // Initialize perturbation
        Hinvf[t] = solver.solve(XMat(f[t]));
        if (solver.info() != Eigen::Success) {
            std::cout << "Solving H^{-1}f failed." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        if(NF != 1) {
            HinvC1[t] = solver.solve(XMat(C1[t]));
            if (solver.info() != Eigen::Success) {
                std::cout << "Solving H^{-1}C_1 failed." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // Initialize measurements
        HinvM[t] = solver.solve(XMat(M[t]));
        if (solver.info() != Eigen::Success) {
            std::cout << "Solving H^{-1}M failed." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        XMat tmp = H * HinvM[t];
        tmp -= M[t];
        
        double error = tmp.cwiseAbs().maxCoeff();
        std::cout << "Max Meas Error: " << error << std::endl;
    }


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

        SMat BU(NDOF, up_list[i].NSM);

        std::vector<Trip > BU_trip_list; 
        for (int k = 0; k < U.outerSize(); ++k) {
            for (SMat::InnerIterator it(U, k); it; ++it) {
                BU_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
            }
        }
        BU.setFromTriplets(BU_trip_list.begin(), BU_trip_list.end());

        XMat HinvU = solver.solve(XMat(BU)); 
        if (solver.info() != Eigen::Success) {
            std::cout << "Solving H^{-1}U failed." << std::endl;
        }

        SM_updates[i] = BU.transpose() * HinvU;
    }
                
}

double LinSolver::solveMeasUpdate(int i, std::vector<std::vector<double> > &meas) {

    
    XVec dK(up_list[i].NSM);

    XVec K = this->K;
    
    // Changes in bond stretch moduli
    for(int s = 0; s < up_list[i].NSM; s++) {
        int b = up_list[i].sm_bonds(s);

        double k1 = K(b);
        double k2 = up_list[i].stretch_mod(s);

        dK(s) = k2 - k1;
        
        K(b) = k2;

    }
    
    
    XMat A = dK.asDiagonal() * SM_updates[i] + XMat::Identity(up_list[i].NSM, up_list[i].NSM);

    double det = fabs(A.determinant());
        
    if(det < 1e-4) {
        // std::cout << "Hessian update creates zero mode..." << std::endl;
        // std::cout << "|det(A)|: " << det << " < " << 1e-3 << std::endl;
        return -1.0;
    }
    
    
    
    // Setup Hessian perturbation
    SMat U(NDOF, up_list[i].NSM); 
    for(int s = 0; s < up_list[i].NSM; s++) {
        int b = up_list[i].sm_bonds(s);

        U.col(s) = Q.col(b);

    }
    
    
    SMat BU(NDOF, up_list[i].NSM);
        
    std::vector<Trip > BU_trip_list; 
    for (int k = 0; k < U.outerSize(); ++k) {
        for (SMat::InnerIterator it(U, k); it; ++it) {
            BU_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    BU.setFromTriplets(BU_trip_list.begin(), BU_trip_list.end());
    
    meas.resize(NF);
    for(int t = 0; t < NF; t++ ) {
        
        XVec m;
        if(NF == 1 || C1[t].nonZeros() == 0) {
            m = (M2K[t] * (K.asDiagonal() * M2K[t].transpose()) + IM[t]) * 
                (M[t].transpose() * Hinvf[t] - HinvM[t].transpose() * BU * dK.asDiagonal() * A.inverse() * BU.transpose() * Hinvf[t]);
        } else {
            XMat CHiC = C1[t].transpose() * HinvC1[t];

            XVec Hpif;
            if(f[t].nonZeros() > 0) {
                Hpif = (Hinvf[t] - HinvC1[t] * CHiC * (C1[t].transpose() * Hinvf[t] + C0[t]));
            } else {
                Hpif = - HinvC1[t] * CHiC * C0[t];
            }

            XVec HpiM = HinvM[t] - HinvC1[t] * CHiC * C1[t].transpose() * HinvM[t];

            m = (M2K[t] * (K.asDiagonal() * M2K[t].transpose()) + IM[t]) * 
                M[t].transpose() * Hpif - HpiM.transpose() * BU * dK.asDiagonal() * A.inverse() * BU.transpose() * Hpif;
        }
                
        eigenToVector(m, meas[t]);
    }
    
    
    
    
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
    
//     SMat H;
//     H.resize(NDOF + NC[0], NDOF + NC[0]);
//     H.setFromTriplets(BH_trip_list.begin(), BH_trip_list.end());
    
    
//     XMat tmp = H * (Hinvf - ((Hinv+dHinv) * BU) * dK.asDiagonal() * A.inverse() * (BU.transpose() * Hinvf));
                
//     tmp -= f;
//     double error = tmp.cwiseAbs().maxCoeff() / XMat(f).cwiseAbs().maxCoeff();
//     // std::cout << "Max Response Error: " << error << std::endl;

//     XMat BHinvU = Hinv * BU;
        
//     double condition =  XMat(H).cwiseAbs().rowwise().sum().maxCoeff() * (Hinv + dHinv - BHinvU * dK.asDiagonal() * A.inverse() * BHinvU.transpose()).cwiseAbs().rowwise().sum().maxCoeff();
    
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
    
    // H^{-1}U
    SMat BU(NDOF, up_list[i].NSM);
        
    std::vector<Trip > BU_trip_list; 
    for (int k = 0; k < U.outerSize(); ++k) {
        for (SMat::InnerIterator it(U, k); it; ++it) {
            BU_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    BU.setFromTriplets(BU_trip_list.begin(), BU_trip_list.end());
    
        
            
    XMat BHinvU = solver.solve(XMat(BU));     
    if (solver.info() != Eigen::Success) {
        std::cout << "Solving H^{-1}U failed." << std::endl;
    }

        
    BHinvU += dHinv * BU;
    
        
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
    dHinv += delta;
        
    std::cout << "Delta Max: " << delta.cwiseAbs().maxCoeff() << std::endl;

    for(int t = 0; t < NF; t++) {
        Hinvf[t] += delta * f[t];
        HinvM[t] += delta * M[t];
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
        
    SMat H_local;
    setupHessian(H_local);
    setupBorderedHessian(H_local);
         
    double error = ((Hinv + dHinv) * H_local - XMat::Identity(H_local.rows(), H_local.cols())).cwiseAbs().maxCoeff();
    std::cout << "Max Bordered Hessian Error: " << error << std::endl;
    
    for(int t = 0; t < NF; t++) {

        XMat tmp = H_local * Hinvf[t];
        tmp -= f[t];
        error = tmp.cwiseAbs().maxCoeff();
        std::cout << "Max Force Error: " << error << std::endl;

        tmp = H_local * HinvM[t];
        tmp -= M[t];
        error = tmp.cwiseAbs().maxCoeff() / XMat(M[t]).maxCoeff();
        std::cout << "Max Meas Error: " << error << std::endl;
    }

    
    double condition =  XMat(H_local).cwiseAbs().rowwise().sum().maxCoeff() * (Hinv + dHinv).cwiseAbs().rowwise().sum().maxCoeff();
    
    std::cout << "Condition Number: " << condition << std::endl;
    
    // if(error > 0.0) {
    //     std::cout << "Resetting..." << std::endl;
    //     prepareUpdateList(up_list);
    // }
    
    
    meas.resize(NF);
    for(int t = 0; t < NF; t++ ) {
        XVec m;
        if(NF == 1 || C1[t].nonZeros() == 0) {
            m = M[t].transpose() * Hinvf[t];
        } else {
            XMat CHiC = C1[t].transpose() * HinvC1[t];

            XVec Hpif;
            if(f[t].nonZeros() > 0) {
                Hpif = (Hinvf[t] - HinvC1[t] * CHiC * (C1[t].transpose() * Hinvf[t] + C0[t]));
            } else {
                Hpif = - HinvC1[t] * CHiC * C0[t];
            }

            m = (M2K[t] * (K.asDiagonal() * M2K[t].transpose()) + IM[t]) * M[t].transpose() * Hpif;
        }
                
        eigenToVector(m, meas[t]);
    }
    
    return error;
    
}


double LinSolver::getConditionNum() {

    setupHessian(H);
    setupBorderedHessian(H);
    
    solver.compute(H);
    
    Hinv = solver.solve(XMat(XMat::Identity(H.rows(), H.cols())));
    
    double condition =  XMat(H).cwiseAbs().rowwise().sum().maxCoeff() * Hinv.cwiseAbs().rowwise().sum().maxCoeff();
    
    return condition;
}




//////////////////////////////// Matrix Setup /////////////////////////////////////////////////////



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
        
        // Rows of rotation matrix that define rotation plane
        // All other axes will be unaffected
        int d = 0;
        for(int d1 = 0; d1 < DIM-1; d1++) {
            for(int d2 = d1+1; d2 < DIM; d2++) {
                DMat dRdTheta = DMat::Identity();
                dRdTheta(d1, d1) = 0;
                dRdTheta(d1, d2) = -1;
                dRdTheta(d2, d1) = 1;
                dRdTheta(d2, d2) = 0;
                
                XVec global_rot = XVec::Zero(NNDOF);
                
                for(int i = 0; i < nw.NN; i++) {
                    DVec pos = nw.node_pos.segment<DIM>(DIM*i) - COM;
                    global_rot.segment<DIM>(DIM*i) = dRdTheta * pos;
                }
                
                global_rot.normalize();
                
                for(int i = 0; i < NNDOF; i++) {
                    G_trip_list.push_back(Trip(i, NDOF-NFGDOF+DIM+d, -global_rot(i)));
                    G_trip_list.push_back(Trip(NDOF-NFGDOF+DIM+d, i, -global_rot(i)));
                }
                
                d++;
            }
        }
        
        
        
        
//         XMat rot_axes = XMat::Zero(DIM*(DIM-1)/2 , 3);
        
//         if(DIM == 2) {
//             rot_axes << 0, 0, 1;
//         } else if(DIM == 3) {
//             rot_axes << 1, 0, 0,
//                         0, 1, 0,
//                         0, 0, 1;
//         }
        
//         for(int d = 0; d < DIM*(DIM-1)/2; d++) {
//             XVec global_rot = XVec::Zero(NNDOF);
            
//             for(int i = 0; i < nw.NN; i++) {
//                 Vec3d pos = Vec3d::Zero();
//                 if(DIM == 2) {
//                     pos.segment<2>(0) = nw.node_pos.segment<2>(2*i) - COM.segment<2>(0);
//                 } else if(DIM  == 3) {
//                     pos = nw.node_pos.segment<3>(3*i) - COM.segment<3>(0);
//                 }
                
//                 Vec3d axis = rot_axes.row(d);
                
//                 Vec3d rot_dir = axis.cross(pos);
                
//                 if(DIM == 2) {
//                     global_rot.segment<2>(2*i) = rot_dir.segment<2>(0);
//                 } else if(DIM  == 3) {
//                     global_rot.segment<3>(3*i) = rot_dir;
//                 }               
//             }
            
//             global_rot.normalize();
                        
//             for(int i = 0; i < NNDOF; i++) {
//                 G_trip_list.push_back(Trip(i, NDOF-NFGDOF+DIM+d, -global_rot(i)));
//                 G_trip_list.push_back(Trip(NDOF-NFGDOF+DIM+d, i, -global_rot(i)));
//             }
            
//         }
        
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
                // Tune extension, not strain
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
            
            // Convert tension to force (force is energy per length)
            f[t] += Q.col(ei) * pert[t].istress(e);
            // Convert stress to force
            // f[t] += Q.col(ei) * pert[t].istress(e) / nw.eq_length(ei);
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
    M2K.resize(NF);
    IM.resize(NF);
    
    NM_tot = 0;
    
    for(int t = 0; t < NF; t++) {
    
        NM[t] = meas[t].NOstrain + meas[t].NOstress + meas[t].NLambda;
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

        M2K[t].resize(NM[t], nw.NE);
        std::vector<Trip> M2K_trip_list;
        
        // Output stress responses
        for(int e = 0; e < meas[t].NOstress; e++) {
            M2K_trip_list.push_back(Trip(meas[t].NOstrain + NMA + e, meas[t].ostress_bonds(e), 1.0));
            
            
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
        
        M2K[t].setFromTriplets(M2K_trip_list.begin(), M2K_trip_list.end());
        
        
        IM[t].resize(NM[t], NM[t]);
        std::vector<Trip> IM_trip_list;
        for(int i = 0; i < meas[t].NOstrain + NMA; i++) {
            IM_trip_list.push_back(Trip(i, i, 1.0));
        }
        
        for(int i = meas[t].NOstrain + NMA + meas[t].NOstress; i < meas[t].NOstrain + NMA + meas[t].NOstress + meas[t].NLambda; i++) {
            IM_trip_list.push_back(Trip(i, i, 1.0));
        }
        IM[t].setFromTriplets(IM_trip_list.begin(), IM_trip_list.end());
    }
    
    meas_index.resize(NF, 0);
    for(int t = 1; t < NF; t++) {
        meas_index[t] = meas_index[t-1] + NM[t];
    }
    
    // std::cout << M[0].transpose() << std::endl;
        
}


void LinSolver::setupHessian(SMat &H) {
    H = Q * (K.asDiagonal() * Q.transpose()) + G;   
}

void LinSolver::setupBorderedHessian(SMat &H) {
    if(NF > 1) {
        return;
    }
    
    
    NDOF = NNDOF + NADOF + NFGDOF + NC[0];
    
    std::vector<Trip > H_trip_list; 
    for (int k = 0; k < H.outerSize(); ++k) {
        for (SMat::InnerIterator it(H, k); it; ++it) {
            H_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }

    for (int k = 0; k < C1[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(C1[0], k); it; ++it) {
            H_trip_list.push_back(Trip(it.row(), NNDOF + NADOF + NFGDOF + it.col(), -it.value()));
            H_trip_list.push_back(Trip(NNDOF + NADOF + NFGDOF + it.col(), it.row(), -it.value()));
        }
    }
    
    H.resize(NDOF, NDOF);
    H.setFromTriplets(H_trip_list.begin(), H_trip_list.end());
}

void LinSolver::extendBorderedSystem() {
    
    if(NF > 1 || bordered) {
        return;
    }
      
    //Setup bordered force
    std::vector<Trip > f_trip_list; 
    for (int k = 0; k < f[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(f[0], k); it; ++it) {
            f_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    for(int k = 0; k < NC[0]; k++) {
        f_trip_list.push_back(Trip(NNDOF + NADOF + NFGDOF+k, 0, C0[0](k)));
    }
    
    f[0].resize(NDOF, 1);
    f[0].setFromTriplets(f_trip_list.begin(), f_trip_list.end());

    //Setup bordered measurement matrix
    std::vector<Trip > M_trip_list; 
    
    for (int k = 0; k < M[0].outerSize(); ++k) {
        for (SMat::InnerIterator it(M[0], k); it; ++it) {
            M_trip_list.push_back(Trip(it.row(), it.col(), it.value()));
        }
    }
    
    for (int l = 0; l < meas[0].NLambda; l++) {
        M_trip_list.push_back(Trip(NNDOF + NADOF + NFGDOF + meas[0].lambda_vars[l], meas[0].NOstrain + meas[0].NOstress + l, 1.0));
    }
    
    M[0].resize(NDOF, NM[0]);
    M[0].setFromTriplets(M_trip_list.begin(), M_trip_list.end());
    
    bordered = true;
    
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

void LinSolver::getEigenvals(std::vector<double> &evals, bool bordered) {
    
    setupHessian(H);
    setupBorderedHessian(H);

    eigen_solver.compute(XMat(H));

        
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
    
    evals.resize(NDOF);
    evals.assign(eigen_solver.eigenvalues().data(), eigen_solver.eigenvalues().data()+NDOF);

}

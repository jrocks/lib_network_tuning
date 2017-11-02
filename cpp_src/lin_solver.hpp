#ifndef LINSOLVER
#define LINSOLVER
    
#include "util.hpp"
#include "network.hpp"
#include "perturb.hpp"
#include "measure.hpp"
#include "lin_solver_state.hpp"
#include "lin_solver_result.hpp"    
  
    
class LinUpdate {
    public:
        
        // Update stretch moduli
        int NdK;
        std::vector<int> dK_edges;
        XVec dK;
    
    LinUpdate(int NdK, std::vector<int> &dK_edges, RXVec dK) {
        this->NdK = NdK;
        this->dK_edges = dK_edges;
        this->dK = dK;
    };
    
};

    
template <int DIM>
class LinSolver {
    
    DEIGEN(DIM);
    DSYMEIGEN(DIM);
    
    public:
    
        static const int dim;
    
        // Network object
        Network<DIM> nw;
    
        // Number of dof in Hessian
        int NDOF;
        // Number of node dofs
        int NNDOF;
        // Number of affine dofs;
        int NADOF;
        // Number of global dofs;
        int NGDOF;
    
    
        // Equilibrium Matrix
        SMat Q;
        // Vector of interaction strengths
        XVec K;
        // Matrix of fixed global dof constraints
        SMat G;
    
        // Hessian matrix
        SMat H;
    
        // Whether to allow zero modes
        // (Only used when exactly one function
        // is being tuned with inputs applied as constraints)
        bool allow_zero;
    
        // Sparse matrix solver
        // Eigen::CholmodSupernodalLLT<SMat > solver;
        Eigen::UmfPackLU<SMat > solver;
        bool is_computed;
    
    
        // Number of independent functions (pert/meas pairs)
        int NF;
    
        // Number of constraints for each function
        std::vector<int> NC;
        // Number of measurements for each function
        std::vector<int> NM;
    
    
        // List of perturbations to apply
        std::vector<Perturb<DIM> > pert;
        // Linear coeffs to perturbs
        std::vector<SMat > C1;
        // Const coeffs to perturbs
        std::vector<XVec > C0;
        // Forces
        std::vector<SMat > f;
    
    
        // List of measurements
        std::vector<Measure<DIM> > meas;
        // Measurement matrices
        std::vector<SMat >  M;
    
        DMat sym_mat_index;
    
    
        // Inverted Hessian times equilibrium matrix (only store columns as needed)
        std::vector<XVec > HiQ;
        // Array of booleans indicating which columns HiQ have been solved for (true indicates it already exists)
        std::vector<bool> have_HiQ;
    
    public:
        LinSolver(Network<DIM> &nw, int NF, 
                  std::vector<Perturb<DIM> > &pert, 
                  std::vector<Measure<DIM> > &meas);
    
        // Set interaction strengths
        void setK(RXVec K);
        void setAllowZero(bool allow_zero);
    
    private:
        // Setup various matrices
        void setupPertMats();
        void setupMeasMats();
        void setupEqMat();
        void setupConMat();
        void setupHessian();
    
        bool computeHessian(LinSolverResult &result);
        bool computeInv(LinSolverState &state, LinSolverResult &result);
        bool computeBlockInv(LinSolverState &state, 
                                     std::vector<XVec > &u, std::vector<XVec > &lamb, LinSolverResult &result);
        bool computeInvUpdate(LinUpdate &up, LinSolverState &state1, LinSolverState &state2, bool save, LinSolverResult &result);
        bool computeResult(std::vector<XVec > &u, std::vector<XVec > &lamb, LinSolverResult &result);
            
    public:
        
        // Solve using current values of interaction strengths
        LinSolverResult* solve();
        // Solve with a linear update to Hessian
        LinSolverResult* solve(LinUpdate &up);
        // Solve using information contained specified in solver state object (solver state must have previously been set, otherwise set it up)
        LinSolverResult*  solve(LinSolverState &state);
        // Solve using information contained specified in solver state object with update to Hessian
        LinSolverResult*  solve(LinUpdate &up, LinSolverState &state); 
    
        
        // Solve and place result into solver state object
        LinSolverState* getSolverState();
        // Update solver state object
        void updateSolverState(LinUpdate &up, LinSolverState &state);
    
        bool computeMeas(LinSolverResult &result); 
        
};

template <int DIM>
const int LinSolver<DIM>::dim = DIM;


//////////////////////////////////////////////////////
// Function Implementations
//////////////////////////////////////////////////////

template<int DIM>
LinSolver<DIM>::LinSolver(Network<DIM> &nw, int NF, 
                          std::vector<Perturb<DIM> > &pert, 
                          std::vector<Measure<DIM> > &meas) {
        
    int p = 0;
    for(int m = 0; m < DIM; m++) {
        for(int n = m; n < DIM; n++) {
            if(m == n) {
                sym_mat_index(m, n) = p;
            } else {
                sym_mat_index(m, n) = p;
                sym_mat_index(n, m) = p;
            }
            
            p++;
        }
    }
    
    NNDOF = DIM * nw.NN;
    NADOF = nw.enable_affine ? DIM*(DIM+1)/2 : 0;
    
    NGDOF = 0;
    if(nw.fix_trans) {
        NGDOF += DIM;
    }
    if(nw.fix_rot) {
        NGDOF += DIM*(DIM-1) / 2;
    }
    
    NDOF = NNDOF + NADOF + NGDOF;
        
    this->nw = nw;
    this->NF = NF;
    this->pert = pert;
    this->meas = meas;
    allow_zero = false;
    
    setupPertMats();
    setupMeasMats();
    setupEqMat();    
    setupConMat();

    K = XVec::Ones(nw.NE);
    setupHessian();

    is_computed = false;
    
    HiQ.resize(nw.NE);
    have_HiQ.resize(nw.NE, false);

}

template<int DIM>
void LinSolver<DIM>::setK(RXVec K) {
    this->K = K;
    
    setupHessian();
    is_computed = false;
}

template<int DIM>
void LinSolver<DIM>::setAllowZero(bool allow_zero) {
    this->allow_zero = allow_zero;
}


//////////////////////////////////////////////////////
// System solver algorithms
//////////////////////////////////////////////////////

template<int DIM>
bool LinSolver<DIM>::computeHessian(LinSolverResult &result) {
    
    solver.compute(H);
    
    if (solver.info() != Eigen::Success) {
        result.success = false;
        result.msg = "Computing LU decomposition failed.";
        std::cout << result.msg << std::endl;
        return false;
    }
    
    is_computed = true;
    
    return true;
}

template<int DIM>
bool LinSolver<DIM>::computeInv(LinSolverState &state, LinSolverResult &result) {
    
    if(!is_computed && !computeHessian(result)) {
        return false;
    }
            
    for(int t = 0; t < NF; t++ ) {
          
        if(f[t].nonZeros() > 0 && NC[t] > 0) {
            
            state.Hif[t] = solver.solve(XMat(f[t]));
            if (solver.info() != Eigen::Success) {
                result.success = false;
                result.msg = "Solving H^{-1}f failed.";
                std::cout << result.msg << std::endl;
                return false;
            }
            
            state.HiC1[t] = solver.solve(XMat(C1[t]));
            if (solver.info() != Eigen::Success) {
                result.success = false;
                result.msg = "Solving H^{-1}C1 failed.";
                std::cout << result.msg << std::endl;
                return false;
            }
            
        } else if (NC[t] > 0) {
            
            state.HiC1[t] = solver.solve(XMat(C1[t]));
            if (solver.info() != Eigen::Success) {
                result.success = false;
                result.msg = "Solving H^{-1}C1 failed.";
                std::cout << result.msg << std::endl;
                return false;
            }
                         
        } else {
            
            state.Hif[t] = solver.solve(XMat(f[t]));
            if (solver.info() != Eigen::Success) {
                result.success = false;
                result.msg = "Solving H^{-1}f failed.";
                std::cout << result.msg << std::endl;
                return false;
            }
                        
        } 
    }
    
    return true;
    
}

template<int DIM>
bool LinSolver<DIM>::computeBlockInv(LinSolverState &state, 
                                     std::vector<XVec > &u, std::vector<XVec > &lamb, LinSolverResult &result) {
    
    u.resize(NF);
    lamb.resize(NF);
    
    for(int t = 0; t < NF; t++ ) {
                
        if(f[t].nonZeros() > 0 && NC[t] > 0) {

            lamb[t] = (C1[t].transpose() * state.HiC1[t]).inverse() * (-C1[t].transpose() * state.Hif[t] + C0[t]);
            u[t] = state.Hif[t] - state.HiC1[t] * lamb[t];
            
        } else if (NC[t] > 0) {
 
            lamb[t] = (C1[t].transpose() * state.HiC1[t]).inverse() * C0[t];
            u[t] = state.HiC1[t] * lamb[t];
                        
        } else {
            u[t] = state.Hif[t];
        } 
    }
    
    return true;
    
}


template<int DIM>
bool LinSolver<DIM>::computeInvUpdate(LinUpdate &up, LinSolverState &state1, LinSolverState &state2, bool save, LinSolverResult &result) {
    
    SMat U(NDOF, up.NdK);
    XMat HiU(NDOF, up.NdK);
    for(int i = 0; i < up.NdK; i++) {
        U.col(i) = Q.col(up.dK_edges[i]);
        
        if(have_HiQ[up.dK_edges[i]]) {
            HiU.col(i) = HiQ[up.dK_edges[i]];
        } else {
                        
            HiU.col(i) = solver.solve(XMat(U.col(i)));
            if (solver.info() != Eigen::Success) {
                result.success = false;
                result.msg = "Solving H^{-1}U failed.";
                std::cout << result.msg << std::endl;
                return false;
            }
            
            have_HiQ[up.dK_edges[i]] = true;
            HiQ[up.dK_edges[i]] = HiU.col(i);
            
        }
        
    }
    
    if(state1.hess_update) {        
        HiU += state1.dHi * U;
    }
    
    XMat A = up.dK.asDiagonal() * U.transpose() * HiU + XMat::Identity(up.NdK, up.NdK);
    
    double det = fabs(A.determinant());
        
    if(det < 1e-4) {
        result.success = false;
        // result.msg = "det: " + std::to_string(det) + " < 1e-4";
        std::cout << result.msg << std::endl;
        return false;
    }
    
    XMat Ai = A.inverse();
    
    std::vector<XVec > u(NF);
    std::vector<XVec > lamb(NF);
    
    for(int t = 0; t < NF; t++ ) {
                
        if(f[t].nonZeros() > 0 && NC[t] > 0) {
            
            state2.Hif[t] = state1.Hif[t] - HiU * up.dK.asDiagonal() * Ai * (HiU.transpose() * f[t]);
            state2.HiC1[t] = state1.HiC1[t] - HiU * up.dK.asDiagonal() * Ai * (HiU.transpose() * C1[t]);
            
        } else if (NC[t] > 0) {
                        
            state2.HiC1[t] = state1.HiC1[t] - HiU * up.dK.asDiagonal() * Ai * (HiU.transpose() * C1[t]);
            
        } else {
            
            state2.Hif[t] = state1.Hif[t] - HiU * up.dK.asDiagonal() * Ai * (HiU.transpose() * f[t]);
            
        } 
    }
    
    if(save) {
        if(!state2.hess_update) {
            state2.hess_update = true;
            state2.K = K;
            state2.dH.resize(NDOF, NDOF);
            state2.dHi = XMat::Zero(NDOF, NDOF);
        }
  
        for(int i = 0; i < up.NdK; i++) {
            state2.K(up.dK_edges[i]) += up.dK(i);
        }
        
        state2.dH += U * (up.dK.asDiagonal() * U.transpose());
        state2.dHi += - HiU * up.dK.asDiagonal() * Ai * HiU.transpose();
                
    }
    
    return true;
    
}


template<int DIM>
bool LinSolver<DIM>::computeResult(std::vector<XVec > &u, std::vector<XVec > &lamb, LinSolverResult &result) {
        
    for(int t = 0; t < NF; t++ ) {
        
        if(meas[t].measure_disp) {
            result.disp[t] = u[t].segment(0, NNDOF);
        }
        
        if(meas[t].measure_strain) {
            result.strain[t] = u[t].segment(NNDOF, NADOF);
        }
        
        if(meas[t].measure_lamb) {
            result.lamb[t] = lamb[t];
        }
        
        int offset = 0;
        XVec m = M[t].transpose() * u[t];
        result.ostrain[t] = m.segment(0, meas[t].N_ostrain);
        
        offset += meas[t].N_ostrain;
        
        if(meas[t].measure_affine_strain) {
            result.affine_strain[t] = m.segment(offset, DIM*(DIM+1)/2);
            offset += DIM*(DIM+1)/2;
        }
        
        XVec ostress = m.segment(offset, meas[t].N_ostress);
        for(int i = 0; i < meas[t].N_ostress; i++) {
            ostress(i) *= K(meas[t].ostress_edges[i]);
        }
        result.ostress[t] = ostress;
        
        XVec olambda(meas[t].N_olambda);
        for(int i = 0; i < meas[t].N_olambda; i++) {
            olambda(i) = lamb[t](meas[t].olambdai[i]);
        }
        
    }
    
    return true;
    
}

template<int DIM>
bool LinSolver<DIM>::computeMeas(LinSolverResult &result) {
    
    int NM_tot = std::accumulate(NM.begin(), NM.end(), 0);
    result.meas.resize(NM_tot);
        
    int index = 0;
    for(int t = 0; t < result.NF; t++) {
        result.meas.segment(index, result.ostrain[t].size()) = result.ostrain[t];
        index += result.ostrain[t].size();
        
        result.meas.segment(index, result.ostress[t].size()) = result.ostress[t];
        index += result.ostress[t].size();
        
        result.meas.segment(index, result.affine_strain[t].size()) = result.affine_strain[t];
        index += result.affine_strain[t].size();
        
        result.meas.segment(index, result.affine_stress[t].size()) = result.affine_stress[t];
        index += result.affine_stress[t].size();
        
        result.meas.segment(index, result.olambda[t].size()) = result.olambda[t];
        index += result.olambda[t].size();
    }
        
    return true;
}


template<int DIM>
LinSolverResult* LinSolver<DIM>::solve() {
    
    LinSolverResult *result = new LinSolverResult(NF);
    
    LinSolverState state(NF);
    if(!computeInv(state, *result)) {
        return result;
    }
        
    std::vector<XVec > u;
    std::vector<XVec > lamb;
    if(!computeBlockInv(state, u, lamb, *result)) {
        return result;
    }
        
    if(!computeResult(u, lamb, *result)) {
        return result;
    }
    
    return result;
     
}


template<int DIM>
LinSolverResult* LinSolver<DIM>::solve(LinSolverState &state) {
    
    LinSolverResult *result = new LinSolverResult(NF);
    
    std::vector<XVec > u;
    std::vector<XVec > lamb;
    if(!computeBlockInv(state, u, lamb, *result)) {
        return result;
    }
    
    if(!computeResult(u, lamb, *result)) {
        return result;
    }
    
    return result;
    
}


template<int DIM>
LinSolverResult* LinSolver<DIM>::solve(LinUpdate &up) {
    
    LinSolverResult *result = new LinSolverResult(NF);
    
    LinSolverState state(NF);
    if(!computeInv(state, *result)) {
        return result;
    }
    
    if(!computeInvUpdate(up, state, state, false, *result)) {
        return result;
    }
        
    std::vector<XVec > u;
    std::vector<XVec > lamb;
    if(!computeBlockInv(state, u, lamb, *result)) {
        return result;
    }
    
    if(!computeResult(u, lamb, *result)) {
        return result;
    }
    
    return result;
     
}


template<int DIM>
LinSolverResult* LinSolver<DIM>::solve(LinUpdate &up, LinSolverState &state) {
    
    LinSolverResult *result = new LinSolverResult(NF);
    
    LinSolverState state2(NF);    
    if(!computeInvUpdate(up, state, state2, false, *result)) {
        return result;
    }
    
    std::vector<XVec > u;
    std::vector<XVec > lamb;
    if(!computeBlockInv(state2, u, lamb, *result)) {
        return result;
    }
    
    if(!computeResult(u, lamb, *result)) {
        return result;
    }
    
    return result;
     
}


template<int DIM>
LinSolverState* LinSolver<DIM>::getSolverState() {
    
    
    LinSolverResult result(NF);
    LinSolverState *state = new LinSolverState(NF);
    computeInv(*state, result);
    
    return state;
    
}

template<int DIM>
void LinSolver<DIM>::updateSolverState(LinUpdate &up, LinSolverState &state) {
    
    LinSolverResult result(NF);
    
    computeInvUpdate(up, state, state, true, result);
     
}



//////////////////////////////////////////////////////
// Matrix Setup
//////////////////////////////////////////////////////



template<int DIM>
void LinSolver<DIM>::setupPertMats() {
    
    NC.resize(NF);
    C0.resize(NF);
    C1.resize(NF);
    f.resize(NF);
    
    for(int t = 0; t < NF; t++) {
        NC[t] = pert[t].N_istrain + DIM*pert[t].NN_fix;
        if(pert[t].apply_affine_strain) {
            NC[t] += DIM*(DIM+1)/2;
        }
        
        // Apply input strains and affine deformations as constraints in an bordered Hessian
        C0[t] = XVec::Zero(NC[t]);
        C1[t].resize(NDOF, NC[t]);
        std::vector<Trip> C1_trip_list;
        // Input strain constraints
        for(int e = 0; e < pert[t].N_istrain; e++) {
            int ei =  pert[t].istrain_nodesi[e];
            int ej =  pert[t].istrain_nodesj[e];
            
            
            DVec Xij = pert[t].istrain_vec.template segment<DIM>(DIM*e);
            DVec Xhatij = Xij.normalized();

            C0[t](e) = pert[t].istrain(e);

            // Choose between strain or extension implementation
            double l0 = pert[t].is_extension ? 1.0 :  Xij.norm();
            for(int m = 0; m < DIM; m++) {
                C1_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m) / l0));
                C1_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m) / l0));
            }
  
            // Components of constraint that depend on affine dofs if enabled
            // C_1^T u = Xhat \cdot du + Xhat Gamma X
            if(nw.enable_affine) {

                SDVec XhatX = SDVec::Zero();
                for(int m = 0; m < DIM; m++) {
                    for(int n = 0; n < DIM; n++) {
                            XhatX(sym_mat_index(m, n)) += Xhatij(m)*Xij(n);
                    }
                }
                                
                for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                    C1_trip_list.push_back(Trip(NNDOF+m, e, XhatX(m) / l0));
                }
            }
        }
        
        // Affine deformation constraints
        if(pert[t].apply_affine_strain) {
        
            SDVec Gamma = SDVec::Zero();
            for(int m = 0; m < DIM; m++) {
                for(int n = 0; n < DIM; n++) {
                    Gamma(sym_mat_index(m, n)) += pert[t].strain_tensor(m, n);                    
                }
            }

            C0[t].segment<DIM*(DIM+1)/2>(pert[t].N_istrain) = Gamma;
            
            for(int m = 0; m < DIM; m++) {
                for(int n = 0; n < DIM; n++) {
                    C1_trip_list.push_back(Trip(NNDOF+sym_mat_index(m, n), 
                                            pert[t].N_istrain+sym_mat_index(m, n), 1.0));                 
                }
            }           
        }
        
        
        
        // Fixed nodes
        for(int n = 0; n < pert[t].NN_fix; n++) {
            int offset = pert[t].N_istrain;
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
        std::vector<Trip> f_trip_list;

        for(int e = 0; e < pert[t].N_istress; e++) {

            int ei =  pert[t].istress_nodesi[e];
            int ej =  pert[t].istress_nodesj[e];

            DVec Xij = pert[t].istress_vec.template segment<DIM>(DIM*e);
            DVec Xhatij = Xij.normalized();

            // Choose between tension and stress (must convert either to a force.
            // Tension already has correct units while stress must be multiplied by a length
            double l0 = pert[t].is_tension ? 1.0 : Xij.norm();
            double sigma = pert[t].istress(e) * l0;
            
            DVec force = sigma * Xhatij;

            for(int m = 0; m < DIM; m++) {
                f_trip_list.push_back(Trip(DIM*ei+m, 0, -force(m)));
                f_trip_list.push_back(Trip(DIM*ej+m, 0, force(m)));
            }
        }

        if(pert[t].apply_affine_stress) {

            for(int m = 0; m < DIM; m++) {
                for(int n = 0; n < DIM; n++) {
                    f_trip_list.push_back(Trip(NNDOF+sym_mat_index(m, n), 0, pert[t].stress_tensor(m, n)));
                }
            }

        }
        
        
        f[t].setFromTriplets(f_trip_list.begin(), f_trip_list.end());
    }
}

template<int DIM>
void LinSolver<DIM>::setupMeasMats() {
    
    NM.resize(NF);
    M.resize(NF);

    for(int t = 0; t < NF; t++) {
    
        NM[t] = meas[t].N_ostrain + meas[t].N_ostress;
        int NMA = 0;
        if(meas[t].measure_affine_strain) {
            NMA = DIM*(DIM+1)/2;
            NM[t] += NMA;
        }

        M[t].resize(NDOF, NM[t]);
        std::vector<Trip> M_trip_list;

        // Output strain responses
        for(int e = 0; e < meas[t].N_ostrain; e++) {
            int ei =  meas[t].ostrain_nodesi[e];
            int ej =  meas[t].ostrain_nodesj[e];

            DVec Xij = meas[t].ostrain_vec.template segment<DIM>(DIM*e);
            DVec Xhatij = Xij.normalized();

            // Choose between strain or extension implementation
            double l0 = meas[t].is_extension ? 1.0 : Xij.norm();
            for(int m = 0; m < DIM; m++) {
                M_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m) / l0));
                M_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m) / l0));
            }

            if(nw.enable_affine) {

                SDVec XhatX = SDVec::Zero();
                for(int m = 0; m < DIM; m++) {
                    for(int n = 0; n < DIM; n++) {
                        XhatX(sym_mat_index(m, n)) += Xhatij(m)*Xij(n);
                    }
                }
                
                for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                    M_trip_list.push_back(Trip(NNDOF+m, e, XhatX(m) / l0));
                }
            }
        }
        
        // Affine deformation responses
        if(meas[t].measure_affine_strain) {           
            for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                M_trip_list.push_back(Trip(NNDOF+m, meas[t].N_ostrain+m, 1.0));
            }             
        }
        
        // Output stress responses
        for(int e = 0; e < meas[t].N_ostress; e++) {
            
            
            int ei =  nw.edgei[meas[t].ostress_edges[e]];
            int ej =  nw.edgej[meas[t].ostress_edges[e]];

            DVec Xij = nw.node_pos.template segment<DIM>(DIM*ej)
                - nw.node_pos.template segment<DIM>(DIM*ei);
            DVec Xhatij = Xij.normalized();

            double l0 = pert[t].is_tension ? 1.0 : Xij.norm();
            for(int m = 0; m<DIM; m++) {
                M_trip_list.push_back(Trip(DIM*ei+m, meas[t].N_ostrain + NMA + e, -Xhatij(m) * l0));
                M_trip_list.push_back(Trip(DIM*ej+m, meas[t].N_ostrain + NMA + e, Xhatij(m) * l0));
            }

            if(nw.enable_affine) {

                SDVec XhatX = SDVec::Zero();
                for(int m = 0; m < DIM; m++) {
                    for(int n = 0; n < DIM; n++) {
                            XhatX(sym_mat_index(m, n)) += Xhatij(m)*Xij(n);
                    }
                }
                
                for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                    M_trip_list.push_back(Trip(NNDOF+m, meas[t].N_ostrain + NMA + e, XhatX(m) * l0));
                }
            }
            
        }
        
        M[t].setFromTriplets(M_trip_list.begin(), M_trip_list.end());
        
    }
    
}

template<int DIM>
void LinSolver<DIM>::setupEqMat() {
    
    std::vector<Trip> Q_trip_list;
    Q.resize(NDOF, nw.NE);
    // Harmonic spring interactions
    for(int e = 0; e < nw.NE; e++) {
                
        int ei =  nw.edgei[e];
        int ej =  nw.edgej[e];
        
        DVec Xij =  nw.bvecij.template segment<DIM>(DIM*e);
        DVec Xhatij = Xij.normalized();
        
        for(int m = 0; m < DIM; m++) {
            Q_trip_list.push_back(Trip(DIM*ei+m, e, -Xhatij(m)));
            Q_trip_list.push_back(Trip(DIM*ej+m, e, Xhatij(m)));
        }
        
        if(nw.enable_affine) {
            
            SDVec XhatX = SDVec::Zero();
            for(int m = 0; m < DIM; m++) {
                for(int n = 0; n < DIM; n++) {
                        XhatX(sym_mat_index(m, n)) += Xhatij(m)*Xij(n);
                }
            }
            
            for(int m = 0; m < DIM*(DIM+1)/2; m++) {
                Q_trip_list.push_back(Trip(NNDOF+m, e, XhatX(m)));
            }

        }
    } 
    
    Q.setFromTriplets(Q_trip_list.begin(), Q_trip_list.end());
    
}

template<int DIM>
void LinSolver<DIM>::setupConMat() {
    
    G.resize(NDOF, NDOF);        
    std::vector<Trip> G_trip_list;
    
    int offset = NNDOF + NADOF;
    // Fix translational dofs
    if(nw.fix_trans) {
        for(int d = 0; d < DIM; d++) {
           for(int i = d; i < NNDOF; i+=DIM) {               
               G_trip_list.push_back(Trip(i, offset+d, -1.0));
               G_trip_list.push_back(Trip(offset+d, i, -1.0));
           }
        }
        
        offset += DIM;
    }
    
    // Fix rotational dofs
    if(nw.fix_rot) {
                
        // center of mass
        DVec COM = DVec::Zero();
        for(int i = 0; i < nw.NN; i++) {
            COM += nw.node_pos.template segment<DIM>(DIM*i);
                    
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
                    DVec pos = nw.node_pos.template segment<DIM>(DIM*i) - COM;
                    global_rot.segment<DIM>(DIM*i) = dRdTheta * pos;
                }
                
                global_rot.normalize();
                
                for(int i = 0; i < NNDOF; i++) {
                    G_trip_list.push_back(Trip(i, offset+d, -global_rot(i)));
                    G_trip_list.push_back(Trip(offset+d, i, -global_rot(i)));
                }
                
                d++;
            }
        }
        
    }

    G.setFromTriplets(G_trip_list.begin(), G_trip_list.end());
    
    
    
}

template<int DIM>
void LinSolver<DIM>::setupHessian() {
        
    H = Q * (K.asDiagonal() * Q.transpose()) + G;
            
    // If allow for zero modes corresponding to constraints,
    // then add extra energy term corresponding to constraint
    if(allow_zero && NF == 1 && C1[0].nonZeros() > 0) {
        for(int i = 0; i < NC[0]; i++) {
            H += C1[0].col(i) * C1[0].col(i).transpose();
        }
    }
}








#endif // LINSOLVER

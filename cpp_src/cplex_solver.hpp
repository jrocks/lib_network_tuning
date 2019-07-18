#ifndef CPLEXSOLVER
#define CPLEXSOLVER
    
#include "util.hpp"
#include "network.hpp"
#include "perturb.hpp"
#include "measure.hpp"
    
#include <ilcplex/ilocplex.h>
    
#include <pybind11/pybind11.h>
namespace py = pybind11;
    
#include "lin_solver.hpp"
#include "lin_solver_result.hpp"
    
    
template <int DIM>
class CPLEXSolver {
    
    DEIGEN(DIM);
    DSYMEIGEN(DIM);
    
    public:
    
        const int dim;
    
        // Network object
        Network<DIM> nw;
    
        // Number of dof in Hessian
        int NDOF;
        // Number of node dofs
        int NNDOF;
    
    
        // Equilibrium Matrix
        SMat Q;
        // Vector of interaction strengths
        XVec K;
        // Hessian matrix
        SMat H;
    
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
    
        // List of measurements
        std::vector<Measure<DIM> > meas;
        // Measurement matrices
        std::vector<SMat >  M;
    
    
        IloEnv env;
        IloModel mod;
        IloNumVarArray vars;
        IloObjective obj;
        IloRangeArray con;
        IloCplex solver;
    
    public:
        CPLEXSolver(Network<DIM> &nw, int NF, 
                  std::vector<Perturb<DIM> > &pert, 
                  std::vector<Measure<DIM> > &meas);
        ~CPLEXSolver() {
            env.end();
        };
    
        // Set interaction strengths
        void setK(RXVec K);
    
    private:
        // Setup various matrices
        void setupPertMats();
        void setupMeasMats();
        void setupEqMat();
        void setupHessian();
        void setupCPLEX();
    
        // Decompose solver solution into displacement, lagrange multipliers, etc.
        bool computeResult(LinSolverResult &result);

    public:
        
        // Solve using current values of interaction strengths
        LinSolverResult solve();
        // Solve with a linear update to Hessian
        LinSolverResult solve(LinUpdate &up);
    
        // Solve for gradient using current values of interaction strengths
        LinSolverResult solveGrad();
    
        // Apply linear update to Hessian
        bool setUpdate(LinUpdate &up, bool undo=false);
    
        // Assemble vector of measurements for objective function
        bool computeMeas(LinSolverResult &result); 
        
};




//////////////////////////////////////////////////////
// Function Implementations
//////////////////////////////////////////////////////

template<int DIM>
CPLEXSolver<DIM>::CPLEXSolver(Network<DIM> &nw, int NF, 
                          std::vector<Perturb<DIM> > &pert, 
                          std::vector<Measure<DIM> > &meas) : dim(DIM) {
        
    NNDOF = DIM * nw.NN;
    
    NDOF = NNDOF;
        
    this->nw = nw;
    this->NF = NF;
    this->pert = pert;
    this->meas = meas;
        
    setupPertMats();
    setupMeasMats();
    setupEqMat();    

    K = XVec::Ones(nw.NE);
    setupHessian();
    setupCPLEX();

}

template<int DIM>
void CPLEXSolver<DIM>::setK(RXVec K) {
    this->K = K;
    setupHessian();
    setupCPLEX();
}


template<int DIM>
bool CPLEXSolver<DIM>::computeResult(LinSolverResult &result) {
    
    
    IloNumArray vals(env);
    solver.getValues(vals, vars);
    
    
        
    for(int t = 0; t < NF; t++ ) {
        
        XVec disp = XVec::Zero(NDOF);
        for(int i = 0; i < NDOF; i++) {
            disp[i] = vals[i];
        }

        XVec lamb = XVec::Zero(NC[t]);
        for(int i = 0; i < NC[t]; i++) {
            lamb[i] = vals[NDOF+i];
        }
        
        if(meas[t].measure_disp) {
            result.disp[t] = disp.segment(0, NNDOF);
        }
        
        if(meas[t].measure_lamb) {
            result.lamb[t] = lamb;
        }
        
        int offset = 0;
        XVec m = M[t].transpose() * disp;
        result.ostrain[t] = m.segment(0, meas[t].N_ostrain);
        
        offset += meas[t].N_ostrain;
        
        if(meas[t].measure_affine_strain) {
            result.affine_strain[t] = m.segment(offset, DIM*(DIM+1)/2);
            offset += DIM*(DIM+1)/2;
        }
        
        std::unordered_map<int,double> K_map;
        for(int i = 0; i < meas[t].N_ostress; i++) {
            K_map.emplace(meas[t].ostress_edges[i], K(meas[t].ostress_edges[i]));
        }
        
        XVec ostress = m.segment(offset, meas[t].N_ostress);
        for(int i = 0; i < meas[t].N_ostress; i++) {
            ostress(i) *= K_map.at(meas[t].ostress_edges[i]);
        }
        result.ostress[t] = ostress;
        
        XVec olambda(meas[t].N_olambda);
        for(int i = 0; i < meas[t].N_olambda; i++) {
            olambda(i) = lamb(meas[t].olambdai[i]);
        }
        
    }
    
    vals.end();
    
    return true;
    
}


template<int DIM>
bool CPLEXSolver<DIM>::computeMeas(LinSolverResult &result) {
    
    int NM_tot = std::accumulate(NM.begin(), NM.end(), 0);
    result.meas.resize(NM_tot);
        
    int index = 0;
    for(int t = 0; t < result.NF; t++) {
        result.meas.segment(index, result.ostrain[t].size()) = result.ostrain[t];
        index += result.ostrain[t].size();
        
        result.meas.segment(index, result.ostress[t].size()) = result.ostress[t];
        index += result.ostress[t].size();
        
    }
        
    return true;
}

template<int DIM>
bool CPLEXSolver<DIM>::setUpdate(LinUpdate &up, bool undo) {
    
    SMat dH(NDOF, NDOF);
    SMat U(NDOF, up.NdK);
    for(int i = 0; i < up.NdK; i++) {
        U.col(i) = Q.col(up.dK_edges[i]);
        
    }
    
    dH = (undo ? -1 : 1) * U * up.dK.asDiagonal() * U.transpose();
    
    
    for(int i = 0; i < NDOF; i++) {
        IloExpr rowExpr = con[i].getExpr();
        
        IloNumVarArray up_vars;
        IloNumArray up_vals;
        // H is symmetric, so use columns of H instead
        for(SMat::InnerIterator it(dH, i); it; ++it) {
            // up_vars.add(vars[it.row()]);
            // up_vals.add(H.coeffRef(i, it.row())+it.value());
            rowExpr += it.value() * vars[it.row()];
        }
        
        // ((IloExpr)con[i].getExpr()).setLinearCoefs(up_vars, up_vals);
        
        con[i].setExpr(rowExpr);
        
        rowExpr.end();
    }
        
    return true;
}


template<int DIM>
LinSolverResult CPLEXSolver<DIM>::solve() {
      
    LinUpdate up;
    
    return solve(up);
         
}


template<int DIM>
LinSolverResult CPLEXSolver<DIM>::solve(LinUpdate &up) {
      
    
    setUpdate(up);
    
    solver.solve();
    
    // std::cout << "solution status = " << solver.getStatus() << std::endl;
    // solver.out() << "solution status = " << solver.getStatus() << std::endl;
    // solver.out() << std::endl;
    // solver.out() << "cost   = " << solver.getObjValue() << std::endl;
    // for (int i = 0; i < vars.getSize(); i++) {
    //     solver.out() << "  x" << i << " = " << solver.getValue(vars[i]) << std::endl;
    // }
    
    // py::print("solution status = ", solver.getStatus());
    
    
    
    
    LinSolverResult result(NF);
    
    if(!computeResult(result)) {
        return result;
    }
    
    result.msg = "Solve successful.";
    
    setUpdate(up, true);
    
    return result;
         
}

template<int DIM>
LinSolverResult CPLEXSolver<DIM>::solveGrad() {
    
    LinSolverResult result(NF);
    
    return result;
     
}

//////////////////////////////////////////////////////
// Matrix Setup
//////////////////////////////////////////////////////



template<int DIM>
void CPLEXSolver<DIM>::setupPertMats() {
    
    NC.resize(NF);
    C0.resize(NF);
    C1.resize(NF);
    
    for(int t = 0; t < NF; t++) {
        NC[t] = pert[t].N_istrain;
        
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

        }
        
        C1[t].setFromTriplets(C1_trip_list.begin(), C1_trip_list.end());

    }
}

template<int DIM>
void CPLEXSolver<DIM>::setupMeasMats() {
    
    NM.resize(NF);
    M.resize(NF);

    for(int t = 0; t < NF; t++) {
    
        NM[t] = meas[t].N_ostrain;

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

        }
        
        M[t].setFromTriplets(M_trip_list.begin(), M_trip_list.end());
        
    }
    
}

template<int DIM>
void CPLEXSolver<DIM>::setupEqMat() {
    
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
        
    } 
    
    Q.setFromTriplets(Q_trip_list.begin(), Q_trip_list.end());
    
}

template<int DIM>
void CPLEXSolver<DIM>::setupHessian() {
        
    H = Q * (K.asDiagonal() * Q.transpose());
    
}

template<int DIM>
void CPLEXSolver<DIM>::setupCPLEX() {
        
    
    int t = 0;
    
    // Initialize model to solve
    mod = IloModel(env);
    
    // Initialize array of variables
    vars = IloNumVarArray(env, NDOF+NC[0], -IloInfinity, IloInfinity, ILOFLOAT);
    
    // Expression for objective function
    IloExpr objExpr(env);
     // Add quadratic components to objective function
    for(int i = 0; i < NDOF; i++) {
        objExpr += 0.5*vars[i]*vars[i];
    }
    
    obj = IloMinimize(env, objExpr);
    mod.add(obj);
    objExpr.end();
    
    // Setup constraint array
    con = IloRangeArray(env);
    
    // Add constraint Hu=f row by row
    SMat C1T = C1[t].transpose();
    for(int i = 0; i < NDOF; i++) {
        IloExpr rowExpr(env);
        
        // H is symmetric, so use columns of H instead
        for(SMat::InnerIterator it(H, i); it; ++it) {
            rowExpr += it.value() * vars[it.row()];
        }
        
        // Add constraint columns
        for(SMat::InnerIterator it(C1T, i); it; ++it) {
            rowExpr += -it.value() * vars[NDOF+it.row()];
        }
        
        con.add(rowExpr == 0.0);
        
        rowExpr.end();
    }
    
    // Add constraint C_1^Tu=C_0 row by row
    for(int i = 0; i < NC[t]; i++) {
        IloExpr rowExpr(env);
        
        for(SMat::InnerIterator it(C1[t], i); it; ++it) {
            rowExpr += -it.value() * vars[it.row()];
        }
        
        con.add(rowExpr == -C0[t](i));
        
        rowExpr.end();
    }
    
    mod.add(con);
    
    solver = IloCplex(mod);
    solver.setParam(IloCplex::Param::Threads, 1);
    
    solver.setOut(env.getNullStream());
    
    // solver.exportModel("test_cplex.lp");
    
}




    
    
    
#endif
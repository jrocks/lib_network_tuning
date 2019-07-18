#ifndef ALGLIBTUNER
#define ALGLIBTUNER

#include "util.hpp"
#include "lin_solver.hpp"
#include "objective_function.hpp"
    
#include "stdafx.h"
#include "optimization.h"
    
#include <pybind11/pybind11.h>
namespace py = pybind11;
    
    
template <int DIM>  
class MinChangeTuner {
          
    
private:
    
    double objFunc(RXVec &K, RXVec &K_init);
    XVec objGrad(RXVec &K, RXVec &K_init);
    
    
public:
    XVec tune(LinSolver<DIM> &solver, LeastSquaresObjFunc &obj_func, RXVec K_init, RXVec K_min, RXVec K_max, double tol=1e-8, bool verbose=true);
    
    
    
    
};

template <int DIM>     
struct Params {
        
        int NE;
        
        RXVec* K_init;
        
        int NM_tot;
    
        MinChangeTuner<DIM>* tuner;
    
        LinSolver<DIM> *solver;
    
        LeastSquaresObjFunc* obj_func;
        
};

template <int DIM>
double MinChangeTuner<DIM>::objFunc(RXVec &K, RXVec &K_init) {
    
    return 0.5 * (K-K_init).squaredNorm();
    
}

template <int DIM>
XVec MinChangeTuner<DIM>::objGrad(RXVec &K, RXVec &K_init) {
    
    return K-K_init;
    
}
    

template <int DIM>
XVec MinChangeTuner<DIM>::tune(LinSolver<DIM> &solver, LeastSquaresObjFunc &obj_func, RXVec K_init, RXVec K_min, RXVec K_max, double tol, bool verbose) {    
    
    alglib::real_1d_array K0;
    // XVec K_tmp =  K_init + 1e-1*XVec::Ones(solver.nw.NE);
    K0.setcontent(solver.nw.NE, K_init.data());
    
    // Algorithm state
    alglib::minnlcstate state;
    // Create algorithm state
    alglib::minnlccreate(solver.nw.NE, K0, state);
    // alglib::minnlccreatef(solver.nw.NE, K0, 1e-6, state);
    
    // Penalty parameter
    double rho = 1e3;
    // Number of iterations to update lagrange multiplier
    alglib::ae_int_t outerits = 1;
    // Set augmented lagrangian parameters
    alglib::minnlcsetalgoaul(state, rho, outerits);

    // Tolerance on gradient, change in function and change in x
    double epsg = 0.0*tol;
    double epsf = 0.0*tol;
    double epsx = tol;
    // Max iterations
    alglib::ae_int_t maxits = 0;
    // Set stopping conditions
    alglib::minnlcsetcond(state, epsg, epsf, epsx, maxits);
    
    // Scale of variables
    alglib::real_1d_array s;
    s.setcontent(solver.nw.NE, K_max.data());
    // Set scale of variables
    alglib::minnlcsetscale(state, s);
    
    // Set boundary constraints
    alglib::real_1d_array bndl;
    bndl.setcontent(solver.nw.NE, K_min.data());
    alglib::real_1d_array bndu;
    bndu.setcontent(solver.nw.NE, K_max.data());
    alglib::minnlcsetbc(state, bndl, bndu);
    
    // Set number of nonlinear equality and inequality constraints
    int NM_tot = std::accumulate(solver.NM.begin(), solver.NM.end(), 0);
    // int NM_tot = 0;
    alglib::minnlcsetnlc(state, NM_tot, 0);
    
    // alglib::minnlcsetgradientcheck(state, 1e-4);
    
    Params<DIM> params;
    params.NE = solver.nw.NE;
    params.K_init = &K_init;
    params.NM_tot = NM_tot;
    params.tuner = this;
    params.solver = &solver;
    params.obj_func = &obj_func;
    
    auto jac = [](const alglib::real_1d_array &x, alglib::real_1d_array &func, 
                                   alglib::real_2d_array &jac, void *ptr) {
        
        Params<DIM>* params = static_cast<Params<DIM>*>(ptr);
        
        XVecConstMap K(x.getcontent(), params->NE);
        
        XVecMap f(func.getcontent(), 1+params->NM_tot);
        
        f(0) = params->tuner->objFunc(K, *(params->K_init));  
        
        XVecMap fg(&jac[0][0], params->NE);
        fg = params->tuner->objGrad(K, *(params->K_init));
        
        params->solver->setK(K);
        LinSolverResult result = params->solver->solveMeasGrad();
//         // LinSolverResult result = params->solver->solveMeas();
        
        // py::print("Meas", result.meas);
        // py::print("Meas grad", result.meas_grad);
        
        f.segment(1, params->NM_tot) = params->obj_func->evalRes(result.meas);
        XMat res_g = params->obj_func->evalResGrad(result.meas, result.meas_grad);
        
        for(int i = 0; i < params->NM_tot; i++) {
            XVecMap rg(&jac[1+i][0], params->NE);
            rg = res_g.row(i);
        }
                
        // py::print("raw f", params->obj_func->evalRes(result.meas));
        // py::print("raw g", params->obj_func->evalResGrad(result.meas, result.meas_grad));
        
        // f.segment(1, params->NM_tot) = XVec::Zero(params->NM_tot);
        // g.block(1, 0, params->NM_tot, params->NE) = XMat::Zero(params->NM_tot, params->NE);
        
        
        // py::print(g);
        
        // func.setcontent(f.size(), &f.data()[0]);
        // jac.setcontent(g.rows(), g.cols(), g.data());

        
//         py::print(func[0], func[1]);
        
//         for(int i = 0; i < g.rows(); i++) {
            
//             for(int j = 0; j < g.cols(); j++) {
//                 py::print(jac[i][j]);
//             }
//         }
        
        
        // py::print("K", K);
        // py::print("f", f);
        // py::print("g", g);
        
        
        
//         for(int i = 0; i < f.size(); i++) {
//             py::print("f", i, f(i), func[i]);
//         }
        
     
//         for(int j = 0; j < fg.size(); j++) {
//             py::print("g", 0, j, fg(j), jac[0][j]);
//         }
                                                             
                                                             
//         for(int i = 0; i < res_g.rows(); i++) {
            
//             for(int j = 0; j < res_g.cols(); j++) {
//                 py::print("g", i+1, j, res_g(i, j), jac[i+1][j]);
//             }
//         }
        
        std::cout << "F: " << func[0] << "\tC: " << func[1] << std::endl;
        // std::cout << "F: " << func[0] << std::endl;
        
    };
        
    
//     XVec tmpvec = XVec::Zero(1+NM_tot);
//     XVec tmpmat = XVec::Zero((1+NM_tot) * solver.nw.NE);
    
//     alglib::real_1d_array x;
    
//     // XVec K_tmp =  K_init + 1e-2*XVec::Ones(solver.nw.NE);
    
//     x.setcontent(solver.nw.NE, K_init.data());
//     alglib::real_1d_array func;
//     func.setcontent(1+NM_tot, tmpvec.data());
//     alglib::real_2d_array grad;
//     grad.setcontent((1+NM_tot), solver.nw.NE, tmpmat.data());
//     void *ptr = &params;
    
//     jac(x, func, grad, ptr);
    
//     py::print("Initial: ", func[0], func[1]);
    
//     for(int i = 0; i < solver.nw.NE; i++) {
//             py::print(i, grad[0][i], grad[1][i]);
//     }
    
//     alglib::real_1d_array func2;
//     func2.setcontent(1+NM_tot, tmpvec.data());
//     alglib::real_2d_array grad2;
//     grad2.setcontent((1+NM_tot), solver.nw.NE, tmpmat.data());
    
//     for(int i = 0; i < solver.nw.NE; i++) {
//         x(i) += 1e-8;
//         jac(x, func2, grad2, ptr);
//         py::print(i, (func2[0] - func[0])/1e-8, grad[0][i], (func2[1] - func[1])/1e-8, grad[1][i], (func2[1] - func[1])/1e-8 - grad[1][i]);
        
//         x(i) -= 1e-8;
//     }
    
    
    // Perform optimization
    alglib::minnlcoptimize(state, jac, NULL, &params);
        
    // Result 
    alglib::real_1d_array K1;
    // Optimization report
    alglib::minnlcreport rep;
    // Retrieve optimization results
    alglib::minnlcresults(state, K1, rep);
    
    if(verbose) {
    
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

    XVecMap K_map(K1.getcontent(), solver.nw.NE);
    
    return K_map;
    
}
    
#endif // ALGLIBTUNER
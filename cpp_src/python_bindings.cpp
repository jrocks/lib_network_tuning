#include "util.hpp"
#include "network.hpp"
#include "perturb.hpp"
#include "measure.hpp"
    
#include "lin_solver.hpp"
#include "lin_solver_state.hpp"
#include "lin_solver_result.hpp"
#include "objective_function.hpp"
    
#ifdef USECPLEX
#include "cplex_solver.hpp"
#endif
    
#ifdef USEALGLIB
#include "alglib_tuner.hpp"
#endif
    
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
    
namespace py = pybind11;


template <int DIM> void init(py::module &m) {
        
    py::class_<Network<DIM> >(m, (std::string("Network")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<int, RXVec, int, std::vector<int> &, std::vector<int> &, RXVec>())
        .def("setInteractions", &Network<DIM>::setInteractions)
        .def_readonly_static("dim", &Network<DIM>::dim)
        .def_readwrite("NN", &Network<DIM>::NN)
        .def_readwrite("node_pos", &Network<DIM>::node_pos)
        .def_readwrite("NE", &Network<DIM>::NE)
        .def_readwrite("edgei", &Network<DIM>::edgei)
        .def_readwrite("edgej", &Network<DIM>::edgej)
        .def_readwrite("L", &Network<DIM>::L)
        .def_readwrite("enable_affine", &Network<DIM>::enable_affine)
        .def_readwrite("fix_trans", &Network<DIM>::fix_trans)
        .def_readwrite("fix_rot", &Network<DIM>::fix_rot)
        .def_readwrite("bvecij", &Network<DIM>::bvecij)
        .def_readwrite("eq_length", &Network<DIM>::eq_length)
        .def_readwrite("K", &Network<DIM>::K);
    
    py::class_<Perturb<DIM> >(m, (std::string("Perturb")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<>())
        .def("setInputStrain", &Perturb<DIM>::setInputStrain)
        .def("setInputStress", &Perturb<DIM>::setInputStress)
        .def("setInputAffineStrain", &Perturb<DIM>::setInputAffineStrain)
        .def("setInputAffineStress", &Perturb<DIM>::setInputAffineStress)
        .def("setFixedNodes", &Perturb<DIM>::setFixedNodes)
        .def_readonly_static("dim", &Perturb<DIM>::dim)
        .def_readwrite("N_istrain", &Perturb<DIM>::N_istrain)
        .def_readwrite("istrain_nodesi", &Perturb<DIM>::istrain_nodesi)
        .def_readwrite("istrain_nodesj", &Perturb<DIM>::istrain_nodesj)
        .def_readwrite("istrain", &Perturb<DIM>::istrain)
        .def_readwrite("istrain_vec", &Perturb<DIM>::istrain_vec)
        .def_readwrite("is_extension", &Perturb<DIM>::is_extension)
        .def_readwrite("N_istress", &Perturb<DIM>::N_istress)
        .def_readwrite("istress_nodesi", &Perturb<DIM>::istress_nodesi)
        .def_readwrite("istress_nodesj", &Perturb<DIM>::istress_nodesj)
        .def_readwrite("istress", &Perturb<DIM>::istress)
        .def_readwrite("istress_vec", &Perturb<DIM>::istress_vec)
        .def_readwrite("is_tension", &Perturb<DIM>::is_tension)
        .def_readwrite("apply_affine_strain", &Perturb<DIM>::apply_affine_strain)
        .def_readwrite("strain_tensor", &Perturb<DIM>::strain_tensor)
        .def_readwrite("apply_affine_stress", &Perturb<DIM>::apply_affine_stress)
        .def_readwrite("stress_tensor", &Perturb<DIM>::stress_tensor)
        .def_readwrite("NN_fix", &Perturb<DIM>::NN_fix)
        .def_readwrite("fixed_nodes", &Perturb<DIM>::fixed_nodes);
    
    py::class_<Measure<DIM> >(m, (std::string("Measure")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<>())
        .def("setOutputDOF", &Measure<DIM>::setOutputDOF)
        .def("setOutputStrain", &Measure<DIM>::setOutputStrain)
        .def("setOutputStress", &Measure<DIM>::setOutputStress)
        .def("setOutputAffineStrain", &Measure<DIM>::setOutputAffineStrain)
        .def("setOutputAffineStress", &Measure<DIM>::setOutputAffineStress)
        .def("setOutputLambda", &Measure<DIM>::setOutputLambda)
       . def("setOutputEnergy", &Measure<DIM>::setOutputEnergy)
        .def_readonly_static("dim", &Measure<DIM>::dim)
        .def_readwrite("measure_disp", &Measure<DIM>::measure_disp)
        .def_readwrite("measure_strain", &Measure<DIM>::measure_strain)
        .def_readwrite("measure_lamb", &Measure<DIM>::measure_lamb)
        .def_readwrite("N_ostrain", &Measure<DIM>::N_ostrain)
        .def_readwrite("ostrain_nodesi", &Measure<DIM>::ostrain_nodesi)
        .def_readwrite("ostrain_nodesj", &Measure<DIM>::ostrain_nodesj)
        .def_readwrite("ostrain_vec", &Measure<DIM>::ostrain_vec)
        .def_readwrite("is_extension", &Measure<DIM>::is_extension)
        .def_readwrite("N_ostress", &Measure<DIM>::N_ostress)
        .def_readwrite("ostress_edges", &Measure<DIM>::ostress_edges)
        .def_readwrite("is_tension", &Measure<DIM>::is_tension)
        .def_readwrite("measure_affine_strain", &Measure<DIM>::measure_affine_strain)
        .def_readwrite("measure_affine_stress", &Measure<DIM>::measure_affine_stress)
        .def_readwrite("measure_energy", &Measure<DIM>::measure_energy);
    
    py::class_<LinSolver<DIM>>(m, (std::string("LinSolver")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<Network<DIM> &, int, std::vector<Perturb<DIM> > &, std::vector<Measure<DIM> > &, double>(), 
             py::arg("nw"), py::arg("NF"), py::arg("pert"), py::arg("meas"), py::arg("tol")=1e-4)
        .def("solve", (LinSolverResult (LinSolver<DIM>::*)()) &LinSolver<DIM>::solve)
        .def("solve", (LinSolverResult (LinSolver<DIM>::*)(LinUpdate &)) &LinSolver<DIM>::solve)
        .def("solve", (LinSolverResult (LinSolver<DIM>::*)(LinSolverState &)) &LinSolver<DIM>::solve)
        .def("solve", (LinSolverResult (LinSolver<DIM>::*)(LinUpdate &, LinSolverState &)) &LinSolver<DIM>::solve)
        .def("getSolverState", &LinSolver<DIM>::getSolverState)
        .def("updateSolverState", &LinSolver<DIM>::updateSolverState)
        .def("computeMeas", &LinSolver<DIM>::computeMeas)
        .def("solveMeas", &LinSolver<DIM>::solveMeas)
        .def("solveMeasGrad", &LinSolver<DIM>::solveMeasGrad)
        .def("setK", &LinSolver<DIM>::setK)
        .def("setAllowZero", &LinSolver<DIM>::setAllowZero)
        .def("getHessian", &LinSolver<DIM>::getHessian)
        .def("getBorderedHessian", &LinSolver<DIM>::getBorderedHessian)
        .def_readonly("dim", &LinSolver<DIM>::dim)
        .def_readonly("tol", &LinSolver<DIM>::tol)
        .def_readonly("nw", &LinSolver<DIM>::nw)
        .def_readonly("NDOF", &LinSolver<DIM>::NDOF)
        .def_readonly("NNDOF", &LinSolver<DIM>::NNDOF)
        .def_readonly("NADOF", &LinSolver<DIM>::NADOF)
        .def_readonly("Q", &LinSolver<DIM>::Q)
        .def_readonly("K", &LinSolver<DIM>::K)
        .def_readonly("G", &LinSolver<DIM>::G)
        .def_readonly("H", &LinSolver<DIM>::H)
        .def_readonly("NF", &LinSolver<DIM>::NF)
        .def_readonly("NC", &LinSolver<DIM>::NC)
        .def_readonly("NM", &LinSolver<DIM>::NM)
        .def_readonly("pert", &LinSolver<DIM>::pert)
        .def_readonly("C1", &LinSolver<DIM>::C1)
        .def_readonly("C0", &LinSolver<DIM>::C0)
        .def_readonly("f", &LinSolver<DIM>::f)
        .def_readonly("meas", &LinSolver<DIM>::meas)
        .def_readonly("M", &LinSolver<DIM>::M);
    
    
#ifdef USECPLEX
    
    py::class_<CPLEXSolver<DIM>>(m, (std::string("CPLEXSolver")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<Network<DIM> &, int, std::vector<Perturb<DIM> > &, std::vector<Measure<DIM> > &>())
        .def("solve", (LinSolverResult (CPLEXSolver<DIM>::*)()) &CPLEXSolver<DIM>::solve, 
             py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
        .def("solve", (LinSolverResult (CPLEXSolver<DIM>::*)(LinUpdate &)) &CPLEXSolver<DIM>::solve, 
             py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>())
        .def("solveGrad", (LinSolverResult (CPLEXSolver<DIM>::*)()) &CPLEXSolver<DIM>::solveGrad)
        .def("computeMeas", &CPLEXSolver<DIM>::computeMeas)
        .def("setK", &CPLEXSolver<DIM>::setK)
        .def("setUpdate", &CPLEXSolver<DIM>::setUpdate)
        .def_readonly("dim", &CPLEXSolver<DIM>::dim)
        .def_readonly("nw", &CPLEXSolver<DIM>::nw)
        .def_readonly("NDOF", &CPLEXSolver<DIM>::NDOF)
        .def_readonly("NNDOF", &CPLEXSolver<DIM>::NNDOF)
        .def_readonly("Q", &CPLEXSolver<DIM>::Q)
        .def_readonly("K", &CPLEXSolver<DIM>::K)
        .def_readonly("H", &CPLEXSolver<DIM>::H)
        .def_readonly("NF", &CPLEXSolver<DIM>::NF)
        .def_readonly("NC", &CPLEXSolver<DIM>::NC)
        .def_readonly("NM", &CPLEXSolver<DIM>::NM)
        .def_readonly("pert", &CPLEXSolver<DIM>::pert)
        .def_readonly("C1", &CPLEXSolver<DIM>::C1)
        .def_readonly("C0", &CPLEXSolver<DIM>::C0)
        .def_readonly("meas", &CPLEXSolver<DIM>::meas)
        .def_readonly("M", &CPLEXSolver<DIM>::M);
    
    
#endif
    
#ifdef USEALGLIB
    
    py::class_<MinChangeTuner<DIM>>(m, (std::string("MinChangeTuner")+std::to_string(DIM)+std::string("D")).c_str())
        .def(py::init<>())
        .def("tune", &MinChangeTuner<DIM>::tune/*, 
             py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>()*/);
    
#endif
    
    
}

PYBIND11_MODULE(network_solver, m) {
    
    
    // Initialize dimension dependent parts of code for each needed dimension
    init<1>(m);
    init<2>(m);
    init<3>(m);
    
    
    py::class_<LinSolverResult>(m, "LinSolverResult")
        .def(py::init<int>())
        .def_readonly("success", &LinSolverResult::success)
        .def_readonly("msg", &LinSolverResult::msg)
        .def_readonly("disp", &LinSolverResult::disp)
        .def_readonly("strain", &LinSolverResult::strain)
        .def_readonly("lamb", &LinSolverResult::lamb)
        .def_readonly("ostrain", &LinSolverResult::ostrain)
        .def_readonly("ostress", &LinSolverResult::ostress)
        .def_readonly("affine_strain", &LinSolverResult::affine_strain)
        .def_readonly("affine_stress", &LinSolverResult::affine_stress)
        .def_readonly("olambda", &LinSolverResult::olambda)
        .def_readonly("energy", &LinSolverResult::energy)
        .def_readonly("meas", &LinSolverResult::meas)
        .def_readonly("meas_grad", &LinSolverResult::meas_grad)
        .def_readonly("update_det", &LinSolverResult::update_det);
    
    py::class_<LinUpdate>(m, "LinUpdate")
        .def(py::init<int, std::vector<int> &, RXVec>())
        .def_readwrite("NdK", &LinUpdate::NdK)
        .def_readwrite("dK_edges", &LinUpdate::dK_edges)
        .def_readwrite("dK", &LinUpdate::dK);
    
    py::class_<LinSolverState>(m, "LinSolverState")
        .def(py::init<int>())
        .def_readonly("hess_update", &LinSolverState::hess_update)
        .def_readonly("dK", &LinSolverState::dK)
        .def_readonly("dH", &LinSolverState::dH)
        .def_readonly("dHi", &LinSolverState::dHi)
        .def_readonly("HiC1", &LinSolverState::HiC1)
        .def_readonly("Hif", &LinSolverState::Hif);
    
    py::class_<LeastSquaresObjFunc>(m, "LeastSquaresObjFunc")
        .def(py::init<int, RXVec>())
        .def("setIneq", &LeastSquaresObjFunc::setIneq)
        .def("setOffset", &LeastSquaresObjFunc::setOffset)
        .def("setNorm", &LeastSquaresObjFunc::setNorm)
        .def("evalFunc", &LeastSquaresObjFunc::evalFunc)
        .def("evalGrad", &LeastSquaresObjFunc::evalGrad)
        .def("evalRes", &LeastSquaresObjFunc::evalRes)
        .def("evalResGrad", &LeastSquaresObjFunc::evalResGrad)
        .def_readonly("NT", &LeastSquaresObjFunc::NT)
        .def_readonly("target", &LeastSquaresObjFunc::target)
        .def_readonly("use_ineq", &LeastSquaresObjFunc::use_ineq)
        .def_readonly("ineq", &LeastSquaresObjFunc::ineq)
        .def_readonly("use_offset", &LeastSquaresObjFunc::use_offset)
        .def_readonly("offset", &LeastSquaresObjFunc::offset)
        .def_readonly("use_norm", &LeastSquaresObjFunc::use_norm)
        .def_readonly("norm", &LeastSquaresObjFunc::norm);
     
};
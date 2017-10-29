#include "util.hpp"
#include "network.hpp"
    
// #include "lin_solver.hpp"
// #include "lin_solver_state.hpp"
// #include "lin_solver_result.hpp"

#include <pybind11/pybind11.h>
    
    
namespace py = pybind11;

PYBIND11_MODULE(network_solver, m) {
    
    py::class_<Network<1> >(m, "Network1D")
        .def(py::init<int, RXVec, int, std::vector<int> &, std::vector<int> &,
                RXVec, bool, bool, bool, RXVec, RXVec, RXVec>())
        .def_readwrite("NN", &Network<1>::NN)
        .def_readwrite("node_pos", &Network<1>::node_pos)
        .def_readwrite("NE", &Network<1>::NE)
        .def_readwrite("edgei", &Network<1>::edgei)
        .def_readwrite("edgej", &Network<1>::edgej)
        .def_readwrite("L", &Network<1>::L)
        .def_readwrite("enable_affine", &Network<1>::enable_affine)
        .def_readwrite("fix_trans", &Network<1>::fix_trans)
        .def_readwrite("fix_rot", &Network<1>::fix_rot)
        .def_readwrite("bvecij", &Network<1>::bvecij)
        .def_readwrite("eq_length", &Network<1>::eq_length)
        .def_readwrite("K", &Network<1>::K);
    
//     py::class_<LinSolver>(m, "LinSolver")
//         .def(py::init<Network &, int NF, std::vector<Perturb> &, std::vector<Measure> &>())
//         .def("setK", &LinSolver::setK)
//         .def("solve", (void (LinSolver::*)(LinSolverResult &)) &LinSolver::solve)
//         .def("solve", (void (LinSolver::*)(LinUpdate &, LinSolverResult &)) &LinSolver::solve)
//         .def("solve", (void (LinSolver::*)(LinSolverState &, LinSolverResult &)) &LinSolver::solve)
//         .def("solve", (void (LinSolver::*)(LinUpdate &, LinSolverState &, LinSolverResult &)) &LinSolver::solve)
//         .def("setSolverState", &LinSolver::setSolverState)
//         .def("updateSolverState", &LinSolver::updateSolverState);
    
//     py::class_<LinUpdate>(m, "LinUpdate")
//         .def(py::init<int, std::vector<int> &, RXVec>)
//         .def_readwrite("NdK", &LinUpdate::NdK)
//         .def_readwrite("dK_edges", &LinUpdate::dK_edges)
//         .def_readwrite("dK", &LinUpdate::dK);
    
    
    
    
};
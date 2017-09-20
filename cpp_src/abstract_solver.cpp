#include "abstract_solver.hpp"
    
void AbstractSolver::solveU(std::vector<std::vector<double> > &u) {
    std::vector<XVec > u_tmp;
    isolveU(u_tmp);
    
    u.resize(u_tmp.size());
    for(int i = 0; i < int(u_tmp.size()); i++) {
        eigenToVector(u_tmp[i], u[i]);
    }  
}

void AbstractSolver::solveM(std::vector<double> &meas) {
    XVec m_tmp;
    isolveM(m_tmp);
    eigenToVector(m_tmp, meas);
}

void AbstractSolver::solveMGrad(std::vector<double> &meas, std::vector<std::vector<double> > &grad) {
    XVec m_tmp;
    std::vector<XVec > g_tmp;
    isolveMGrad(m_tmp, g_tmp);
    
    eigenToVector(m_tmp, meas);
    
    grad.resize(g_tmp.size());
    for(int i = 0; i < m_tmp.size(); i++) {
        eigenToVector(g_tmp[i], grad[i]);
    }
}

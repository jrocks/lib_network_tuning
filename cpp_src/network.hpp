#ifndef NETWORK
#define NETWORK
    
#include "util.hpp"

class Network {
    public:
    
        // Number of nodes
        int NN;
        // Node positions
        XVec node_pos;

        // Number of edges
        int NE;
        // Nodes for each edge
        XiVec edgei;
        XiVec edgej;
        
        // Number of global dofs
        int NGDOF;
        // Box dimensions
        DVec L;
    
        // Enable box dofs
        bool enable_affine;
    
        // Bond vectors
        XVec bvecij;
        // Bond equilibrium lengths
        XVec eq_length;
        // Stretch moduli / spring stiffnesses
        XVec stretch_mod;
     

        Network() {};
        Network(int NN, std::vector<double> &node_pos, 
                int NE, std::vector<int> &edgei, std::vector<int> &edgej,
                int NGDOF, std::vector<double> &L, bool enable_affine,
                std::vector<double> &bvecij, std::vector<double> &eq_length, std::vector<double> &stretch_mod) {
            
            this->NN = NN;
            this->NE = NE;
            this->NGDOF = NGDOF;
            this->enable_affine = enable_affine;

            vectorToEigen(node_pos, this->node_pos);
            vectorToEigen(edgei, this->edgei);
            vectorToEigen(edgej, this->edgej);
            vectorToEigen(L, this->L);
            
            vectorToEigen(bvecij, this->bvecij);
            vectorToEigen(eq_length, this->eq_length);
            vectorToEigen(stretch_mod, this->stretch_mod);
            
        }
};

#endif

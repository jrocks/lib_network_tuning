#ifndef NETWORK
#define NETWORK
    
#include "util.hpp"

template <int DIM>
class Network {
    
    DEIGEN(DIM);
    
    public:
    
        // Number of nodes
        int NN;
        // Node positions
        XVec node_pos;

        // Number of edges
        int NE;
        // Nodes for each edge
        std::vector<int> edgei;
        std::vector<int> edgej;
        
        // Box dimensions
        DVec L;
    
        // Enable box dofs
        bool enable_affine;
    
        bool fix_trans, fix_rot;
    
        // Bond vectors
        XVec bvecij;
        // Bond equilibrium lengths
        XVec eq_length;
        // Stretch moduli / spring stiffnesses
        XVec K;
     

        Network() {};
        Network(int NN, RXVec node_pos, 
                int NE, std::vector<int> &edgei, std::vector<int> &edgej,
                RXVec L, bool enable_affine, bool fix_trans, bool fix_rot, 
                RXVec bvecij, RXVec eq_length, RXVec K) {
            
            this->NN = NN;
            this->NE = NE;
            this->enable_affine = enable_affine;
            this->fix_trans = fix_trans;
            this->fix_rot = fix_rot;
            
            this->node_pos = node_pos;
            this->edgei = edgei;
            this->edgej = edgej;
            // this->L = L;
            
            this->bvecij = bvecij;
            this->eq_length = eq_length;
            this->K = K;
            
        }
};

#endif

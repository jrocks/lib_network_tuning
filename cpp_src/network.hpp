#ifndef NETWORK
#define NETWORK
    
#include "util.hpp"

template <int DIM>
class Network {
    
    DEIGEN(DIM);
    
    public:
    
        static const int dim;
    
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
    
        // Pairwise edge interaction parameters
        // Bond vectors
        XVec bvecij;
        // Bond equilibrium lengths
        XVec eq_length;
        // Stretch moduli / spring stiffnesses
        XVec K;
     
        Network() {
            NN = 0;
            NE = 0;
        };
    
        Network(int NN, RXVec node_pos, int NE, std::vector<int> &edgei, std::vector<int> &edgej, RXVec L) {
            
            this->NN = NN;
            this->NE = NE;
            
            enable_affine = false;
  
            this->node_pos = node_pos;
            this->edgei = edgei;
            this->edgej = edgej;
            this->L = L; 
        };
    
        void setInteractions(RXVec bvecij, RXVec eq_length, RXVec K) {
            this->bvecij = bvecij;
            this->eq_length = eq_length;
            this->K = K;
        };
};

template <int DIM>
const int Network<DIM>::dim = DIM;

#endif

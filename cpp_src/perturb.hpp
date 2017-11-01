#ifndef PERTURB
#define PERTURB
    
#include "util.hpp"
 
template <int DIM>
class Perturb {
    
    DEIGEN(DIM);
    
	public:
        static const int dim;
    
        // Input strains imposed as constraints (strain = extension / l0)
        int N_istrain;
        std::vector<int> istrain_nodesi;
        std::vector<int> istrain_nodesj;
        XVec istrain;
        XVec istrain_vec;
        // Whether to interpret strains as extensions
        bool is_extension;
    
        // Input stresses imposed as forces
        int N_istress;
        std::vector<int> istress_nodesi;
        std::vector<int> istress_nodesj;
        XVec istress;
        XVec istress_vec;
        // Whether to interpret stresses as tensions (stress = tension * l0)
        bool is_tension;
    
        // Affine strain
        bool apply_affine_strain;
        DMat strain_tensor;
    
        // Affine stress
        bool apply_affine_stress;
        DMat stress_tensor;
    
        int NN_fix;
        std::vector<int> fixed_nodes;
    
        Perturb() { 
            N_istrain = 0;
            is_extension = false;
            N_istress = 0;
            is_tension = false;
            apply_affine_strain = false;
            apply_affine_stress = false;
            NN_fix = 0;
        };
    
        void setInputStrain(int N_istrain, std::vector<int> &istrain_nodesi, std::vector<int> &istrain_nodesj, RXVec istrain, RXVec istrain_vec, bool is_extension) {
            this->N_istrain = N_istrain;
            this->istrain_nodesi = istrain_nodesi;
            this->istrain_nodesj = istrain_nodesj;
            this->istrain = istrain;
            this->istrain_vec = istrain_vec;
            this->is_extension = is_extension;
        };
    
        void setInputStress(int N_istress, std::vector<int> &istress_nodesi, std::vector<int> &istress_nodesj, RXVec istress, RXVec istress_vec, bool is_tension) {
            this->N_istress = N_istress;
            this->istress_nodesi = istress_nodesi;
            this->istress_nodesj = istress_nodesj;
            this->istress = istress;
            this->istress_vec = istress_vec;
            this->is_tension = is_tension;
        };
    
        void setInputAffineStrain(RXMat strain_tensor) {
            apply_affine_strain = true;
            this->strain_tensor = strain_tensor;
        };
    
        void setInputAffineStress(RXMat stress_tensor) {
            apply_affine_stress = true;
            this->stress_tensor = stress_tensor;
        };
    
        void setFixedNodes(int NN_fix, std::vector<int> &fixed_nodes) {
            this->NN_fix = NN_fix;
            this->fixed_nodes = fixed_nodes;
        };
        
};

template <int DIM>
const int Perturb<DIM>::dim = DIM;
    
    
#endif // PERTURB

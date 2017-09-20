#ifndef PERTURB
#define PERTURB
    
#include "util.hpp"
    
class Perturb {
	public:
    
        // Input strains
        int NIstrain;
        XiVec istrain_nodesi;
        XiVec istrain_nodesj;
        XVec istrain;
        XVec istrain_vec;
        XiVec istrain_bonds;
    
        // Input stresses
        int NIstress;
        XiVec istress_bonds;
        XVec istress;
    
        // Affine strain
        bool apply_affine_strain;
        DMat strain_tensor;
    
        // Affine stress
        bool apply_affine_stress;
        DMat stress_tensor;
    
        int NFix;
        XiVec fixed_nodes;
    
        Perturb() {
            NIstrain = 0;
            NIstress = 0;
            apply_affine_strain = false;
            apply_affine_stress = false;
        };
    
		Perturb(int NIstrain, std::vector<int> &istrain_nodesi, std::vector<int> &istrain_nodesj, 
                std::vector<int> &istrain_bonds,
                std::vector<double> &istrain, std::vector<double> &istrain_vec,
                int NIstress, std::vector<int> &istress_bonds, std::vector<double> &istress,
                bool apply_affine_strain, std::vector<double> &strain_tensor, 
                bool apply_affine_stress, std::vector<double> &stress_tensor,
               int NFix, std::vector<int> &fixed_nodes) {
            
            this->NIstrain = NIstrain;
            this->apply_affine_strain = apply_affine_strain;
            this->NIstress = NIstress;
            this->apply_affine_stress = apply_affine_stress;
            this->NFix = NFix;
            
            vectorToEigen(istrain_nodesi, this->istrain_nodesi);
            vectorToEigen(istrain_nodesj, this->istrain_nodesj);
            vectorToEigen(istrain_bonds, this->istrain_bonds);
            vectorToEigen(istrain, this->istrain);
            vectorToEigen(istrain_vec, this->istrain_vec);
            
            vectorToEigen(istress_bonds, this->istress_bonds);
            vectorToEigen(istress, this->istress);
            

            DMatMap strain_tensor_map(strain_tensor.data());

            DMatMap stress_tensor_map(stress_tensor.data());

            // Eigen matrices are stored in column-major order by default so need to transpose
            this->strain_tensor = strain_tensor_map.transpose();
            this->stress_tensor = stress_tensor_map.transpose();
            
            vectorToEigen(fixed_nodes, this->fixed_nodes);
            
        };
        
};
    
    
#endif

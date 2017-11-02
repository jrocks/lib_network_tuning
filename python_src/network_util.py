import network_solver as ns
import numpy as np
import numpy.linalg as la

    
def closestEdge(net, pos):
    
    NE = net.NE
    L = net.L
    dim = net.dim
    
    dist = np.zeros(NE, float)
    
    for i in range(NE):
        bi = net.edgei[i]
        bj = net.edgej[i]
        
        posi = net.node_pos[dim*bi:dim*bi+dim]
        posj = net.node_pos[dim*bj:dim*bj+dim]
        
        bvec = posj - posi
        bvec -= np.round(bvec/net.L)*net.L
        
        bpos = posi + bvec/2.0
        
        vec = pos - bpos
        vec -= np.rint(vec / L) * L
        
        dist[i] = la.norm(vec)
                
    return np.argsort(dist)
        
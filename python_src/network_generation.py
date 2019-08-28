import os, sys

sys.path.insert(0, '../')
sys.path.insert(0, '../python_src/')

import numpy as np
import numpy.linalg as la
import numpy.random as rand
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from netCDF4 import Dataset, chartostring, stringtoarr
import itertools as it
import shelve
import copy

import network_solver as ns


def convert_jammed_state_to_network(label, index, DIM=2):

#     directory="/data1/home/rocks/data/network_states"
    directory="/home/rocks/data/network_states"
    
    fn = "{1}/{0}.nc".format(label, directory)

    data = Dataset(fn, 'r')

    # print data

    NN = len(data.dimensions['NP'])

    node_pos = data.variables['pos'][index]
        
    rad = data.variables['rad'][index]
    box_mat = data.variables['BoxMatrix'][index]

    L = np.zeros(DIM, float)
    for d in range(DIM):
        L[d] = box_mat[d *(DIM+1)]
        
    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] *= L

    edgei = []
    edgej = []

    NE = 0
    edgespernode = np.zeros(NN, int)
        
    gridL = np.max([int(round(NN**(1.0/DIM))/4.0), 1])
    NBINS = gridL**DIM
    
    print("Grid Length:", gridL, "Number Bins:", NBINS, "Nodes per Bin:", 1.0 * NN / NBINS)
    
    tmp = []
    for d in range(DIM):
        tmp.append(np.arange(gridL))
    
    bin_to_grid = list(it.product(*tmp))
    
    grid_to_bin = {x: i for i, x in enumerate(bin_to_grid)}
    
    grid_links = set()
                
    for i in range(NBINS):
        bini = bin_to_grid[i]
        for j in range(i+1, NBINS):
            binj = bin_to_grid[j]
            
            link = True
                        
            for d in range(DIM):
                dist = bini[d] - binj[d]
                dist -= np.rint(1.0*dist/gridL) * gridL
                                
                if np.abs(dist) > 1:
                    link = False
                    
            if link:
                grid_links.add(tuple(sorted([i,j])))
      
        
    bin_nodes = [[] for b in range(NBINS)]
        
    for n in range(NN):
        pos = node_pos[DIM*n:DIM*n+DIM]
        ipos = tuple(np.floor(pos / L * gridL).astype(int))
        
        bin_nodes[grid_to_bin[ipos]].append(n)
                
    # add edges within each bin
    for ibin in range(NBINS):
        for i in range(len(bin_nodes[ibin])):
            for j in range(i+1,len(bin_nodes[ibin])):
                
                ni = bin_nodes[ibin][i]
                nj = bin_nodes[ibin][j]
                
                posi = node_pos[DIM*ni:DIM*ni+DIM]
                posj = node_pos[DIM*nj:DIM*nj+DIM]
                bvec = posj - posi
                bvec -= np.rint(bvec / L) * L
                l0 = la.norm(bvec)
                
                if l0 < rad[ni] + rad[nj]:
                    NE += 1
                    edgei.append(ni)
                    edgej.append(nj)
                    edgespernode[ni] += 1
                    edgespernode[nj] += 1
     
    # add edge between bins
    for (bini, binj) in grid_links:
        for ni in bin_nodes[bini]:
            for nj in bin_nodes[binj]:
                
                posi = node_pos[DIM*ni:DIM*ni+DIM]
                posj = node_pos[DIM*nj:DIM*nj+DIM]
                bvec = posj - posi
                bvec -= np.rint(bvec / L) * L
                l0 = la.norm(bvec)
                
                if l0 < rad[ni] + rad[nj]:
                    NE += 1
                    edgei.append(ni)
                    edgej.append(nj)
                    edgespernode[ni] += 1
                    edgespernode[nj] += 1
                    
                    
    node_pos_tmp = np.copy(node_pos)
    edgei_tmp = np.copy(edgei)
    edgej_tmp = np.copy(edgej)
    rad_tmp = np.copy(rad)

    index_map = list(range(NN))
    rattlers = set()
    for i in range(NN):
        if edgespernode[i] < DIM+1:
            print("Removing", i, edgespernode[i])
            index_map.remove(i)
            rattlers.add(i)        

    rev_index_map = -1 * np.ones(NN, int)
    for i in range(len(index_map)):
        rev_index_map[index_map[i]] = i

    NN = len(index_map)
    node_pos = np.zeros(DIM*NN, float)
    rad = np.zeros(NN)

    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] = node_pos_tmp[DIM*index_map[i]:DIM*index_map[i]+DIM]
        rad[i] = rad_tmp[index_map[i]]

    edgei = []
    edgej = []
    for i in range(NE):
        if edgei_tmp[i] not in rattlers and edgej_tmp[i] not in rattlers:
            edgei.append(rev_index_map[edgei_tmp[i]])
            edgej.append(rev_index_map[edgej_tmp[i]])

    NE = len(edgei)
    
    print("Number Rattlers:", len(rattlers))
    
    print("NN", NN)
    print("NE", NE) 
        
    net = {}
    net['source'] = fn
    
    net['DIM'] = DIM
    net['box_L'] = L
    
    net['NN'] = NN
    net['node_pos'] = node_pos
    net['rad'] = rad
    
    net['NE'] = NE
    net['edgei'] = np.array(edgei)
    net['edgej'] = np.array(edgej)
    
    return net


def load_network(db_fn, index):
    
    with shelve.open(db_fn) as db:
        net = db["{}".format(index)]
        
    return net


def convert_to_network_object(net, periodic=True):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = np.array(net['node_pos'])
    box_mat = net['box_mat']
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    
    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] = box_mat.dot(node_pos[DIM*i:DIM*i+DIM])
        
    
    L = box_mat.diagonal()
    
    bvecij = np.zeros(DIM*NE, float)
    eq_length = np.zeros(NE, float)
    for i in range(NE):
        bvec =  node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]-node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]
        bvec -= np.rint(bvec / L) * L
        
        bvecij[DIM*i:DIM*i+DIM] = bvec
        eq_length[i] = la.norm(bvec)
    
        
    if DIM == 2:
        cnet = ns.Network2D(NN, node_pos, NE, edgei, edgej, L)
    elif DIM == 3:
        cnet = ns.Network3D(NN, node_pos, NE, edgei, edgej, L)
        
    cnet.setInteractions(bvecij, eq_length, np.ones(NE, float) / eq_length)
    
#     print("convert", cnet.NE)
    
    return cnet





def prune_network(net, rem_nodes, rem_edges):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    rad = net['rad']
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    if 'boundary' in net:
        boundary = net['boundary']
    else:
        boundary = set()
    
    # map from original periodic network to current network
    if 'node_map' in net:
        node_map = net['node_map']
    else:
        node_map = {i:i for i in range(NN)}
    
    rem_nodes = set(rem_nodes)
    rem_edges = set(rem_edges)
    
#     print("Removing", len(rem_nodes), "/", NN, "nodes and", len(rem_edges), "/", NE, "edges...")
    
    
    local_node_map = {}
    
    NN_tmp = 0
    node_pos_tmp = []
    boundary_tmp = set()
    rad_tmp = []
    for v in range(NN):
        
        if v not in rem_nodes:
        
            node_pos_tmp.extend(node_pos[DIM*v:DIM*v+DIM])
            rad_tmp.append(rad[v])
            
            if v in boundary:
                boundary_tmp.add(NN_tmp)
                
            local_node_map[v] = NN_tmp

            NN_tmp += 1
            
    NE_tmp = 0
    edgei_tmp = []
    edgej_tmp = []
    for e in range(NE):
        
        if edgei[e] not in local_node_map or edgej[e] not in local_node_map:
            rem_edges.add(e)
        
        if e in rem_edges:
            if edgei[e] in local_node_map:
                boundary_tmp.add(local_node_map[edgei[e]])
            if edgej[e] in local_node_map:
                boundary_tmp.add(local_node_map[edgej[e]])
        
        else :
            edgei_tmp.append(local_node_map[edgei[e]])
            edgej_tmp.append(local_node_map[edgej[e]])
            NE_tmp += 1
      
    node_map_tmp = {}
    
    for v in node_map:
        if node_map[v] in local_node_map:
            node_map_tmp[v] = local_node_map[node_map[v]]
            
    
    
    new_net = copy.deepcopy(net)
    
    new_net['NN'] = NN_tmp
    new_net['node_pos'] = np.array(node_pos_tmp)
    new_net['rad'] = np.array(rad_tmp)
    
    new_net['NE'] = NE_tmp
    new_net['edgei'] = np.array(edgei_tmp)
    new_net['edgej'] = np.array(edgej_tmp)
    
    new_net['boundary'] = boundary_tmp
    new_net['node_map'] = node_map_tmp
    
    
#     print("Removed", NN-NN_tmp, "/", NN, "nodes and", NE-NE_tmp, "/", NE, "edges.")
    
    return new_net
    
    
    


def make_finite(net):
    
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    box_mat = net['box_mat']
        
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    rem_edges = set()
    
    for b in range(NE):
        posi = node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
        posj = node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]
        
        bvec = posj-posi
        bvec -= np.rint(bvec)
        
        if not ((posi+bvec <= 1.0).all() and (posi+bvec >= 0.0).all()):
            rem_edges.add(b)
    
    return prune_network(net, set(), rem_edges)
            

def make_ball(net, radius, center=None):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    box_mat = net['box_mat']
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    if 'boundary' in net:
        boundary = net['boundary']
    else:
        boundary = set()
    
    if center is None:
        center = 0.5
        
    rem_nodes = set()      
        
    for v in range(NN):
        
        posi = node_pos[DIM*v:DIM*v+DIM]
        
        bvec = posi - center
        bvec -= np.rint(bvec)
        
        if la.norm(bvec) > radius / 2.0:
            rem_nodes.add(v)
        
    
    
    return prune_network(net, rem_nodes, set())
    

    
def prune_zero_modes(net, threshold=1e-12):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    L = net['box_mat'].diagonal()
        
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    boundary = net['boundary']
    node_map = net['node_map']
    
    pert = []
    if DIM == 2:
            pert.append(ns.Perturb2D())
    elif DIM == 3:
        pert.append(ns.Perturb3D())
        
    meas = []
    if DIM == 2:
            meas.append(ns.Measure2D())
    elif DIM == 3:
        meas.append(ns.Measure3D())

    
    znet = copy.deepcopy(net)
            
    while True:
            
        if DIM == 2:
            solver = ns.LinSolver2D(convert_to_network_object(znet, periodic=False), 1, pert, meas)
        elif DIM == 3:
            solver = ns.LinSolver3D(convert_to_network_object(znet, periodic=False), 1, pert, meas)
        
        NGDOF = int(DIM*(DIM+1)/2)
        
        H = solver.getBorderedHessian()
        
        (evals, evecs) = sla.eigsh(H, k=NGDOF+16, which='SA')
        
#         print(evals)
        
        print("Min eval:", evals[NGDOF])
        
        if evals[NGDOF] > threshold:
            return znet
        
        irem = np.argmax(la.norm(evecs[:DIM*znet['NN'], NGDOF].reshape(znet['NN'], DIM), axis=1))
                
        znet = prune_network(znet, {irem}, set())
                    

    
    
    
def choose_boundary_edge(net, theta, phi=0):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    L = net['box_mat'].diagonal()
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    boundary = net['boundary']
    
    
    Z = np.zeros(NN, float)
    
    for b in range(NE):
        Z[edgei[b]] += 1
        Z[edgej[b]] += 1
    
    
    
    vec = np.zeros(DIM, float)
    
    if DIM == 2:
        theta = theta*np.pi/180
        vec[0] = np.cos(theta)
        vec[1] = np.sin(theta)
    if DIM == 3:
        theta = theta*np.pi/180
        phi = phi * np.pi / 180
        
        vec[0] = np.sin(theta)*np.cos(phi)
        vec[1] = np.sin(theta)*np.sin(phi)
        vec[2] = np.cos(theta)
    
    
    angles = np.zeros(len(boundary), float)
    center = 0.5
    
    boundary_edges = []
    angles = []
    for b in range(NE):
        
        if edgei[b] in boundary and edgej[b] in boundary:
            boundary_edges.append(b)
            
            posi = node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
            posj = node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]
            
            bvec = posj - posi
            bvec -= np.rint(bvec)
            
            pos = posi + 0.5*bvec - center
            pos /= la.norm(pos)
            angles.append(np.arccos(np.dot(pos, vec)))
           
    asort = np.argsort(angles)
        
    for i in asort:
        
        b = boundary_edges[i]
                
        if Z[edgei[b]] >= DIM + 1 and Z[edgej[b]] >= DIM + 1:
            bmin = boundary_edges[i]
            break
            
        print("skipping", b, "Z:", Z[edgei[b]], Z[edgej[b]])
    
    
    return (bmin, edgei[bmin], edgej[bmin])



def choose_boundary_nodes(net, theta, edge_map, phi=0):
    
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
    L = net['box_mat'].diagonal()
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    boundary = list(net['boundary'])
    
    
    vec = np.zeros(DIM, float)
    
    if DIM == 2:
        theta = theta*np.pi/180
        vec[0] = np.cos(theta)
        vec[1] = np.sin(theta)
    if DIM == 3:
        theta = theta*np.pi/180
        phi = phi * np.pi / 180
        
        vec[0] = np.sin(theta)*np.cos(phi)
        vec[1] = np.sin(theta)*np.sin(phi)
        vec[2] = np.cos(theta)
    
    
    angles = np.zeros(len(boundary), float)
    center = 0.5
    
    angles = []
    for b in boundary:
            
            pos = node_pos[DIM*b:DIM*b+DIM] - center
            pos -= np.rint(pos)
            
            pos /= la.norm(pos)
            angles.append(np.arccos(np.dot(pos, vec)))
           
    asort = np.argsort(angles)
    
    bi = boundary[asort[0]]
    
    for j in asort[1:]:
        
        bj = boundary[j]
        
    
        if tuple(sorted([bi, bj])) not in edge_map:
            break
        
    
    return (bi, bj)






# def bondExists(nw, nodei, nodej):
# 	NE = nw.NE
# 	edgei = nw.edgei
# 	edgej = nw.edgej
    
# 	for b in range(NE):
# 		if (edgei[b] == nodei and edgej[b] == nodej) or (edgei[b] == nodej and edgej[b] == nodei):
# 			return True
		
# 	return False

    
    
    
    
    
    
    
    
#old

def load_jammed_network(db_fn, index):
    
    with shelve.open(db_fn) as db:
        net = db["{}".format(index)]
            
            
    
            
    DIM = net['DIM']
   
    NN = net['NN']
    node_pos = net['node_pos']
#     L = net['box_mat'].diagonal()
    L = net['box_L']
    
#     for i in range(NN):
#         node_pos[DIM*i:DIM*i+DIM] = L * node_pos[DIM*i:DIM*i+DIM]
    
    
    NE = net['NE']
    edgei = net['edgei']
    edgej = net['edgej']
    
    bvecij = np.zeros(DIM*NE, float)
    eq_length = np.zeros(NE, float)
    for i in range(NE):
        bvec =  node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]-node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]
        bvec -= np.rint(bvec / L) * L
        
        bvecij[DIM*i:DIM*i+DIM] = bvec
        eq_length[i] = la.norm(bvec)
    
    
    if DIM == 2:
        cnet = ns.Network2D(NN, node_pos, NE, edgei, edgej, L)
    elif DIM == 3:
        cnet = ns.Network3D(NN, node_pos, NE, edgei, edgej, L)
        
    cnet.setInteractions(bvecij, eq_length, np.ones(NE, float) / eq_length)
#     cnet.fix_trans = True
#     cnet.fix_rot = False
    
    return cnet

    

def convertToFlowNetwork(net):
    DIM = 1
    NN = net.NN
    NE = net.NE
    node_pos = np.arange(0, 1, 1.0/NN)
    edgei = net.edgei
    edgej = net.edgej
    
    L = np.array([1.0])
    
    bvecij = np.zeros(NE, float)
    eq_length = np.zeros(NE, float)
    for b in range(NE):
        bvec =  node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]-node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
        bvec -= np.rint(bvec / L) * L
        bvec /= la.norm(bvec)
        
        eq_length[b] = 1.0
        
        bvecij[DIM*b:DIM*b+DIM] = bvec
        
        
    fnet = ns.Network1D(NN, node_pos, NE, edgei, edgej, L)
    fnet.setInteractions(bvecij, eq_length, np.ones(NE, float))
#     fnet.fix_trans = True
#     fnet.fix_rot = False
        
    return fnet




    

    
    
# def chooseNodesPos(nw, pos):
    
#     NN = nw.NN
#     L = nw.L
#     DIM = nw.DIM
    
#     dist = np.zeros(NN, float)
    
#     for i in range(NN):
#         posi = nw.node_pos[DIM*i:DIM*i+DIM]
#         vec = pos - posi
#         vec -= np.rint(vec / L) * L
        
#         dist[i] = la.norm(vec)
        
        
#     return np.argsort(dist)



# def distortNetworkPos(net, sigma=1.0, seed=None):
    
#     rand.seed(seed)
    
#     DIM = net.DIM
#     NN = net.NN
#     node_pos = net.node_pos
#     NE = net.NE
#     edgei = net.edgei
#     edgej = net.edgej
#     NGDOF = net.NGDOF
#     L = net.L
    
#     pert_node_pos = np.zeros(DIM*NN, float)
#     for i in range(NN):
#         pos = node_pos[DIM*i:DIM*i+DIM]
#         pert_node_pos[DIM*i:DIM*i+DIM] = rand.normal(loc=pos, scale=sigma, size=DIM)

        
#     bvecij = np.zeros(DIM*NE, float)
#     eq_length = np.zeros(NE, float)
#     for b in range(NE):
#         bvec =  pert_node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]-pert_node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
#         bvec -= np.rint(bvec / L) * L
#         bvecij[DIM*b:DIM*b+DIM] = bvec
        
#         eq_length[b] = la.norm(bvec)
        
        
        

#     pert_net = network.Network(DIM, NN, pert_node_pos, NE, edgei, edgej, NGDOF, L)
    
#     pert_net.setStretchInt(bvecij, eq_length, np.ones(NE, float) / eq_length)
    
    
#     return pert_net

# def lower_DZ(net, DZ, seed=None, remove=False, local=False):
    
#     rand.seed(seed)
    
#     DIM = net.DIM
#     NN = net.NN
#     node_pos = net.node_pos
#     NE = net.NE
#     edgei = net.edgei
#     edgej = net.edgej
#     NGDOF = net.NGDOF
#     L = net.L
    
#     Q = calcEqMat(net)
        
#     K = np.ones(NE, float)
    
# #     H = Q.dot(sparse.diags(K).dot(Q.transpose()))

# #     vals, vecs = sparse.linalg.eigsh(H, k=np.min([16, DIM*NN-1]), which='SA')
# #     print vals
   

#     A = AdjList(NN)
#     A.add_edges(NE, edgei, edgej)
         
#     DZ_current = 2.0 * NE / NN - 2.0*DIM + 2.0*NGDOF/NN
            
#     print 0, "NE:", NE, "DZ:", DZ_current
        
#     edge_list = range(NE)
#     rand.shuffle(edge_list)
        
#     keep_set = set(range(NE))
        
#     i = 0
#     while DZ_current > DZ:
                
#         i += 1
        
#         if len(edge_list) > 0:
#             b_test = edge_list.pop()
#         else:
#             break
        
#         if local:
            
#             A.remove_edge(edgei[b_test], edgej[b_test])
            
#             is_constrained = True
#             for j in range(NN):
#                 if A.get_degree(j) < DIM + 1:
#                     is_constrained = False
                    
#             A.add_edge(edgei[b_test], edgej[b_test])
            
#             if not is_constrained:
#                 continue
            
#         else:
        
#             K_test = np.copy(K)
#             K_test[b_test] = 0.0

#             H_test = Q.dot(sparse.diags(K_test).dot(Q.transpose()))

#             vals, vecs = sparse.linalg.eigsh(H_test, k=np.min([16, DIM*NN-1]), which='SA')

#             print vals
            
#             if vals[NGDOF] < np.sqrt(np.finfo(float).eps):
#                 continue
            
            
#         A.remove_edge(edgei[b_test], edgej[b_test])
            
#         keep_set.remove(b_test)
#         K[b_test] = 0.0

#         DZ_current = 2.0 * np.sum(K) / NN - 2.0*DIM + 2.0*NGDOF/NN
       
    
    
# #     H = Q.dot(sparse.diags(K).dot(Q.transpose()))

# #     vals, vecs = sparse.linalg.eigsh(H, k=np.min([16, DIM*NN-1]), which='SA')
# #     print vals
    
#     if remove:
    
#         NE = len(keep_set)
#         edgei = edgei[list(keep_set)]
#         edgej = edgej[list(keep_set)]


#         bvecij = np.zeros(DIM*NE, float)
#         eq_length = np.zeros(NE, float)
#         for b in range(NE):
#             bvec =  node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]-node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
#             bvec -= np.rint(bvec / L) * L
#             bvecij[DIM*b:DIM*b+DIM] = bvec

#             eq_length[b] = la.norm(bvec)

#         print i, "NE:", NE, "DZ:", DZ_current


#         low_net = network.Network(DIM, NN, node_pos, NE, edgei, edgej, NGDOF, L)

#         low_net.setStretchInt(bvecij, eq_length, np.ones(NE, float) / eq_length)

        
#     else:
        
#         bvecij = np.zeros(DIM*NE, float)
#         eq_length = np.zeros(NE, float)
#         for b in range(NE):
#             bvec =  node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]-node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
#             bvec -= np.rint(bvec / L) * L
#             bvecij[DIM*b:DIM*b+DIM] = bvec

#             eq_length[b] = la.norm(bvec)

#         print i, "NE:", NE, "DZ:", DZ_current
        
#         low_net = network.Network(DIM, NN, node_pos, NE, edgei, edgej, NGDOF, L)
        
#         low_net.setStretchInt(bvecij, eq_length, K / eq_length)
        
#     return low_net
          
        
        
# def calcEqMat(net):
#     node_pos = net.node_pos
#     NN = net.NN

#     edgei = net.edgei
#     edgej = net.edgej
#     NE = net.NE
#     L = net.L

#     DIM = net.DIM

#     Q = np.zeros([DIM*NN, NE], float)

#     for i in range(NE):
#         posi = node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]
#         posj = node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]
#         bvec = posj - posi
#         bvec -= np.rint(bvec / L) * L
#         bvec /= la.norm(bvec)

#         Q[DIM*edgei[i]:DIM*edgei[i]+DIM, i] = -bvec
#         Q[DIM*edgej[i]:DIM*edgej[i]+DIM, i] = bvec

#     sQ = sparse.csc_matrix(Q)
        
#     return sQ







# class AdjList(object):
    
#     def __init__(self, NN):
           
#         self.NN = NN
#         self.adj_list = [set() for _ in range(NN)]
        
        
#     def add_edges(self, NE, bi_list, bj_list):
#         for b in range(NE):
#             self.add_edge(bi_list[b], bj_list[b])
   
#     def add_edge(self, bi, bj):
#         self.adj_list[bi].add(bj)
#         self.adj_list[bj].add(bi)
        
#     def remove_edge(self, bi, bj):
#         self.adj_list[bi].discard(bj)
#         self.adj_list[bj].discard(bi)
        
#     def get_degree(self, bi):
#         return len(self.adj_list[bi])
        
# def getLineGraph(NN, NE, bi_list, bj_list):
    
#     # sorted edges by node
#     buckets = [set() for _ in range(NN)]
    
#     for b in range(NE):
#         buckets[bi_list[b]].add(b)
#         buckets[bj_list[b]].add(b)
        
#     # connect all edges sharing  node
#     lg_bi_list = []
#     lg_bj_list = []
    
#     lg_NE = 0
    
#     for n in range(NN):
#         for pair in it.combinations(buckets[n], 2):
#             lg_NE += 1
#             lg_bi_list.append(pair[0])
#             lg_bj_list.append(pair[1])
            
#     adj = AdjList(NE)
#     adj.add_edges(lg_NE, lg_bi_list, lg_bj_list)
    
#     return adj
        
       
# def getGraphDistances(adj, start):

#     visited = set()
#     queue = [start]
    
#     dists = np.full(adj.NN, np.inf, float)
#     dists[start] = 0.0
    
#     # print dists
    
#     while queue:
#         vertex = queue.pop()
#         if vertex not in visited:
#             visited.add(vertex)
#             neighbors = adj.adj_list[vertex] - visited
#             queue.extend(neighbors)
                        
#             for neighb in neighbors:
#                 if dists[vertex] + 1 < dists[neighb]:
#                     dists[neighb] = dists[vertex] + 1
            
#     return dists
    
    
        
        
# class PebbleGame(object):
    
#     def __init__(self, NN):
        
#         self.DIM = 2
        
#         self.NN = NN
#         self.NB = 0
        
#         # Total number of free degrees of freedom
#         self.NDOF = self.DIM * NN
              
#         # Number of global degrees of freedom allowed (assuming free BCs)
#         self.NGDOF = self.DIM*(self.DIM+1)/2
        
#         # Adjacency list of directed graph composed only of independent degrees of freedom
#         self.ind_adj_list = [set() for _ in range(NN)]
        
#         # Adjecency list of directed graph composed only of dependent degrees of freedom
#         self.dep_adj_list = [set() for _ in range(NN)]
        
#     # Return true if number of free dofs is equal to number of global dofs
#     def is_rigid(self):
#         return self.NDOF == self.NGDOF
     
#     def edge_exists(self, bi, bj):
#         return (bj in self.ind_adj_list[bi] or bi in self.ind_adj_list[bj] 
#                 or bj in self.dep_adj_list[bi] or bi in self.dep_adj_list[bj])
        
#     # Add all edges from lists if possible
#     def add_edges(self, NB, bi_list, bj_list):
#         for b in range(NB):
#             self.add_edge(bi_list[b], bj_list[b])
    
#     # Add edge (bi, bj) to network and return true if edge can be added as independent constraint
#     # Return False if edge already exists or cannot be added as independent
#     def add_edge(self, bi, bj):
         
#         # If edge already exists return false
#         if self.edge_exists(bi, bj):
#             return False
        
#         # Otherwise try to add bond as independent constraint
#         else:
            
#             self.NB += 1

#             if not self.is_rigid() and self.test_add_edge(bi, bj):
#                 self.ind_adj_list[bi].add(bj)
#                 self.NDOF -= 1
#                 return True
#             else:
#                 self.dep_adj_list[bi].add(bj)
#                 return False
    
#     # Return true if a relative degree of freedom exists between nodes bi and bj
#     # If successful, then there will be DIM pebbles on each of node bi and bj
#     # Assumes edge does not exist, but if it does will return False
#     def test_add_edge(self, bi, bj):
        
#         # Attempt to collect DIM pebbles on node bi while holding bj fixed
#         while(len(self.ind_adj_list[bi]) > 0):
#             if not self.collect_pebble(bi, bj):
#                 return False
        
#         # Attempt to collect DIM pebbles on node bj while holding bi fixed
#         while(len(self.ind_adj_list[bj]) > 0):
#             if not self.collect_pebble(bj, bi):
#                 return False    
                        
#         return True
     
#     # Remove all edges from lists if possible
#     def remove_edges(self, NB, bi_list, bj_list):
#         for b in range(NB):
#             self.remove_edge(bi_list[b], bj_list[b])
        
#     # Remove edge (bi, bj) and return True if edge removal introduces new dof
#     def remove_edge(self, bi, bj):
        
#         if not self.edge_exists(bi, bj):
#             print "edge doesn't exist"
#             return False
        
#         self.NB -= 1

#         # If edge is dependent, return False
#         if bj in self.dep_adj_list[bi]:
#             self.dep_adj_list[bi].remove(bj)
#             return False
#         elif bi in self.dep_adj_list[bj]:
#             self.dep_adj_list[bj].remove(bi)
#             return False
                
#         # Otherwise remove from independent list and try collecting pebbles on one of the dependent edges
#         else:
            
#             if bj in self.ind_adj_list[bi]:
#                 vi = bi
#                 vj = bj
#             else:
#                 vi = bj
#                 vj = bi
            
#             # Remove edge from independent list and free up pebble
#             self.ind_adj_list[vi].remove(vj)
#             # Check to see if the pebble can be moved to a dependent bond
#             for i in range(self.NN):
#                 for j in self.dep_adj_list[i]:
#                     # If free dof exists, move the pebble
#                     if self.test_add_edge(i, j):
#                         self.dep_adj_list[i].remove(j)
#                         self.ind_adj_list[i].add(j)
#                         return False
                         
                            
#             self.NDOF += 1 
#             return True
        
#     def test_remove_edge(self, bi, bj):
#         rtest = self.remove_edge(bi, bj)
                
#         atest = self.add_edge(bi, bj)
        
#         if rtest != atest:
#             print "Bond removal test error"
        
#         return rtest
    
#     def collect_pebble(self, start, exclude):
#         visited = set()
#         stack = [start]
#         parent_map = dict()
        
#         while stack:
            
#             vertex = stack.pop()
                        
#             if vertex not in visited:
                
#                 if len(self.ind_adj_list[vertex]) < self.DIM and vertex != start and vertex != exclude:
#                     self.reverse_path(self.get_path(start, vertex, parent_map))
                    
#                     return True
                
#                 visited.add(vertex)
#                 children = self.ind_adj_list[vertex] - visited
#                 stack.extend(children)
#                 for child in children:
#                     parent_map[child] = vertex
                
#         return False
                 
#     def get_path(self, start, finish, parent_map):
#         path = []
#         vertex = finish
#         while vertex != start:
#             path.append(vertex)
#             vertex = parent_map[vertex]
            
#         path.append(start)
#         path.reverse()
        
#         return path
                
#     def reverse_path(self, path):
        
#         bi = path[0]
#         for i in range(1, len(path)):
#             bj = path[i]
            
#             self.ind_adj_list[bi].remove(bj)
#             self.ind_adj_list[bj].add(bi)
            
#             bi = bj
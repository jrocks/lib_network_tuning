import numpy as np
import numpy.linalg as la
from netCDF4 import Dataset, chartostring, stringtoarr
import network
import itertools as it

import mech_network_solver as mns


class Network(object):
    def __init__(self, DIM, NN, node_pos, NE, edgei, edgej, NGDOF, L):
        self.DIM = DIM
        self.NN = NN
        self.node_pos = np.array(node_pos)
        self.NE = NE
        self.edgei = np.array(edgei)
        self.edgej = np.array(edgej)
        self.NGDOF = NGDOF
        self.L = np.array(L)
        self.enable_affine = False
        
        self.bvecij = np.array([0], dtype=np.double)
        self.eq_length = np.array([0], dtype=np.double)
        self.stretch_mod = np.array([0], dtype=np.double)
        
        self.NSN = 0
        self.surface_nodes = np.array([0], dtype=np.double)
        
    def setStretchInt(self, bvecij, eq_length, stretch_mod):
        self.bvecij = np.array(bvecij)
        self.eq_length = np.array(eq_length)
        self.stretch_mod = np.array(stretch_mod)
        
    def enableAffine(self):
        self.enable_affine = True
        
    def disableAffine(self):
        self.enable_affine = False
        
    def setSurfaceNodes(self, NSN, surface_nodes):
        self.NSN = NSN
        self.surface_nodes = np.array(surface_nodes)
        
    def getCyNetwork(self):
        return mns.CyNetwork(self.NN, self.node_pos, self.NE, self.edgei, self.edgej, self.NGDOF, self.L, self.enable_affine,
                            self.bvecij, self.eq_length, self.stretch_mod)
    
    
def closestEdge(net, pos):
    
    NE = net.NE
    L = net.L
    DIM = net.DIM
    
    dist = np.zeros(NE, float)
    
    for i in range(NE):
        bi = net.edgei[i]
        bj = net.edgej[i]
        
        posi = net.node_pos[DIM*bi:DIM*bi+DIM]
        posj = net.node_pos[DIM*bj:DIM*bj+DIM]
        
        bvec = posj - posi
        bvec -= np.round(bvec/net.L)*net.L
        
        bpos = posi + bvec/2.0
        
        vec = pos - bpos
        vec -= np.rint(vec / L) * L
        
        dist[i] = la.norm(vec)
                
    return np.argsort(dist)
        
    
    
def chooseNodesPos(nw, pos):
    
    NN = nw.NN
    L = nw.L
    DIM = nw.DIM
    
    dist = np.zeros(NN, float)
    
    for i in range(NN):
        posi = nw.node_pos[DIM*i:DIM*i+DIM]
        vec = pos - posi
        vec -= np.rint(vec / L) * L
        
        dist[i] = la.norm(vec)
        
        
    return np.argsort(dist)

def chooseNodesAngle(nw, theta, phi=0):
	DIM = nw.DIM
	
	
	vec = np.zeros(DIM, float)
	
	if DIM == 2:

		theta = theta*np.pi/180

		vec[0] = np.cos(theta)
		vec[1] = np.sin(theta)
	elif DIM == 3:
        
		theta = theta*np.pi/180
		phi = phi *np.pi / 180
		vec[0] = np.sin(theta)*np.cos(phi)
		vec[1] = np.sin(theta)*np.sin(phi)
		vec[2] = np.cos(theta)
		
		
	# print vec
    
	angles = np.zeros(nw.NSN, float)
    
	center = np.zeros(DIM, float)
	for i in range(nw.NN):
		center += nw.node_pos[DIM*i:DIM*i+DIM]
    
	center /= nw.NN
	
	surface_nodes = nw.surface_nodes  
	
	node_pos = nw.node_pos
	for i in range(nw.NSN):
		pos = node_pos[DIM*surface_nodes[i]:DIM*surface_nodes[i]+DIM] - center
		pos /= la.norm(pos)
		angles[i] = np.arccos(np.dot(pos, vec))

	arg = np.argsort(angles)

	nodei = surface_nodes[arg[0]]
	nodej = surface_nodes[arg[1]]
	
	i = 2
	while(bondExists(nw, nodei, nodej)):
		nodei = surface_nodes[arg[i]]
		i += 1
    
    
	return (nodei, nodej)

def bondExists(nw, nodei, nodej):
	NE = nw.NE
	edgei = nw.edgei
	edgej = nw.edgej
    
	for b in range(NE):
		if (edgei[b] == nodei and edgej[b] == nodej) or (edgei[b] == nodej and edgej[b] == nodei):
			return True
		
	return False

def loadFiniteRandomNetwork(label, irec):
    
    
    directory="/data1/home/rocks/data/allostery/"
    data = Dataset("{1}/{0}.nc".format(label, directory), 'r')
    
    DIM = len(data.dimensions['DIM'])
    NN = data.variables['NN'][irec]
    NDOF = data.variables['NDOF'][irec]
    NB = data.variables['NB'][irec]
    NSN = data.variables['NSN'][irec]

    NDOF_index = data.variables['NDOF_index'][irec]
    NB_index = data.variables['NB_index'][irec]
    NSN_index = data.variables['NSN_index'][irec]

    node_pos = data.variables['node_pos'][NDOF_index:NDOF_index+NDOF]
    
    pmin = np.zeros(DIM, float)
    for d in range(DIM):
        pmin[d] = np.min(node_pos[d::DIM])
    
    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] -= pmin
        
    
    if NB > 0:
        bondi = data.variables['bondi'][NB_index:NB_index+NB]
        bondj = data.variables['bondj'][NB_index:NB_index+NB]
        l0 = data.variables['eq_length'][NB_index:NB_index+NB]
    else:
        bondi = np.array([0], dtype=np.int32)
        bondj = np.array([0], dtype=np.int32)
        l0 = np.array([0], dtype=np.double)
        
    bvecij = np.zeros(DIM*NB, float)
    for b in range(NB):
        bvecij[DIM*b:DIM*b+DIM] = node_pos[DIM*bondj[b]:DIM*bondj[b]+DIM]-node_pos[DIM*bondi[b]:DIM*bondi[b]+DIM]
    
    
    # box_mat = data.variables['box_mat'][irec]
    # L = np.zeros(DIM, float)

    # if DIM == 2:
    #     L[0] = box_mat[0]
    #     L[1] = box_mat[3]
    # elif DIM == 3:
    #     L[0] = box_mat[0]
    #     L[1] = box_mat[4]
    #     L[2] = box_mat[8]
    
    L = np.zeros(DIM, float)
    for d in range(DIM):
        L[d] = np.max(node_pos[d::DIM]) - np.min(node_pos[d::DIM])

    if NSN > 0:
        surface_nodes = data.variables['surface_nodes'][NSN_index:NSN_index+NSN]
    else:
        surface_nodes = np.array([])
    
    nw = network.Network(DIM, NN, node_pos, NB, bondi, bondj, DIM*(DIM+1)/2, L)
    nw.setStretchInt(bvecij, l0, np.ones(NB, float) / l0)
    nw.setSurfaceNodes(NSN, surface_nodes)
    
    return nw

def loadPeriodicRandomNetwork(label, seed):
    
    directory="/data1/home/rocks/data/network_states/"

    DIM = 2

    irec = seed % 512

    seed = irec

    data = Dataset("{1}/{0}.nc".format(label, directory), 'r')

    # print data

    NN = len(data.dimensions['NP'])

    node_pos = data.variables['pos'][irec]
    rad = data.variables['rad'][irec]
    box_mat = data.variables['BoxMatrix'][irec]

    L = np.zeros(DIM, float)
    L[0] = box_mat[0]
    L[1] = box_mat[3]


    bondi = []
    bondj = []
    eq_length = []


    NB = 0
    
    NBINS = np.max([int(np.sqrt(NN)/4.0), 1])
    # NBINS = 4
    print "NBINS", NBINS
    
    grid = [[[] for iy in range(NBINS)] for ix in range(NBINS)] 

    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] *= L

        posi = node_pos[DIM*i:DIM*i+DIM]
        ix = int(np.floor(posi[0]/L[0] * NBINS))
        iy = int(np.floor(posi[1]/L[1] * NBINS))

        grid[ix][iy].append(i)


    bondspernode = np.zeros(NN, int)

    grid_links = np.zeros([NBINS, NBINS, NBINS, NBINS], bool)
    
    for ix in range(NBINS):
        for iy in range(NBINS):

            for n in range(len(grid[ix][iy])):
                for m in range(n+1,len(grid[ix][iy])):
                    i = grid[ix][iy][n]
                    j = grid[ix][iy][m]
                    posi = node_pos[DIM*i:DIM*i+DIM]
                    posj = node_pos[DIM*j:DIM*j+DIM]
                    bvec = posj - posi
                    bvec -= np.rint(bvec / L) * L
                    l0 = la.norm(bvec)

                    if l0 < rad[i] + rad[j]:
                        NB += 1
                        bondi.append(i)
                        bondj.append(j)
                        eq_length.append(l0)
                        bondspernode[i] += 1
                        bondspernode[j] += 1
                        
            grid_links[ix][iy][ix][iy] = True
            
            
            if not grid_links[ix][iy][(ix+1)%NBINS][iy] and not grid_links[(ix+1)%NBINS][iy][ix][iy]:
                grid_links[ix][iy][(ix+1)%NBINS][iy] = True
                grid_links[(ix+1)%NBINS][iy][ix][iy] = True
                for i in grid[ix][iy]:
                    for j in grid[(ix+1)%NBINS][iy]:
                        posi = node_pos[DIM*i:DIM*i+DIM]
                        posj = node_pos[DIM*j:DIM*j+DIM]
                        bvec = posj - posi
                        bvec -= np.rint(bvec / L) * L
                        l0 = la.norm(bvec)

                        if l0 < rad[i] + rad[j]:
                            NB += 1
                            bondi.append(i)
                            bondj.append(j)
                            eq_length.append(l0)
                            bondspernode[i] += 1
                            bondspernode[j] += 1

            if not grid_links[ix][iy][ix][(iy+1)%NBINS] and not grid_links[ix][(iy+1)%NBINS][ix][iy]:
                grid_links[ix][iy][ix][(iy+1)%NBINS] = True
                grid_links[ix][(iy+1)%NBINS][ix][iy] = True
                for i in grid[ix][iy]:
                    for j in grid[ix][(iy+1)%NBINS]:
                        posi = node_pos[DIM*i:DIM*i+DIM]
                        posj = node_pos[DIM*j:DIM*j+DIM]
                        bvec = posj - posi
                        bvec -= np.rint(bvec / L) * L
                        l0 = la.norm(bvec)

                        if l0 < rad[i] + rad[j]:
                            NB += 1
                            bondi.append(i)
                            bondj.append(j)
                            eq_length.append(l0)
                            bondspernode[i] += 1
                            bondspernode[j] += 1

            if not grid_links[ix][iy][(ix+1)%NBINS][(iy+1)%NBINS] and not grid_links[(ix+1)%NBINS][(iy+1)%NBINS][ix][iy]:
                grid_links[ix][iy][(ix+1)%NBINS][(iy+1)%NBINS] = True
                grid_links[(ix+1)%NBINS][(iy+1)%NBINS][ix][iy] = True
                for i in grid[ix][iy]:
                    for j in grid[(ix+1)%NBINS][(iy+1)%NBINS]:
                        posi = node_pos[DIM*i:DIM*i+DIM]
                        posj = node_pos[DIM*j:DIM*j+DIM]
                        bvec = posj - posi
                        bvec -= np.rint(bvec / L) * L
                        l0 = la.norm(bvec)

                        if l0 < rad[i] + rad[j]:
                            NB += 1
                            bondi.append(i)
                            bondj.append(j)
                            eq_length.append(l0)
                            bondspernode[i] += 1
                            bondspernode[j] += 1
                            
            if not grid_links[ix][iy][(ix-1+NBINS)%NBINS][(iy+1)%NBINS] and not grid_links[(ix-1+NBINS)%NBINS][(iy+1)%NBINS][ix][iy]:
                grid_links[ix][iy][(ix-1+NBINS)%NBINS][(iy+1)%NBINS] = True
                grid_links[(ix-1+NBINS)%NBINS][(iy+1)%NBINS][ix][iy] = True
                for i in grid[ix][iy]:
                    for j in grid[(ix-1+NBINS)%NBINS][(iy+1)%NBINS]:
                        
                        posi = node_pos[DIM*i:DIM*i+DIM]
                        posj = node_pos[DIM*j:DIM*j+DIM]
                        bvec = posj - posi
                        bvec -= np.rint(bvec / L) * L
                        l0 = la.norm(bvec)

                        if l0 < rad[i] + rad[j]:
                            NB += 1
                            bondi.append(i)
                            bondj.append(j)
                            eq_length.append(l0)
                            bondspernode[i] += 1
                            bondspernode[j] += 1


    node_pos_tmp = np.copy(node_pos)
    bondi_tmp = np.copy(bondi)
    bondj_tmp = np.copy(bondj)
    eq_length_tmp = np.copy(eq_length)

    index_map = range(NN)
    rattlers = set()
    for i in range(NN):
        if bondspernode[i] < DIM+1:
            print "Removing", i, bondspernode[i]
            index_map.remove(i)
            rattlers.add(i)        

    rev_index_map = -1 * np.ones(NN, int)
    for i in range(len(index_map)):
        rev_index_map[index_map[i]] = i

    NN = len(index_map)
    node_pos = np.zeros(DIM*NN, float)

    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] = node_pos_tmp[DIM*index_map[i]:DIM*index_map[i]+DIM]

    bondi = []
    bondj = []
    eq_length = []
    for i in range(NB):
        if bondi_tmp[i] not in rattlers and bondj_tmp[i] not in rattlers:
            bondi.append(rev_index_map[bondi_tmp[i]])
            bondj.append(rev_index_map[bondj_tmp[i]])
            eq_length.append(eq_length_tmp[i])

    NB = len(bondi)

    bvecij = np.zeros(DIM*NB, float)
    for b in range(NB):
        bvec =  node_pos[DIM*bondj[b]:DIM*bondj[b]+DIM]-node_pos[DIM*bondi[b]:DIM*bondi[b]+DIM]
        bvec -= np.rint(bvec / L) * L
        
        bvecij[DIM*b:DIM*b+DIM] = bvec
    
    print "NN", NN
    print "NB", NB 
        
    net = Network(DIM, NN, node_pos, NB, bondi, bondj, DIM, L)
    net.setStretchInt(bvecij, eq_length, np.ones(NB, float) / eq_length)
    
    return net


def create2DTriLattice(NX, NY, a):
    DIM = 2
    
    NX -= NX % 2
    NY -= NY % 2
    
    L = np.zeros(DIM, float)
    L[0] = NX * a
    L[1] = np.sqrt(3.0) / 2.0 * NY * a
     
    NN = NX*NY
    NB = 3*NN
    
    node_pos = np.zeros(DIM*NN, float)
    bondi = np.zeros(NB, int)
    bondj = np.zeros(NB, int)
    
    a1 = np.array([a, 0])
    a2 = np.array([a / 2.0 , a * np.sqrt(3.0) / 2.0])
    
    for ix in range(NX):
        for iy in range(NY):
            pos = ix*a1 + iy*a2
            pos -= np.floor(pos / L) * L
            node_pos[DIM*(ix*NY+iy): DIM*(ix*NY+iy)+DIM] = pos
        
    
    for ix in range(NX-1):
        for iy in range(NY):
            bondi[3*(ix*NY+iy)] = ix*NY+iy
            bondj[3*(ix*NY+iy)] = (ix+1)*NY+iy
            
    ix = NX-1
    for iy in range(NY):
        bondi[3*(ix*NY+iy)] = ix*NY+iy
        bondj[3*(ix*NY+iy)] = iy
          
    
    for ix in range(NX):
        for iy in range(NY-1):
            bondi[3*(ix*NY+iy)+1] = ix*NY+iy
            bondj[3*(ix*NY+iy)+1] = ix*NY+iy+1
        
    iy = NY-1
    for ix in range(NX):
        bondi[3*(ix*NY+iy)+1] = ix*NY+iy
        bondj[3*(ix*NY+iy)+1] = (ix+(iy+1)/2)%NX * NY
    
    
    for ix in range(1, NX):
        for iy in range(NY-1):
            bondi[3*(ix*NY+iy)+2] = ix*NY+iy
            bondj[3*(ix*NY+iy)+2] = (ix-1)*NY+iy+1
    
    ix = 0
    for iy in range(NY-1):
        bondi[3*(ix*NY+iy)+2] = ix*NY+iy
        bondj[3*(ix*NY+iy)+2] = (NX-1)*NY+iy+1
      
    iy = NY-1
    for ix in range(1, NX):
        bondi[3*(ix*NY+iy)+2] = ix*NY+iy
        bondj[3*(ix*NY+iy)+2] = (ix+iy/2)%NX * NY
        
    ix = 0
    iy = NY-1
    bondi[3*(ix*NY+iy)+2] = ix*NY+iy
    bondj[3*(ix*NY+iy)+2] = (ix+iy/2)%NX * NY
          
    bvecij = np.zeros(DIM*NB, float)
    for b in range(NB):
        bvec =  node_pos[DIM*bondj[b]:DIM*bondj[b]+DIM]-node_pos[DIM*bondi[b]:DIM*bondi[b]+DIM]
        bvec -= np.rint(bvec / L) * L
        
        bvecij[DIM*b:DIM*b+DIM] = bvec
        
        
    print "NN", NN
    print "NB", NB 
    
    eq_length = a * np.ones(NB, float)
        
    net = Network(DIM, NN, node_pos, NB, bondi, bondj, DIM, L)
    net.setStretchInt(bvecij, eq_length, np.ones(NB, float) / eq_length)
    
    return net





def convertToFlowNetwork(network):
    DIM = 1
    NN = network.NN
    NE = network.NE
    node_pos = np.arange(0, 1, 1.0/NN)
    edgei = network.edgei
    edgej = network.edgej

    L = np.array([1.0])
    
    bvecij = np.zeros(NE, float)
    eq_length = np.zeros(NE, float)
    for b in range(NE):
        bvec =  node_pos[DIM*edgej[b]:DIM*edgej[b]+DIM]-node_pos[DIM*edgei[b]:DIM*edgei[b]+DIM]
        bvec -= np.rint(bvec / L) * L
        bvec /= la.norm(bvec)
        
        eq_length[b] = 1.0
        
        bvecij[DIM*b:DIM*b+DIM] = bvec
        
        
    fnw = Network(DIM, NN, node_pos, NE, edgei, edgej, DIM, L)
    fnw.setStretchInt(bvecij, eq_length, np.ones(NE, float) / eq_length)
    
    return fnw










class AdjList(object):
    
    def __init__(self, NN):
           
        self.NN = NN
        self.adj_list = [set() for _ in range(NN)]
        
        
    def add_edges(self, NE, bi_list, bj_list):
        for b in range(NE):
            self.add_edge(bi_list[b], bj_list[b])
   
    def add_edge(self, bi, bj):
        self.adj_list[bi].add(bj)
        self.adj_list[bj].add(bi)
        
    def remove_edge(self, bi, bj):
        self.adj_list[bi].discard(bj)
        self.adj_list[bj].discard(bi)
        
    def get_degree(self, bi):
        return len(self.adj_list[bi])
        
def getLineGraph(NN, NE, bi_list, bj_list):
    
    # sorted edges by node
    buckets = [set() for _ in range(NN)]
    
    for b in range(NE):
        buckets[bi_list[b]].add(b)
        buckets[bj_list[b]].add(b)
        
    # connect all edges sharing  node
    lg_bi_list = []
    lg_bj_list = []
    
    lg_NE = 0
    
    for n in range(NN):
        for pair in it.combinations(buckets[n], 2):
            lg_NE += 1
            lg_bi_list.append(pair[0])
            lg_bj_list.append(pair[1])
            
    adj = AdjList(NE)
    adj.add_edges(lg_NE, lg_bi_list, lg_bj_list)
    
    return adj
        
       
def getGraphDistances(adj, start):

    visited = set()
    queue = [start]
    
    dists = np.full(adj.NN, np.inf, float)
    dists[start] = 0.0
    
    # print dists
    
    while queue:
        vertex = queue.pop()
        if vertex not in visited:
            visited.add(vertex)
            neighbors = adj.adj_list[vertex] - visited
            queue.extend(neighbors)
                        
            for neighb in neighbors:
                if dists[vertex] + 1 < dists[neighb]:
                    dists[neighb] = dists[vertex] + 1
            
    return dists
    
    
        
        
class PebbleGame(object):
    
    def __init__(self, NN):
        
        self.DIM = 2
        
        self.NN = NN
        self.NB = 0
        
        # Total number of free degrees of freedom
        self.NDOF = self.DIM * NN
              
        # Number of global degrees of freedom allowed (assuming free BCs)
        self.NGDOF = self.DIM*(self.DIM+1)/2
        
        # Adjacency list of directed graph composed only of independent degrees of freedom
        self.ind_adj_list = [set() for _ in range(NN)]
        
        # Adjecency list of directed graph composed only of dependent degrees of freedom
        self.dep_adj_list = [set() for _ in range(NN)]
        
    # Return true if number of free dofs is equal to number of global dofs
    def is_rigid(self):
        return self.NDOF == self.NGDOF
     
    def edge_exists(self, bi, bj):
        return (bj in self.ind_adj_list[bi] or bi in self.ind_adj_list[bj] 
                or bj in self.dep_adj_list[bi] or bi in self.dep_adj_list[bj])
        
    # Add all edges from lists if possible
    def add_edges(self, NB, bi_list, bj_list):
        for b in range(NB):
            self.add_edge(bi_list[b], bj_list[b])
    
    # Add edge (bi, bj) to network and return true if edge can be added as independent constraint
    # Return False if edge already exists or cannot be added as independent
    def add_edge(self, bi, bj):
         
        # If edge already exists return false
        if self.edge_exists(bi, bj):
            return False
        
        # Otherwise try to add bond as independent constraint
        else:
            
            self.NB += 1

            if not self.is_rigid() and self.test_add_edge(bi, bj):
                self.ind_adj_list[bi].add(bj)
                self.NDOF -= 1
                return True
            else:
                self.dep_adj_list[bi].add(bj)
                return False
    
    # Return true if a relative degree of freedom exists between nodes bi and bj
    # If successful, then there will be DIM pebbles on each of node bi and bj
    # Assumes edge does not exist, but if it does will return False
    def test_add_edge(self, bi, bj):
        
        # Attempt to collect DIM pebbles on node bi while holding bj fixed
        while(len(self.ind_adj_list[bi]) > 0):
            if not self.collect_pebble(bi, bj):
                return False
        
        # Attempt to collect DIM pebbles on node bj while holding bi fixed
        while(len(self.ind_adj_list[bj]) > 0):
            if not self.collect_pebble(bj, bi):
                return False    
                        
        return True
     
    # Remove all edges from lists if possible
    def remove_edges(self, NB, bi_list, bj_list):
        for b in range(NB):
            self.remove_edge(bi_list[b], bj_list[b])
        
    # Remove edge (bi, bj) and return True if edge removal introduces new dof
    def remove_edge(self, bi, bj):
        
        if not self.edge_exists(bi, bj):
            print "edge doesn't exist"
            return False
        
        self.NB -= 1

        # If edge is dependent, return False
        if bj in self.dep_adj_list[bi]:
            self.dep_adj_list[bi].remove(bj)
            return False
        elif bi in self.dep_adj_list[bj]:
            self.dep_adj_list[bj].remove(bi)
            return False
                
        # Otherwise remove from independent list and try collecting pebbles on one of the dependent edges
        else:
            
            if bj in self.ind_adj_list[bi]:
                vi = bi
                vj = bj
            else:
                vi = bj
                vj = bi
            
            # Remove edge from independent list and free up pebble
            self.ind_adj_list[vi].remove(vj)
            # Check to see if the pebble can be moved to a dependent bond
            for i in range(self.NN):
                for j in self.dep_adj_list[i]:
                    # If free dof exists, move the pebble
                    if self.test_add_edge(i, j):
                        self.dep_adj_list[i].remove(j)
                        self.ind_adj_list[i].add(j)
                        return False
                         
                            
            self.NDOF += 1 
            return True
        
    def test_remove_edge(self, bi, bj):
        rtest = self.remove_edge(bi, bj)
                
        atest = self.add_edge(bi, bj)
        
        if rtest != atest:
            print "Bond removal test error"
        
        return rtest
    
    def collect_pebble(self, start, exclude):
        visited = set()
        stack = [start]
        parent_map = dict()
        
        while stack:
            
            vertex = stack.pop()
                        
            if vertex not in visited:
                
                if len(self.ind_adj_list[vertex]) < self.DIM and vertex != start and vertex != exclude:
                    self.reverse_path(self.get_path(start, vertex, parent_map))
                    
                    return True
                
                visited.add(vertex)
                children = self.ind_adj_list[vertex] - visited
                stack.extend(children)
                for child in children:
                    parent_map[child] = vertex
                
        return False
                 
    def get_path(self, start, finish, parent_map):
        path = []
        vertex = finish
        while vertex != start:
            path.append(vertex)
            vertex = parent_map[vertex]
            
        path.append(start)
        path.reverse()
        
        return path
                
    def reverse_path(self, path):
        
        bi = path[0]
        for i in range(1, len(path)):
            bj = path[i]
            
            self.ind_adj_list[bi].remove(bj)
            self.ind_adj_list[bj].add(bi)
            
            bi = bj
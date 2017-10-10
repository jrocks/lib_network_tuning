import os
from IPython.display import display, HTML

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import collections as mc
import matplotlib as mpl
import seaborn as sns


def createGIF(iname, oname, show=True):
    
    os.system("convert -delay 5 -loop 0 {0} {1}".format(iname, oname))

    if show:
        IMG_TAG = """<img src="data:image/gif;base64,{0}" alt="some_text">"""

        data = open("{0}".format(oname), "rb").read()
        data = data.encode("base64")
        display(HTML(IMG_TAG.format(data)) ) 
        
        
        
        
        
gray = "#969696"
blue = "#377eb8"
red = "#e41a1c"


def show_network(ax, net, disp=None, strain=None, styles={}, box_mult=1.0):

        
    boxsize=0.5
    padding=0.0
    
    edgei = np.copy(net.edgei)
    edgej = np.copy(net.edgej)
    node_pos = np.copy(net.node_pos)
    DIM = net.DIM
    NE = net.NE
    NN = net.NN
    L = np.copy(net.L)
    center = 0.5 * np.ones(DIM, float)
    
    
    if disp is None:
        disp = np.zeros(DIM*NN, float)
    
    if strain is None:
        strain = np.zeros([DIM, DIM], float)
    else:
        boxL = 0.5 + padding
        corners = np.array([[-boxL, -boxL], 
                            [boxL, -boxL], 
                            [boxL, boxL], 
                            [-boxL, boxL]])
        corners[0] = def_tensor[0:2, 0:2].dot(corners[0])
        corners[1] = def_tensor[0:2, 0:2].dot(corners[1])
        corners[2] = def_tensor[0:2, 0:2].dot(corners[2])
        corners[3] = def_tensor[0:2, 0:2].dot(corners[3])

        ax.add_patch(mpatches.Polygon(corners, True, fill=False, lw=4.0))

    patches = []


    def_tensor = np.identity(DIM) + strain 



    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] += padding * L
    
    L += 2.0 * padding * L
    
    edges = []
    edge_index = []
    
    #change to showing 9 copies of networks

    for i in range(NE):
        posi = node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]+disp[DIM*edgei[i]:DIM*edgei[i]+DIM]
        posj = node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]+disp[DIM*edgej[i]:DIM*edgej[i]+DIM]

        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        
        edges.append([tuple(posi),tuple(posi+bvec)])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([1.0, 0.0])),tuple(posi+bvec+np.array([1.0, 0.0]))])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([1.0, -1.0])),tuple(posi+bvec+np.array([1.0, -1.0]))])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([0.0, -1.0])),tuple(posi+bvec+np.array([0.0, -1.0]))])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([-1.0, -1.0])),tuple(posi+bvec+np.array([-1.0, -1.0]))])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([-1.0, 0.0])),tuple(posi+bvec+np.array([-1.0, 0.0]))])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([-1.0, 1.0])),tuple(posi+bvec+np.array([-1.0, 1.0]))])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([0.0, 1.0])),tuple(posi+bvec+np.array([0.0, 1.0]))])    
        edge_index.append(i)
        
        edges.append([tuple(posi+np.array([1.0, 1.0])),tuple(posi+bvec+np.array([1.0, 1.0]))])    
        edge_index.append(i)


    for i, b in enumerate(edges):
        edges[i] = [def_tensor.dot(b[0])[0:2], def_tensor.dot(b[1])[0:2]]
    
    ls = []
    colors = []
    lw = []
    
    for i, b in enumerate(edge_index):
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append(gray)
            
            
        if b in styles and 'ls' in styles[b]:
            ls.append(styles[b]['ls'])
        else:
            ls.append('solid')
            
        if b in styles and 'lw' in styles[b]:
            lw.append(styles[b]['lw'])
        else:
            lw.append(2.0)
                    
    
    lc = mc.LineCollection(edges, zorder=-1, linestyle=ls, lw=lw, alpha=0.8, color=colors)  
    ax.add_collection(lc)
    
    
    if box_mult > 1.0:
        boxL = 0.5 + padding
        corners = np.array([[-boxL, -boxL], 
                            [boxL, -boxL], 
                            [boxL, boxL], 
                            [-boxL, boxL]])
        corners[0] = def_tensor[0:2, 0:2].dot(corners[0])
        corners[1] = def_tensor[0:2, 0:2].dot(corners[1])
        corners[2] = def_tensor[0:2, 0:2].dot(corners[2])
        corners[3] = def_tensor[0:2, 0:2].dot(corners[3])

        ax.add_patch(mpatches.Polygon(corners, True, fill=False, lw=4.0, ls='dashed'))
        
    
    ax.set_xlim(-box_mult*boxsize, box_mult*boxsize)
    ax.set_ylim(-box_mult*boxsize, box_mult*boxsize)
    
    
def show_nodes(ax, net, nodes, disp=None, strain=None, styles={}, marker='o'):
    
    
    edgei = np.copy(net.edgei)
    edgej = np.copy(net.edgej)
    node_pos = np.copy(net.node_pos)
    DIM = net.DIM
    NE = net.NE
    NN = net.NN
    L = np.copy(net.L)
    center = 0.5 * np.ones(DIM, float)
    
    if disp is None:
        disp = np.zeros(DIM*NN, float)
    
    if strain is None:
        strain = np.zeros([DIM, DIM], float)

    def_tensor = np.identity(DIM) + strain 
        
    x1 = []
    y1 = []

    for i in range(len(nodes)):
        
        posi = node_pos[DIM*nodes[i]:DIM*nodes[i]+DIM]+disp[DIM*nodes[i]:DIM*nodes[i]+DIM]

        posi /= L

        posi -= np.floor(posi)

        posi -= center

        posi = def_tensor.dot(posi)

        x1.append(posi[0])
        y1.append(posi[1])

    colors = []
    sizes = []

    for b in nodes:
        
        if b in styles and 'color' in styles[b]:
            colors.append(styles[b]['color'])
        else:
            colors.append('k')
            
        if b in styles and 'size' in styles[b]:
            sizes.append(styles[b]['size'])
        else:
            sizes.append(200)
                        
    ax.scatter(x1, y1, marker=marker , s=sizes, facecolor=colors, alpha=1.0)

def show_vecs(ax, net, u, strain=None, stream=False):
    
    edgei = np.copy(net.edgei)
    edgej = np.copy(net.edgej)
    node_pos = np.copy(net.node_pos)
    DIM = net.DIM
    NE = net.NE
    NN = net.NN
    L = np.copy(net.L)
    center = 0.5 * np.ones(DIM, float)
    
    disp = np.copy(u)
        
    if strain is None:
        strain = np.zeros([DIM, DIM], float)

    def_tensor = np.identity(DIM) + strain 
    
    X = np.zeros(NN, float)
    Y = np.zeros(NN, float)
    U = np.zeros(NN, float)
    V = np.zeros(NN, float)
    
    for i in range(NN):
        
        pos = node_pos[DIM*i:DIM*i+DIM]

        pos /= L

        pos -= np.floor(pos)

        pos -= center
        
        pos = def_tensor.dot(pos)
        
        u = disp[DIM*i:DIM*i+DIM]
        u /= L
        
        X[i] = pos[0]
        Y[i] = pos[1]
        U[i] = u[0]
        V[i] = u[1]
     
    if not stream:
        ax.quiver(X, Y, U, V, units='xy', scale=1.0/2.0)
    else:
        
        mag = np.sqrt(U*U + V*V) 
        
        x = np.linspace(X.min(), X.max(), 1000)
        y = np.linspace(Y.min(), Y.max(), 1000)

        xi, yi = np.meshgrid(x,y)

        #then, interpolate your data onto this grid:

        gu = sp.interpolate.griddata(zip(X,Y), U, (xi,yi))
        gv = sp.interpolate.griddata(zip(X,Y), V, (xi,yi))
        gmag = sp.interpolate.griddata(zip(X,Y), mag, (xi,yi))

        lw = 6*gmag/np.nanmax(gmag)
         #now, you can use x, y, gu, gv and gspeed in streamplot:

        ax.streamplot(xi, yi, gu, gv, density=1.0, color='k', linewidth=lw)
        
        
        
def scalar_field(ax, x, y, z, sigma, NGRID, L, cmap=mpl.cm.viridis):
        
    x_grid = np.linspace(0, L[0], NGRID)
    y_grid = np.linspace(0, L[1], NGRID)
    
    weights = np.exp(-((x_grid[:,None,None]-x[None, None, :])**2 
                  + (y_grid[None,:,None]-y[None, None, :])**2) / (2.0 * sigma**2))
    
    z_est = np.sum(weights * z[None, None, :], axis=2) / np.sum(weights, axis=2)
        
#     z_est /= np.sum(np.exp(-((x_grid[:,None,None]-x[None, None, :])**2 
#                   + (y_grid[None,:,None]-y[None, None, :])**2) / (2.0 * sigma)), axis=2)
            
    im = ax.imshow(z_est.T, cmap=cmap, origin='lower',
                vmin=np.min(z_est), vmax=np.max(z_est), extent=[0, L[0], 0, L[1]])
    im.set_interpolation('bilinear')
    
    return im

#     cf = ax1.contourf(z_est.T, cmap=mpl.cm.Blues, levels=np.linspace(0, 1.0, 11))
#     return cf
        
         
        
        
        
        
        
        
        
        

def frame(network, disp, Gamma, K, pert, meas, label, boxsize=0.5, padding=0.0, save=False, show_removed=False, ostrain=[]):

    sns.set(color_codes=True)
    sns.set_context('poster', font_scale=1.75)
    # sns.set_palette("hls", 6)
    sns.set_palette("bright")
    sns.set_style('ticks', {'xtick.direction': 'in','ytick.direction': 'in', 'axes.linewidth': 2.0})
    gray = "#969696"
    blue = "#377eb8"
    red = "#e41a1c"
    
    edgei = network.edgei
    edgej = network.edgej
    node_pos = np.copy(network.node_pos)
    DIM = network.DIM
    NE = network.NE
    NN = network.NN
    L = np.copy(network.L)
    center = 0.5 * np.ones(DIM, float)
    
    fig = plt.figure(1, (8, 8))

    ax = fig.add_axes([0,0,1,1], frameon=True, aspect=1.0)

    patches = []


    def_tensor = np.identity(DIM) + Gamma 

    boxL = 0.5 + padding
    corners = np.array([[-boxL, -boxL], 
                        [boxL, -boxL], 
                        [boxL, boxL], 
                        [-boxL, boxL]])
    corners[0] = def_tensor[0:2, 0:2].dot(corners[0])
    corners[1] = def_tensor[0:2, 0:2].dot(corners[1])
    corners[2] = def_tensor[0:2, 0:2].dot(corners[2])
    corners[3] = def_tensor[0:2, 0:2].dot(corners[3])

    ax.add_patch(mpatches.Polygon(corners, True, fill=False, lw=2.0))


    for i in range(NN):
        node_pos[DIM*i:DIM*i+DIM] += padding * L
    
    L += 2.0 * padding * L
    # center -= padding
    
#     center = np.array([0.0, 0.0])
#     for i in range(NN):
#         center += node_pos[DIM*i:DIM*i+DIM] / L
#     center /= NN
    
    edges = []
    edge_index = []

    for i in range(NE):
        posi = node_pos[DIM*edgei[i]:DIM*edgei[i]+DIM]+disp[DIM*edgei[i]:DIM*edgei[i]+DIM]
        posj = node_pos[DIM*edgej[i]:DIM*edgej[i]+DIM]+disp[DIM*edgej[i]:DIM*edgej[i]+DIM]

        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center


        if posi[0]+bvec[0] > 0.5 and posi[1]+bvec[1] > 0.5:
            c1  = (0.5 - posi[0]) / bvec[0]
            c2  = (0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([-1.0 ,0.0])+c1*bvec),tuple(posi+np.array([-1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,-1.0])+c1*bvec),tuple(posi+np.array([0.0 ,-1.0])+c2*bvec)]) 

            c  = np.min([-(-0.5 - posj[0]) / bvec[0],-(-0.5 - posj[1]) / bvec[1]])
            edges.append([tuple(posj),tuple(posj-c*bvec)])  
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] > 0.5 and posi[1]+bvec[1] < -0.5:
            c1  = (0.5 - posi[0]) / bvec[0]
            c2  = (-0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([-1.0 ,0.0])+c1*bvec),tuple(posi+np.array([-1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,1.0])+c1*bvec),tuple(posi+np.array([0.0 ,1.0])+c2*bvec)]) 
            c  = np.min([-(-0.5 - posj[0]) / bvec[0],-(0.5 - posj[1]) / bvec[1]])
            edges.append([tuple(posj),tuple(posj-c*bvec)]) 
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] < -0.5 and posi[1]+bvec[1] > 0.5:
            c1 = (-0.5 - posi[0]) / bvec[0]
            c2 = (0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([1.0 ,0.0])+c1*bvec),tuple(posi+np.array([1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,-1.0])+c1*bvec),tuple(posi+np.array([0.0 ,-1.0])+c2*bvec)])            

            c  = np.min([-(-0.5 - posj[1]) / bvec[1],-(0.5 - posj[0]) / bvec[0]])
            edges.append([tuple(posj),tuple(posj-c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] < -0.5 and posi[1]+bvec[1] < -0.5:
            c1  = (-0.5 - posi[0]) / bvec[0]
            c2  = (-0.5 - posi[1]) / bvec[1]
            c  = np.min([c1, c2])
            edges.append([tuple(posi),tuple(posi+c*bvec)])

            if c1 < c2:
                edges.append([tuple(posi+np.array([1.0 ,0.0])+c1*bvec),tuple(posi+np.array([1.0 ,0.0])+c2*bvec)])  
            else:
                edges.append([tuple(posi+np.array([0.0 ,1.0])+c1*bvec),tuple(posi+np.array([0.0 ,1.0])+c2*bvec)])

            c  = np.min([-(0.5 - posj[0]) / bvec[0],-(0.5 - posj[1]) / bvec[1]])
            edges.append([tuple(posj),tuple(posj-c*bvec)])  
            
            edge_index.append(i)
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] > 0.5:
            c  = (0.5 - posi[0]) / bvec[0]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (-0.5 - posj[0]) / bvec[0]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        elif posi[1]+bvec[1] > 0.5:
            c  = (0.5 - posi[1]) / bvec[1]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (-0.5 - posj[1]) / bvec[1]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        elif posi[0]+bvec[0] < -0.5:
            c  = (-0.5 - posi[0]) / bvec[0]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (0.5 - posj[0]) / bvec[0]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        elif posi[1]+bvec[1] < -0.5:
            c  = (-0.5 - posi[1]) / bvec[1]
            edges.append([tuple(posi),tuple(posi+c*bvec)])
            c  = (0.5 - posj[1]) / bvec[1]
            edges.append([tuple(posj),tuple(posj+c*bvec)])
            
            edge_index.append(i)
            edge_index.append(i)

        else:
            edges.append([tuple(posi),tuple(posi+bvec)])
            
            edge_index.append(i)

    for i, b in enumerate(edges):
        edges[i] = [def_tensor.dot(b[0])[0:2], def_tensor.dot(b[1])[0:2]]

    #show input nodes
    inodesi = pert.istrain_nodesi
    inodesj = pert.istrain_nodesj
    
    x1 = []
    y1 = []

    for i in range(pert.NIstrain):
        
        posi = node_pos[DIM*inodesi[i]:DIM*inodesi[i]+DIM]+disp[DIM*inodesi[i]:DIM*inodesi[i]+DIM]
        posj = node_pos[DIM*inodesj[i]:DIM*inodesj[i]+DIM]+disp[DIM*inodesj[i]:DIM*inodesj[i]+DIM]
        
        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        posi = def_tensor.dot(posi)
        posj = def_tensor.dot(posj)
        
        x1.append(posi[0])
        x1.append(posj[0])
        
        y1.append(posi[1])
        y1.append(posj[1])
        
        
    ax.scatter(x1, y1, marker='o' , edgecolor='', s=200, linewidth=0.0, facecolor=blue, alpha=1.0)

    #show output nodes
    onodesi = meas.ostrain_nodesi
    onodesj = meas.ostrain_nodesj
    
    x1 = []
    y1 = []

    for i in range(meas.NOstrain):
        if ostrain[i] < 0:
            continue
        
        posi = node_pos[DIM*onodesi[i]:DIM*onodesi[i]+DIM]+disp[DIM*onodesi[i]:DIM*onodesi[i]+DIM]
        posj = node_pos[DIM*onodesj[i]:DIM*onodesj[i]+DIM]+disp[DIM*onodesj[i]:DIM*onodesj[i]+DIM]
        
        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        posi = def_tensor.dot(posi)
        posj = def_tensor.dot(posj)
        
        x1.append(posi[0])
        x1.append(posj[0])
        
        y1.append(posi[1])
        y1.append(posj[1])

    ax.scatter(x1, y1, marker='s' , edgecolor='', s=200, linewidth=0.0, facecolor='r', alpha=1.0)
    
    x1 = []
    y1 = []

    for i in range(meas.NOstrain):
        if ostrain[i] > 0:
            continue
        
        posi = node_pos[DIM*onodesi[i]:DIM*onodesi[i]+DIM]+disp[DIM*onodesi[i]:DIM*onodesi[i]+DIM]
        posj = node_pos[DIM*onodesj[i]:DIM*onodesj[i]+DIM]+disp[DIM*onodesj[i]:DIM*onodesj[i]+DIM]
        
        posi /= L
        posj /= L

        bvec = posj - posi
        bvec -= np.rint(bvec)

        posi -= np.floor(posi)
        posj -= np.floor(posj)

        posi -= center
        posj -= center
        
        posi = def_tensor.dot(posi)
        posj = def_tensor.dot(posj)
        
        x1.append(posi[0])
        x1.append(posj[0])
        
        y1.append(posi[1])
        y1.append(posj[1])

    ax.scatter(x1, y1, marker='o' , edgecolor='', s=200, linewidth=0.0, facecolor='k', alpha=1.0)
        
    carray = []
    for b in edge_index:
        carray.append(K[b])

    #add lines to plot
    if not show_removed:
        ls = ['solid' for b in range(len(edges))]
        lw = (4*np.array(carray)+1.0)
        lc = mc.LineCollection(edges, zorder=-1, linestyle=ls, lw=lw, alpha=0.8,color=gray)
    else:
        ls = [(0.0, (8.0,4.0)) if K[b] < 0.1 else 'solid' for b in edge_index]
        lw = [3.0 if K[b] < 0.1 else 5.0 for b in edge_index]
        colors = ["r" if K[b] < 0.1 else gray for b in edge_index]
        lc = mc.LineCollection(edges, zorder=-1, linestyle=ls, lw=lw, alpha=0.8,color=colors)
    
    
#     lw = 3

#     sns.set_color_codes('bright')
#     # c = mcolors.ColorConverter().to_rgb
#     # cmap = nplot.make_colormap([c('r'), 0.5, c(gray), c(gray), 0.7, c('b')])

#     cmap = mcolors.LinearSegmentedColormap.from_list("customMap", ['r', gray])

#     cmap = mcolors.LinearSegmentedColormap.from_list("customMap", ['r', u'#ff8080', gray])

    
    
    
        
    
#     lc.set_array(np.array(carray))
    ax.add_collection(lc)

#     cb = plt.colorbar(lc)
#     cb.set_clim(0, 1.0)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlim(-boxsize, boxsize)
    ax.set_ylim(-boxsize, boxsize)
    
    if save:
        plt.savefig(label, bbox_inches='tight')

    plt.show()
import numpy as np

class Histogram(object):

    def __init__(self, data, hist_type="FW", use_FD=False, scale="linear", NBINS=1, NPPB=1, bin_width=1.0, verbose=False):
        data = np.array(data)
        
        if scale == "log":
            data = np.log(data)
        
        
        self.N = len(data)
        self.dmax = np.max(data)
        self.dmin = np.min(data)        
        
        self.NPPB = np.max([np.min([NPPB, self.N]), 1])
        self.NBINS = np.max([np.min([NBINS, self.N]), 1])
        self.bin_width = np.min([bin_width, self.dmax - self.dmin])
        if self.bin_width < 0:
            self.bin_width = 1.0
        
        if hist_type == "PPB":
            if use_FD:
                (FD_width, FD_NBINS) = self.getFreedmanDiaconis(data)
                self.NBINS = np.max([FD_NBINS, 1])
                self.NPPB = np.max([int(self.N / self.NBINS), 1])
            else:
                if NPPB > 1:
                    self.NBINS = int(self.N / self.NPPB)
                elif NBINS > 1:
                    self.NPPB = int(self.N / self.NBINS)                
                
            if verbose: 
                print "Using fixed number of points per bin with {} bins and {} points per bin.".format(self.NBINS, self.NPPB)
                    
                    
                    
            NREM = self.N % self.NBINS

            self.hist = np.full(self.NBINS, self.NPPB, int)

            rng = np.random.RandomState(42)

            index = np.arange(self.NBINS)
            rng.shuffle(index)

            for i in np.arange(NREM):
                self.hist[index[i]] += 1

            self.bin_edges = np.zeros(self.NBINS+1, float)
            sdata = np.sort(data)

            self.bin_edges[0] = sdata[0]
            self.bin_edges[-1] = sdata[-1]
            iedge = 0
            for i in np.arange(1, self.NBINS):
                iedge += self.hist[i-1]
                self.bin_edges[i] = (sdata[iedge-1] + sdata[iedge])/2.0  
                    
        elif hist_type == "FW":
            if use_FD:
                (FD_width, FD_NBINS) = self.getFreedmanDiaconis(data)
                self.NBINS = np.max([FD_NBINS, 1])
                self.bin_width = FD_width
                
            else:
                if bin_width != 1.0:
                    self.NBINS = int(np.ceil((self.dmax - self.dmin) / self.bin_width))
                elif NBINS > 1:
                    self.bin_width = (self.dmax - self.dmin) / self.NBINS                 
                
            if verbose:
                print "Using fixed bin width with {} bins and a bin width of {}.".format(self.NBINS, self.bin_width)
            
         
            self.hist = np.zeros(self.NBINS, int)
            for i in np.arange(self.N):
                ibin = int((data[i]-self.dmin) / self.bin_width)
                if ibin == self.NBINS:
                    ibin -= 1
                self.hist[ibin] += 1

            self.bin_edges = np.zeros(self.NBINS+1, float)
            self.bin_edges[0] = self.dmin;
            for i in np.arange(1, self.NBINS+1):
                self.bin_edges[i] = self.bin_edges[i-1] + self.bin_width
        
        
            
        self.bin_centers = np.zeros(self.NBINS, float)
        for i in np.arange(self.NBINS):
            self.bin_centers[i] = (self.bin_edges[i]+self.bin_edges[i+1])/2.0
        
        if scale=="log":
            self.bin_edges = np.exp(self.bin_edges)
            self.bin_centers = np.exp(self.bin_centers)
        
        
        self.N = len(self.hist)
        self.pdf = np.zeros(self.N, float)
        for i in np.arange(self.N):
            self.pdf[i] = self.hist[i] / (self.bin_edges[i+1]-self.bin_edges[i])

        self.norm = np.sum(self.hist)
        self.pdf /= self.norm
        
        
            
            
            
    def getFreedmanDiaconis(self, x):
        
        if self.dmax == self.dmin:
            return (1.0, 1)
        
        Q75, Q25 = np.percentile(x, [75.0 ,25.0], interpolation='midpoint')
        
        if Q75 == Q25:
            return (1.0, 1)
        
        
        IQR = Q75 - Q25
        FD_width =  2.0 * IQR / self.N**(1.0/3.0)
        FD_NBINS = int(np.ceil((self.dmax - self.dmin) / FD_width))
        
        return (FD_width, FD_NBINS)
            


# class PPBHistogram(object):

# 	def __init__(self, data, binparam, bintype='nbins', verbose=False):
# 		self.NBINS = 1
# 		self.ppb = 1
# 		self.N = len(data)
# 		if bintype == 'nbins':
# 			self.NBINS = binparam
# 			self.ppb = int(self.N / self.NBINS)
# 			if self.ppb == 0:
# 				if verbose:
# 					print "Not enough points per bin."
# 				self.ppb = 1
# 				self.NBINS = self.N
# 		elif bintype == 'ppb':
# 			self.ppb = binparam
# 			self.NBINS = int(self.N / self.ppb)
# 			if self.NBINS == 0:
# 				if verbose:
# 					print "Not enough bins."
# 				self.NBINS = 1
		
		
		
# 		NREM = self.N % self.NBINS
		
# 		self.hist = np.full(self.NBINS, self.ppb, int)

# 		rng = np.random.RandomState(42)
		
# 		index = np.arange(self.NBINS)
# 		rng.shuffle(index)
		
# 		for i in np.arange(NREM):
# 			self.hist[index[i]] += 1
		
# 		self.bin_edges = np.zeros(self.NBINS+1, float)
# 		sdata = np.sort(data)
		
# 		self.bin_edges[0] = sdata[0]
# 		self.bin_edges[-1] = sdata[-1]
# 		iedge = 0
# 		for i in np.arange(1, self.NBINS):
# 			iedge += self.hist[i-1]
# 			self.bin_edges[i] = (sdata[iedge-1] + sdata[iedge])/2.0  
		
# 		self.bin_centers = np.zeros(self.NBINS, float)
# 		for i in np.arange(self.NBINS):
# 			self.bin_centers[i] = (self.bin_edges[i]+self.bin_edges[i+1])/2.0
		
		
# 		self.N = len(self.hist)
# 		self.pdf = np.zeros(self.N, float)
# 		for i in np.arange(self.N):
# 			self.pdf[i] = self.hist[i] / (self.bin_edges[i+1]-self.bin_edges[i])

# 		self.norm = np.sum(self.hist)
# 		self.pdf /= self.norm

# class FWHistogram(object):

# 	def __init__(self, data, binparam, bintype='nbins', verbose=False):
# 		self.NBINS = 1
# 		self.binw = 1.0
# 		self.N = len(data)
        
# 		dmin = np.min(data)
# 		dmax = np.max(data)
        
# 		if bintype == 'nbins':
# 			self.NBINS = binparam
# 			self.binw = (dmax - dmin) / self.NBINS

# 		elif bintype == 'binw':
# 			self.binw = binparam
# 			self.NBINS = int(np.ceil((dmax-dmin) / self.binw))
		
# 		self.hist = np.zeros(self.NBINS, int)
# 		for i in np.arange(len(data)):
# 			ibin = int((data[i]-dmin) / self.binw)
# 			if ibin == self.NBINS:
# 				ibin -= 1
# 			self.hist[ibin] += 1

# 		self.bin_edges = np.zeros(self.NBINS+1, float)
# 		self.bin_edges[0] = dmin;
# 		for i in np.arange(1, self.NBINS+1):
# 			self.bin_edges[i] = self.bin_edges[i-1] + self.binw;

# 		self.bin_centers = np.zeros(self.NBINS, float)
# 		for i in np.arange(self.NBINS):
# 			self.bin_centers[i] = (self.bin_edges[i]+self.bin_edges[i+1])/2.0
		
		
# 		self.N = len(self.hist)
# 		self.pdf = np.zeros(self.N, float)
# 		for i in np.arange(self.N):
# 			self.pdf[i] = self.hist[i] / (self.bin_edges[i+1]-self.bin_edges[i])

# 		self.norm = np.sum(self.hist)
# 		self.pdf /= self.norm


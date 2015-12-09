import numpy as np
import matplotlib.pyplot as plt
import seaborn


class ARMA:
    def __init__(self,data):
        data = np.array(data)
        self.mean = data.mean()
        self.data = data - self.mean
        
        #cache first 100 values of acf
        self.acf = [None] * 100
        self.acf = [self.sample_autocovariance_function(k) for k in range(100)]
    
    def sample_autocovariance_function(self,lag):
        lag = abs(lag)
        if lag < 100 and self.acf[lag] is not None:
            return self.acf[lag]
        n = len(self.data)
        if lag >= n:
            raise ValueError('lag out of range')
        return (1.0/n)*sum(self.data[-(n-lag):]*self.data[:n-lag])

    def sample_autococorrelation_function(self,lag):
        return self.sample_autocovariance_function(lag)/self.sample_autocovariance_function(0)

    def sample_covariance_matrix(self,k):
        row = []
        for l in range(k):
            row.append(self.sample_autocovariance_function(l))
        matrix = []
        for l in range(k):
            matrix.append(row[1:1+l][::-1]+row[:k-l])
        return np.matrix(matrix)

    def partial_autocorrelation_function(self,k):
        if k==0:
            return 1
        Gamma_k = self.sample_covariance_matrix(k)
        gamma_k = np.matrix([self.sample_autocovariance_function(l) for l in range(1,k+1)]).T
        return (np.linalg.inv(Gamma_k)*gamma_k).item((k-1,0))

    def plot_ACF(self,limit=None):
        if not limit:
            limit=len(self.data)
        AC = []
        for lag in range(1,limit):
            AC.append(self.sample_autococorrelation_function(lag))
        plt.bar(list(range(1,limit)),AC)
        plt.axhline(1.96/np.sqrt(len(self.data)), linestyle='--',alpha=0.6)
        plt.axhline(-1.96/np.sqrt(len(self.data)), linestyle='--',alpha=0.6)
        plt.xlabel('lag')
        plt.ylabel('ACF')
        
    def plot_PACF(self,limit=None):
        if not limit:
            limit=len(self.data)
        PAC = []
        for lag in range(1,limit):
            PAC.append(self.partial_autocorrelation_function(lag))
        plt.bar(list(range(1,limit)),PAC)
        plt.axhline(1.96/np.sqrt(len(self.data)), linestyle='--',alpha=0.6)
        plt.axhline(-1.96/np.sqrt(len(self.data)), linestyle='--',alpha=0.6)
        plt.xlabel('lag')
        plt.ylabel('PACF')
        
    def fit_ar(self,p):
        Gamma = self.sample_covariance_matrix(p)
        gamma = np.matrix([self.sample_autocovariance_function(l) for l in range(1,p+1)]).T
        self.coefs = np.linalg.inv(Gamma)*gamma
        self.sigma = self.sample_autocovariance_function(0) - self.coefs.T*gamma
        
    def fit_ar_durbin_levinson(self,p):
        self.nu = np.zeros(p)
        self.phi = [np.zeros(m+1) for m in range(p)]
        self.nu[0] = self.sample_autocovariance_function(0)*(1-self.sample_autococorrelation_function(1)**2)
        self.phi[0][0] = self.sample_autococorrelation_function(1)
        for m in range(1,p):
            self.phi[m][m] = (self.sample_autocovariance_function(m+1)-sum(self.phi[m-1][j]*self.sample_autocovariance_function(m-j) for j in range(m)))/self.nu[m-1]
            for j in range(m):
                self.phi[m][j] = self.phi[m-1][j] - self.phi[m][m]*self.phi[m-1][m-1-j]
            self.nu[m] = self.nu[m-1]*(1-self.phi[m][m]**2)
        
    def get_confidence_interval_ar_d_l(self,p):
        return [1.96*np.sqrt((self.nu[p-1]*np.linalg.inv(self.sample_covariance_matrix(p)))[j,j])/np.sqrt(len(self.data)) for j in range(p)]
            
    def fit_ma_durbin_levinson(self,q):
        self.nu_ma = np.zeros(q+1)
        self.theta_ma = [np.zeros(m+1) for m in range(q)]
        self.nu_ma[0] = self.sample_autocovariance_function(0)
        for m in range(q):
            for k in range(m+1):
                self.theta_ma[m][m-k] = (self.sample_autocovariance_function(m-k+1)-sum(self.theta_ma[m][m-j]*self.theta_ma[k-1][k-j-1]*self.nu_ma[j] for j in range(k)))/self.nu_ma[k]
            self.nu_ma[m+1] = self.sample_autocovariance_function(0) - sum(self.theta_ma[m][m-j]**2*self.nu_ma[j] for j in range(m+1))
            
    def get_confidence_interval_ma_d_l(self,q):
        return [1.96*np.sqrt(sum(self.theta_ma[q-1][k]**2 for k in range(j+1)))/np.sqrt(len(self.data)) for j in range(q)]
            
    def fit_arma_preliminary(self,p,q,m=0):
        m = max(m,p+q)
        self.fit_ma_durbin_levinson(m)
        estimation_matrix = np.matrix([[self.theta_ma[m-1][q+j-i-1] for i in range(p)] for j in range(p)])
        estimation_matrix = np.linalg.inv(estimation_matrix)
        self.phi = estimation_matrix*np.matrix([self.theta_ma[m-1][q+i] for i in range(p)]).T
        self.theta = np.array([self.theta_ma[m-1][j]-self.phi[j]-sum(self.phi[i]*self.theta_ma[m-1][j-i] for i in range(min(j,p))) for j in range(min(p+2,q,max(p,q)))]+
                             [self.theta_ma[m-1][j]-sum(self.phi[i]*self.theta_ma[m-1][j-i] for i in range(min(j,p))) for j in range(min(p+2,q),q)])
 
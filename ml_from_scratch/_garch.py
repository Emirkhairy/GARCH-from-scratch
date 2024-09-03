import numpy as np
from scipy.optimize import minimize

class GARCH:
    """
    A function to predict volatility:

    data: A time series representing the return over a specific period.
        Ensure that the data represents returns, not actual raw data.

    p: The number of lags in the data.
        Default value of P is 1


    q: The number of lags in the predicted data.
        Default value of q is 1
    """
    def __init__(self,data,p=1,q=1):

        self.data = data
        self.p = p
        self.q = q
        
        
    
    def define_params(self,params):
        """
        A function to determine parameters by maximizing the negative likelihood.

        Parameters to be determined:

        alpha0: Constant variable
        alpha1: Parameter 1
        beta1: Parameter 2
        mu: Mean of the data
        """
        #define params
        alpha0 = params[0]
        alpha1 = params[1:self.p+1]
        beta1 = params[self.p+1:self.p+1+self.q]
        mu = params[-1]
        
        var = np.var(self.data)
        long_run = var
        

        #define baseline params
        resid = self.data - mu
        n = len(self.data)
        sigma2 = np.zeros(n)
        sigma2[0] = long_run
        log_likelihood = 0

        #GARH equation
        for t in range(max(self.p,self.q),n):
            sigma2[t] = alpha0

            #define alpha-nol equation
            for i in range(1,self.p+1):
                sigma2[t] += alpha1[i-1] * resid[t-i]**2

            #define beta-one equation
            for j in range(1,self.q+1):
                sigma2[t] += beta1[j-1] * sigma2[t-j]**2

            sigma2[t] = sigma2[t]**(1/2)
       
        #defines Maximum Log Likelihood
        for t in range(max(self.p,self.q),n):      
            log_likelihood += -0.5 *(np.log(2*np.pi)+np.log(sigma2[t]**2)+(resid[t]**2/sigma2[t]**2))
        
        return -log_likelihood
    
    def fit(self):
        """
        Perfoming a MLE (Maximum Log Likelihood).

        Maximizing it by finding the minimum value of the negative log likelihood
        """
        #estimate values of params
        initial_guess = [np.var(self.data)] + [0.1]*self.p + [0.8]*self.q + [np.average(self.data)]
        

        #optimize log-likehood function
        res = minimize(self.define_params,initial_guess,method='Nelder-Mead')

        params = res.x
        lle = res.fun

        return params,lle
    
    
    def predict(self):
        """
        Predict the data

        determined the sigma squared of given data.
        """
        params, lle = self.fit()

        #Get paramaters from the calculation before
        alpha0 = params[0]
        alpha1 = params[1:1+self.p]
        beta1 = params[self.p+1:1+self.p+self.q]
        mu = params[-1]

        # Define other variable that needed to do some prediction.
        var = np.var(self.data)
        resid = self.data - mu
        n = len(self.data)
        sigma2 = np.zeros(n)
        sigma2[0] = var

        for t in range(max(self.p,self.q), n):

            #Calculate the GARCH model
            #First, import alpha0 as a constant variable into the equation
            sigma2[t] = alpha0

            # seconde, add alpha1 into the equation.
            for i in range(1, self.p+1):
                sigma2[t] += alpha1[i-1] * resid[t-i]**2
            # Third, add beta1 to the equation
            for j in range(1, self.q+1):
                sigma2[t] += beta1[j-1] * sigma2[t-j]**2
            # Squared root the result to complete the equation.
            sigma2[t] = sigma2[t]**(1/2)

        return sigma2
    
    def forcast(self,steps=10):
        """
        We will perform a forecasting task, predicting values for n steps to obtain the result.

        Steps: The default value is 10.
        This determines how many steps the model needs to predict the volatility."
        """

        params,lle = self.fit()
        sigma2 = self.predict() 

        alpha0 = params[0]
        alpha1 = params[1:1+self.p]
        beta1 = params[self.p+1:1+self.p+self.q]
        mu = params[-1]

        fsigma2 = np.zeros(1+steps)
        fsigma2[0] = sigma2[-1]
        fdata = np.zeros(1+steps)
        fdata[0] = self.data[-1]
        resid = fdata - mu
        

        for t in range(max(self.p,self.q), 1+steps):

            zt = (fdata[t-1]-mu)/fsigma2[t-1]
            fdata[t] = mu + fsigma2[t-1]*(zt**(1/2))

            fsigma2[t] = alpha0
            for i in range(1, self.p+1):
                fsigma2[t] += alpha1[i-1] * resid[t-i]**2
            for j in range(1, self.q+1):
                fsigma2[t] += beta1[j-1] * sigma2[t-j]**2
            
            fsigma2[t] = fsigma2[t]**(1/2)

        return fsigma2


            


        
#clustered standard error computation
#Uses the methods described in
#A. Colin Cameron, Jonah B. Gelbach & Douglas L. Miller (2011) Robust Inference With Multiway
#Clustering, Journal of Business & Economic Statistics, 29:2, 238-249, DOI: 10.1198/jbes.2010.07136


import numpy
from scipy.stats import norm

N = norm.cdf
log = numpy.log

def grad(f,x,tol=1e-6):
    '''
    compute approximate gradient vector of function f at x
    '''
    f0 = f(x)
    x= numpy.array(x)
    g= numpy.zeros(list(numpy.shape(x,)) + list(f0.shape))
    for i in range(len(x)):
        x1 = x.copy()
        x1[i]+=tol
        g[i] = (f(x1) - f0) / tol
    return g


def clustered_se_from_model(model,params,cluster_var):
    '''
    Computes one-way clustered standard errors from maximum likelihood object model,
    with estimated params, clustering on cluster_var
    returns a matrix containing the clustered variance-covariance terms
    '''
    groups = numpy.unique(cluster_var)
    B=params
    score_cpts = grad(model.loglikeobs,B).transpose() #probit_score_cpts(y,X,B).transpose()
    try:
        try:
            A = numpy.matrix(model.information(B))
        except NotImplementedError:
            A = numpy.matrix(grad(model.score,B)).transpose()
            print 'WARNING: Using approximate gradient of score to compute information matrix'
    except NotImplementedError:
        score = lambda b: grad(model.loglike,b)
        A = numpy.matrix(grad(score,B)).transpose()
        print 'WARNING: Using approximate hessian to compute information matrix'
    D = numpy.zeros((len(B),len(B)))
    for g in groups:
        h_g = score_cpts[cluster_var == g].sum(0)
        D += numpy.outer(h_g,h_g) #was outer
    Ainv = A**(-1)
    D = numpy.matrix(D)
    V = Ainv*D*Ainv.T

    #degrees of freedom correction
    M = len(groups)
    N = len(score_cpts)
    K = len(B)
    dfc = 1.0*M/(M-1)*(N-1)/(N-K)

    return dfc*V

def clustered_output(mod,fit,group):
    '''
    Format a pandas table of output with clustered standard errors
    '''
    import pandas
    cse= numpy.diag(clustered_se_from_model(mod,fit.params,group))**0.5
    scse=pandas.Series(cse,index=fit.params.index)
    outp = pandas.DataFrame([fit.params,fit.bse,scse]).transpose()
    outp.columns = ['Coef','SE','Cl. SE']
    return outp


def test_probit_logit():
    import pandas
    import statsmodels.api as sm

    #SAMPLE PROGRAM COMPARING CLUSTERED AND REGULAR STANDARD ERRORS FOR PROBIT AND LOGIT
    print 'TEST OF PROBIT/LOGIT CLUSTERED STANDARD ERROR CORRECTION'

    #generate probit data with cluster correlated error structure
    gps = 30 # number of clusters (groups)
    obs = 1000 # number of observations per group

    #generate errors
    e1 = numpy.random.randn(obs,gps) #iid errors across groups and observations
    e2 = numpy.random.randn(gps) #errors correlated across observations within a group
    u = 1+2*numpy.random.rand(gps) #scaling of errors with a group (scale heteroskedasticity by group)
    e = (e1*u + e2).ravel()
    e = e / (e.dot(e)/len(e))**0.5 #normalize to make it easier to interpret parameters

    #generate regressor, dependant variable and group variable
    x = 5*(numpy.random.randn(obs,gps) + 2.0*numpy.random.randn(gps)).ravel() #regressor (has group correlation)
    gp = (numpy.ones((obs,gps))*numpy.arange(1,gps+1)).ravel()
    X = pandas.DataFrame([numpy.ones(obs*gps),x]).transpose() #put the regressor and constant into a dataframe
    X.columns = ['one','x']
    y_lat = 2 * x + e  #latent variable
    y = pandas.Series(1*(y_lat>0)) #observed dependent variable
#    for i in range(2,gps+1):
#        X['G%i'%(i,)]=1*(gp == i)

    print 'LOGIT'
    modl = sm.Logit(y,X)
    resl = modl.fit()
    print clustered_output(modl,resl,gp)
    print

    print 'PROBIT'
    modp = sm.Probit(y,X)
    resp = modp.fit()
    print clustered_output(modp,resp,gp)
    print

if __name__ == '__main__':
    test_probit_logit()
    import clustering_petersen_example
    clustering_petersen_example.test_petersen()


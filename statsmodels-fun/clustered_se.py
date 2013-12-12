#clustered standard error computation
#Uses the methods described in
#A. Colin Cameron, Jonah B. Gelbach & Douglas L. Miller (2011) Robust Inference With Multiway
#Clustering, Journal of Business & Economic Statistics, 29:2, 238-249, DOI: 10.1198/jbes.2010.07136


import numpy
from scipy.stats import norm
import itertools

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

def as_cluster_var(data):
    '''
    converts a 2d array to a 1d array of "tuples" treating each row as a group for clustering
    '''
    return data.view(data.dtype.descr * data.shape[1]).ravel()


def _cluster_A(fit_results):
    model = fit_results.model
    B=fit_results.params
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
    return A

def _cluster_D(fit_results,cluster_var):
    model = fit_results.model
    B=fit_results.params
    groups = numpy.unique(cluster_var)
    score_cpts = grad(model.loglikeobs,B).transpose() #probit_score_cpts(y,X,B).transpose()
    D = numpy.zeros((len(B),len(B)))
    for g in groups:
        h_g = score_cpts[cluster_var == g].sum(0)
        D += numpy.outer(h_g,h_g) #was outer
    #degrees of freedom correction
    M = len(groups)
    N = len(score_cpts)
    K = len(B)
    dfc = 1.0*M/(M-1)*(N-1)/(N-K)

    return dfc*numpy.matrix(D)

def clustered_se(fit_results,cluster_var):
    '''
    Computes one-way clustered standard errors from maximum likelihood object model,
    with estimated params, clustering on cluster_var
    returns a matrix containing the clustered variance-covariance terms
    '''
    A =  _cluster_A(fit_results)
    D = _cluster_D(fit_results, cluster_var)
    Ainv = A**(-1)
    V = Ainv*D*Ainv.T
    return V

def multiway_clustered_se(fit_results,cluster_vars):
    '''
    Computes multi-way clustered standard errors from maximum likelihood object model,
    with estimated params, clustering on the variables in 2d array (or dataframe) cluster_vars
    returns a matrix containing the clustered variance-covariance terms
    '''
    cluster_vars=numpy.array(cluster_vars)
    D = 0
    A = _cluster_A(fit_results)
    Ainv = A**(-1)
    for v in itertools.product([False,True],repeat=cluster_vars.shape[1]):
        if sum(v)==0:
            continue
        v=numpy.array(v,dtype=bool)
        cv = as_cluster_var(cluster_vars[:,v])
        D0 = _cluster_D(fit_results,cv)
        D = D - (-1)**sum(v) * D0
    V = Ainv*D*Ainv.T
    return V


def clustered_output(fit_results,group):
    '''
    Format a pandas table of output with clustered standard errors
    '''
    import pandas
    cse= numpy.diag(clustered_se(fit_results,group))**0.5
    scse=pandas.Series(cse,index=fit_results.params.index)
    outp = pandas.DataFrame([fit_results.params,fit_results.bse,scse]).transpose()
    outp.columns = ['Coef','SE','Cl. SE']
    return outp



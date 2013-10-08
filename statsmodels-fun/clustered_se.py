#clustered standard error computation
#Uses the methods described in
#A. Colin Cameron, Jonah B. Gelbach & Douglas L. Miller (2011) Robust Inference With Multiway
#Clustering, Journal of Business & Economic Statistics, 29:2, 238-249, DOI: 10.1198/jbes.2010.07136


import numpy
from scipy.stats import norm
import itertools

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

def as_cluster_var(data):
    '''
    converts a 2d array to a 1d array of "tuples" treating each row as a group for clustering
    '''
    return data.view(data.dtype.descr * data.shape[1]).ravel()


def _cluster_A(model,params):
    B=params
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

def _cluster_D(model,params,cluster_var):
    groups = numpy.unique(cluster_var)
    B=params
    score_cpts = grad(model.loglikeobs,B).transpose() #probit_score_cpts(y,X,B).transpose()
    D = numpy.zeros((len(B),len(B)))
    for g in groups:
        h_g = score_cpts[cluster_var == g].sum(0)
        D += numpy.outer(h_g,h_g) #was outer
    #degrees of freedom correction
    M = len(groups)
    print 'M',M
    N = len(score_cpts)
    K = len(B)
    dfc = 1.0*M/(M-1)*(N-1)/(N-K)

    return dfc*numpy.matrix(D)

def clustered_se_from_model(model,params,cluster_var):
    '''
    Computes one-way clustered standard errors from maximum likelihood object model,
    with estimated params, clustering on cluster_var
    returns a matrix containing the clustered variance-covariance terms
    '''
    A =  _cluster_A(model, params)
    D = _cluster_D(model, params, cluster_var)
    Ainv = A**(-1)
    V = Ainv*D*Ainv.T
    return V

def multiway_clustered_se_from_model(model,params,cluster_vars):
    '''
    Computes one-way clustered standard errors from maximum likelihood object model,
    with estimated params, clustering on the variables in 2d array (or dataframe) cluster_vars
    returns a matrix containing the clustered variance-covariance terms
    '''
    cluster_vars=numpy.array(cluster_vars)
    D = 0
    A = _cluster_A(model,params)
    Ainv = A**(-1)
    for v in itertools.product([False,True],repeat=cluster_vars.shape[1]):
        if sum(v)==0:
            continue
        v=numpy.array(v,dtype=bool)
        cv = as_cluster_var(cluster_vars[:,v])
        D0 = _cluster_D(model,params,cv)
        D = D - (-1)**sum(v) * D0
    V = Ainv*D*Ainv.T
    return V


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



if __name__ == '__main__':
    test_probit_logit()
    import clustering_petersen_example
    clustering_petersen_example.test_petersen()


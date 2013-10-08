'''Test module for clustered standard errors'''

def test_probit_logit():
    '''
    SAMPLE PROGRAM COMPARING CLUSTERED AND REGULAR STANDARD ERRORS FOR PROBIT AND LOGIT
    '''

    import pandas
    import statsmodels.api as sm

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


def test_petersen():
    '''
    A test of the clustered standard error module using Mitch Petersen's test data
    http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm
    '''
    import statsmodels.api as sm
    import clustered_se
    import pandas
    import numpy
    from scipy.stats import norm

    print 'TEST OF PETERSEN CLUSTERED STANDARD ERROR CORRECTIONS'
    df = pandas.read_csv('petersen-data/test_data.txt')
    '''
    OLS Coefficients and Standard Errors
    Reported at http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm
    '''
    b = numpy.array([0.0297,1.0348])
    se = numpy.array([0.028359,0.028583])
    se_by_firm= numpy.array([0.067013,0.050596])
    se_by_yr = numpy.array([0.0233387,0.033389])
    se_by_firm_and_yr = numpy.array([0.065064,0.053558])


    #add a constant and run the regression
    df['one']=1
    mod = sm.OLS(df['y'],df[['one','x']])
    res = mod.fit()
    print res.summary()

    #now generate the clustered standard errors
    #becasue loglikeobs is not implemented in the default OLS implementation, we'll monkey patch it here
    def llo(b):
        y = numpy.array(df['y'])
        X = numpy.array(df[['one','x']])
        b=numpy.array(b)
        e = y - numpy.inner(X,b)
        s = (e.dot(e)/(len(e)))**0.5
        return numpy.log(norm.pdf(e/s)/s)
    def info(b):
        y = numpy.array(df['y'])
        X = numpy.array(df[['one','x']])
        b=numpy.array(b)
        e = y - numpy.inner(X,b)
        s2 = (e.dot(e)/(len(e)))
        return numpy.dot(X.T,X)/s2

    mod.loglikeobs =llo
    mod.information =info

    print
    print 'CLUSTERED STANDARD ERRORS'

    print 'BY YR'
    my_se_yr = numpy.diag(clustered_se.clustered_se_from_model(mod,res.params,df['yr']))**0.5
    print 'Mine',my_se_yr
    print 'Petersen',se_by_yr
    print

    print 'BY FIRM'
    my_se_firm = numpy.diag(clustered_se.clustered_se_from_model(mod,res.params,df['firmid']))**0.5
    print 'Mine',my_se_firm
    print 'Petersen',se_by_firm
    print

    print 'BY FIRM AND YEAR'
    my_se_firm_and_yr = numpy.diag(clustered_se.multiway_clustered_se_from_model(mod,res.params,df[['firmid','yr']]))**0.5
    print 'Mine',my_se_firm_and_yr
    print 'Petersen',se_by_firm_and_yr
    print

    print 'Running assertions'
    assert(numpy.abs(b-res.params).sum()<1e-4)
    assert(numpy.abs(se-res.bse).sum()<1e-4)
    assert(numpy.abs(se_by_yr-my_se_yr).sum()<1e-4)
    assert(numpy.abs(se_by_firm-my_se_firm).sum()<1e-4)
    assert(numpy.abs(se_by_firm_and_yr-my_se_firm_and_yr).sum()<1e-4)
    print 'Petersen test passed'


if __name__ == '__main__':
    test_petersen()

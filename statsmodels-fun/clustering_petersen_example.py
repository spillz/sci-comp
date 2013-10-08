'''
A test of the clustered standard error module using Mitch Petersen's test data
http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.htm
'''


def test_petersen():
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

    print 'Running assertions'
    assert(numpy.abs(b-res.params).sum()<1e-4)
    assert(numpy.abs(se-res.bse).sum()<1e-4)
    assert(numpy.abs(se_by_yr-my_se_yr).sum()<1e-4)
    assert(numpy.abs(se_by_firm-my_se_firm).sum()<1e-4)
    print 'Petersen test passed'


if __name__ == '__main__':
    test_petersen()

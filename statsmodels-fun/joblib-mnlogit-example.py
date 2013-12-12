# joblib-mnlogit-example.py
# test the performance of statsmodels on parallel tasks - in this case running mnlogit on variants of a model with the same dataset
# Adapted from the single process mnlogit fitting example at
# http://statsmodels.sourceforge.net/devel/examples/generated/example_discrete.html#multinomial-logit

import numpy
import statsmodels.api as sm
from itertools import permutations
from joblib import Parallel, delayed
import time

#Get the data
anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_endog = anes_data.endog
anes_exog = sm.add_constant(anes_exog, prepend=True)
anes_exog = anes_exog

#We are going run the model on permutations of the first four exogenous vars and all of the remaining vars
varlist = range(5)
ovars = tuple(range(5,10))

#This function, which carries out the regression, is called on variants of the x vars
def reg(y,x):
    mlogit_mod = sm.MNLogit(y, x)
    mlogit_res = mlogit_mod.fit()
#    mlogit_margeff = mlogit_reg.get_margeff()
    return mlogit_res.params#,mlogit_margeff.summary()


#sets up and runs the regressions using multiple jobs
def run_par(n_jobs,output=False):
    print 'running with %s jobs'%(n_jobs,)
    s=time.time()
    results = Parallel(n_jobs = n_jobs)(delayed(reg)(anes_endog,anes_exog[:,p+ovars])  for p in permutations(varlist))
    t=time.time()-s
    if output:
        for parms in results:
                print parms
                print margeffs
                print
    print 'took',t
    print '###############################################'
    print
    return t

#run using between 1 and 8 jobs and output the times
max_jobs = 10
times = [run_par(n+1) for n in range(max_jobs)]
print '# Jobs\ttime(s)\tSpeedup multiple vs 1 job'
for n in range(max_jobs):
    print '%i\t%3.2f\t%3.2f'%(n+1,times[n],times[0]/times[n])
print

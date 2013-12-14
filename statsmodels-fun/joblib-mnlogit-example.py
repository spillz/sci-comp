# joblib-mnlogit-example.py
# test the performance of statsmodels on parallel tasks - in this case running mnlogit on variants of a model with the same dataset
# Adapted from the single process mnlogit fitting example at
# http://statsmodels.sourceforge.net/devel/examples/generated/example_discrete.html#multinomial-logit

import time
import numpy
import statsmodels.api as sm
from itertools import chain, combinations, permutations

def powerset(iterable):
  xs = list(iterable)
  # note we return an iterator rather than a list
  return chain.from_iterable( combinations(xs,n+1) for n in range(len(xs)) )

work = 10

#This function, which carries out the regression, is called on variants of the x vars
def reg(y,x):
    mlogit_mod = sm.MNLogit(y, x)
    for x in range(work): ##NB: DO TEN TIMES TO GET ENOUGH WORK TO SEE PARALLEL PERFORMANCE GAIN
        mlogit_res = mlogit_mod.fit()
#    mlogit_margeff = mlogit_reg.get_margeff()
    return mlogit_res.params#,mlogit_margeff.summary()

def reg2(data):
    y,x = data
    mlogit_mod = sm.MNLogit(y, x)
    for x in range(work): ##NB: DO TEN TIMES TO GET ENOUGH WORK TO SEE PARALLEL PERFORMANCE GAIN ON WINDOWS
        mlogit_res = mlogit_mod.fit(disp=False)
#    mlogit_margeff = mlogit_reg.get_margeff()
    return mlogit_res.params#,mlogit_margeff.summary()


def load():
    global anes_exog, anes_endog, varlist, ovars
    #Get the data
    anes_data = sm.datasets.anes96.load()
    anes_exog = anes_data.exog
    anes_endog = anes_data.endog
    anes_exog = sm.add_constant(anes_exog, prepend=True)
    anes_exog = anes_exog


#runs the regressions using job lib
def run_par(n_jobs,output=False):
    from joblib import Parallel, delayed
#    print 'running with %s jobs'%(n_jobs)
    s=time.time()
    results = Parallel(n_jobs = n_jobs)(delayed(reg)(anes_endog,anes_exog[:,p])  for p in powerset(varlist))
    t=time.time()-s
    if output:
        for parms in results:
                print (parms)
                print (margeffs)
                print ()
    print ('took',t)
    print ('###############################################')
    print ()
    return t

def exog_iter(varlist):
    for p in permutations(varlist): ##NOTE: THIS ISN'T DOING ANYTHING OTHER THAN SWAPPING COLUMNS AROUND - SUPERSET WOULD CHOOSE ALL SUBSETS OF THE COLUMNS
        yield (anes_endog,anes_exog[:,p])

#runs the regression using multiprocessing
def run_mp(n_jobs):
    import multiprocessing
    pool = multiprocessing.Pool(n_jobs)#,load)
    s=time.time()
    pool.map(reg2, exog_iter(varlist),1)
    t=time.time()-s
    return t

def run(n_jobs):
    s=time.time()
    for x in exog_iter(varlist):
        reg2(x)
    t=time.time()-s
    return t


def output_results(jobs, times):
    print('# Jobs\ttime(s)\tSpeedup multiple vs 1 job')
    for n,t in zip(jobs,times):
        print ('%i\t%3.2f\t%3.2f'%(n,t,times[0]/t))
    print ()

if __name__ == '__main__':
    #We are going run the model on permutations or supersets of the first four exogenous vars and all of the remaining vars
    load()
    varlist = range(5)
    ovars = tuple(range(5,10))

    max_jobs = 10

    print ('Running without multiprocessing')
    print (run(1))


    jobs = [1,2,4,8]
    times_mp = [run_mp(n) for n in jobs]
##    times_par = [run_par(n+1) for n in range(max_jobs)]

    print ('Runs with multiprocessing')
    output_results(jobs,times_mp)
##    print ('Runs with joblib')
##    output_results(jobs,times_par)


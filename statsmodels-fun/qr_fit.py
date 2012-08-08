import  numpy
import numpy.random
from numpy.linalg import qr
import statsmodels.api as sm

def qr_sol(R,qy):
    s=R.shape[0]
    x=numpy.zeros(s)
    for i in range(s):
        x[s-i-1]=(qy[s-i-1] - x[s-i:].dot(R[s-i-1,s-i:]))/R[s-i-1,s-i-1]
    return x

def singular_indices(Z):
    mapi=[]
    mapu=[]
    for i in range(len(Z)):
        if Z[i]:
            mapu.append(i)
        else:
            mapi.append(i)
    return mapi,mapu

## Sets up X~(10,6) with rank 3
X = numpy.ones((10,6))
X[:,4]= numpy.arange(1,11)
X[:,3] = [1,1,1,2,2,2,3,3,3,4]
X[:,5] = 2*X[:,0] + 4*X[:,3]
y= X[:,1]+X[:,4]+X[:,5] + numpy.random.randn(10)/1000


Q,R = qr(X)
Z=(numpy.abs(numpy.diag(R))<1e-10)
Ri=R[~Z][:,~Z]
Qi=Q[:,~Z]
Ru=R[~Z][:,Z]

mapi,mapu=singular_indices(Z)
print 'Identified params',mapi
print 'Unidentified params',mapu
print

K=numpy.ones((len(mapu),len(mapi)))
for i in range(Ru.shape[1]):
    bu=qr_sol(Ri,Ru[:,i])
    K[i,:]=bu
    eq=' + '.join(['%5.4f*b_%i'%(bu[j],mapi[j]) for j in range(len(bu))])
    print 'unidentified coefficient',mapu[i],'restricted as b_%i'%mapu[i],'= ',eq
print 'QR OLS fit of identified params',qr_sol(Ri,Qi.transpose().dot(y))
print 'Ri'
print Ri

print
print 'K'
print K
##Note that we get different results than if we had dropped the unidentifiable columns

X1=X[:,~Z]
Q1,R1 = qr(X1)
print
print 'QR OLS fit params - dropping unidentifiable columns of X',qr_sol(R1,Q1.transpose().dot(y))
print 'R1'
print R1 ##Ri is different from R1

print
print 'Statsmodels OLS fit - dropping unidentifiable columns of X',sm.OLS(y,X1).fit().params

print 'X'
print X
print

print 'R'
print R
print


'''
X=QR
XB = y
QR B = y
RB = Q'y

[R1 R2][B1 B2]' = [Q1 Q2]'y
[R3 R4]

R1B1 + R2B2 = Q1'y
R3B1 + R4B2 = Q2'y

B2 = KB1
K solves R1 K = R2

R1B1 + R2KB1 = Q1'y
(R1 + R2K)B1 = Q1'y

'''

print
#KR = R[:,~Z] + R[:,Z].dot(K)
#print 'KR',KR
KR = Ri + Ru.dot(K)
print 'KR',KR
print 'Alt QR OLS fit of identified params',qr_sol(KR,Qi.transpose().dot(y))

import numpy as np 

dusttype = 1
synctype = 1
N=50

name ='resultso1bt_moms_full_d%ss%sc_fix0'%(dusttype,synctype)

res=np.load('Best-fits/%s/res%s.npy'%(name,0),allow_pickle=True).item()
keys=list(res.keys())

for i in range(1,N):
	restemp=np.load('Best-fits/%s/res%s.npy'%(name,i),allow_pickle=True).item()
	for k in range(len(keys)):
		res[keys[k]]=res[keys[k]]+restemp[keys[k]]

if synctype==None:
    np.save('Best-fits/resultso1bt_moms_full_d%sc_fix0.npy'%(dusttype),res)
else:
    np.save('Best-fits/resultso1bt_moms_full_d%ss%sc_fix0.npy'%(dusttype,synctype),res)

import sys
from pathlib import Path
sys.path.append("./lib")
import numpy as np 
import analys_lib as an

dusttype = 0
synctype = 0
N=50

def merge_parallel(name,N,dusttype,synctype):
	res=np.load('./Best-fits/%s/res%s.npy'%(name,0),allow_pickle=True).item()
	keys=list(res.keys())
	for i in range(1,N):
		restemp=np.load('./Best-fits/%s/res%s.npy'%(name,i),allow_pickle=True).item()
		for k in range(len(keys)):
			res[keys[k]]=res[keys[k]]+restemp[keys[k]]

	if synctype==None:
	    np.save('Best-fits/%s.npy'%(name),res)
	else:
	    np.save('Best-fits/%s.npy'%(name),res)
	an.plotr_gaussproduct(res,Nmax=15,debug=False,color='darkorange',save=True,kwsave=name)

merge_parallel('resultsmbb_PL_d%ss%sc'%(dusttype,synctype),N,dusttype,synctype)
merge_parallel('resultso1bt_PL_d%ss%sc_fix0_p0'%(dusttype,synctype),N,dusttype,synctype)
#merge_parallel('resultso1bt_moms_full_d%ss%sc_fix0'%(dusttype,synctype),N,dusttype,synctype)
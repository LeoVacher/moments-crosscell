import numpy as np
from mpfit import mpfit
import mpfitlib as mpl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib

#contains all function to plot results.

#PLOT FUNCTIONS ##################################################################################################################

def getr_analytical(results,Nmin=0,Nmax=20):
    """
    return r and sigma(r) computed analytically
    :param results: output of moment fitting
    :Nmin: minimal bin of ell in which to fit the Gaussians
    :Nmax: maximal bin of ell in which to fit the Gaussians
    """
    rl = results['r'][Nmin:Nmax]
    sig=np.std(rl,axis=1)
    mean=np.mean(rl,axis=1)
    rstd= np.sqrt(1/(np.sum(1/sig**2)))
    rmean = np.sqrt(np.sum(mean**2/sig**2))*rstd
    return rmean,rstd

def plotr_gaussproduct_analytical(results,Nmin=0,Nmax=20,label='MBB',color='darkblue',debug=False,r=0,quiet=True,save=False,kwsave='',show=False):
    rmean,rstd=getr_analytical(results,Nmin=Nmin,Nmax=Nmax)
    x = np.linspace(-1,1,10000)
    intervall = 0.014
    fig,ax = plt.subplots(1,1, figsize=(10,7))
    ax.plot(x,(func.Gaussian(x,rmean,rstd))/np.max(func.Gaussian(x,rmean,rstd)),color=color,linewidth= 5,label='$%s \\pm %s$'%(np.round(rmean,5),np.round(rstd,5)))
    ax.fill_between(x,(func.Gaussian(x,rmean,rstd))/np.max(func.Gaussian(x,rmean,rstd)),color=color,alpha=0.2,linewidth=5)
    ax.fill_between(x,(func.Gaussian(x,rmean,rstd))/np.max(func.Gaussian(x,rmean,rstd)),facecolor="none",edgecolor=color,linewidth=5)
    ax.axvline(r, 0, 1, color = 'black', linestyle = "--",linewidth=3,zorder=1)
    ax.plot(x, np.zeros(len(x)), color = 'black', linewidth=5,linestyle='--',zorder=10000000)
    ax.set_xlim([r-intervall,r+intervall])
    ax.legend()
    ax.set_xlabel(r"$\hat{r}$")
    ax.set_ylim([0,1.03])
    if save==True:
        show=False
        plt.savefig("./plot-gauss/"+kwsave+"_analytical.pdf")
    if show==True:
        plt.show()
 
def plotr_gaussproduct(results,Nmin=0,Nmax=20,label='MBB',color='darkblue',debug=False,r=0,quiet=True,save=False,kwsave='',show=False):
    """
    Fit a Gaussian curve for r(ell) in each bin of ell and plot the product of all of them as a final result
    :param results: output of moment fitting
    :Nmin: minimal bin of ell in which to fit the Gaussians
    :Nmax: maximal bin of ell in which to fit the Gaussians
    :debug: plot the gaussian fit in each ell to ensure its working well, default: False
    :label: label for the plot
    """
    rl = results['r']
    Nell,N = rl.shape
    K_r = int(2*N**(0.33))
    moytemp=[]
    sigtemp=[]
    for ell in range(Nmin,Nmax):
        y1_cond , bins_cond = np.histogram(rl[ell,:],K_r)
        x1_cond = [.5*(b1+b2) for b1,b2 in zip(bins_cond[:-1],bins_cond[1:])] # Milieu du bin
        ysum_cond = scipy.integrate.simps(y1_cond,x1_cond)
        coeffunit = y1_cond[np.argmax(y1_cond)]/ysum_cond

        pl0=[np.mean(y1_cond),np.std(y1_cond)]
        parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0}]
        fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 1/(np.sqrt(y1_cond)*ysum_cond)}
        m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)
        if m.params[1]>0.01:
            fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 1000/(np.sqrt(y1_cond)*ysum_cond)}
            m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
        if m.params[1]>0.01:
            fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 0.0001/(np.sqrt(y1_cond)*ysum_cond)}
            m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)            
        
        if debug==True:
            plt.plot(x1_cond,y1_cond/ysum_cond)
            plt.plot(x1_cond,func.Gaussian(x1_cond,m.params[0],m.params[1]))
            plt.show()
        moytemp.append(m.params[0])
        sigtemp.append(m.params[1])
    moy = np.array(moytemp)
    sig = np.array(sigtemp)

    x = np.linspace(-1,1,10000)
    intervall = 0.014
    fig,ax = plt.subplots(1,1, figsize=(10,7))
    gausstot = 1
    for i in range(Nmax-Nmin):
        gausstot = gausstot*func.Gaussian(x,moy[i],sig[i])
    Norm = scipy.integrate.simps(gausstot,x)
    coeffunit = gausstot[np.argmax(gausstot)]/Norm
    pl0=[np.mean(gausstot/Norm/coeffunit),np.std(gausstot/Norm/coeffunit)]
    parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0}]
    fa = {'x':x, 'y':gausstot/Norm/coeffunit, 'err': 1000/(np.sqrt(gausstot/Norm))}
    m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
    if m.params[1]>0.01:
                fa = {'x':x, 'y':gausstot/Norm, 'err': 0.0001/(np.sqrt(gausstot/Norm))}
                m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
    if m.params[1]>0.01:
                fa = {'x':x, 'y':gausstot/Norm, 'err': 1000/(np.sqrt(gausstot/Norm))}
                m = mpfit(mpl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)            
    ax.plot(x,(func.Gaussian(x,m.params[0],m.params[1])/coeffunit)/np.max(func.Gaussian(x,m.params[0],m.params[1])/coeffunit),color=color,linewidth= 5,label='$%s \\pm %s$'%(np.round(m.params[0],5),np.round(m.params[1],5)))
    ax.fill_between(x,(func.Gaussian(x,m.params[0],m.params[1])/coeffunit)/np.max(func.Gaussian(x,m.params[0],m.params[1])/coeffunit),color=color,alpha=0.2,linewidth=5)
    ax.fill_between(x,(func.Gaussian(x,m.params[0],m.params[1])/coeffunit)/np.max(func.Gaussian(x,m.params[0],m.params[1])/coeffunit),facecolor="none",edgecolor=color,linewidth=5)
    #ax.errorbar(x1_cond,y1_cond/ysum_cond/coeffunit,xerr=0,yerr=1/(np.sqrt(y1_cond)*ysum_cond)/coeffunit,fmt='^',color=c[i],ecolor=c[i],zorder=300001,label='$%s \\pm %s$'%(m.params[0],m.params[1]))
    ax.axvline(r, 0, 1, color = 'black', linestyle = "--",linewidth=3,zorder=1)
    ax.plot(x, np.zeros(len(x)), color = 'black', linewidth=5,linestyle='--',zorder=10000000)
    ax.set_xlim([r-intervall,r+intervall])
    ax.legend()
    ax.set_xlabel(r"$\hat{r}$")
    ax.set_ylim([0,1.03])
    if save==True:
        show=False
        plt.savefig("./plot-gauss/"+kwsave+".pdf")
    if show==True:
        plt.show()
 
# Plot results

def plotmed(ell,label,res,color='darkblue',marker="D",show=True,legend=''):
    """
    Plot median and median absolute deviation of best-fits as a function of ell
    :ell: bandpower array
    :label: string indicating the name of the quantity
    :color: color of the plot
    :marker: marker of the plot
    :show: show the plot
    legend: legend to add to the plot
    """
    ellbound=ell.shape[0]
    name={'A':r'$A^d$','beta':r'$\beta^d$','temp':r'$T^d$','beta_s':r'$\beta^s$','A_s':r'$A^s$','A_sd':r'$A^{sd}$','r':r'$\hat{r}$','X2red':r'$\chi^2$','Aw1b':r'$\mathcal{D}_\ell^{A\times\omega_1^{\beta}}$','Aw1t':r'$\mathcal{D}_\ell^{A\times\omega_1^{T}}$','Asw1bs':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta^s}}$','w1bw1b':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^\beta}$','w1tw1t':r'$\mathcal{D}_\ell^{\omega_1^T\times\omega_1^T}$','w1bw1t':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^T}$','w1bsw1bs':r'$\mathcal{D}_\ell^{\omega_1^{\beta^s}\times\omega_1^{\beta^s}}$', 'Asw1b':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta}}$','Asw1t':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{T}}$','Adw1s':r'$\mathcal{D}_\ell^{A\times\omega_1^{\beta^s}}$'}
    edgecolor="#80AAF3"
    plt.errorbar(ell,np.median(res[label],axis=1)[:ellbound],yerr=scipy.stats.median_abs_deviation(res[label],axis=1)[:ellbound],c=color,fmt=marker,linestyle='',label=legend)
    plt.scatter(ell,np.median(res[label],axis=1)[:ellbound],s=175,c=color,marker=marker,edgecolor=edgecolor)
    plt.ylabel(name[label],fontsize=20)
    plt.xlabel(r"$\ell$",fontsize=20)
    plt.legend()
    plt.tight_layout()
    if show==True:
        plt.show()

def plotrespdf(l,res,legs,colors):
    """
    return a pdf with all the quantities of interest
    :l: bandpower array
    :res: list of all the results to plot
    :legs: list of all legends 
    :colors: list of all colors
    """
    namesave=''
    for i in range(len(legs)):
        namesave += legs[i] +'-'
    pdf = matplotlib.backends.backend_pdf.PdfPages("./pdf_plots/%s.pdf"%(namesave))

    if len(res)==1:
        common_keys = res[0].keys()
        unique_keys=[]
    if len(res)==2:
        common_keys = res[0].keys() & res[1].keys()
        unique_keys = res[0].keys() ^ res[1].keys()
    if len(res)==3:
        common_keys = res[0].keys() & res[1].keys() & res[2].keys()
        unique_keys = res[0].keys() ^ res[1].keys() ^ res[2].keys()

    for k in common_keys:
        plt.figure(figsize=(10,7))
        for i in range(len(res)):
            plotmed(l+i,k,res[i],show=False,color=colors[i],legend=legs[i])        
            if k =='A':
                plt.loglog()
            if k =='A_s':
                plt.loglog()
            if k=='X2red':
                plt.plot(l,np.ones(len(l)),c='k',linestyle='--')
            if k=='r':
                plt.plot(l,np.zeros(len(l)),c='k',linestyle='--')
            if k=='beta_s':
                plt.plot(l,-3*np.ones(len(l)),c='k',linestyle='--')
            if k=='temp':
                plt.plot(l,20*np.ones(len(l)),c='k',linestyle='--')
            if k=='beta':
                plt.plot(l,1.54*np.ones(len(l)),c='k',linestyle='--')
        pdf.savefig()

    for i in range(len(res)):
        for k in unique_keys:
            if k in res[i]:
                plt.figure(figsize=(10,7))
                plotmed(l+i,k,res[i],show=False,color=colors[i],legend=legs[i])        
                pdf.savefig()

    for i in range(len(res)):
        plotr_gaussproduct(res[i],color=colors[i],label=legs[i],show=False,Nmax=len(l))
        pdf.savefig()

    pdf.close()

import numpy as np
from mpfit import mpfit
import fitlib as ftl
import scipy
import matplotlib.pyplot as plt 
import basicfunc as func
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib
import seaborn
from getdist import plots, MCSamples

#contains all function to plot results.

#PLOT FUNCTIONS ##################################################################################################################

def plotr_hist(results,color='darkblue',debug=False,r=0,quiet=True,save=False,kwsave='',show=False):
    """
    plot histogramm of r and chi^2 for the all-ell case
    :param results: output of moment fitting for the all-ell case
    """
    rn=results['r']
    chi2=results['X2red']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    seaborn.histplot(rn,stat="probability",kde=True,ax=ax[0])
    ax[0].text(0.95, 0.95, r"$r=%s\pm%s$"%(np.round(np.mean(rn),6), np.round(np.std(rn),6)), transform=ax[0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
    seaborn.histplot(chi2,stat="probability",kde=True,ax=ax[1])
    ax[1].text(0.95, 0.95, r"$\chi^2_{\rm red}=%s\pm%s$"%(np.round(np.mean(chi2),6), np.round(np.std(chi2),6)), transform=ax[1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.tight_layout()
    if save == True:
        show = False
        plt.savefig("./plot_gauss/"+kwsave+".pdf")
    if show == True:
        plt.show()
 
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

def plotr_gaussproduct_analytical(results,Nmin=0,Nmax=20,color='darkblue',debug=False,r=0,quiet=True,save=False,kwsave='',show=False):
    """
    compute r and sigma(r) analytically and plot a corresponding Gaussian curve
    :param results: output of moment fitting
    :Nmin: minimal bin of ell in which to fit the Gaussians
    :Nmax: maximal bin of ell in which to fit the Gaussians
    """
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
    if save == True:
        show = False
        plt.savefig("./plot_gauss/"+kwsave+"_analytical.pdf")
    if show == True:
        plt.show()
 
def plotr_gaussproduct(results,Nmin=0,Nmax=20,color='darkblue',debug=False,r=0,quiet=True,save=False,kwsave='',show=False,ax=None,alpha=1):
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
    
    #fit mean and std in each bin of ell

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
        m = mpfit(ftl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)
        if m.params[1]>0.01:
            fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 1000/(np.sqrt(y1_cond)*ysum_cond)}
            m = mpfit(ftl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
        if m.params[1]>0.01:
            fa = {'x':x1_cond, 'y':y1_cond/ysum_cond, 'err': 0.0001/(np.sqrt(y1_cond)*ysum_cond)}
            m = mpfit(ftl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)            
        
        if debug==True:
            plt.plot(x1_cond,y1_cond/ysum_cond)
            plt.plot(x1_cond,func.Gaussian(x1_cond,m.params[0],m.params[1]))
            plt.show()
        moytemp.append(m.params[0])
        sigtemp.append(m.params[1])
    moy = np.array(moytemp)
    sig = np.array(sigtemp)

    #compute gaussian product and fit gauss curve

    x = np.linspace(-1,1,10000)
    intervall = 0.014
    gausstot = 1
    for i in range(Nmax-Nmin):
        gausstot = gausstot*func.Gaussian(x,moy[i],sig[i])
    Norm = scipy.integrate.simps(gausstot,x)
    coeffunit = gausstot[np.argmax(gausstot)]/Norm
    pl0=[np.mean(gausstot/Norm/coeffunit),np.std(gausstot/Norm/coeffunit)]
    parinfopl = [{'value':pl0[0], 'fixed':0},{'value':pl0[1],'fixed':0}]
    fa = {'x':x, 'y':gausstot/Norm/coeffunit, 'err': 1000/(np.sqrt(gausstot/Norm))}
    m = mpfit(ftl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
    if m.params[1]>0.01:
                fa = {'x':x, 'y':gausstot/Norm, 'err': 0.0001/(np.sqrt(gausstot/Norm))}
                m = mpfit(ftl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)        
    if m.params[1]>0.01:
                fa = {'x':x, 'y':gausstot/Norm, 'err': 1000/(np.sqrt(gausstot/Norm))}
                m = mpfit(ftl.Gaussian,parinfo= parinfopl ,functkw=fa,quiet=quiet)            
    
    #plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(x,(func.Gaussian(x,m.params[0],m.params[1])/coeffunit)/np.max(func.Gaussian(x,m.params[0],m.params[1])/coeffunit),color=color,linewidth= 5,label='$%s \\pm %s$'%(np.round(m.params[0],5),np.round(m.params[1],5)),alpha=alpha)
    ax.axvline(r, 0, 1, color = 'black', linestyle = "--",linewidth=3,zorder=1)
    ax.plot(x, np.zeros(len(x)), color = 'black', linewidth=5,linestyle='--',zorder=10000000)
    ax.set_xlim([r-intervall,r+intervall])
    ax.legend()
    ax.set_xlabel(r"$\hat{r}$")
    ax.set_ylim([0,1.03])
    if save == True:
        show = False
        plt.savefig("./plot_gauss/"+kwsave+".pdf")
    if show == True:
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
    name = {'A_d':r'$A^d$','beta_d':r'$\beta^d$','T_d':r'$T^d$','beta_s':r'$\beta^s$','A_s':r'$A^s$','A_sd':r'$A^{sd}$','r':r'$\hat{r}$','X2red':r'$\chi^2_{\rm red}$','Aw1b':r'$\mathcal{D}_\ell^{A_d\times\omega_1^{\beta}}$','Aw1t':r'$\mathcal{D}_\ell^{A_d\times\omega_1^{1/T}}$','Asw1bs':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta^s}}$','w1bw1s':r'$\mathcal{D}_\ell^{\omega_1^{\beta^d}\times\omega_1^{\beta^s}}$','w1sw1T':r'$\mathcal{D}_\ell^{\omega_1^{1/T}\times\omega_1^{\beta^s}}$','w1bw1b':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^\beta}$','w1tw1t':r'$\mathcal{D}_\ell^{\omega_1^{1/T}\times\omega_1^{1/T}}$','w1bw1t':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^{1/T}}$','w1bsw1bs':r'$\mathcal{D}_\ell^{\omega_1^{\beta^s}\times\omega_1^{\beta^s}}$', 'Asw1b':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta}}$','Asw1t':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{1/T}}$','Adw1s':r'$\mathcal{D}_\ell^{A_d\times\omega_1^{\beta^s}}$'}

    edge_colors = {'darkblue': "#80AAF3",'darkred': "#FF7F7F",'darkorange': "#FED8B1",'forestgreen': "#D1FFBD"}
    edgecolor = edge_colors.get(color, "white")
    plt.errorbar(ell,np.median(res[label],axis=1)[:ellbound],yerr=scipy.stats.median_abs_deviation(res[label],axis=1)[:ellbound],c=color,fmt=marker,linestyle='',label=legend)
    plt.scatter(ell,np.median(res[label],axis=1)[:ellbound],s=175,c=color,marker=marker,edgecolor=edgecolor)
    plt.ylabel(name[label],fontsize=20)
    plt.xlabel(r"$\ell$",fontsize=20)
    plt.legend()
    plt.tight_layout()
    if show==True:
        plt.show()

def plothist(label,res,colors='darkblue',r=0):
    
    #legends:
    nameo0 = {'A_d':r'$A^d$','beta_d':r'$\beta^d$','T_d':r'$T^d$','beta_s':r'$\beta^s$','A_s':r'$A^s$','A_sd':r'$A^{sd}$','r':r'$\hat{r}$','X2red':r'$\chi^2_{\rm red}$'}
    namemom = {'Aw1b':r'$\mathcal{D}_\ell^{A_d\times\omega_1^{\beta}}$','Aw1t':r'$\mathcal{D}_\ell^{A_d\times\omega_1^{1/T}}$','Asw1bs':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta^s}}$','w1bw1s':r'$\mathcal{D}_\ell^{\omega_1^{\beta^d}\times\omega_1^{\beta^s}}$','w1sw1T':r'$\mathcal{D}_\ell^{\omega_1^{1/T}\times\omega_1^{\beta^s}}$','w1bw1b':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^\beta}$','w1tw1t':r'$\mathcal{D}_\ell^{\omega_1^{1/T}\times\omega_1^{1/T}}$','w1bw1t':r'$\mathcal{D}_\ell^{\omega_1^\beta\times\omega_1^{1/T}}$','w1bsw1bs':r'$\mathcal{D}_\ell^{\omega_1^{\beta^s}\times\omega_1^{\beta^s}}$', 'Asw1b':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{\beta}}$','Asw1t':r'$\mathcal{D}_\ell^{A_s\times\omega_1^{1/T}}$','Adw1s':r'$\mathcal{D}_\ell^{A_d\times\omega_1^{\beta^s}}$'}
    keyspl = ['alpha_'+s for s in list(namemom.keys())]
    valuespl = [s.replace('mathcal{D}_\\ell','alpha') for s in list(namemom.values())]
    namepl = dict(zip(keyspl, valuespl))
    name = {**nameo0,**namemom,**namepl}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    seaborn.histplot(res[label],stat="probability",kde=True,ax=ax,color=colors)
    plt.text(0.95, 0.95, name[label]+r"$=%s\pm%s$"%(np.round(np.mean(res[label]),6), np.round(np.std(res[label]),6)), transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right')
    plt.title("%s"%name[label])
    if label!='X2red' and label!='beta_d' and label!='T_d' and label!='beta_s':
        print(1)
        ax.axvline(r, 0, 1, color = 'black', linestyle = "--",linewidth=3,zorder=1)


def plotrespdf(l, res, legs, colors,mom_an=None,plot_contours=False,betadbar=1.54,tempbar=20.,betasbar=-3.):
    """
    Generate a PDF with plots for all quantities of interest.
    :param l: Bandpower array
    :param res: List of all result dictionaries to plot
    :param legs: List of legends
    :param colors: List of colors
    """
    # Construct file name
    namesave = "-vs-".join(legs) if len(legs) > 1 else legs[0]
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"./pdf_plots/{namesave}.pdf")
    
    # Determine common and unique keys across all dictionaries
    all_keys = [set(resi.keys()) for resi in res]
    common_keys = set.intersection(*all_keys)
    unique_keys = set.union(*all_keys) - common_keys
    # Check if all dictionaries have only common keys
    only_common_keys = all(len(resi.keys() - common_keys) == 0 for resi in res)
    
    # Plot common keys
    for k in common_keys:
        plt.figure(figsize=(10, 7))
        if only_common_keys:
            for i, resi in enumerate(res):
                if resi[k].ndim == 1:
                    plothist(k, resi)
                else:
                    plotmed(l+i, k, resi, show=False, color=colors[i], legend=legs[i])
            if mom_an != None and k in mom_an:
                plt.plot(l,mom_an[k][:len(l)],color='k',linestyle="--",linewidth=3)
        else:
            for i, resi in enumerate(res):
                if resi[k].ndim == 1:
                    plothist(k, resi)
                else:
                    plotmed(l + i, k, resi, show=False, color=colors[i], legend=legs[i])
                    if mom_an != None and k in mom_an:
                        plt.plot(l,mom_an[k][:len(l)],color='k',linestyle="--",linewidth=3)
        
        if res[0][k].ndim != 1:
            if k in {'A_d', 'A_s'}:
                plt.loglog()
            elif k in {'X2red', 'r', 'beta_s', 'T_d', 'beta_d'}:
                ref_values = {'X2red': 1, 'r': 0, 'beta_s': betasbar, 'T_d': tempbar, 'beta_d': betadbar}
                plt.plot(l, ref_values[k] * np.ones(len(l)), c='k', linestyle='--')
            else:
                plt.plot(l, np.zeros(len(l)), c='k', linestyle='--')

        pdf.savefig()
        plt.close()
    # Plot unique keys
    for i, resi in enumerate(res):
        for k in unique_keys:
            if k in resi:
                plt.figure(figsize=(10, 7))
                if resi[k].ndim == 1:
                    plothist(k, resi)
                else:
                    plotmed(l + i, k, resi, show=False, color=colors[i], legend=legs[i])
                    
                if mom_an != None and k in mom_an:
                    plt.plot(l,mom_an[k][:len(l)],color='k',linestyle="--",linewidth=3)
                pdf.savefig()
                plt.close()

    # Plot additional Gaussian product analyses if applicable
    fig, ax = plt.subplots(figsize=(10, 7))  
    for i, resi in enumerate(res):
        if 'r' in resi and resi['r'].ndim != 1:
            plotr_gaussproduct(resi, color=colors[i], show=False, Nmax=len(l),ax=ax,alpha=0.8)
    pdf.savefig()
    plt.close()

    if plot_contours:
        param_names = list(resi.keys())[:-1]
        nameo0 = {'A_d':"A_d",'beta_d':"\\beta_d",'T_d':"T_d",'beta_s':"\\beta_s",'A_s':"A_s",'A_sd':"A_{sd}",'r':'r'}
        param_names = [nameo0.get(name, name) for name in param_names]
        for ell in range(len(l)):
            samples = []

            for i, resi in enumerate(res):
                data = np.column_stack([resi[key][ell] for key in list(resi.keys())[:-1]]) 
                samples.append(MCSamples(samples=data, names=param_names, labels=param_names))

            g = plots.get_subplot_plotter()
            g.settings.lab_fontsize = 20
            g.settings.legend_fontsize = 20
            g.settings.alpha_filled_add=0.6
            g.triangle_plot(samples, 
                filled=True, 
                contour_colors=colors, 
                contour_levels=[0.68, 0.95],
                title_limit=1,
                legend_labels=[],
                )
            plt.suptitle(r"$\ell_{\rm bin}=%s$"%ell, fontsize=18, fontweight='bold')
            plt.savefig('./param_testplot.pdf')
            pdf.savefig()
            plt.close()
    pdf.close()


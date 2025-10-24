#CAN ALSO BE EASILY COPIED AND PASTED IN A JUPYTER NOTEBOOK FOR CONVINIENCE
sys.path.append("../lib")
import plotlib as plib 

xlabel= "$f_{sky}$" #label for the different cases 
cases = ["0.5","0.6", "0.7","0.8"] #different scenarios you want to plot on x axis
subcases = ["o0", "1bts", "adapt."] #different scenarios you want to plot for each point of x
colors = ["blue", "darkorange", "red"] #colors of each subcase 
kw_save= "fsky_vs_fittingscheme" #name to save the pdf
offset = 0.05 #size of the offset between the different subcases

#load the results:
res_c0sc0 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_maskGWD_ds_o%s_fix%s.npy'%(1,1,0.5,'0',0),allow_pickle=True).item()
res_c0sc1 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_maskGWD_ds_o%s_fix%s.npy'%(1,1,0.5,'1bts',1),allow_pickle=True).item()
res_c0sc2 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_maskGWD_ds_o%s_fix%s_adaptative.npy'%(1,1,0.5,'1bts',1),allow_pickle=True).item()

res_c1sc0 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(1,1,0.6,'0',0),allow_pickle=True).item()
res_c1sc1 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(1,1,0.6,'1bts',1),allow_pickle=True).item()
res_c1sc2 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s_adaptative.npy'%(1,1,0.6,'1bts',1),allow_pickle=True).item()

res_c2sc0 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(1,1,0.7,'0',0),allow_pickle=True).item()
res_c2sc1 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(1,1,0.7,'1bts',1),allow_pickle=True).item()
res_c2sc2 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s_adaptative.npy'%(1,1,0.7,'1bts',1),allow_pickle=True).item()

res_c3sc0 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(1,1,0.8,'0',0),allow_pickle=True).item()
res_c3sc1 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s.npy'%(1,1,0.8,'1bts',1),allow_pickle=True).item()
res_c3sc2 = np.load('../best_fits/results_d%ss%s_%s_Nmt-fg_ds_o%s_fix%s_adaptative.npy'%(1,1,0.8,'1bts',1),allow_pickle=True).item()


res_list = [
    [res_c0sc0, res_c1sc0, res_c2sc0, res_c3sc0],
    [res_c0sc1, res_c1sc1, res_c2sc1, res_c3sc1],
    [res_c0sc2, res_c1sc2, res_c2sc2, res_c3sc2]
]

r_all, sigma_r_all = [], []
X_all, sigma_X_all = [], []

for res_group in res_list:
    r_group, sigma_r_group = [], []
    X_group, sigma_X_group = [], []
    for res in res_group:
        rmean, rstd = plib.getr_analytical(res, Nmin=0, Nmax=20)
        Xmean, Xstd = plib.getX2_analytical(res, Nmin=0, Nmax=20)
        r_group.append(rmean)
        sigma_r_group.append(rstd)
        X_group.append(Xmean)
        sigma_X_group.append(Xstd)
    r_all.append(np.array(r_group))
    sigma_r_all.append(np.array(sigma_r_group))
    X_all.append(np.array(X_group))
    sigma_X_all.append(np.array(sigma_X_group))

x = np.arange(len(cases))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'hspace': 0.0})
ax1.axhline(1, linestyle="--", color="k", linewidth=1)
ax2.axhline(0, linestyle="--", color="k", linewidth=1)

# Plot des erreurs
for i in range(len(res_list)):
    ax1.errorbar(x + (i-1)*offset, X_all[i], yerr=sigma_X_all[i], fmt="o", label=subcases[i],
                 c=colors[i], markersize=8, capsize=4)
    ax2.errorbar(x + (i-1)*offset, r_all[i], yerr=sigma_r_all[i], fmt="o", label=subcases[i],
                 c=colors[i], markersize=8, capsize=4)

ax1.set_ylabel(r"$\chi^2_{\rm red} \pm \sigma(\chi^2_{\rm red})$", fontsize=16)
ax1.legend(fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(cases, fontsize=14, rotation=45, ha="right")
ax2.set_ylabel(r"$\hat{r}\pm \sigma_{\hat{r}}$", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("%s" % xlabel, fontsize=16)
plt.tight_layout()
plt.savefig(f"../pdf_plots/sigma_r_{kw_save}.pdf", bbox_inches="tight")
plt.show()
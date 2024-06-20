
case=["d0s0","d1s0","d1s1","d10s0","d12s0"]
r0= [2e-05,0.01319,0.01318,0.02243,0.02468]
sigmar0=[0.00043,0.00059,0.00059,0.00072,0.0008 ]
r1= [0.00014,0.00022,0.00092,0.00137,0.00239]
sigmar1=[0.00125,0.00172,0.00185,0.0021,0.00258 ]

plt.plot(case,np.zeros(len(r1)),linestyle="--",c='k')
plt.errorbar(case,r0,yerr=sigmar0,linestyle='',label='o0',fmt='o')
plt.errorbar(case,r1,yerr=sigmar1,linestyle='',label='o1bt',fmt='o')
plt.ylabel(r"$\hat{r}\pm \sigma_{\hat{r}}$")
plt.legend()
plt.savefig("./pdf_plots/sigma_r.pdf")
plt.show()
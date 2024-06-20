import basicfunc as leo
import pysm3

nside = 64
 
dusttype=0
Pathsave='./plot-maps/'
 
instr_name='LiteBIRD_full'
instr =  np.load("./lib/instr_dict/%s.npy"%instr_name,allow_pickle=True).item()
freq= instr['frequencies']
 
N_freqs =len(freq)
Ncross=int(N_freqs*(N_freqs+1)/2)
sky = pysm3.Sky(nside=512, preset_strings=['d%s'%dusttype])#,'s%s'%synctype])
 
#freq_maps = get_observation(instrument, sky, unit='uK_CMB')
modeld10= np.array([sim.downgrade_map(sky.get_emission(freq[f] * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq[f]*u.GHz)),nside_in=512,nside_out=nside) for f in range(len(freq))])
 
dust = sky.components[0]
nu0I=dust.freq_ref_I.value
nu0P=dust.freq_ref_P.value
 
betamap= dust.mbb_index.value
tempmap= dust.mbb_temperature.value
 
skyrefI = sim.downgrade_map(sky.get_emission(nu0I * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0P*u.GHz)).value,nside_in=512,nside_out=nside)
 
skyrefP = sim.downgrade_map(sky.get_emission(nu0P * u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0P*u.GHz)).value,nside_in=512,nside_out=nside)
 
 
#mbbQ=leo.MBBpysm(freq,skyrefP[1],betamap,tempmap,nu0P)
mbbQ=skyrefP[1,40000]*leo.MBB_fit(freq,betamap,tempmap)/leo.MBB_fit(nu0P,betamap,tempmap)
plt.plot(mbbQ,label="extr")
plt.plot(modeld10[:,1,40000],label="pysm",linestyle='--')
plt.xlabel(r'$\nu$')
plt.ylabel(r'SED')
plt.legend()
plt.show()
 
diffQ=np.array(mbbQ/modeld10[:,1])
 
plt.plot(freq,diffQ[:,0],label='Qex/Qd0')
#plt.plot(freq,leo.planck_dBnu_dT(freq*1e9,2.725)/leo.planck_dBnu_dT(freq[3]*1e9,2.725)/diffQ[:,0],label=r'$d B_\nu(T_{\rm CMB})/dT$')
plt.xlabel(r"$\nu$")
plt.legend()
plt.show()
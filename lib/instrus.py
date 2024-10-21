import numpy as np

#LITEBIRD-full///////////////////////////////////////////

freq_LB_full=np.array([40,50,60,68,68,78,78,89,89,100,119,140,100,119,140,166,195,195,235,280,337,402])
sens_P_LB_full =np.array([37.42, 33.46, 21.31, 19.91, 31.77, 15.55, 19.13,12.28,28.77, 10.34,7.69,7.25,8.48,5.70,6.38, 5.57, 7.05, 10.50, 10.79, 13.80, 21.95, 47.45])

litebird = {
        'frequencies': freq_LB_full,
        'sens_I': sens_P_LB_full /1.41,
        'sens_P': sens_P_LB_full,
        'beams': np.zeros(len(freq_LB_full)),
        'add_noise': False,
        'noise_seed': int(np.random.uniform()),
        'use_bandpass': False,
        'output_units': 'uK_CMB',
        'output_directory': '/dev/null/',
        'output_prefix': 'litebird',
        'use_smoothing': True,
    }

np.save("./lib/instr_dict/litebird_full",litebird)

#LITEBIRD-reduced///////////////////////////////////////////

maxbeams = np.array([70.5, 58.5, 51.1, 47.1, 43.8, 41.5, 37.8, 33.6, 30.8, 28.9, 28.6, 24.7, 22.5, 20.9, 17.9])
freq_LB = np.array([40.0, 50.0, 60.0, 68.0, 78.0, 89.0, 100.0, 119, 140.0, 166.0, 195.0, 235, 280.0, 337.0, 402.0])
freq_overl = [3,4,5,6,7,8,10]
fwhm_overl = np.array([[41.6,47.1],[36.9,43.8],[33.0,41.5],[30.2,37.8],[26.3,33.6],[23.7,30.8],[28.0,28.6]])
NET_overl = np.array([[19.91,31.77],[15.55,19.13],[12.28,28.77],[10.34,8.48,],[7.69,5.70],[7.25,6.38],[7.05,10.50]])
sens_P_LB =np.array([37.42, 33.46, 21.31, 15.38, 10.81, 9.26, 5.92, 4.15, 4.20, 5.57, 5.78, 10.79, 13.80, 21.95, 47.45])

# combine overlapping freqs:
t=0
for i in freq_overl:
        sens_P_LB[i] = np.sqrt(1/( (fwhm_overl[t][1]/fwhm_overl[t][0])**2*NET_overl[t][0]**-2 + NET_overl[t][1]**-2))
        t=t+1

litebird = {
        'frequencies': freq_LB,
        'sens_I': sens_P_LB /1.41,
        'sens_P': sens_P_LB,
        'beams': maxbeams,
        'add_noise': False,
        'noise_seed': int(np.random.uniform()),
        'use_bandpass': False,
        'output_units': 'uK_CMB',
        'output_directory': '/dev/null/',
        'output_prefix': 'litebird',
        'use_smoothing': True,
    }


np.save("./lib/instr_dict/litebird_reduced",litebird)

#QUBIC///////////////////////////////////////////


freqnsub1= [150,220] 
freqnsub2 = [140.04, 158.79,205.39,232.89] 
freqnsub3 = [136.98,148.95,161.97,200.91,218.47,237.56] 
freqnsub4 = [135.51,144.29,153.65,163.61,198.74,211.63,225.35,239.96]
freqnsub5 = [134.63, 141.57, 148.87, 156.54, 164.61,197.46, 207.64, 218.34, 229.6,  241.43 ]
freqnsub6 = [134.06, 139.79, 145.77, 152.01, 158.51, 165.29, 196.62, 205.03, 213.8,  222.94, 232.48, 242.42]
freqnsub7 = [133.65, 138.53, 143.6,  148.85, 154.29, 159.93, 165.77, 196.02, 203.18, 210.61, 218.31, 226.29, 234.56, 243.14]
freqnsub8 = [133.34, 137.6,  141.99, 146.52, 151.2 , 156.02, 161.  , 166.14, 195.57, 201.81, 208.25, 214.9,  221.76, 228.83, 236.14, 243.67]

np.save("./lib/instr_dict/freq_QUBIC_Nsub1",freqnsub1)
np.save("./lib/instr_dict/freq_QUBIC_Nsub2",freqnsub2)
np.save("./lib/instr_dict/freq_QUBIC_Nsub3",freqnsub3)
np.save("./lib/instr_dict/freq_QUBIC_Nsub4",freqnsub4)
np.save("./lib/instr_dict/freq_QUBIC_Nsub5",freqnsub5)
np.save("./lib/instr_dict/freq_QUBIC_Nsub6",freqnsub6)
np.save("./lib/instr_dict/freq_QUBIC_Nsub7",freqnsub7)
np.save("./lib/instr_dict/freq_QUBIC_Nsub8",freqnsub8)

#PICO///////////////////////////////////////////

# https://sites.google.com/umn.edu/picomission/home
PICO = {
        'frequencies': np.array([21.0, 25.0, 30.0, 36.0, 43.2, 51.8, 62.2, 74.6, 89.6, 107.5, 129.0, 154.8, 185.8, 222.9, 267.5, 321.0, 385.2, 462.2, 554.7, 665.6, 798.7]),
        'sens_I': np.array([16.9, 11.8, 8.1, 5.7, 5.8, 4.1, 3.8, 2.9, 2.0, 1.6, 1.6, 1.3, 2.6, 3.0, 2.1, 2.9, 3.5, 7.4, 34.6, 143.7, 896.4]) / 1.41,
        'sens_P': np.array([16.9, 11.8, 8.1, 5.7, 5.8, 4.1, 3.8, 2.9, 2.0, 1.6, 1.6, 1.3, 2.6, 3.0, 2.1, 2.9, 3.5, 7.4, 34.6, 143.7, 896.4]),
        'beams': np.array([40.9, 34.1, 28.4, 23.7, 19.7, 16.4, 13.7, 11.4, 9.5, 7.9, 6.6, 5.5, 4.6, 3.8, 3.2, 2.7, 2.2, 1.8, 1.5, 1.3, 1.1]),
        'add_noise': True,
        'noise_seed': int(np.random.uniform()*1e6),
        'use_bandpass': False,
        'output_directory': '/dev/null',
        'output_prefix': 'pico',
        'use_smoothing': False,
    }

np.save("./lib/instr_dict/PICO",PICO)

#SIMONS///////////////////////////////////////////

sens_P_SO = np.array([52., 27., 5.8, 6.3, 15., 37.])
   
SO_LAT =  {
        'frequencies': np.array([27.,39.,93.,145.,225.,280.]),
        'sens_I': sens_P_SO / np.sqrt(2),
        'sens_P': sens_P_SO,
        'beams': np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]),
        'add_noise': True,
        'noise_seed': int(np.random.uniform()*1e6),
        'use_bandpass': False,
        'output_units': 'uK_CMB',
        'output_directory': '/dev/null',
        'output_prefix': 'so_lat',
        'use_smoothing': False,
    }

freq_SO = np.array([27.,39.,93.,145.,225.,280.])
beams_SO = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])

np.save("./lib/instr_dict/SO_LAT",SO_LAT)
np.save("./lib/instr_dict/sens_P_SO",sens_P_SO)
np.save("./lib/instr_dict/freq_SO",freq_SO)
np.save("./lib/instr_dict/beams_SO",beams_SO)


#CMB-S4///////////////////////////////////////////


# https://cmb-s4.org/wiki/index.php/Survey_Performance_Expectations

CMBS4 = {
        'frequencies': np.array([20, 30, 40, 85, 95, 145, 155, 220, 270]),
        'sens_I': np.array([16.66, 10.62, 10.07, 2.01, 1.59, 4.53, 4.53, 11.61, 15.84]),
        'sens_P': np.array([13.6, 8.67, 8.22, 1.64, 1.30, 2.03, 2.03, 5.19, 7.08]),
        'beams': np.array([11.0, 76.6, 57.5, 27.0, 24.2, 15.9, 14.8, 10.7, 8.5]),
        'add_noise': True,
        'noise_seed': 1234,
        'use_bandpass': False,
        'output_directory': '/dev/null',
        'output_prefix': 'cmbs4',
        'use_smoothing': False,
    }

np.save("./lib/instr_dict/CMBS4",CMBS4)

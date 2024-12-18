# $\mathcal{C}_{\ell}$ based moment expansion of polarized foregrounds SED.

Component separation method fitting a moment expansion of the foreground SED on cross-frequencies angular power spectra (intensity, $E$- or $B$-modes) in order to recover an unbiased value of the tensor-to-scalar ratio $r$.

## About this work

The original formalism was proposed by [Chluba et al (2017)](https://academic.oup.com/mnras/article/472/1/1195/4064377) and extended to the cross-frequency angular power-spectra in intensity by [Mangilli et al (2021)](https://www.aanda.org/articles/aa/abs/2021/03/aa37367-19/aa37367-19.html) and in polarization in [Vacher et al (2023a)](https://www.aanda.org/articles/aa/full_html/2023/01/aa43913-22/aa43913-22.html) and [Vacher et al (2023b)](https://www.aanda.org/articles/aa/full_html/2023/04/aa45292-22/aa45292-22.html).

The present folder contains the codes applied to the *[LiteBIRD](https://academic.oup.com/ptep/article/2023/4/042F01/6835420?login=false)* instrument in [Vacher et al (2022)](https://www.aanda.org/articles/aa/abs/2022/04/aa42664-21/aa42664-21.html).

## How to use

Moments and r can be fitted by running the "fitmom_dust_sync.py" file. More documentation will be deployed soon.

## Dependencies

Dependencies required: Original simulations were made using [Healpy](https://healpy.readthedocs.io/en/latest/) for map based computations, foreground models were created with [Pysm](https://github.com/galsci/pysm) while the CMB angular power-spectra were generated with [CAMB](https://camb.info/). In the present module, the instrument is called through the [fgbuster](https://github.com/fgbuster/fgbuster) class. Cross-$\mathcal{C}_{\ell}$ are computed using [Namaster](https://github.com/LSSTDESC/NaMaster). Moment coefficients are fited using [mpfit](https://github.com/segasai/astrolibpy/blob/master/mpfit/mpfit.py).

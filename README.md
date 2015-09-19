# fft-comparison
compare different FFT implementations

# Latest results

```
generating input signals
initializing FFTW
        duration: 637.193
Armadillo FFT:
        duration: 4.37427
convolution:
        duration: 471.615
Armadillo FFT-Pow2-convolution:
        duration: 7.50905
        maximum difference of result: 0.0197754
FFTW FFT-Pow2-convolution:
        duration: 0.528795
        maximum difference of result: 0.020142
Armadillo FFT-convolution:
        duration: 390.823
        maximum difference of result: 0.0200195
Armadillo convolution:
        duration: 487.721
        maximum difference of result: 0
```

import numpy as np

def make_root_nyquist_filter(winfunc, Ntaps, Nos, designRatio=128):
    fs = np.zeros(Ntaps * designRatio)
    fs[:Ntaps*designRatio//Nos//2] = 1
    fs[-Ntaps*designRatio//Nos//2:] = 1
    impResp = np.fft.ifft(fs)
    impResp = np.real(impResp)
    impResp = impResp * np.fft.ifft(np.fft.fftshift(winfunc(np.linspace(-Nos/2, Nos/2, num=Ntaps*designRatio, endpoint=False))))
    fs = np.sqrt(np.abs(np.fft.fft(impResp)))
    impResp = np.fft.ifft(fs)
    impResp = np.real(impResp)
    impResp = np.hstack((impResp[:Ntaps//2], impResp[-Ntaps//2:]))
    impResp /= np.sum(impResp)
    return np.fft.fftshift(impResp)

from pympedance.UNSW import waveform_to_spectrum, spectrum_to_waveform
import matplotlib.pyplot as pl
import numpy as np

npts=2**6
x=np.random.rand(npts)
fx=np.fft.fft(x)
spec_x = fx[:int(npts/2+1)]/npts*2
spec_x[0] /=2
xi=spectrum_to_waveform(spec_x)
pl.figure()
pl.plot(x)
pl.plot(xi)

harm_lo=2
harm_hi = int(npts/4)
xi=spectrum_to_waveform(spec_x[harm_lo:harm_hi],harm_lo=harm_lo,num_points=npts)


pl.plot(xi)

pl.title('waveform from fft')

spec_x = waveform_to_spectrum(x)
fig,ax=pl.subplots(2)
ax[0].semilogy(np.abs(fx))
ax[1].plot(np.angle(fx))
ax[0].semilogy(np.abs(fx/npts*2))
ax[1].plot(np.angle(fx/npts*2))
ax[0].semilogy(np.abs(spec_x))
ax[1].plot(np.angle(spec_x))
ax[1].legend(('fft','fft/npts','spec'))

fig = pl.figure()
pl.plot(x)
pl.plot(spectrum_to_waveform(waveform_to_spectrum(x)))
pl.plot(np.fft.ifft(np.fft.fft(x)))
pl.plot(np.fft.ifft(np.fft.fft(x,n=npts),n=npts))
pl.plot(np.fft.ifft(np.fft.fft(x,n=int(npts/2)),n=int(npts/2)))
#pl.plot(spectrum_to_waveform(waveform_to_spectrum(x),num_points=int(npts/2)))




pl.show()

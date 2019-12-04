#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from apply_ltspice_filter import \
  apply_ltspice_filter,\
  convolution_filter,\
  get_impulse_response
import scipy.io.wavfile

##################################################
##             read input waveform              ##
##################################################

rate, data = scipy.io.wavfile.read("348275__bigmanjoe__fantasy-orchestra.wav")

# bring to desired numerical format and normalize 
data = data.astype('float32')
# just take left channel
data = data[:,0]
data = data/np.max(np.abs(data))
print("rate: {:d}".format(rate))


##################################################
##             get impulse response             ##
##################################################

delta_t = 1./rate

kernel_delay = 5e-3
kernel_sample_width = 25e-3

# all values in SI units
filter_configuration = {
  "C": 200e-9,  # 200 nF
  "L": 200e-3,  # 200 mH
  "R": 1e3     # 1 kOhm
}

kernel_time, kernel = get_impulse_response(
        "filter_circuit_2.asc",
        params = filter_configuration,
        sample_width = kernel_sample_width,
        delta_t = delta_t,
        kernel_delay = kernel_delay
        )

##################################################
##            plot impulse response             ##
##################################################

plt.plot(kernel_time, kernel, label="impulse response of filter_circuit_2.asc")
plt.xlabel("time (s)")
plt.ylabel("voltage (V)")
plt.title("impulse response")

plt.legend()
plt.show()


##################################################
##      apply IR filter to input waveform       ##
##################################################

filtered = convolution_filter(
  data,
  kernel,
  delta_t = delta_t,
  kernel_delay = kernel_delay
)

##################################################
##        write filtered signal to file         ##
##################################################

scipy.io.wavfile.write("sound_filtered.wav",rate,filtered.astype('float32') )


##################################################
##      plot original and filtered signal       ##
##################################################

time = np.linspace(0,1./rate * len(data) ,len(data) ) # time vector for x axis of plot
plt.plot(time,data,label="original signal",alpha=0.6)
plt.plot(time,filtered, label="filtered signal",alpha=0.6)
plt.xlabel("time (s)")
plt.ylabel("normalized amplitude")
plt.title("waveform before and after filter")
plt.legend()
plt.show()


# Raptopoulos Petros, Team 11 (No Partner)
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi as pi
import math

# global declared constants
fm = 1000 # f = 1kHz (1+4+5=10, 1+0=1)
T = 0.001 # T = 10^(-3)sec
conStep = 0.000005 # Step small enough to consider
                   # the outcome continuous
t = np.arange(0,4*T, conStep) # define time axis
# span the time values over 4 periods
# we consider the time axis continuous despite
# its discrete representation

def sine(t, f = fm): # define sine function for each time point
    return np.sin(2*pi*f*t)

def sinc(x): # define sinc function for each time point
    if x == 0:
        return 1
    else:
        return math.sin(pi*x)/(pi*x)

def tri(t): # define triangular function for each time point
    t = t % 0.001 # find the mod in order to apply
                  # formulas for the first pulse
    if 0 <= t and t <= 0.0005: # 0 <= t <= T/2
        return 16000*t-4
    elif 0.0005 <= t and t <= 0.001: # T/2 <= t <= T
        return -16000*t+12

vtri = np.vectorize(tri)
# make the tri() function an element-wise one

def q(t):
    return sine(t) + sine(t, fm+1000)
    # fm = 1kHz, Λ = 1kHz, Α = 1V

# Exercise 1

funct = ["", vtri, sine, q]
# list the functions used in each iteration
# start the functions from index 1
for iterNum in range(1,4):
# a for loop to avoid code repetition
# question (a'),(b'): iterNum = 1
# question (c'.i): iterNum = 2
# question (c'.ii): iterNum = 3

    if iterNum != 3:
        xAxisMax = 4*T
        # start from 0 and end just before 4*T
    else:
        xAxisMax = T
        # start from 0 and end just before T
        # for (c.ii) we want one period to be plotted
        # four periods otherwise

    y = funct[iterNum](t) # the continuous signal
    # y_i = funct(t_i), for all i

    fig1 = plt.figure(2*iterNum-1) # create a new figure
    # first iteration uses fig 1,2 , second call uses fig 3,4 ...
    ax = fig1.add_subplot(221) # create one subplot
    plt.xlabel("Time(sec)")
    plt.ylabel("Amplitude(V)")
    plt.title("Given Periodic Signal (fm = 1kHz)")
    plt.grid() # use the grid
    ax.plot(t,y)
    ax.set_xlim(left = 0, right = xAxisMax) # start from 0 and end
                                            # just before xAxisMax

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # use the scientific (exponential) notation for the x axis

    # Question (a')
    # (i)

    # first sampling process
    fs1 = 30*fm
    Ts1 = 1/fs1
    t1 = np.arange(0,4*T+Ts1, Ts1) # define time axis
    # the instruction np.arange() isn't end-inclusive
    # we must add Ts1 to the end in order to include
    # the sample value at t = 4*T

    y1 = funct[iterNum](t1) # y_i = funct(t_i), for all i

    ax1 = fig1.add_subplot(222) # create one subplot
    plt.xlabel("Time(sec)")
    plt.ylabel("Amplitude(V)")
    plt.title("Sampled Signal (fs1 = 30kHz)")
    plt.grid() # use the grid
    ax1.scatter(t1,y1, s=20, marker='o') # adjust size and marker
    ax1.set_xlim(left = 0, right = xAxisMax) # start from 0 and end
                                             # just before xAxisMax

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # use the scientific (exponential) notation for the x axis

    # (ii)

    # second sampling process
    fs2 = 50*fm
    Ts2 = 1/fs2
    t2 = np.arange(0,4*T+Ts2, Ts2) # define time axis
    # the instruction np.arange() isn't end-inclusive
    # we must add Ts1 to the end in order to include
    # the sample value at t = 4*T

    y2 = funct[iterNum](t2) # y_i = funct(t_i), for all i

    ax2 = fig1.add_subplot(223) # create one subplot
    plt.xlabel("Time(sec)")
    plt.ylabel("Amplitude(V)")
    plt.title("Sampled Signal (fs2 = 50kHz)")
    plt.grid() # use the grid
    ax2.scatter(t2,y2, s=20, marker='o') # adjust size and marker
    ax2.set_xlim(left = 0, right = xAxisMax) # start from 0 and end
                                             # just before xAxisMax

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # use the scientific (exponential) notation for the x axis

    # (iii) show both sampled signals in one plot

    ax3 = fig1.add_subplot(224) # create one subplot
    plt.xlabel("Time(sec)")
    plt.ylabel("Amplitude(V)")
    plt.title("Both Sampled Signals")
    plt.grid() # use the grid
    ax3.scatter(t1,y1, s=10, c='b', marker='o', label='fs1 = 30kHz')
    # adjust size, marker, color and label
    ax3.scatter(t2,y2, s=10, c='r', marker='o', label='fs2 = 50kHz')
    # adjust size, marker, color and label
    ax3.set_xlim(left = 0, right = xAxisMax) # start from 0 and end
                                             # just before xAxisMax
    ax3.legend() # show the legends on the plot
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # use the scientific (exponential) notation for the x axis

    fig1.tight_layout() # adjust the spacing between subplots in
                        # order to avoid text overlapping

    # Question (b')
    if iterNum != 3:
        fs = [4*fm, 3*fm, 1.9*fm]
        # 3 sample frequencies for all questions except (c.ii)
    else:
        fs = [4*fm, 4.1*fm, 3.5*fm]
    fig2 = plt.figure(2*iterNum) # create a new figure
    for j in range(0,3): # j = 1 for fs = 4*fm ...
    # Using a for loop to avoid code repetitions
        Ts = 1/fs[j]
        ts = np.arange(0,400*T+Ts, Ts) # define time axis
        # for each sample frequency

        ys = funct[iterNum](ts) # ys_i = funct(ts_i), for all i
        ax1 = fig2.add_subplot(2, 3, 1+j) # create one subplot
        plt.xlabel("Time(sec)")
        plt.ylabel("Amplitude(V)")
        plt.title("Sampled Signal (fs = {}kHz)".format(fs[j]/1000))
        plt.grid() # use the grid
        ax1.scatter(ts,ys, s=10, c='b', marker='o')
        # adjust size, marker, color and label
        ax1.set_xlim(left = 0, right = xAxisMax)
        # start from 0 and end just before xAxisMax

        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # use the scientific (exponential) notation for the x axis

        # Try to reconstruct the signal from its samples
        yrecon = np.empty(t.size)
        for i in range(t.size):
            yrecon[i] = 0
            for n in range(ys.size):
                yrecon[i] += ys[n]*sinc((i*conStep-n*Ts)/Ts)
        # From the theory we have y(t)=sum(y[nTs]sinc((t-nTs)/Ts))
        # for each n (integer) from -00 to +00
        ax2 = fig2.add_subplot(2, 3, 4+j) # create one subplot
        plt.xlabel("Time(sec)")
        plt.ylabel("Amplitude(V)")
        plt.title("Reconstructed Signal (fs = {}kHz)".format(fs[j]/1000))
        plt.grid() # use the grid
        ax2.plot(t,yrecon) # plot the reconstructed signal
        ax2.set_xlim(left = 0, right = xAxisMax) # start from 0 and end
                                                 # just before xAxisMax

        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # use the scientific (exponential) notation for the x axis
        fig2.tight_layout() # adjust the spacing between subplots in
                            # order to avoid text overlapping


# Exercise 2
def gray(x): # x: input int, output binary gray code
    x = int(x) # must do this because some values
               # are stored as floats (2.) and the
               # bitwise operations dont work
    return "{0:b}".format(x^(x>>1)).zfill(5)
# return the gray code as a string consisting of 5
# binary digits. ^ -> bitwise xor, >> -> shift right

vgray = np.vectorize(gray)
# make the gray() function an element-wise one

fs1 = 30*fm
Ts1 = 1/fs1
ts = np.arange(0,4*T+Ts1, Ts1)
ys = vtri(ts) # sampling process
bits = 5 # fm = 1kHz odd
levels = 2**bits # levels of quantization
D = 2*4/levels # quantization step = 2*maxValue/levels

def quantize(x): # apply classification rule
    temp = np.floor(x/D)
    return temp-1 if temp == levels/2 else temp
    # level nums for negative sample-values = -16...-1
    # level nums for positive sample-values = 0...15
    # we notice that the 15th level is overloaded
    # as the classification rule np.floor(x/D) for
    # x = +m_max gives 16, which is rejected.
    # (we have 2^bits levels not 2^bits+1)
    # so we assign for x = +m_max the 15th level

vquantize = np.vectorize(quantize)
# make the quantize() function an element-wise one

# plot the quantization diagram
In = np.linspace(-4, 4, 3000)
qOut = D*(vquantize(In) + 1/2)
# we have a mid rise uniform quantizer
# according to the theory the above formula
# must be used

fig = plt.figure(7) # create a new figure
ax1 = fig.add_subplot(111) # create one subplot
ax2 = ax1.twinx() # create a second y axis
                  # but on the same plot

num = -4
xticks = []
while(num <= 4):
    xticks.append(num)
    num += D
plt.xticks(xticks) # plot ticks of x axis
plt.title("Mid Rise Uniform Quantization Diagram")
ax2.plot(In, qOut)
plt.ylabel("Output Gray-Coded Level")
ax1.set_xlabel("Input Sample")
ax1.set_ylabel("Output Quantized Value")
ax1.set_yticks(np.array(xticks) + D/2) # ticks of first y axis
ax1.plot(In, qOut)
ax1.grid() # use the grid for the first axis
ax1.set_xlim(left = -4, right = 4) # set axis limits
y2ticks = vgray(vquantize(In) + levels/2)
plt.yticks(qOut, y2ticks) # plot ticks for second y axis
fig.autofmt_xdate() # rotate xticks to avoid text overlapping


yqLvl = vquantize(ys) + levels/2 # y_quantization_Level
# we want to count the levels from the bottom
fig = plt.figure(8) # create a new figure
ax3 = fig.add_subplot(111) # create one subplot
plt.yticks([i for i in range(0,levels)], [gray(i) for i in range(32)])
# plot ticks for y axis (gray code)
plt.title("Quantizer Output of Sampled Signal ")
ax3.scatter(ts,yqLvl, s=10, c='r', marker='o', label='Quantized')
ax3.legend(loc = 'upper right') # show the legends on the plot
ax4 = ax3.twinx() # create a second y axis
                  # but on the same plot
ax4.scatter(ts,ys, s=10, c='b', marker='o', label='Sampled')
ax4.plot(t,vtri(t), linewidth=1, label='Analog')
ax4.legend(loc = 'lower right') # show the legends on the plot
ax4.set_ylabel("Amplitude(V)")
ax3.set_xlabel("Time(sec)")
ax3.set_ylabel("Gray Code")
ax3.grid() # use the grid
ax3.set_xlim(left = 0, right = 4*T) # start from 0 and end
                                         # just before 4*T
ax4.set_xlim(left = 0, right = 4*T) # start from 0 and end
                                    # just before 4*T
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

yq = D*(vquantize(ys) + 1/2) # output values of quantizer
error = yq - ys # quantization error

def mean(itemsNum, array): # return arithmetic mean
    sum = 0
    for i in range(itemsNum):
        sum += array[i]
    return sum/itemsNum

def stdDev(itemsNum, array): # return Standard Deviation
    return (mean(itemsNum, (array - mean(itemsNum, array))**2))**(1/2.0)
# σ^2 = Ε[(error_i - μ)^2]

print("Standard Deviation (10 samples):", stdDev(10, error))
print("Standard Deviation (20 samples):", stdDev(20, error))

def SNRq(itemsNum): # return Signal-Noise Ration
    return (stdDev(itemsNum, ys)/stdDev(itemsNum, error))**2

print("SNRq (10 samples):",SNRq(10),",in dBs:",10*math.log10(SNRq(10)))
print("SNRq (20 samples):",SNRq(20),",in dBs:",10*math.log10(SNRq(20)))
print("SNRq (30 samples):",SNRq(30),",in dBs:",10*math.log10(SNRq(30)))

yc = vgray(yqLvl) # y_coded
Tc = 0.002 # duration of a bit
tc = np.linspace(0,bits*Tc*(fs1/fm), 10000)
# we want to plot only one period
def polarRZ(t):
    bitsCount = int(t/Tc)           # count all bits encoded
                                    # starting from zero
    bitIndex = bitsCount % bits     # at which bit are we inside
                                    # one gray code
    codeIndex = int(bitsCount/bits) # at which gray code are we
                                    # inside yc
    if 1/2 <= t/Tc - bitsCount and t/Tc - bitsCount < 1:
        return 0
    else:
        return 1 if yc[codeIndex][bitIndex] == '1' else -1

vpolarRZ = np.vectorize(polarRZ)
# make the quantize() function an element-wise one

fig = plt.figure(9) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("Bits Stream-POLAR RZ encoding")
ax.plot(tc, vpolarRZ(tc))
plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax.set_xlim(left = 0, right = bits*Tc*(fs1/fm))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

# Exercise 3
bits = 36
bitSeq = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0])
# bitSeq = np.random.randint(2, size = bits)
# generate random 36-bits sequence
# discrete uniform distribution
print(bitSeq) # print the sequence
Tb = 0.25 # symbol duration (sec)
fc = 2 # carrier frequency (Hz)
A = 1 # Amplitude (V)

t = np.linspace(0, bits*Tb, 10000, endpoint = False)
# consider continuous t, don't include t = bits*Tb
tb = np.arange(0, bits*Tb+Tb, Tb) # used for ticks

# Question (b')
def BPAM(t): # return amplitude, time used as input
    return A if bitSeq[int(t/Tb)] == 1 else -A

vBPAM = np.vectorize(BPAM)
# make the vBPAM() function an element-wise one

fig = plt.figure(10) # create a new figure
ax1 = fig.add_subplot(221) # create one subplot
plt.title("B-PAM Modulation of Binary Sequence")
ax1.plot(t, vBPAM(t)) # plot BPAM signal
plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax1.set_xlim(left = 0, right = bits*Tb)
ax1.set_xticks(tb) # set x axis' ticks
ax1.set_yticks([-1, 0, 1]) # set y axis' ticks

# Question (a')
def BPSK_bit(t): # return current bit
    return bitSeq[int(t/Tb)]

def BPSK(t): # which formula to use (gray coded)
    if BPSK_bit(t) == 1:
        return A*math.cos(2*pi*fc*t)
    elif BPSK_bit(t) == 0:
        return -A*math.cos(2*pi*fc*t)

vBPSK = np.vectorize(BPSK)
# make the BPSK() function an element-wise one

ax2 = fig.add_subplot(222) # create one subplot
plt.title("Binary Phase-Shift Keying")
prevBit = BPSK_bit(t[0])
lastPlotted = 0
# plot the color-legends
ax2.plot([],[], c='b', label='0')
ax2.plot([],[], c='r', label='1')
colors = ['b','r']
for i in range(1, t.size):
    if BPSK_bit(t[i]) != prevBit or i == t.size - 1:
        ax2.plot(t[lastPlotted:i], vBPSK(t[lastPlotted:i]), c=colors[prevBit])
        prevBit = BPSK_bit(t[i])
        lastPlotted = i
# we plot zeros and ones with different colors

plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax2.set_xlim(left = 0, right = bits*Tb)
ax2.set_xticks(tb)
ax2.legend(loc = 'lower right')
plt.setp(ax2.get_xticklabels(), rotation=90)
plt.setp(ax1.get_xticklabels(), rotation=90)
# we rotate the axis' ticks in order to avoid overlapping

def QPSK_bits(t): # return current bits
    return str(bitSeq[2*int(t/(2*Tb))])+str(bitSeq[2*int(t/(2*Tb))+1])

def QPSK(t): # which formula to use (gray coded)
    if QPSK_bits(t) == '00':
        return A*math.cos(2*pi*fc*t)
    elif QPSK_bits(t) == '01':
        return A*math.sin(2*pi*fc*t)
    elif QPSK_bits(t) == '11':
        return -A*math.cos(2*pi*fc*t)
    elif QPSK_bits(t) == '10':
        return -A*math.sin(2*pi*fc*t)

vQPSK = np.vectorize(QPSK)
# make the QPSK() function an element-wise one

ax3 = fig.add_subplot(223) # create one subplot
plt.title("Quadrature Phase-Shift Keying")
prevBits = QPSK_bits(t[0])
lastPlotted = 0
colors = {'00':'b', '01':'r','10':'g','11':'m'}
for i in colors:
    ax3.plot([],[], c=colors[i], label=i)
# plot color-legends

for i in range(1, t.size):
    if QPSK_bits(t[i]) != prevBits or i == t.size - 1:
        ax3.plot(t[lastPlotted:i], vQPSK(t[lastPlotted:i]), c=colors[prevBits])
        prevBits = QPSK_bits(t[i])
        lastPlotted = i
# we plot different colors for each gray code

plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax3.set_xlim(left = 0, right = bits*Tb)
ax3.set_xticks(tb)
ax3.legend(loc = 'lower right')
plt.setp(ax3.get_xticklabels(), rotation=90)
# we rotate the axis' ticks in order to avoid overlapping

def gray(x): # redefine gray function
    x = int(x)
    return "{0:b}".format(x^(x>>1)).zfill(3)

# use / to make spread among many lines
def PSK8_bits(t): # return current bit
    return str(bitSeq[3*int(t/(3*Tb))]) + \
    str(bitSeq[3*int(t/(3*Tb))+1]) + \
    str(bitSeq[3*int(t/(3*Tb))+2])

def PSK8(t): # which formula to use
    phase = {gray(i):i*pi/4 for i in range(8)}
    return A*math.cos(2*pi*fc*t - phase[PSK8_bits(t)])

vPSK8 = np.vectorize(PSK8)
# make the QPSK() function an element-wise one

ax4 = fig.add_subplot(224) # create one subplot
plt.title("8 Phase-Shift Keying")
prevBits = PSK8_bits(t[0])
lastPlotted = 0
colors = {'000':'b', '001':'r','011':'g','010':'m',\
'110':'c','111':'y','101':'k','100':'tab:purple'}
for i in colors:
    ax4.plot([],[], c=colors[i], label=i)
# plot color-legends

for i in range(1, t.size):
    if PSK8_bits(t[i]) != prevBits or i == t.size - 1:
        ax4.plot(t[lastPlotted:i], vPSK8(t[lastPlotted:i]), c=colors[prevBits])
        prevBits = PSK8_bits(t[i])
        lastPlotted = i
# we plot different colors for each gray code

plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax4.set_xlim(left = 0, right = bits*Tb)
ax4.set_xticks(tb)
ax4.legend(loc = 'lower right')
plt.setp(ax4.get_xticklabels(), rotation=90)
# we rotate the axis' ticks in order to avoid overlapping
fig.tight_layout() # adjust the spacing between subplots in
                    # order to avoid text overlapping

# Question (c')
A = 1 # A = 1V
phaseBPAM = [0, pi] # The signals' phase
eBPAM = A*A*Tb # Energy of bit in BPAM modulation
Q = [math.sqrt(eBPAM)*math.sin(phaseBPAM[i]) for i in range(2)]
# Quadrature Component
I = [math.sqrt(eBPAM)*math.cos(phaseBPAM[i]) for i in range(2)]
# In Phase Component
figCon = plt.figure(11) # create a new figure
ax = figCon.add_subplot(221) # create one subplot
plt.title("Constellation Diagram-BPAM Signal")
ax.scatter(I, Q, marker=(5, 1), s = 100)
plt.ylabel("Quadrature Component-φ2(t)")
plt.xlabel("In Phase Component-φ1(t)")
plt.grid() # use the grid
ax.set_xlim(left = -0.75, right = +0.75)
ax.set_ylim(bottom = -0.75, top = +0.75)

# Question (d')
# plot again BPAM signal
fig = plt.figure(12) # create a new figure
ax = fig.add_subplot(221) # create one subplot
plt.title("B-PAM Modulation of Binary Sequence")
ax.plot(t, vBPAM(t))
plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax.set_xlim(left = 0, right = bits*Tb)
ax.set_xticks(tb)
ax.set_yticks([-1, 0, 1])
plt.setp(ax.get_xticklabels(), rotation=90)

# one sided spectral noise density N0/2
M = 2 # We have B-PAM
k = math.log(M, 2)
ratioIndB = np.array([5, 15]) # ratio(dB) = 10log(Eb/(N0/2))
ratio = 10**(ratioIndB/10)
SNR = 2*k*ratio
noisePower = stdDev(t.size, vBPAM(t))/SNR
noiseMean = 0
for i in range(2):
    noiseX = np.random.normal(noiseMean, noisePower[i]**(1/2.0), t.size)
    noiseY = np.random.normal(noiseMean, noisePower[i]**(1/2.0), t.size)
    sigWnoise = noiseX + vBPAM(t)
    ax1 = fig.add_subplot(2,2,i+2) # create one subplot
    ax1.set_title("Modulated Signal with AWG Noise (Eb/N0 = {}dB)".format(ratioIndB[i]))
    ax1.set_ylabel("Amplitude(V)")
    ax1.set_xlabel("Time(sec)")
    ax1.plot(t, sigWnoise, linewidth=0.3)
    ax1.grid() # use the grid
    ax1.set_xlim(left = 0, right = bits*Tb)
    ax1.set_xticks(tb)
    plt.setp(ax1.get_xticklabels(), rotation=90)
    # Question (e')
    ax2 = figCon.add_subplot(2, 2, 2+i) # create one subplot
    ax2.set_title("Constellation Diagram of the B-PAM Signal (Eb/N0 = {}dB)".format(ratioIndB[i]))
    ax2.set_ylabel("Quadrature Component-φ2(t)")
    ax2.set_xlabel("In Phase Component-φ1(t)")
    ax2.scatter(sigWnoise*math.sqrt(eBPAM), noiseY*math.sqrt(eBPAM), s = 1)
    ax2.grid() # use the grid
    ax2.set_xlim(left = -1, right = +1)
    ax2.set_ylim(bottom = -1, top = +1)

figCon.tight_layout()
fig.tight_layout() # adjust the spacing between subplots in
                    # order to avoid text overlapping

# Question (f')
def Q(x):
    return 1/2.0 - math.erf(x/(2)**(1/2.0))/2.0

vQ = np.vectorize(Q)
# make the Q() function an element-wise one

fig = plt.figure(13) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("B-PAM Error Rates")
plt.ylabel("Bit Error Rate")
plt.xlabel("Eb/N0 (dB)")
plt.grid() # use the grid
ax.set_xlim(left = 0, right = 15)
ax.set_yscale('log')

# calculate the theoretical probability error
ratioIndB = np.arange(0, 16, 0.01)
ratio = 10**(ratioIndB/10)
Perror = vQ((2*2*ratio)**(1/2.0)) # Probability error in BPAM
ax.plot(ratioIndB, Perror, c='b', label='Theoretical')

# calculate the empirical probability error
bits = 10**6
bitSeq = np.random.randint(2, size = bits) # binary sequence
divs = 2 # how many division of time in Tb
t = np.linspace(0, bits*Tb, divs*bits, endpoint=False) # time
ratioIndB = np.arange(16) # ratio values for plotting
for i in range(ratioIndB.size):
    ratio = 10**(ratioIndB[i]/10)
    noisePower = stdDev(t.size, vBPAM(t))/(2*ratio)
    # we must multiply with 2 because we have N0/2 one sided
    # spectral density for the noise, so N0/4 two sided
    noise = np.random.normal(0, noisePower**(1/2.0), t.size)
    sigWnoise = noise + vBPAM(t) # add noise to our signal
    sums = np.zeros(bits)
    for time in range(t.size): # using matched filter
        sums[int(t[time]/Tb)] += sigWnoise[time]

    errors = 0 # count the errors
    for bit in range(bits): # step = i*Tb
        if sums[bit]*BPAM(bit*Tb)<0:
            errors += 1
        elif sums[bit] == 0 and np.random.randint(2, size = 1) == 1:
            errors += 1
    errorFreq = errors / bits # calculate error frequency
    ax.scatter(ratioIndB[i], errorFreq,  c='r', s=10)

ax.plot([],[], c='r', label='Empirical')
ax.legend(loc = 'lower left')


# Exercise 4
# Question (a)
bits = 36
bitSeq = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0])
# test the sequence included in the lab report
# bitSeq = np.random.randint(2, size = bits)
# generate random 36-bits sequence
# discrete uniform distribution
print(bitSeq) # print the sequence
Tb = 0.25 # bit duration (sec), Tsymbol = 2Tb
A = 1 # Amplitude (V)
eQPSK = A*A*Tb # Bit Energy for QPSK, SymbolEnergy = 2*eQPSK
phaseQPSK = [-3*pi/4, 3*pi/4, -pi/4, pi/4] # The signals' phase
grayCode = ['00', '01', '10', '11'] # Gray coded signals
Q = [math.sqrt(2*eQPSK)*math.sin(phaseQPSK[i]) for i in range(4)]
# Quadrature Component
I = [math.sqrt(2*eQPSK)*math.cos(phaseQPSK[i]) for i in range(4)]
# In Phase Component
figCon = plt.figure(14) # create a new figure
ax = figCon.add_subplot(221) # create one subplot
plt.title("Constellation Diagram-QPSK Signal")
ax.scatter(I, Q, marker=(5, 1), s = 100)
for i in range(4):
    ax.annotate(grayCode[i], (I[i],Q[i]), fontsize=10)
plt.ylabel("Quadrature Component-φ2(t)")
plt.xlabel("In Phase Component-φ1(t)")
plt.grid() # use the grid
ax.set_xlim(left = -0.75, right = +0.75)
ax.set_ylim(bottom = -0.75, top = +0.75)

# Question (b)
t = np.linspace(0, bits*Tb, 10000, endpoint = False)
# consider continuous t, don't include t = bits*Tb
grayMapped = {'00':-3*pi/4, '01':3*pi/4, '10':-pi/4, '11':pi/4}
M = 4 # We have QPSK
k = math.log(M, 2)
ratioIndB = np.array([5, 15]) # ratio(dB) = 10log(Eb/(N0/2))
ratio = 10**(ratioIndB/10)
SNR = 2*k*ratio
def QPSK_bits(t):
    return str(bitSeq[2*int(t/(2*Tb))])+str(bitSeq[2*int(t/(2*Tb))+1])

noisePower = eQPSK/Tb/SNR # power of given QPSK signal = eQPSK/Tb
noiseMean = 0
for i in range(2):
    noiseX = np.random.normal(noiseMean, noisePower[i]**(1/2.0), t.size)
    noiseY = np.random.normal(noiseMean, noisePower[i]**(1/2.0), t.size)
    ax2 = figCon.add_subplot(2, 2, 2+i) # create one subplot
    ax2.set_title("Constellation Diagram of the QPSK Signal (Eb/N0 = {}dB)".format(ratioIndB[i]))
    ax2.set_ylabel("Quadrature Component-φ2(t)")
    ax2.set_xlabel("In Phase Component-φ1(t)")
    ax2.grid() # use the grid
    ax2.set_xlim(left = -1, right = +1)
    ax2.set_ylim(bottom = -1, top = +1)
    for j in range(t.size):
        sigWnoiseX = noiseX[j] + math.cos(grayMapped[QPSK_bits(t[j])])
        sigWnoiseY = noiseY[j] + math.sin(grayMapped[QPSK_bits(t[j])])
        ax2.scatter(sigWnoiseX*math.sqrt(2*eQPSK), sigWnoiseY*math.sqrt(2*eQPSK), s = 1, c='b')
figCon.tight_layout()

# Question (c)
def QPSK_bitsRecon(x, y):
    if x == 0:
        x = 2*np.random.randint(2, size=1)-1
    if y == 0:
        y = 2*np.random.randint(2, size=1)-1
# if x == 0 or y == 0 we consider them positive
# or negative randomly
    if x < 0 and y < 0:
        return '00'
    elif x < 0 and y > 0:
        return '01'
    elif x > 0 and y > 0:
        return '11'
    elif x > 0 and y < 0:
        return '10'

fig = plt.figure(15) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("QPSK Error Rates")
plt.ylabel("Bit Error Rate")
plt.xlabel("Eb/N0 (dB)")
plt.grid() # use the grid
ax.set_xlim(left = 0, right = 15)
ax.set_yscale('log')

# calculate the theoretical probability error
ratioIndB = np.arange(0, 16, 0.01)
ratio = 10**(ratioIndB/10)
Perror = vQ((2*2*ratio)**(1/2.0)) # Probability error in BPAM
ax.plot(ratioIndB, Perror, c='b', label='Theoretical')

# calculate the empirical probability error
bits = 10**6
bitSeq = np.random.randint(2, size = bits) # binary sequence
divs = 1 # how many division of time in Tb
t = np.linspace(0, bits*Tb, divs*bits, endpoint=False) # time
ratioIndB = np.arange(16) # ratio values for plotting
for i in range(ratioIndB.size):
    ratio = 10**(ratioIndB[i]/10)
    noisePower = eQPSK/Tb/(2*k*ratio)
    # we must multiply with 2 because we have N0/2 one sided
    # spectral density for the noise, so N0/4 two sided
    noiseX = np.random.normal(noiseMean, noisePower**(1/2.0), t.size)
    noiseY = np.random.normal(noiseMean, noisePower**(1/2.0), t.size)
    Xs = np.zeros(int(bits/2))
    Ys = np.zeros(int(bits/2))
    for time in range(t.size): # using matched filter
        Xs[int(t[time]/(2*Tb))] += noiseX[time] + math.cos(grayMapped[QPSK_bits(t[time])])
        Ys[int(t[time]/(2*Tb))] += noiseY[time] + math.sin(grayMapped[QPSK_bits(t[time])])
    errors = 0
    for j in range(Xs.size):
        if str(bitSeq[2*j])+str(bitSeq[2*j+1]) != QPSK_bitsRecon(Xs[j], Ys[j]):
            errors += 1
    errorFreq = errors / bits # calculate error frequency
    ax.scatter(ratioIndB[i], errorFreq,  c='r', s=10)

ax.plot([],[], c='r', label='Empirical')
ax.legend(loc = 'lower left')


# Question (d)
# (i)
import binascii
f = open("rice_odd.txt", "r") # el19145
# 1 + 4 + 5 = 10, 1 + 0 = 1 (odd)
inputText = f.read()
binary = bin(int.from_bytes(inputText.encode(), 'big'))
binary = binary[0] + binary[2:]
# remove b character which denotes binary number in python

print("Input Text:")
print(inputText)
f.close()

# (ii) Quantization of the signal

bits = 8 # fm = 1kHz odd
levels = 2**bits # levels of quantization
D = 2**bits/levels # quantization step=valuesRange/levels
# the characters (8-digit binary numbers) can take the
# discrete values from 00000000 to 11111111 (not Gray coded)

# So we can infer that the output quantized value
# is equal to the input value (8-digit binary number),
# as the quantizer and each character uses 8 bits and
# the string is already a discrete signal

fig = plt.figure(16) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("Text Quantization")
plt.ylabel("Quantization Binary Level")
plt.xlabel("Characters' Succession In Text")
plt.grid() # use the grid
plt.yticks([i for i in range(levels)], [(bin(i)[0]+bin(i)[2:]).zfill(8) for i in range(levels)])
plt.yticks(fontsize=6)
for i in range(0, len(binary), 8):
    ax.scatter(i/8, int(binary[i:i+8], 2), c='b', s=5)
ax.set_xlim(left = 0, right = 500)

# (iii) QPSK modulation
fc = 1
Tb = 0.25
A = 1
def QPSK_bits(t):
    return str(binary[2*int(t/(2*Tb))])+str(binary[2*int(t/(2*Tb))+1])

def QPSK(t): # which formula to use (gray coded)
    if QPSK_bits(t) == '00':
        return A*math.cos(2*pi*fc*t)
    elif QPSK_bits(t) == '01':
        return A*math.sin(2*pi*fc*t)
    elif QPSK_bits(t) == '11':
        return -A*math.cos(2*pi*fc*t)
    elif QPSK_bits(t) == '10':
        return -A*math.sin(2*pi*fc*t)

vQPSK = np.vectorize(QPSK)
# make the QPSK() function an element-wise one

t = np.linspace(0, len(binary)*Tb, 100000, endpoint = False)
# consider continuous t, don't include t = bits*Tb

fig = plt.figure(17) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("Modulated(QPSK) Text")
prevBits = QPSK_bits(t[0])
lastPlotted = 0
colors = {'00':'b', '01':'r','10':'g','11':'m'}
for i in colors:
    ax.plot([],[], c=colors[i], label=i)
# plot color-legends

for i in range(1, t.size):
    if QPSK_bits(t[i]) != prevBits or i == t.size - 1:
        ax.plot(t[lastPlotted:i], vQPSK(t[lastPlotted:i]), c=colors[prevBits])
        prevBits = QPSK_bits(t[i])
        lastPlotted = i
# we plot different colors for each gray code

plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax.set_xlim(left = 0, right = len(binary)*Tb)
ax.legend(loc = 'lower right')
plt.setp(ax.get_xticklabels(), rotation=90)
# we rotate the axis' ticks in order to avoid overlapping

# (iv) Noise Addition
ratioIndB = np.array([5, 15]) # ratio(dB) = 10log(Es/(N0/2))
ratio = 10**(ratioIndB/10)
SNR = 2*ratio # we have N0/2 one-sided noise power spectral density
eQPSK = A*A*Tb # Energy of bit in QPSK modulation
noisePower = eQPSK/Tb/SNR # power of given QPSK signal = eQPSK/Tb
noiseMean = 0
fig = plt.figure(18) # create a new figure
figCon = plt.figure(19) # create a new figure
for i in range(2):
    noise = np.random.normal(noiseMean, noisePower[i]**(1/2.0), t.size)
    ax = fig.add_subplot(2,1,1+i) # create one subplot
    ax.set_title("QPSK signal with noise (Es/N0 = {}dB)".format(ratioIndB[i]))
    ax.set_ylabel("Amplitude(V)")
    ax.set_xlabel("Time(sec)")
    ax.grid() # use the grid
    ax.set_xlim(left = 0, right = len(binary)*Tb)
    sigWnoise = noise + vQPSK(t) # noise added to original signal
    ax.plot(t, sigWnoise)
    # (v) Constellation Diagrams
    Xs = np.zeros(int(len(binary)/2)) # Demodulation of the signal
    Ys = np.zeros(int(len(binary)/2))
    for time in range(t.size):
        Xs[int(t[time]/(2*Tb))] += sigWnoise[time]*math.cos(2*pi*fc*t[time])
        Ys[int(t[time]/(2*Tb))] += sigWnoise[time]*math.sin(2*pi*fc*t[time])
    ax = figCon.add_subplot(2,1,1+i) # create one subplot
    ax.set_title("Constellation Diagram of the QPSK Signal (Es/N0 = {}dB)".format(ratioIndB[i]))
    ax.set_ylabel("Quadrature Component-φ2(t)")
    ax.set_xlabel("In Phase Component-φ1(t)")
    ax.grid() # use the grid
    for bitPair in range(Xs.size):
        ax.scatter(Xs[bitPair], Ys[bitPair], s = 1, c='b')

    # (vi) BER calculation
    # Reconstruction of the received signal
    phases = [0, pi/2, pi, 3*pi/2]
    codes = ['00', '01', '11', '10']
    sI = [math.cos(k) for k in phases]
    sQ = [math.sin(k) for k in phases]
    binaryRecon = ''
    for bitPair in range(Xs.size):
        min = 10000 # one big number for min var init
        minJ = 0
        for j in range(4): # find the min distance among original signals
            if (Xs[bitPair]-sI[j])**2 + (Ys[bitPair]-sQ[j])**2 < min:
                min = (Xs[bitPair]-sI[j])**2 + (Ys[bitPair]-sQ[j])**2
                minJ = j
        binaryRecon += codes[minJ] # s_j decided

    errors = 0
    for bit in range(len(binaryRecon)):
        if binaryRecon[bit] != binary[bit]:
            errors += 1
    BERempirical = errors/len(binaryRecon)
    BERtheoretical = vQ(math.sqrt(2*ratio[i]))
    print("For Es/N0(db)={}".format(ratioIndB[i]))
    print("Empirical BER: ", BERempirical)
    print("Theoretical BER: ", BERtheoretical)

    # (vii) Text Reconstruction
    fOut = open("reconstructedText{}dB.txt".format(ratioIndB[i]), "w")
    out = ''
    for bit in range(int(len(binaryRecon)/8)):
        temp = int(binaryRecon[8*bit:8*bit+8], 2) # convert binary to chars
        out += temp.to_bytes((temp.bit_length() + 7) // 8, 'big').decode()
    print("Output Text:")
    print(out) # print reconstructed text to terminal
    fOut.write(out) # write reconstructed text to output file
    fOut.close()
fig.tight_layout()
figCon.tight_layout()


# Exercise 5
# Question (a)
from scipy.io.wavfile import read

input_data = read("soundfile1_lab2.wav")
audio = input_data[1]
fig = plt.figure(20) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("Given wavfile Waveform")
plt.grid() # use the grid
plt.plot(audio)
plt.ylabel("Amplitude")
plt.xlabel("Time")

# Question (b)

bits = 8
levels = 2**bits # levels of quantization
maxValue = max(audio)
minValue = np.min(audio)
D = (maxValue-minValue)/levels # quantization step
def quantize(x): # apply classification rule
    temp = np.floor(x/D)
    return temp-1 if x == maxValue else temp

vquantize = np.vectorize(quantize)
# make the quantize() function an element-wise one

q = D*(vquantize(audio) + 1/2) # output values of quantizer
fig = plt.figure(21) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("Quantized Waveform")
plt.grid() # use the grid
plt.plot(q)
plt.ylabel("Amplitude")
plt.xlabel("Time")

qLvl = vquantize(audio) # quantization_Level
qLvl = [qLvl[i]+abs(quantize(minValue)) for i in range(qLvl.size)]
# we want positive quantization levels
qBinary = ''
for lvl in qLvl:
    temp = bin(int(lvl))
    temp = temp[0]+temp[2:]
    qBinary += temp.zfill(8)

# (c) QPSK base band modulation
fc = 1
Tb = 0.25
A = 1
def QPSK_bits(t):
    return str(qBinary[2*int(t/(2*Tb))])+str(qBinary[2*int(t/(2*Tb))+1])

def QPSK(t): # which formula to use (gray coded)
    if QPSK_bits(t) == '00':
        return A*math.cos(2*pi*fc*t)
    elif QPSK_bits(t) == '01':
        return A*math.sin(2*pi*fc*t)
    elif QPSK_bits(t) == '11':
        return -A*math.cos(2*pi*fc*t)
    elif QPSK_bits(t) == '10':
        return -A*math.sin(2*pi*fc*t)

vQPSK = np.vectorize(QPSK)
# make the QPSK() function an element-wise one

t = np.linspace(0, len(qBinary)*Tb, 100000, endpoint = False)

fig = plt.figure(22) # create a new figure
ax = fig.add_subplot(111) # create one subplot
plt.title("Modulated(QPSK) Signal")
prevBits = QPSK_bits(t[0])
lastPlotted = 0
colors = {'00':'b', '01':'r','10':'g','11':'m'}
for i in colors:
    ax.plot([],[], c=colors[i], label=i)
# plot color-legends

for i in range(1, t.size):
    if QPSK_bits(t[i]) != prevBits or i == t.size - 1:
        ax.plot(t[lastPlotted:i], vQPSK(t[lastPlotted:i]), c=colors[prevBits])
        prevBits = QPSK_bits(t[i])
        lastPlotted = i
# we plot different colors for each gray code

plt.ylabel("Amplitude(V)")
plt.xlabel("Time(sec)")
plt.grid() # use the grid
ax.set_xlim(left = 0, right = len(qBinary)*Tb)
ax.legend(loc = 'lower right')
plt.setp(ax.get_xticklabels(), rotation=90)
# we rotate the axis' ticks in order to avoid overlapping

# (d) Noise Addition
ratioIndB = np.array([4, 14]) # ratio(dB) = 10log(Es/(N0/2))
ratio = 10**(ratioIndB/10)
SNR = 2*ratio # we have N0/2 one-sided noise power spectral density
eQPSK = A*A*Tb # Energy of bit in QPSK modulation
noisePower = eQPSK/Tb/SNR # power of given QPSK signal = eQPSK/Tb
noiseMean = 0
fig = plt.figure(23) # create a new figure
figCon = plt.figure(24) # create a new figure
for i in range(2):
    noise = np.random.normal(noiseMean, noisePower[i]**(1/2.0), t.size)
    ax = fig.add_subplot(2,1,1+i) # create one subplot
    ax.set_title("QPSK signal with noise (Es/N0 = {}dB)".format(ratioIndB[i]))
    ax.set_ylabel("Amplitude(V)")
    ax.set_xlabel("Time(sec)")
    ax.grid() # use the grid
    sigWnoise = noise + vQPSK(t) # noise added to original signal
    ax.plot(sigWnoise)
    # (v) Constellation Diagrams
    Xs = np.zeros(int(len(qBinary)/2)) # Demodulation of the signal
    Ys = np.zeros(int(len(qBinary)/2))
    for time in range(t.size):
        Xs[int(t[time]/(2*Tb))] += sigWnoise[time]*math.cos(2*pi*fc*t[time])
        Ys[int(t[time]/(2*Tb))] += sigWnoise[time]*math.sin(2*pi*fc*t[time])
    ax = figCon.add_subplot(2,1,1+i) # create one subplot
    ax.set_title("Constellation Diagram of the QPSK Signal (Es/N0 = {}dB)".format(ratioIndB[i]))
    ax.set_ylabel("Quadrature Component-φ2(t)")
    ax.set_xlabel("In Phase Component-φ1(t)")
    ax.grid() # use the grid
    for bitPair in range(1000):
        ax.scatter(Xs[bitPair], Ys[bitPair], s = 1, c='b')


    # (vi) BER calculation
    # Reconstruction of the received signal
    phases = [0, pi/2, pi, 3*pi/2]
    codes = ['00', '01', '11', '10']
    sI = [math.cos(k) for k in phases]
    sQ = [math.sin(k) for k in phases]
    binaryRecon = ''
    for bitPair in range(Xs.size):
        min = 10000 # one big number for min var init
        minJ = 0
        for j in range(4): # find the min distance among original signals
            if (Xs[bitPair]-sI[j])**2 + (Ys[bitPair]-sQ[j])**2 < min:
                min = (Xs[bitPair]-sI[j])**2 + (Ys[bitPair]-sQ[j])**2
                minJ = j
        binaryRecon += codes[minJ] # s_j decided

    errors = 0
    for bit in range(len(binaryRecon)):
        if binaryRecon[bit] != qBinary[bit]:
            errors += 1
    BERempirical = errors/len(binaryRecon)
    BERtheoretical = vQ(math.sqrt(2*ratio[i]))
    print("For Es/N0(db)={}".format(ratioIndB[i]))
    print("Empirical BER: ", BERempirical)
    print("Theoretical BER: ", BERtheoretical)

fig.tight_layout()
figCon.tight_layout()

plt.show() # show the figures and their plots

import sys
sys.path.append("./client")

import argparse
import ezsdr
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button
from scipy import signal as spsignal
import filterlib

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', '--mod', default='bpsk')
argparser.add_argument('-n', '--nos', default=8, type=int)
argparser.add_argument('-b', '--beta', default=0.5, type=float)
argparser.add_argument('-r', '--rrc', default=1, type=int)
argparser.add_argument('--nfilt', default=10, type=int)
argparser.add_argument('--seed', default=-1, type=int)
argparser.add_argument('--ipaddr', default="127.0.0.1")
argparser.add_argument('--port', default=8888, type=int)
argparser.add_argument('--onlyTx', default=0, type=int)
argparser.add_argument('--onlyRx', default=0, type=int)
argparser.add_argument('--cnstl', default="[]")
# argparser.add_argument('--wirefmt', default="fc32")


args = argparser.parse_args()

if args.onlyTx > 0 and args.onlyRx > 0:
    raise ValueError("args.onlyTx と args.onlyRx が同時に指定されています")

# サーバーIPとポート番号
IPADDR = args.ipaddr;
PORT = args.port;

# wirefmt = np.complex64
# if args.wirefmt == "sc16":
#     wirefmt = ezsdr.complex_int16
# elif args.wirefmt == "sc8":
#     wirefmt = ezsdr.complex_int8

bUseTx = not args.onlyRx
bUseRx = not args.onlyTx

nTXUSRP = 1 if bUseTx else 0
nRXUSRP = 1 if bUseRx else 0

nSamples = 2**10
nOverSample = args.nos
nRRCSamples = nOverSample*args.nfilt

if args.rrc > 0:
    useRRC = True
else:
    useRRC = False

bpsk_constellation = np.array([1+0j, -1+0j])
qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

if args.mod == "bpsk":
    constellation = bpsk_constellation
elif args.mod == "qpsk":
    constellation = qpsk_constellation
elif args.mod == "user":
    constellation = eval(args.cnstl)
    assert len(constellation).bit_count() == 1 and len(constellation) > 1, "The size of given len(cnstl) is not 2**n where n > 0"
    constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
else:
    assert False, "Invalid parameter 'mod'"


def make_rrc_filter(Ntaps, Nos, beta, designRatio=128):

    @np.vectorize
    def rc_window(x):
        if -beta/2 < x and x <= beta/2:
            return np.cos(np.pi * x / beta)
        else:
            return 0.0
        
    return filterlib.make_root_nyquist_filter(rc_window, Ntaps, Nos, designRatio)


def calc_delay(tx, rx):
    tx_freq = np.fft.fft(tx)
    rx_freq = np.fft.fft(rx)
    rxy = np.abs(np.fft.ifft(np.conj(tx_freq) * rx_freq))
    return np.argmax(rxy)

rrcImpResp = make_rrc_filter(nRRCSamples, nOverSample, args.beta)

def gen_transmit_signal(gain):
    if args.seed > -1:
        np.random.seed(args.seed)

    selected_symbols = np.random.choice(constellation, nSamples//nOverSample)

    if useRRC:
        signals = np.zeros((1, nSamples), dtype=np.complex128)
        signals[0, 0::nOverSample] = selected_symbols

        signals = [
            spsignal.lfilter(rrcImpResp, 1, signals[0]) * gain
        ]

    else:
        signals = [
            np.repeat(selected_symbols, nOverSample) * gain
        ]

    return signals

# with multiusrp.SimpleClient(IPADDR, PORT, [nTXUSRP], [nRXUSRP]) as usrp:
with ezsdr.EzSDRClient(IPADDR, PORT) as client:
    signals = gen_transmit_signal(0.1)

    if bUseRx:
        RX0 = ezsdr.CyclicReceiver(client, "RX0", dtype_wire=None, dtype_cpu=np.complex64)
        RX0.changeAlignSize(nSamples)

    if bUseTx:
        TX0 = ezsdr.CyclicTransmitter(client, "TX0", dtype_wire=None, dtype_cpu=np.complex64)
        TX0.transmit(signals)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax_phase = fig.add_axes([0.1, 0, 0.8, 0.03])
    phase_slider = Slider(
        ax=ax_phase,
        label='RX Phase (rad)',
        valmin=-np.pi,
        valmax=np.pi,
        valinit=0,
    )
    ax_gain = fig.add_axes([0.1, 0.9, 0.8, 0.03])
    gain_slider = Slider(
        ax=ax_gain,
        label='TX Gain (dB)',
        valmin=-100,
        valmax=10,
        valinit=-20,
    )

    frame_count = 0
    starttime = 0
    mean_psd = np.zeros(nSamples, dtype=np.double)

    def plot(data):
        global frame_count
        global starttime
        global mean_psd
        global signals

        if frame_count == 0:
            starttime = time.time()
        elif frame_count % 100 == 0:
            print("{} fps".format(frame_count / (time.time() - starttime)))

        ax1.clear()
        ax2.clear()
        ax3.clear()

        if bUseRx:
            recv = RX0.receive(nSamples*2)[0] * np.exp(1j*phase_slider.val)
        else:
            recv = np.tile(signals[0].copy(), 2)

        recvOriginal = recv.copy()

        if useRRC:
            recv = spsignal.lfilter(rrcImpResp, 1, recv)

        nDelay = calc_delay(signals[0], recv[:nSamples])
        recv = recv[nDelay:nDelay+nSamples]
        recvOriginal = recvOriginal[nDelay:nDelay+nSamples]

        if frame_count == 0:
            mean_psd = np.abs(np.fft.fftshift(np.fft.fft(recvOriginal * np.blackman(nSamples))))**2
        else:
            mean_psd = mean_psd * (1-0.3) + np.abs(np.fft.fftshift(np.fft.fft(recvOriginal * np.blackman(nSamples))))**2 * 0.3

        RImax = np.max(np.hstack((np.abs(np.real(recv)), np.abs(np.imag(recv)))))

        startDelay = 0
        if useRRC:
            startDelay = nRRCSamples

        recvTiming = recv[startDelay::nOverSample]

        ax1.scatter(np.real(recvTiming), np.imag(recvTiming))
        ax1.set_xlim([-RImax*1.1, RImax*1.1])
        ax1.set_ylim([-RImax*1.1, RImax*1.1])

        ax2.plot(np.linspace(-0.5, 0.5, nSamples)*nOverSample, 10 * np.log10(mean_psd))

        # ax3.plot(np.arange(nSamples), np.real(signals[0]), label="TX:Re")
        # ax3.plot(np.arange(nSamples), np.imag(signals[0]), label="TX:Im")
        ax3.plot(np.arange(nSamples), np.real(recv), label="RX:Re")
        ax3.plot(np.arange(nSamples), np.imag(recv), label="RX:Im")

        startDelay = 0
        if useRRC:
            startDelay = nRRCSamples//2

        recvTiming = recv[startDelay::nOverSample]
        ax3.scatter(np.arange(nSamples)[startDelay::nOverSample], np.real(recvTiming), marker='x', color='red')
        ax3.scatter(np.arange(nSamples)[startDelay::nOverSample], np.imag(recvTiming), marker='x', color='red')

        ax3.legend(loc='lower left')
        # ax2.set_ylim([-70, 0])

        signals = gen_transmit_signal(10**(gain_slider.val/20))

        if bUseTx:
            TX0.transmit(signals)

        frame_count += 1

    ani = animation.FuncAnimation(fig, plot, interval=50, blit=False)
    plt.show()

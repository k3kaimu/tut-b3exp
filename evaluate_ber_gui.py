import sys
sys.path.append("./client")

import argparse
import ezsdr
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', '--mod', default='bpsk')
argparser.add_argument('--ipaddr', default="127.0.0.1")
argparser.add_argument('--port', default=8888, type=int)
argparser.add_argument('--onlyTx', default=0, type=int)
argparser.add_argument('--onlyRx', default=0, type=int)
argparser.add_argument('--ebn0', default=10, type=float)
argparser.add_argument('--amplitude', default=0.1, type=float)
argparser.add_argument('--sigma2', default=-1, type=float)
argparser.add_argument('--delaythr', default=1000, type=int)
argparser.add_argument('--nsym', default=10, type=int)
argparser.add_argument('--nrep', default=100, type=int)
argparser.add_argument('--cnstl', default="[]")
argparser.add_argument('--wirefmt', default="fc32")


args = argparser.parse_args()

if args.onlyTx > 0 and args.onlyRx > 0:
    raise ValueError("args.onlyTx と args.onlyRx が同時に指定されています")

# サーバーIPとポート番号
IPADDR = args.ipaddr;
PORT = args.port;

bUseTx = not args.onlyRx
bUseRx = not args.onlyTx

wirefmt = np.complex64
if args.wirefmt == "sc16":
    wirefmt = ezsdr.complex_int16
elif args.wirefmt == "sc8":
    wirefmt = ezsdr.complex_int8

# 送受信の遅延時間を何サンプルまで許すか
nDelayThr = args.delaythr if args.delaythr > 0 else 2**62

nTXUSRP = 1 if bUseTx else 0
nRXUSRP = 1 if bUseRx else 0

nOS = 4                 # OFDMのオーバーサンプリング率
nSC = 64               # OFDMの有効サブキャリア数
nFFT = nSC * nOS        # OFDM変調のFFTサイズ
nCP = nFFT//4           # CPのサイズ

bpsk_constellation = np.array([1+0j, -1+0j])
qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)


def getNBitsPerSym(cnstl):
    return int(np.log2(len(cnstl)))


def checkNonDuplicated(cnstl):
    cnt = 0
    for c1 in cnstl:
        for c2 in cnstl:
            if c1 == c2:
                cnt += 1

    return cnt == len(cnstl)

if args.mod == "bpsk":
    constellation = bpsk_constellation
    nBitsPerSymbol = 1
elif args.mod == "qpsk":
    constellation = qpsk_constellation
    nBitsPerSymbol = 2
elif args.mod == "user":
    constellation = eval(args.cnstl)
    assert len(constellation).bit_count() == 1 and len(constellation) > 1, "The size of given len(cnstl) is not 2**n where n > 0"
    assert checkNonDuplicated(constellation), "There are some duplicated constellation"
    constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
    nBitsPerSymbol = getNBitsPerSym(constellation)
else:
    assert False, "Invalid parameter 'mod'"


def calc_delay(tx, rx):
    tx_freq = np.fft.fft(tx)
    rx_freq = np.fft.fft(rx)
    rxy = np.abs(np.fft.ifft(np.conj(tx_freq) * rx_freq))
    return np.argmax(rxy)


# OFDM変調
def mod_ofdm(scs):
    nSYM = len(scs)//nSC
    scs = scs.reshape([nSYM, nSC])
    scs = np.hstack((np.zeros((nSYM,1)), scs[:,:nSC//2], np.zeros((nSYM, nFFT - nSC - 1)), scs[:,nSC//2:]))
    sym = np.fft.ifft(scs, norm="ortho") * np.sqrt(nFFT / nSC)  # IFFT
    sym = np.hstack((sym[:,nFFT-nCP:], sym))                    # add CP
    return sym.reshape((nFFT+nCP)*nSYM)


# OFDM復調
def demod_ofdm(sym):
    nSYM = len(sym)//(nFFT + nCP)
    sym = sym.reshape([nSYM, nFFT + nCP])
    sym = sym[:,nCP:]                           # remove CP
    scs = np.fft.fft(sym, norm="ortho") / np.sqrt(nFFT / nSC)   # FFT
    scs = np.hstack((scs[:,1:nSC//2+1], scs[:,nFFT-nSC//2:]))
    return scs.reshape(nSYM * nSC)


def demod_nearest(signal, cnstl):
    cnstl = np.asarray(cnstl)
    def find_nearest(value):
        return (np.abs(cnstl - value)).argmin()

    Nbits = getNBitsPerSym(cnstl)
    return np.unpackbits(np.array([find_nearest(v) for v in signal]).astype(np.uint8)).reshape([len(signal), 8])[:, -Nbits:].flatten()


txGain = args.amplitude

if args.sigma2 < 0:
    # EbN0から計算
    noiseSigma2 = txGain**2 / 10**(args.ebn0/10) * (nFFT / nSC) / nBitsPerSymbol
else:
    noiseSigma2 = args.sigma2 * (nFFT / nSC)



# with ezsdr.SimpleClient(IPADDR, PORT, nTXUSRP, nRXUSRP, wirefmt=wirefmt) as usrp:
with ezsdr.EzSDRClient(IPADDR, PORT) as client:
    TX0 = ezsdr.CyclicTransmitter(client, "TX0", dtype_wire=wirefmt, dtype_cpu=np.complex64)
    RX0 = ezsdr.CyclicReceiver(client, "RX0", dtype_wire=wirefmt, dtype_cpu=np.complex64)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    iTrial = 0
    sumTotBits = 0
    sumErrBits = 0
    lastDem = []
    
    def plot(data):
        global iTrial
        global sumTotBits
        global sumErrBits
        global lastDem

        ax1.clear()

        # 1回の送受信で10シンボル伝送する
        nTxSYM = args.nsym
        nRep = args.nrep

        # BPSK,QPSK変調したサブキャリア
        np.random.seed(iTrial)
        subcarriers = np.random.choice(constellation, nSC*nTxSYM)

        # OFDM変調した信号
        modulated = mod_ofdm(subcarriers) * txGain
        nModulated = len(modulated)

        # チャネル推定のためのサブキャリアと信号
        subcarriers_forEst = np.random.choice(qpsk_constellation, nSC*nTxSYM)
        modulated_forEst = mod_ofdm(subcarriers_forEst) * txGain

        txSignal = []

        np.random.seed(iTrial)
        for iTx in range(nRep):
            noiseSignal = np.random.normal(loc=0, scale=np.sqrt(noiseSigma2/2), size=nModulated) + 1j*np.random.normal(loc=0, scale=np.sqrt(noiseSigma2/2), size=nModulated)
            txSignal = np.hstack((txSignal, modulated_forEst, modulated + noiseSignal, np.zeros(nModulated)))


        nTxSignal = len(txSignal) // nRep
        # print("S = ", np.sum(np.abs(modulated)**2) / nModulated)
        # print("N = ", np.sum(np.abs(noiseSignal)**2) / (nFFT / nSC) / nModulated )

        RX0.changeAlignSize(nTxSignal)   # USRPの受信バッファのサイズを信号長に合わせる
        TX0.transmit([txSignal])           # 送信信号を設定する

        # usrp.sync()
        # # # 破棄
        # recv = usrp.receive(nTxSamples)[0]
        # recv = usrp.receive(nTxSamples)[0]
        nDelayFirst = 0
        # チャネル推定用に受信+データ復調用に受信
        recv = RX0.receive(nTxSignal*2*nRep)[0]

        # 遅延時間推定
        nDelay = calc_delay(np.hstack((txSignal, np.zeros(len(txSignal)))), recv)
        nDelay += 1
        recv = recv[nDelay:]

        for iTx in range(nRep):
            if len(recv) < nModulated*2:
                continue

            recv_forEst = recv[:nModulated]
            recv_forDec = recv[nModulated:nModulated*2]
            print(nDelay)
            # if iTx == 0:
            #     nDelayFirst = nDelay

            # if nDelay < 0 or nDelay > nDelayThr or nDelay != nDelayFirst:
            #     break

            demodulated = demod_ofdm(recv_forEst)

            # チャネル推定
            channel_resp = [
                np.mean((demodulated / subcarriers_forEst).reshape([nTxSYM, nSC]), axis=0),
            ]

            # 復調&等化
            demodulated = demod_ofdm(recv_forDec) / np.tile(channel_resp[0], nTxSYM)
            lastDem = demodulated.copy()

            # if args.mod == "bpsk":
            txbits = demod_nearest(subcarriers, constellation)
            detected = demod_nearest(demodulated, constellation)
                # txbits = bpskToBits(subcarriers)
                # detected = bpskToBits(demodulated)
            # else:
                # txbits = qpskToBits(subcarriers)
                # detected = qpskToBits(demodulated)

            print("Tx[:100]: ", txbits[:100])
            print("Rx[:100]: ", detected[:100])
            
            # print("Error:", err, ", Total:", len(txbits))
            sumTotBits += len(txbits)
            sumErrBits += np.sum(txbits != detected)
            print("Error:", sumErrBits, ", Total:", sumTotBits, ", BER:", sumErrBits / sumTotBits)

            recv = recv[:nTxSignal]

        if len(lastDem) > 0:
            ax1.scatter(np.real(lastDem), np.imag(lastDem))
            ax1.set_xlim([-2, 2])
            ax1.set_ylim([-2, 2])

        iTrial += 1
        # ax2.clear()
        # ax3.clear()
        
    
    ani = animation.FuncAnimation(fig, plot, interval=50, blit=False)
    plt.show()
        

        # # 受信結果表示（先頭1シンボルだけ）
        # for i in range(nSC):
        #     print("{},{},{}".format(i, np.real(demodulated[i]), np.imag(demodulated[i])))

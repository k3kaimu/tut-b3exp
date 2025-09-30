import socket
import numpy as np
import scipy
from collections import namedtuple
import sigdatafmt
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


complex_int8 = np.dtype([('real', np.int8), ('imag', np.int8)])
complex_int16 = np.dtype([('real', np.int16), ('imag', np.int16)])


class EzSDRClient:
    def __init__(self, ipaddr, port):
        self.sock = None
        self.ipaddr = ipaddr
        self.port = port
        self.enterCount = 0
        self.ifaceVersion = "3.0.11"

    def __enter__(self):
        self.enterCount += 1
        if self.enterCount > 1:
            return self

        try:
            if self.sock is None:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            if self.ipaddr is not None:
                self.sock.__enter__()
                self.sock.connect((self.ipaddr, self.port))
                sigdatafmt.writeStringToSock(self.sock, self.ifaceVersion)
        except Exception as e:
            self.enterCount = 0
            raise e

        return self

    def __exit__(self, *args):
        if self.enterCount == 0:
            return
        
        self.enterCount -= 1
        if self.enterCount == 0:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
            self.sock.__exit__(args)
            self.sock = None

    # def connect(self):
    #     if self.ipaddr is not None:
    #         self.sock.connect((self.ipaddr, self.port))

    def sendMsg(self, target, msg):
        with self:
            sigdatafmt.writeStringToSock(self.sock, target)
            sigdatafmt.writeIntToSock(self.sock, len(msg), np.uint64)
            self.sock.sendall(msg)

    def resumeController(self, target):
        with self:
            msg = sigdatafmt.valueToBytes(0b00000001, np.uint8)
            msg += sigdatafmt.valueToBytes(len(target), np.uint64)
            msg += target.encode(encoding="utf-8")
            self.sendMsg("@server", msg)

    def stopController(self, target):
        with self:
            msg = sigdatafmt.valueToBytes(0b00000010, np.uint8)
            msg += sigdatafmt.valueToBytes(len(target), np.uint64)
            msg += target.encode(encoding="utf-8")
            self.sendMsg("@server", msg)

    def resumeAllController(self):
        with self:
            msg = sigdatafmt.valueToBytes(0b00000011, np.uint8)
            self.sendMsg("@server", msg)

    def stopAllController(self):
        with self:
            msg = sigdatafmt.valueToBytes(0b00000100, np.uint8)
            self.sendMsg("@server", msg)

    def setParamToDevice(self, target, key, value):
        with self:
            msg = sigdatafmt.valueToBytes(0b00000000, np.uint8)
            msg += sigdatafmt.valueToBytes(len(key), np.uint64)
            msg += key.encode(encoding="utf-8")
            msg += sigdatafmt.valueToBytes(len(value), np.uint64)
            msg += value.encode(encoding="utf-8")
            return self.sendMsg(target, msg)

    def setParamToAllDevice(self, key, value):
        with self:
            self.setParamToDevice("@alldevs", key, value)

    def getParamFromDevice(self, target, key):
        with self:
            msg = sigdatafmt.valueToBytes(0b00000001, np.uint8)
            msg += sigdatafmt.valueToBytes(len(key), np.uint64)
            msg += key.encode(encoding="utf-8")
            self.sendMsg(target, msg)

            # Read the response
            ret = sigdatafmt.readInt64FromSock(self.sock)
            if ret == 0:
                return None
            else:
                return sigdatafmt.readStringFromSock(self.sock, ret)


def onTime(t):
    nsec = int(t * 1000000000)
    tag = 0x16C002AF
    msg = sigdatafmt.valueToBytes(nsec, np.uint64)
    return sigdatafmt.valueToBytes(len(msg), np.uint64) + sigdatafmt.valueToBytes(tag, np.uint32) + msg



def typeConvert(signals, dtype_in, dtype_out):
    # print(dtype_in, "->", dtype_out)
    # print(np.dtype(dtype_in), "->", np.dtype(dtype_out))
    if dtype_in == dtype_out:
        return signals

    if dtype_in == np.complex64 and dtype_out == np.complex128:
        return signals.astype(np.complex128)
    elif dtype_in == np.complex128 and dtype_out == np.complex64:
        return signals.astype(np.complex64)
    elif dtype_in == complex_int8 and dtype_out == complex_int16:
        return signals.astype(complex_int16)
    elif dtype_in == complex_int16 and dtype_out == complex_int8:
        return signals.astype(complex_int8)
    elif dtype_in == complex_int8:
        signals = signals['real'] / 127.0 + 1j * signals['imag'] / 127.0
        signals = signals.astype(np.complex128)
        return typeConvert(signals, np.complex128, dtype_out)
    elif dtype_in == complex_int16:
        signals = signals['real'] / 32767.0 + 1j * signals['imag'] / 32767.0
        signals = signals.astype(np.complex128)
        return typeConvert(signals, np.complex128, dtype_out)
    elif dtype_out == complex_int8:
        dst = np.zeros(signals.shape, dtype=complex_int8)
        dst['real'] = (np.real(signals) * 127).astype(np.int8)
        dst['imag'] = (np.imag(signals) * 127).astype(np.int8)
        return dst
    elif dtype_out == complex_int16:
        dst = np.zeros(signals.shape, dtype=complex_int16)
        dst['real'] = (np.real(signals) * 32767).astype(np.int16)
        dst['imag'] = (np.imag(signals) * 32767).astype(np.int16)
        return dst
    else:
        raise ValueError(f"Unsupported type conversion from {dtype_in} to {dtype_out}.")
    



class CyclicTransmitter:
    def __init__(self, client, target, dtype_wire=np.complex64, dtype_cpu=np.complex64):
        self.client = client
        self.target = target
        self.dtype_wire = dtype_wire
        self.dtype_cpu = dtype_cpu

    def sendMsgWQ(self, msg, qs):
        with self.client:
            self.client.sendMsg(self.target, sigdatafmt.valueToBytes(len(qs), np.uint64) + qs + msg)

    def setTransmitSignal(self, signals, qs=b''):
        signals = np.array(signals, dtype=self.dtype_cpu)
        signals = typeConvert(signals, self.dtype_cpu, self.dtype_wire)
        # print(self.dtype_wire)

        with self.client:
            msg = sigdatafmt.valueToBytes(0b00010000, np.uint8)
            for i in range(len(signals)):
                msg += sigdatafmt.valueToBytes(len(signals[i]), np.uint64)
                msg += sigdatafmt.arrayToBytes(signals[i], self.dtype_wire)
            
            self.sendMsgWQ(msg, qs)
    
    def startTransmitLoop(self, qs=b''):
        with self.client:
            msg = sigdatafmt.valueToBytes(0b00010001, np.uint8)
            self.sendMsgWQ(msg, qs)

    def stopTransmitLoop(self, qs=b''):
        with self.client:
            msg = sigdatafmt.valueToBytes(0b00010010, np.uint8)
            self.sendMsgWQ(msg, qs)
    
    def transmit(self, signals, qs1=b'', qs2=b''):
        with self.client:
            self.setTransmitSignal(signals, qs1)
            self.startTransmitLoop(qs2)


class CyclicReceiver:
    def __init__(self, client, target, dtype_wire=np.complex64, dtype_cpu=np.complex64):
        self.client = client
        self.target = target
        self.dtype_wire = dtype_wire
        self.dtype_cpu = dtype_cpu

    def sendMsgWQ(self, msg, qs):
        with self.client:
            self.client.sendMsg(self.target, sigdatafmt.valueToBytes(len(qs), np.uint64) + qs + msg)
    
    def startReceiveLoop(self, qs=b''):
        with self.client:
            msg = sigdatafmt.valueToBytes(0b00010001, np.uint8)
            self.sendMsgWQ(msg, qs)

    def stopReceiveLoop(self, qs=b''):
        with self.client:
            msg = sigdatafmt.valueToBytes(0b00010010, np.uint8)
            self.sendMsgWQ(msg, qs)

    def receive(self, size, qs=b''):
        with self.client:
            self.receiveRequestOnly(size, qs)
            return self.receiveResponseOnly()

    def receiveRequestOnly(self, size, qs=b''):
        with self.client:
            msg = sigdatafmt.valueToBytes(0b00010100, np.uint8)
            msg += sigdatafmt.valueToBytes(size, np.uint64)
            self.sendMsgWQ(msg, qs)
            ack = sigdatafmt.readIntFromSock(self.client.sock, np.uint8)
            if ack != 83:  # 'S' in ASCII
                raise RuntimeError(f"Unexpected response: {ack}. Expected 'S' for .receiveRequestOnly().")

    def receiveResponseOnly(self):
        with self.client:
            msg = sigdatafmt.valueToBytes(0b00010101, np.uint8)
            msg += sigdatafmt.valueToBytes(1, np.uint64)  # 最大でも1つだけ結果を取得
            msg += sigdatafmt.valueToBytes(1, np.uint64)  # 最小は1にする（必ず1つは結果を得る）
            self.sendMsgWQ(msg, b'')
            nresp = sigdatafmt.readIntFromSock(self.client.sock, np.uint64) # 結果の数を取得
            assert nresp == 1, f"Expected 1 response, got {nresp} at .receiveResponseOnly()"

            nbuf = sigdatafmt.readInt64FromSock(self.client.sock)
            ret = []
            for i in range(nbuf):
                nsamples = sigdatafmt.readInt64FromSock(self.client.sock)
                ret.append(sigdatafmt.readSignalFromSock(self.client.sock, nsamples, dtype=self.dtype_wire))
            
            ret = np.array(ret, dtype=self.dtype_wire)
            return typeConvert(ret, self.dtype_wire, self.dtype_cpu)

    def changeAlignSize(self, value):
        with self.client:
            msg = sigdatafmt.valueToBytes(0b0010011, np.uint8)
            msg += sigdatafmt.valueToBytes(value, np.uint64)
            self.sendMsgWQ(msg, b'')


def syncUSRPLoopTXRX(client, devs, txlist, rxlist, loopStartTime=0.2, sleepTime=1):
    for e in txlist:
        e.stopTransmitLoop()
    
    for e in rxlist:
        e.stopReceiveLoop()

    time.sleep(sleepTime)
    for e in devs:
        client.setParamToDevice(e, "set_time_unknown_pps_to_zero", "[]")

    time.sleep(sleepTime)
    client.getParamFromDevice(devs[0], "wait_set_time_unknown_pps")

    for e in txlist:
        e.startTransmitLoop(onTime(loopStartTime))

    for e in rxlist:
        e.startReceiveLoop(onTime(loopStartTime))


class SimpleClient:
    def __init__(self, ipaddr, port, nTXUSRPs, nRXUSRPs):
        if type(nTXUSRPs) is int:
            nTXUSRPs = [nTXUSRPs]
        else:
            nTXUSRPs = nTXUSRPs

        if type(nRXUSRPs) is int:
            nRXUSRPs = [nRXUSRPs]
        else:
            nRXUSRPs = nRXUSRPs

        self.client = EzSDRClient(ipaddr, port)
        self.txs = []
        self.rxs = []

        for i in range(len(nTXUSRPs)):
            self.txs.append(CyclicTransmitter(self.client, f"TX{i}"))

        for i in range(len(nRXUSRPs)):
            self.rxs.append(CyclicReceiver(self.client, f"RX{i}"))

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, *args):
        self.client.__exit__(*args)

    # def connect(self):
    #     self.client.connet()

    def transmit(self, signals, **kwargs):
        with self.client:
            tidx = kwargs.get("tidx", 0)
            self.txs[tidx].transmit(signals)

    def receive(self, nsamples, **kwargs):
        with self.client:
            ridx = kwargs.get("ridx", 0)

            if ('onlyResponse' not in kwargs) or (not kwargs['onlyResponse']):
                self.rxs[ridx].receiveRequestOnly(nsamples)

            if ('onlyRequest' not in kwargs) or (not kwargs['onlyRequest']):
                return self.rxs[ridx].receiveResponseOnly()
            else:
                return None

    def changeRxAlignSize(self, newAlign, **kwargs):
        with self.client:
            ridx = kwargs.get("ridx", 0)
            self.rxs[ridx].changeAlignSize(newAlign)

    def sync(self):
        with self.client:
            for e in self.txs:
                e.stopTransmitLoop()
            
            for e in self.rxs:
                e.stopReceiveLoop()

            time.sleep(1)
            self.client.setParamToAllDevice("set_time_unknown_pps_to_zero", "[]")
            time.sleep(1)
            self.client.getParamFromDevice("USRP0", "wait_set_time_unknown_pps")

            for e in self.txs:
                e.startTransmitLoop(onTime(0.2))

            for e in self.rxs:
                e.startReceiveLoop(onTime(0.2))

        # time.sleep(1)


    # def rxPowerThr(self, p, m):
    #     ridx = kwargs.get("ridx", 0)
    #     self.sock.sendall(b'p')
    #     sigdatafmt.writeInt32ToSock(self.sock, ridx)
    #     sigdatafmt.writeFloat32ToSock(self.sock, p)
    #     sigdatafmt.writeFloat32ToSock(self.sock, m)

    # def clearCmdQueue(self):
    #     self.sock.sendall(b'q')

    # def stopTxStreaming(self, idx):
    #     self.sock.sendall(b'\x81')
    #     sigdatafmt.writeInt32ToSock(self.sock, idx)
    
    # def startTxStreaming(self, idx):
    #     self.sock.sendall(b'\x82')
    #     sigdatafmt.writeInt32ToSock(self.sock, idx)

    # def stopRxStreaming(self, idx):
    #     self.sock.sendall(b'\x83')
    #     sigdatafmt.writeInt32ToSock(self.sock, idx)
    
    # def startRxStreaming(self, idx):
    #     self.sock.sendall(b'\x84')
    #     sigdatafmt.writeInt32ToSock(self.sock, idx)


class SimpleMockClient:
    def __init__(self, nTXUSRP, nRXUSRP, impRespMatrix=np.array([[[1]]]), SIGMA2=0, delay=0):
        self.nTXUSRP = nTXUSRP
        self.nRXUSRP = nRXUSRP
        self.sampleIndex = 0
        self.alignSize = 4096
        self.txsignals = np.zeros((nTXUSRP, 4096), dtype=np.complex64)
        self.impRespMatrix = impRespMatrix
        self.rxsignals = np.zeros((nRXUSRP, 4096), dtype=np.complex64)
        self.SIGMA2 = SIGMA2
        self.delay = delay
    
    def __enter__(self):
        self.sampleIndex = 0
        self.alignSize = 4096
        return self

    def __exit__(self, *args):
        pass
    
    def connect(self):
        pass

    def makeRxSignals(self):
        self.rxsignals = np.zeros((self.nRXUSRP, len(self.txsignals[0])), dtype=np.complex64)
        N = len(self.txsignals[0])
        for i in range(self.nTXUSRP):
            for j in range(self.nRXUSRP):
                txFreq = np.fft.fft(self.txsignals[i])
                irFreq = np.fft.fft(np.hstack((np.zeros(self.delay), self.impRespMatrix[i, j], np.zeros(N)))[:N])
                rxFreq = txFreq * irFreq
                self.rxsignals[j,:] = self.rxsignals[j,:] + np.fft.ifft(rxFreq)
    
    def transmit(self, signals):
        self.txsignals = signals
        self.makeRxSignals()

    def receive(self, nsamples, **kwargs):
        if ('onlyRequest' not in kwargs) or (not kwargs['onlyRequest']):
            return self.receiveImpl(nsamples)
        else:
            return None
    
    def receiveImpl(self, nsamples):
        dst = np.zeros((self.nRXUSRP, nsamples), dtype=np.complex128)

        # 次のアライメント（受信バッファの先頭）を計算する
        self.sampleIndex += self.alignSize - (self.sampleIndex % self.alignSize)

        N = len(self.rxsignals[0])
        D = self.sampleIndex % N
        for i in range(self.nRXUSRP):
            dst[i,:] = np.tile(self.rxsignals[i], (D + nsamples)//N + 1)[D : D+nsamples]
            dst[i,:] += np.random.normal(0, np.sqrt(self.SIGMA2/2), size=nsamples) + np.random.normal(0, np.sqrt(self.SIGMA2/2), size=nsamples)*1j

        self.sampleIndex += nsamples
        return dst

    def shutdown(self):
        pass
    
    def changeRxAlignSize(self, newAlign):
        self.alignSize = newAlign

    def skipRx(self, delay):
        D = delay % N
        for i in range(self.nTXUSRP):
            self.signals[i] = np.roll(self.signals[i], -D)

    def sync(self):
        self.sampleIndex = 0



class SimpleClientWithTimeSeriesPlot(SimpleClient):
    def __init__(self, ipaddr, port, nTXUSRPs, nRXUSRPs):
        super().__init__(ipaddr, port, nTXUSRPs, nRXUSRPs)
        self.txprocesslist = []
        self.txsenderlist = []
        for i, n in enumerate(self.nTXUSRPs):
            prx, ptx = mp.Pipe(duplex=False)
            pplot = mp.Process(target=plotTimeSeries, args=[prx, f"TX{i}"])
            self.txprocesslist.append(pplot)
            self.txsenderlist.append(ptx)

        self.rxprocesslist = []
        self.rxsenderlist = []
        for i, n in enumerate(self.nRXUSRPs):
            prx, ptx = mp.Pipe(duplex=False)
            pplot = mp.Process(target=plotTimeSeries, args=[prx, f"RX{i}"])
            self.rxprocesslist.append(pplot)
            self.rxsenderlist.append(ptx)


    def __enter__(self):
        super().__enter__()
        for p in [*self.txprocesslist, *self.rxprocesslist]:
            p.start()

        return self
    

    def __exit__(self, *args):
        super().__exit__()
        for p in [*self.txsenderlist, *self.rxsenderlist]:
            p.send([])


    def transmit(self, signals, **kwargs):
        super().transmit(signals, **kwargs)
        tidx = kwargs.get("tidx", 0)
        self.txsenderlist[tidx].send(signals)


    def receive(self, nsamples, **kwargs):
        ridx = kwargs.get("ridx", 0)
        ret = super().receive(nsamples, **kwargs)
        if ret is not None:
            self.rxsenderlist[ridx].send(ret)
        
        return ret
    
    def receiveNBResponse(self, **kwargs):
        ridx = kwargs.get("ridx", 0)
        ret = super().receiveNBResponse(**kwargs)
        if ret[0]:
            self.rxsenderlist[ridx].send(ret[1])
        
        return ret

    def receiveNBResponseToFn(self, fn, bufferSize=0, **kwargs):
        ridx = kwargs.get("ridx", 0)
        rxdata = [np.array([]) for _ in range(self.nRXUSRPs[ridx])]

        def proxyfunc(i, j, data):
            rxdata = np.hstack(rxdata[i], data)
            fn(i, j, data)
        
        ret = super().receiveNBResponseToFn(proxyfunc, bufferSize, **kwargs)
        if ret:
            self.rxsenderlist[ridx].send(rxdata)

        return ret




def plotTimeSeries(p, name):
    plt.ion()
    plt.figure(num=name)
    while True:
        if p.poll():
            data = p.recv()

            # データ長が0なら終了する
            if len(data) == 0:
                break

            # データをプロットする
            plt.clf()
            for i in range(len(data)):
                plt.plot(np.arange(len(data[i])), np.real(data[i]), label=f"Re,{i}")
                plt.plot(np.arange(len(data[i])), np.imag(data[i]), label=f"Im,{i}")
            plt.legend()
            plt.draw()

        plt.pause(0.05)
import socket
import numpy as np
import struct
import zlib
import soundfile
import io

# ソケットから信号を読む
def readSignalFromSock(sock, size = None, dtype=np.complex64):

    if size is None:
        # 受信サンプルのサイズを取得
        size = int.from_bytes(sock.recv(4), 'little');

    # 受信サンプルを取得
    data = bytearray();
    while len(data) < size * np.dtype(dtype).itemsize:
        # print(len(data), "/ (", size, "*", np.dtype(dtype).itemsize, ")")
        data += sock.recv(min(4096, size*np.dtype(dtype).itemsize - len(data)));

    # バイト列を複素数I+jQの配列へ変換
    data = np.frombuffer(data, dtype=dtype);
    return data;


def valueToBytes(val, dtype):
    return np.array([val], dtype=dtype).tobytes()

def arrayToBytes(arr, dtype):
    # print(arr.dtype, "->", dtype)
    return np.array(arr, dtype=dtype).tobytes()

# ソケットからInt32の値を読む
def readInt32FromSock(sock):
    return int.from_bytes(sock.recv(4), 'little');

# ソケットからInt64の値を読む
def readInt64FromSock(sock):
    return int.from_bytes(sock.recv(8), 'little');

# ソケットにInt32の値を書き込む
def writeInt32ToSock(sock, value):
    writeIntToSock(sock, value, np.uint32)

def readIntFromSock(sock, dtype):
    # ソケットからデータを受信
    data = bytearray()
    while len(data) < np.dtype(dtype).itemsize:
        data += sock.recv(min(4096, np.dtype(dtype).itemsize - len(data)))

    # データを指定された型に変換して返す
    return np.frombuffer(data, dtype=dtype)[0]

# ソケットにIntの値を書き込む
def writeIntToSock(sock, value, dtype):
    data = np.array([value], dtype=dtype).tobytes()
    sock.sendall(data)

# ソケットから文字列を読む
def readStringFromSock(sock, size):
    data = bytearray()
    while len(data) < size:
        data += sock.recv(min(4096, size - len(data)))
    return data.decode(encoding="utf-8")

# ソケットに信号を書き込む
def writeSignalToSock(sock, signal, withHeader = True):
    size = len(signal)

    # データをバイト列に変換する
    if size != 0:
        data = np.concatenate(np.vstack((np.real(signal), np.imag(signal))).T);
    else:
        data = np.array([])

    data = data.astype(np.float32).tobytes()

    # サンプル数をヘッダーとして付与
    header = np.array([size], dtype=np.uint32).tobytes()

    if withHeader:
        response = header + data
    else:
        response = data

    # クライアント側に返答
    txbytes = 0
    while txbytes != len(response):
        txbytes += sock.send(response[txbytes:])


def writeFloat32ToSock(sock, value):
    data = np.array([value], dtype=np.float32).tobytes()
    sock.sendall(data)

def writeStringToSock(sock, str):
    bs = str.encode(encoding="utf-8")
    writeIntToSock(sock, len(bs), np.uint16)
    sock.sendall(bs)

def getMinStep(rsignal):
    rsignal = np.abs(rsignal)
    return np.min(rsignal[rsignal != 0])

def getMaxError(rsignal, scale):
    return np.max(np.abs(rsignal - (np.rint(rsignal * scale).astype(np.int16).astype(np.float32) / scale)))

def compress(signal, scale=-1):

    if scale < 0:
        scale = 32767
        ps = np.real(signal[:min(10000000, len(signal))])
        minStep = getMinStep(ps)
        if minStep < (1/32766) and minStep > (1/32768):
            scale = 1/minStep

        e1 = getMaxError(ps, 32767)
        e2 = getMaxError(ps, scale)
        if e1 < e2:
            scale = 32767

    data = np.hstack((np.real(signal), np.imag(signal))).astype(np.float32)
    data *= scale
    data = np.rint(data).astype(np.int16)
    data += 2**7
    data = data.tobytes()
    header = np.array([scale], dtype=np.float32).tobytes()
    data = header + data[::2] + data[1::2]
    return zlib.compress(data)


def decompress(data, scale=-1):
    data = zlib.decompress(data)
    if scale < 0:
        scale = np.frombuffer(data[:4], np.float32)[0]

    data = data[4:]
    bdata = bytearray(len(data))
    bdata[0::2] = data[:len(data)//2]
    bdata[1::2] = data[len(data)//2:]
    data = np.frombuffer(bdata, dtype=np.int16)
    data = data - 2**7
    data = data.astype(np.float32) / scale
    return data[:len(data)//2] + data[len(data)//2:] * 1j


def compress_flac(signal, scale=-1):
    if scale < 0:
        scale = 32767
        ps = np.real(signal[:min(10000000, len(signal))])
        minStep = getMinStep(ps)
        if minStep < (1/32766) and minStep > (1/32768):
            scale = 1/minStep

        e1 = getMaxError(ps, 32767)
        e2 = getMaxError(ps, scale)
        if e1 < e2:
            scale = 32767

    data = np.array([np.real(signal), np.imag(signal)]).astype(np.float32)
    data *= scale
    data = np.rint(data).astype(np.int16)

    flac_file = io.BytesIO()
    flac_file.name = "file.flac"
    fmt = "FLAC"
    stype = "PCM_16"
    soundfile.write(flac_file, data.T, 44100, format=fmt, subtype=stype)
    flac_file.seek(0)

    header = np.array([scale], dtype=np.float32).tobytes()
    return header + flac_file.read()

def decompress_flac(data, scale=-1):
    if scale < 0:
        scale = np.frombuffer(data[:4], np.float32)[0]

    data = data[4:]

    flac_file = io.BytesIO()
    flac_file.name = "file.flac"
    flac_file.write(data)
    flac_file.seek(0)
    dst = soundfile.read(flac_file, dtype='int16')[0].T
    dst = dst.astype(np.float32)
    dst = dst[0] + dst[1]*1j
    return dst / scale

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def filter_signal(source, lower_freq, upper_freq, signal_duration=2.0E-2) -> (np.ndarray, np.ndarray):
    freqs = np.fft.rfft(source)
    upper_index = upper_freq * signal_duration
    lower_index = lower_freq * signal_duration
    f_freqs = freqs.copy()
    for i in range(int(lower_index), int(upper_index) + 1):
        f_freqs[i] = 0
    result = np.fft.irfft(f_freqs)
    return f_freqs, result


def filter_signal_low_freq(source, upper_freq, signal_duration=2.0E-2) -> (np.ndarray, np.ndarray):
    freqs = np.fft.rfft(source)
    upper_index = int(upper_freq * signal_duration)
    f_freqs = np.concatenate((np.zeros((upper_index,)), freqs[upper_index:]))
    result = np.fft.irfft(f_freqs)
    return result


def plot_all(y, size=(13, 4), dpi=211.82, start=0, end=-1):
    n = len(y)
    fig, axs = plt.subplots(n, 1, sharex='all')
    fig.subplots_adjust(hspace=0)
    fig.set_dpi(dpi)
    fig.set_size_inches(size[0], size[1] * n)
    if end == -1:
        end = len(y[0])
    for i in range(n):
        axs[i].plot(y[i][start:end])
    plt.ion()
    plt.show(block=True)


def avg_fft(source):
    avg_freqs = None

    for col in source:
        freqs = np.fft.rfft(source[col])
        if avg_freqs is None:
            avg_freqs = freqs
        else:
            avg_freqs = np.add(avg_freqs, freqs)
    avg_freqs /= len(source)
    return avg_freqs


def analyse():
    train_meta = pd.read_csv('../input/metadata_train.csv')

    pos_meta = train_meta.loc[train_meta['target'] == 1]
    neg_meta = train_meta.loc[train_meta['target'] == 0]

    print("Positives: ", pos_meta.shape)
    print("Negatives: ", neg_meta.shape)

    pos_cols = pos_meta['signal_id'].tolist()
    pos_cols_str = map(lambda x: str(x), pos_cols)
    pos_train = pq.read_pandas('../input/train.parquet', columns=pos_cols_str).to_pandas()
    avg_pos_freqs = avg_fft(pos_train)
    print("Finished positives.")

    neg_cols = neg_meta['signal_id'].tolist()
    neg_cols_str = map(lambda x: str(x), neg_cols)
    neg_train = pq.read_pandas('../input/train.parquet', columns=neg_cols_str).to_pandas()
    avg_neg_freqs = avg_fft(neg_train)
    print("Finished negatives.")

    diff_freqs = np.subtract(avg_pos_freqs, avg_neg_freqs)

    plot_all([avg_pos_freqs, avg_neg_freqs, diff_freqs])


def filtering():
    cols = [str(i) for i in range(0, 6, 3)]
    train = pq.read_pandas('../input/train.parquet', columns=cols).to_pandas()

    all_arrays = []

    for i in range(len(cols)):
        all_arrays.append(train[cols[i]])
        fq, ft = filter_signal(train[cols[i]], 0.0, 1000.1)
        all_arrays.append(ft)
        all_arrays.append(fq)

    plot_all(all_arrays)


def spectrogram_it(x):
    dt = 2.0E-2 / 800000.0
    t = np.arange(0.0, 2.0E-2, dt, np.float32)

    nfft = 1024  # the length of the windowing segments
    fs = int(1.0 / dt)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(t, x)
    pxx, freqs, bins, im = ax2.specgram(x, NFFT=nfft, Fs=fs, noverlap=900)
    # The `specgram` method returns 4 objects. They are:
    # - pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot
    plt.show()


def spectrogram(col=0):
    cols = [str(i) for i in range(col, col + 1)]
    train = pq.read_pandas('../input/train.parquet', columns=cols).to_pandas()

    spectrogram_it(train[str(col)])


def wavelet(col=0):
    from scipy import signal
    cols = [str(i) for i in range(col, col + 1)]
    train = pq.read_pandas('../input/train.parquet', columns=cols).to_pandas()

    dt = 2.0E-2 / 800000.0
    t = np.arange(0.0, 2.0E-2, dt, np.float32)

    # t = np.linspace(-1, 1, 200, endpoint=False)
    fq, sig = filter_signal(train[str(col)], 0.0, 1000.1)
    widths = np.arange(1, 20)
    cwtmatr = signal.cwt(sig, signal.morlet, widths)
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(t, sig)
    # ax2.imshow(cwtmatr, extent=[0, 2.0E-2, 1, 20], cmap='Greys', aspect='auto',
    #           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    ax2.imshow(cwtmatr, extent=[0, 2.0E-2, 1, 20], cmap='Greys', aspect='auto')

    plt.show()


def wavelet_cardio(col=0):
    import pylab as py
    # from pylab import *
    import pywt
    from scipy import signal

    def lepow2(x):
        return 2 ** py.floor(py.log2(x))

    # Load the signal, take the first channel, limit length to a power of 2 for simplicity.
    # sig  = np.cos(2  np.pi  4  t) + np.cos(2  np.pi  3  t])

    cols = [str(i) for i in range(col, col + 1)]
    train = pq.read_pandas('../input/train.parquet', columns=cols).to_pandas()

    # sig = pywt.data.ecg()
    sig = train[str(col)][:1024]
    t = np.linspace(0, 2.0E-2, sig.shape[0], endpoint=False)

    length = int(lepow2(len(sig)))
    sig = sig[0: length]
    w = pywt.Wavelet('db4')
    # tree = pywt.wavedec(sig, w)
    (cA, cD) = pywt.dwt(sig, w)

    f_min = 1
    f_max = 50
    widths = np.arange(f_min, f_max)
    coefs, freqs = pywt.cwt(sig, widths, 'mexh')

    # Plotting.
    plt.plot(sig)
    plt.legend(['Original signal'])
    plt.show()

    # Plot the approximation coefficients
    plt.figure(2)
    plt.plot(np.array(range(1, len(cA) + 1)), cA, color='red')
    plt.title('Approximation values cA')
    plt.xlabel('Coefficient ID value')
    plt.ylabel('Approximation coefficient value')
    plt.grid(True)

    # Plot the detail coefficients
    plt.figure(3)
    plt.plot(np.array(range(1, len(cD) + 1)), cD, color='green')
    plt.title('Detail values cD')
    plt.xlabel('Coefficient ID value')
    plt.ylabel('Detail coefficient value')
    plt.grid(True)

    # CWT
    fig = plt.figure(4)
    ax = fig.add_subplot(111)

    print(coefs.shape)
    t1 = np.linspace(0, 2.0E-2, coefs.shape[1], endpoint=False)
    x, y = np.meshgrid(t1, np.logspace(np.log10(f_min), np.log10(f_max), coefs.shape[0]))

    ax.pcolormesh(x, y, np.abs(coefs), cmap="plasma")

    ax.set_xlabel("Time, s")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_yscale('log')
    ax.set_ylim(f_min, f_max)
    plt.show()
    # plt.imshow(cwtmatr, extent=[-1, 1, 1, 15], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())


def scipy_spectrogram(x):
    sampling_frequency = 800000*100/2  # number of samples divided by signal duration 800000 / 20ms
    lower_frequency = 1.5E7  # 15Mhz
    f, t, sxx = signal.spectrogram(x, sampling_frequency)
    start_index = int(lower_frequency*f.shape[0]*2/sampling_frequency)+1  # cut off lower 15Mhz. x2 as one-sided
    # start_index = 0
    plt.pcolormesh(t, f[start_index:], np.exp(sxx[start_index:][:]), cmap=plt.get_cmap('plasma'))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def tw_transformation_features(s, sdt=2E-2, wdt=2E-6, overlap=0.25):
    """
    :param s: signal
    :param sdt: signal duration time
    :param wdt: window width in seconds 2000ns (2micro sec) time-frame
    :param overlap: windows overlap (0.25 is 25%)
    http://www.innoconsulting.com.ar/innorep/html/pdf/Bangalore_08.pdf
    """
    n = len(s)
    # time between measurements
    dt = sdt / n
    win = int(wdt/dt)
    # overlap 25%
    step = int(win * (1 - overlap))
    ts = np.zeros(int((n-win)/step), np.float32)
    ws = np.zeros(int((n-win)/step), np.float32)
    i = 0
    for start in range(0, n - win, step):
        sw = s[start:start + win].astype(np.float32)
        tau = np.arange(start, start + win, 1, dtype=np.float32)
        tau = tau * dt
        ns = sw / np.sqrt(np.trapz(np.abs(np.square(sw)), tau))
        # L2 normalized signal
        # ns = sw / np.linalg.norm(x=sw, ord=2)
        t0 = np.trapz(tau * ns, tau)
        # t0 = np.sum(tau * np.abs(ns))/win
        t = np.sqrt(np.trapz(np.square(tau - t0)*np.square(sw), tau))
        # t = np.sum(np.square(tau - t0)*np.square(ns)/win)
        sf = np.fft.fft(ns)
        f = np.fft.fftfreq(win, dt)
        mid = int(len(f)/2)
        f = np.concatenate([f[mid:], f[:mid]])
        sf = np.concatenate([sf[mid:], sf[:mid]])
        w = np.sqrt(np.trapz(np.square(f)*np.square(sw)*np.abs(np.square(sf)), f))
        ts[i] = t
        ws[i] = w
        i += 1
    return ts, ws


def tw_transformation_features_stat(s, sdt=2E-2, wdt=2E-6, overlap=0.25):
    """
    :param s: signal
    :param sdt: signal duration time
    :param wdt: window width in seconds (by default 2000ns (2micro sec) time-frame)
    :param overlap: windows overlap (0.25 is 25%)
    http://www.innoconsulting.com.ar/innorep/html/pdf/Bangalore_08.pdf
    """
    n = len(s)
    # time between measurements
    dt = sdt / n
    win = int(wdt/dt)
    # overlap 25%
    step = int(win * (1 - overlap))
    ts = np.zeros(int((n-win)/step), np.float32)
    ws = np.zeros(int((n-win)/step), np.float32)
    i = 0
    for start in range(0, n - win, step):
        sw = s[start:start + win].astype(np.float32)
        tau = np.arange(start, start + win, 1, dtype=np.float32)
        tau = tau * dt
        # ns = sw / np.sqrt(np.trapz(np.abs(np.square(sw)), tau))
        # L2 normalized signal
        ns = sw / np.linalg.norm(x=sw, ord=2)
        # t0 = np.trapz(tau * ns, tau)
        # t0 = np.sum(tau * ns)/np.sum(ns)
        # t0 = np.sum(tau * np.abs(ns))/win
        # t = np.sqrt(np.trapz(np.square(tau - t0)*np.square(sw), tau))
        sm = np.mean(ns)
        t = np.sqrt(np.sum(np.square(ns - sm))/win)
        sf = np.fft.rfft(ns)
        # f = np.fft.fftfreq(win, dt)
        # mid = int(len(f)/2)
        # f = np.concatenate([f[mid:], f[:mid]])
        # sf = np.concatenate([sf[mid:], sf[:mid]])
        sfm = np.mean(sf)
        w = np.sqrt(np.sum(np.square(sf - sfm))/win)
        # w = np.sqrt(np.trapz(np.square(f)*np.square(sw)*np.abs(np.square(sf)), f))
        ts[i] = t
        ws[i] = np.abs(w)
        i += 1
    return ts, ws


def plot_dots(x, y):
    plt.scatter(x=x, y=y, s=1)
    plt.show()


def main():
    # wavelet(0)
    # filtering()
    # spectrogram(0)
    # spectrogram(3)

    start_col = 2
    n_cols = 10
    cols = [str(i) for i in range(start_col, start_col + n_cols)]
    train = pq.read_pandas('../../input/train.parquet', columns=cols).to_pandas()

    import sounddevice as sd

    plots = []
    for c in cols:
        #  filtered_signal = filter_signal_low_freq(train[c], 1.5E7)
        filtered_signal = train[c]
        # x = np.arange(0, 2E-2, 2E-2/800000)
        # filtered_signal = np.sin(x)
        # scipy_spectrogram(filtered_signal)
        # fs = 8000
        # sd.play(filtered_signal, fs)
        # sd.wait()
        # plots.append(filtered_signal)
        ts, ws = tw_transformation_features(filtered_signal, 2E-2, 1E-5, 0.25)
        plot_dots(ws, ts)


    # plots = [train[c], filtered_signal]
    # plot_all(plots)


    # spectrogram_it(train[str(col)])
    # spectrogram_it(filtered_signal)
    # scipy_spectrogram(train[str(col)])


    # plots = [train[str(col)], filtered_signal]
    # plot_all(plots)





if __name__ == "__main__":
    main()

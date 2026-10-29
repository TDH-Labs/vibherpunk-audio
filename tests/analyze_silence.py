"""Measure leading/trailing silence in generated WAVs (10 ms windows)."""
import sys
import numpy as np
import scipy.io.wavfile

for path in sys.argv[1:]:
    sr, data = scipy.io.wavfile.read(path)
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    else:
        x = data.astype(np.float64)
    win = sr // 100  # 10 ms
    n = len(x) // win
    rms = np.array([np.sqrt(np.mean(x[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    peak = rms.max() if n else 0.0
    for label, thr in (("abs0.01", 0.01), ("rel-40dB", peak * 0.01)):
        above = np.nonzero(rms > thr)[0]
        lead = int(above[0]) * 10 if len(above) else -1
        trail = (n - 1 - int(above[-1])) * 10 if len(above) else -1
        print(f"{path.split('/')[-1]} thr={label} peak_rms={peak:.3f} "
              f"leading_silence={lead}ms trailing={trail}ms dur={len(x)/sr:.2f}s")

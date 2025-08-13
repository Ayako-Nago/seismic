import numpy as np
import matplotlib.pyplot as plt
# Radon2Dの代わりにChirpRadon2Dをインポート
from pylops.signalprocessing import ChirpRadon2D
from scipy.ndimage import uniform_filter1d
from pylops.signalprocessing import Radon2D
import numpy as np
import matplotlib.pyplot as plt
import time



def compute_snr(reference, estimate):
    # 信号対雑音比（Signal-to-Noise Ratio）を計算する関数
    reference = np.asarray(reference).flatten()
    estimate = np.asarray(estimate).flatten()

    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - estimate) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-12))  # 安定化項
    return snr

def compute_cc(reference, estimate):
    # 相関係数（Cross-Correlation）を計算する関数
    reference = np.asarray(reference).flatten()
    estimate = np.asarray(estimate).flatten()

    ref_mean = np.mean(reference)
    est_mean = np.mean(estimate)
    
    numerator = np.sum((reference - ref_mean) * (estimate - est_mean))
    denominator = np.sqrt(np.sum((reference - ref_mean) ** 2) * np.sum((estimate - est_mean) ** 2))
    cc = numerator / (denominator + 1e-12)
    return cc

def compute_rmse(reference, estimate):
    # 平均二乗誤差（Root Mean Square Error）を計算する関数
    mse = np.mean((reference.astype(float) - estimate.astype(float)) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# この関数は元の線形モデルのままで正しいため、変更は不要です
def compute_true_semblance_weights(d_obs, t, x, p, L):
    """
    観測データに基づく Semblance 重みを計算 (論文式7に基づく)
    """
    nt, nx = d_obs.shape
    np_ = len(p)
    dt = t[1] - t[0]

    weights = np.zeros((nt, np_))

    for ip, pp in enumerate(p):
        numerator = np.zeros(nt)
        denominator = np.zeros(nt)

        for k in range(nx):
            trace = np.zeros(nt)
            trace_sq = np.zeros(nt)

            for l in range(-L, L + 1):
                # 時間シフト量 t = τ + p x_k + l dt に相当 (線形モデル)
                shift = pp * x[k] + l * dt
                idx = np.arange(nt) + int(round(shift / dt))
                idx = np.clip(idx, 0, nt - 1)

                trace += d_obs[idx, k]
                trace_sq += d_obs[idx, k] ** 2

            numerator += trace ** 2
            denominator += trace_sq

        s_raw = numerator / (nx * denominator + 1e-10)
        s_smooth = uniform_filter1d(s_raw, size=2 * L + 1)
        weights[:, ip] = s_smooth

    return weights

def ISTA(Rop, d, niter=50000, tol=1e-5, eps=1e-4, lambda_=1.0):
    print("Running ISTA...")
    # print(f"d shape: {d.shape}")

    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse reconstruction.

    Parameters:
        Rop (Radon2D): Radon operator.
        d (np.ndarray): Observed data.
        niter (int): Number of iterations.
        tol (float): Tolerance for convergence.
        eps (float): Step size for gradient descent.
        lambda_ (float): Regularization parameter for l1 norm.

    Returns:
        np.ndarray: Reconstructed sparse data.
    """
    x = Rop * d  # Initial guess using adjoint operator
    print(f"Initial x shape: {x.shape}, d shape: {d.shape}")

    for i in range(niter):
        # Forward operator application
        Ax = Rop.H * x
        # Gradient step
        grad = Rop * (Ax - d)
        x_tmp = x - eps * grad

        # print(f"norm: {np.linalg.norm(x_tmp, ord=1)}")
        # Proximal operator for l1 norm
        x_new = np.sign(x_tmp) * np.maximum(np.abs(x_tmp) - (eps * lambda_), 0)

        if i % 1000 == 0 or i == niter - 1:
            print(f"Iteration {i+1}/{niter}")
            print(f"residual norm: {np.linalg.norm(x_new - x)/np.linalg.norm(x)} fidelity: {np.linalg.norm(Ax - d)}")

        # Check convergence
        if np.linalg.norm(x_new - x)/np.linalg.norm(x) < tol:
            print(f"tolerance reached: {np.linalg.norm(x_new - x)} < {tol}")
            print(f"Converged after {i+1} iterations.")
            break
        else:
            x = x_new

    return x



def compute_psnr(img_1, img_2, data_range):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)


# Load data
truth = np.load("ground_truth_cmp_600x100.npz")
d, t, x, gt = truth["GT"], truth["t"], truth["x"], truth["GT"]

noise_amplitude = 0.3 * np.max(np.abs(d))
noise = np.random.normal(0, noise_amplitude, d.shape)
d = d + noise

print(f"Data shape: {d.shape}, Time: {t[0]:.3f} to {t[-1]:.3f} s, Offset: {x[0]} to {x[-1]} m")
nt, nx = d.shape


# Radon Transform Setting
slope_max = 0.0003
npx = 100
px = np.linspace(-slope_max, slope_max, npx)


# Radon2D operator  
rop = Radon2D(t, x, px, centeredh=False, kind='linear', interp=True,engine='numpy')

start = time.time()  
m_flat = ISTA(rop, d.ravel())
m_sparse = m_flat.reshape(nt, npx)
# semblance weighting
semblance = compute_true_semblance_weights(d, t, x, px, L=5)
m_weighted = m_sparse * semblance

#Processing time
end = time.time()
time_diff = end - start
print(time_diff)  


fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
axs[0].imshow(m_sparse.T, aspect='auto', cmap='seismic')
plt.title("Linear Radon Transform (Chirp)")
axs[1].imshow(m_weighted.T, aspect='auto', cmap='gray')
axs[1].set_title("Weighted Data")
#plt.show()
np.save("a_SRT.npy", m_flat)
np.save("a_SRTW.npy", m_weighted)

# Inverse Radon transform
d_denoised_flat = rop * m_flat
d_denoised = d_denoised_flat.reshape(nt, nx)
d_weighted = rop * m_flat
d_weighted = d_weighted.reshape(nt, nx)



np.save("d_SRT.npy", d_denoised)
np.save("d_SRTW.npy", d_weighted)

# Metrics and Plotting
print(f"radon:")
print(f"psnr : {compute_psnr(gt.flatten(), d_denoised.flatten(), data_range=1)} snr : {compute_snr(gt.flatten(), d_denoised.flatten())} cc : {compute_cc(gt.flatten(), d_denoised.flatten())} rmse : {compute_rmse(gt.flatten(), d_denoised.flatten())}")
print(f"weighted : ")
print(f"psnr : {compute_psnr(gt.flatten(), d_weighted.flatten(), data_range=1)} snr : {compute_snr(gt.flatten(), d_weighted.flatten())} cc : {compute_cc(gt.flatten(), d_weighted.flatten())} rmse : {compute_rmse(gt.flatten(), d_weighted.flatten())}")
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
axs[0].imshow(d, aspect='auto', cmap='gray', extent=[x[0], x[-1], t[-1], t[0]])
axs[0].set_title("Original Data")
axs[1].imshow(d_denoised, aspect='auto', cmap='gray', extent=[x[0], x[-1], t[-1], t[0]])
axs[1].set_title("Denoised Data")
axs[2].imshow(d_weighted, aspect='auto', cmap='gray', extent=[x[0], x[-1], t[-1], t[0]])
axs[2].set_title("Denoised Data(Weighted)")
for ax in axs: ax.set_xlabel("Offset (m)")
axs[0].set_ylabel("Time (s)")
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from pylops.signalprocessing import Radon2D
from pylops import FirstDerivative
from matplotlib.colors import Normalize
from scipy.sparse.linalg import svds
import time


def ProjL2ball(x, x_0, epsilon):
    # projection on l2 ball
    val = np.copy(x)
    if(np.linalg.norm(x - x_0) > epsilon):
        val = x_0 + epsilon * (x - x_0) / np.linalg.norm(x - x_0)
    return val


def Prox_l1norm(A, gamma):
    return np.sign(A) * np.maximum(np.abs(A) - gamma, 0)


def f_d(a,d,lam,radonop):
    val = lam * (d - radonop * a)
    return val

def f_a(a,d,lam,radonop):
    val = - (lam * radonop.H * (d - radonop * a))
    return val

def compute_psnr(img_1, img_2, data_range):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)


def compute_snr(reference, estimate):
    # Signal-to-Noise Ratio
    reference = np.asarray(reference).flatten()
    estimate = np.asarray(estimate).flatten()

    signal_power = np.sum(reference ** 2)
    noise_power = np.sum((reference - estimate) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
    return snr

def compute_cc(reference, estimate):
    # Cross-Correlation
    reference = np.asarray(reference).flatten()
    estimate = np.asarray(estimate).flatten()
    ref_mean = np.mean(reference)
    est_mean = np.mean(estimate)
    
    numerator = np.sum((reference - ref_mean) * (estimate - est_mean))
    denominator = np.sqrt(np.sum((reference - ref_mean) ** 2) * np.sum((estimate - est_mean) ** 2))
    cc = numerator / (denominator + 1e-12)
    return cc

def compute_rmse(reference, estimate):
    # Root Mean Square Error
    reference = np.asarray(reference).flatten()
    estimate = np.asarray(estimate).flatten()
    rmse = np.sqrt(np.mean((reference - estimate) ** 2))
    return rmse

#Noisy Data
data = np.load("noisy_10.npz")
Z = data['GT']     # Noisy Data
t = data['t']      # Time
h = data['x']      # Offset
nt, nx = Z.shape
#Ground Truth Data
data_GT = np.load("ground_truth_10.npz") 
Z_GT = data_GT['GT']     # Ground Truth Data
t_GT = data_GT['t']      # Time
h_GT = data_GT['x']      # Offset
nt_GT, nx_GT = Z_GT.shape


# Radon Transform Setting
slope_max = 0.0003
npx = 100
px = np.linspace(-slope_max, slope_max, npx)


dt = t[1] - t[0]
dx = h[1] - h[0]

# Radon2D operator  
radonop = Radon2D(t, h, px, centeredh=False, kind='linear', interp=True,engine='numpy')

#radon transform
d_in = Z.flatten()
d_GT = Z_GT.flatten()
m = radonop.H * d_in

S_max_r = 57.757149818729566 # (slope_max = 0.0003)
S_max_l = 1 + S_max_r
S_max_i = 1

epsilon_alpha = 0.95
sigma = 0.1
epsilon = epsilon_alpha * sigma * np.sqrt(np.size(d_in))

maxiter = 50001
a = np.zeros_like(m)
d = d_in
y = np.zeros_like(d_in)
obj = 1
obj_a = 1
obj_d = 1
subject = 1
residual = 1

lam_1 = 0.1
lam_2 = 0.5
obj_aarr = []
obj_darr = []
obj_arr = []
sub_arr = []
res_arr = []

# Step Size ( > 0)
gamma_1 = 0.000573
gamma_2 = 0.99/(gamma_1 * (S_max_i ** 2)) - (((lam_2 ** 2) *  (S_max_l ** 2))/ (2 * (S_max_i ** 2)))


print(f"gamma_1 : {gamma_1}")
print(f"gamma_2 : {gamma_2}")
print(f"gamma_1 * (beta / 2 + gamma_2 * S_max^2) : {gamma_1 * (((lam_2 ** 2 * (S_max_l ** 2)) / 2) + (gamma_2 * (S_max_i ** 2)))}")
print(f"epsilon : {epsilon}")


start = time.time()  
for i in range(maxiter):
    #終了条件
    if residual < 1e-5:
        print(f"i:{i:04d} a:{obj_a:.5f} d:{obj_d:.5f} obj:{obj:.5f} fidelity:{subject:.5f} residual:{residual:.5f}")
        print(f"psnr : {compute_psnr(d_GT, d, data_range=1)} snr : {compute_snr(d_GT, d)} cc : {compute_cc(d_GT, d)} rmse : {compute_rmse(d_GT, d)}")
        break

    #PDS Algorithm
    d_bef = d
    a_bef = a
    d = ProjL2ball(d_bef - gamma_1 * (f_d(a_bef,d_bef,lam_2,radonop) + y),d_in,epsilon)
    a = Prox_l1norm(a_bef - gamma_1 * f_a(a_bef,d_bef,lam_2,radonop) ,gamma_1 * lam_1)
    y_tmp = y + gamma_2 * (2 * d - d_bef)
    y = y_tmp - gamma_2 * Prox_l1norm(y_tmp/gamma_2, 1/gamma_2)

    #表示する値たち
    obj_a = np.linalg.norm(a, ord = 1)
    obj_aarr.append(obj_a)
    obj_d = np.linalg.norm(d, ord = 1)
    obj_darr.append(obj_d)
    obj = np.linalg.norm(d, ord = 1) + lam_1 * np.linalg.norm(a, ord = 1) + (lam_2 * np.linalg.norm(d - radonop * a) ** 2)/ 2
    obj_arr.append(obj)
    subject = np.linalg.norm(d - d_in)
    sub_arr.append(subject)
    residual = np.linalg.norm(d - d_bef)/np.linalg.norm(d_bef)
    res_arr.append(residual)


    if i%100 == 0:
        print(f"i:{i:04d} a:{obj_a:.5f} d:{obj_d:.5f} obj:{obj:.5f} fidelity:{subject:.5f} residual:{residual:.6f}")

#Processing time
end = time.time()
time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print(time_diff)  

np.save('d_proposed', d)
np.save('a_proposed', a)


fig, axs = plt.subplots(1, 5, figsize=(18, 6))
scale = 0.5

axs[0].plot(obj_aarr)
axs[0].set_yscale('log')
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("a(L1)")
axs[0].set_title("a(L1)")

axs[1].plot(obj_darr)
axs[1].set_yscale('log')
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("d(L1)")
axs[1].set_title("d(L1)")

axs[2].plot(obj_arr)
axs[2].set_yscale('log')
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("obj")
axs[2].set_title("obj")

axs[3].plot(sub_arr)
axs[3].set_xlabel("Iteration")
axs[3].set_ylabel("fidelity")
axs[3].set_title("fidelity")

axs[4].plot(res_arr)
axs[4].set_yscale('log')
axs[4].set_xlabel("Iteration")
axs[4].set_ylabel("residual")
axs[4].set_title("residual")


plt.show()


# Radon domain
M = a.reshape(nt,len(px)).T

# Time-Space domain
dp = d.reshape(nt, nx)

# Ground Truth Datas
m_GT = radonop.H * d_GT
M_GT = m_GT.reshape(nt_GT, len(px)).T
dp_GT = (radonop * m_GT).reshape(nt_GT, nx_GT)

# Visualization
fig, axs = plt.subplots(2, 3, figsize=(18, 6))
scale = 0.5

# Noisy Data
im = axs[0,0].imshow(Z / np.max(np.abs(Z)),
                   extent=[h[0], h[-1], t[0], t[-1]],
                   aspect='auto', cmap='gray',
                   norm=Normalize(vmin=-1, vmax=1))
axs[0,0].set_title('Noisy Data')
axs[0,0].set_xlabel('Distance (km)')
axs[0,0].set_ylabel('Time (sec)')

# Radon Domain
im = axs[0,1].imshow(M / np.max(np.abs(M)),
                   extent=[px[0], px[-1], t[-1], t[0]],
                   aspect='auto', cmap='seismic',
                   norm=Normalize(vmin=-1, vmax=1))
axs[0,1].set_title('Radon Domain')
axs[0,1].set_xlabel('p (s/km)')
axs[0,1].set_ylabel('Time (sec)')
fig.colorbar(im, ax=axs[0,1])

# Denoised Data
im = axs[0,2].imshow(dp / np.max(np.abs(dp)),
                   extent=[h[0], h[-1], t[0], t[-1]],
                   aspect='auto', cmap='gray',
                   norm=Normalize(vmin=-1, vmax=1))
axs[0,2].set_title('Denoised Data')
axs[0,2].set_xlabel('Distance (km)')
axs[0,2].set_ylabel('Time (sec)')


# Ground Truth
im = axs[1,0].imshow(Z_GT / np.max(np.abs(Z_GT)),
                   extent=[h[0], h[-1], t[0], t[-1]],
                   aspect='auto', cmap='gray',
                   norm=Normalize(vmin=-1, vmax=1))
axs[1,0].set_title('CMP Section(raw)')
axs[1,0].set_xlabel('Distance (km)')
axs[1,0].set_ylabel('Time (sec)')

# Radon Domain
im = axs[1,1].imshow(M_GT / np.max(np.abs(M_GT)),
                   extent=[px[0], px[-1], t[-1], t[0]],
                   aspect='auto', cmap='seismic',
                   norm=Normalize(vmin=-1, vmax=1))
axs[1,1].set_title('Radon')
axs[1,1].set_xlabel('p (s/km)')
axs[1,1].set_ylabel('Time (sec)')
fig.colorbar(im, ax=axs[1,1])

# Ground Truth
im = axs[1,2].imshow(dp_GT / np.max(np.abs(dp_GT)),
                   extent=[h[0], h[-1], t[0], t[-1]],
                   aspect='auto', cmap='gray',
                   norm=Normalize(vmin=-1, vmax=1))
axs[1,2].set_title('CMP Section(raw)')
axs[1,2].set_xlabel('Distance (km)')
axs[1,2].set_ylabel('Time (sec)')


plt.tight_layout()
plt.show()

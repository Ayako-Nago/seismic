import numpy as np
import matplotlib.pyplot as plt
from pylops.signalprocessing import Radon2D
from pylops import FirstDerivative
from matplotlib.colors import Normalize
from scipy.sparse.linalg import svds



def ProjL2ball(x, x_0, epsilon):
    # projection on l2 ball
    val = np.copy(x)
    if(np.linalg.norm(x - x_0) > epsilon):
        val = x_0 + epsilon * (x - x_0) / np.linalg.norm(x - x_0)
    return val


def Prox_l1norm(A, gamma):
    return np.sign(A) * np.maximum(np.abs(A) - gamma, 0)


def f_d(a,d,lam,radonop):
    val = lam * (d - np.fft.fft(a))
    return val

def f_a(a,d,lam,radonop):
    val = - (lam * np.fft.ifft(d - np.fft.fft(a)))
    return val

def compute_psnr(img_1, img_2, data_range):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)


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



import time

start = time.time()  

# Load data
truth = np.load("ground_truth_10.npz")
d, t, x, gt = truth["GT"], truth["t"], truth["x"], truth["GT"]

sigma = 0.1
noise_amplitude = sigma * np.max(np.abs(truth))
noise = np.random.normal(0, noise_amplitude, truth.shape)
d = truth + noise

Z = d.copy()
nt, nx = Z.shape
t = np.arange(nt) * 10  # 時間軸 [s] → 10秒ステップ × 600点 = 1時間
h = np.arange(nx) * 2.0  # 距離軸 [km]（仮に2km間隔）


# ② スローネス軸（p軸）の設定
# 傾き（px）の範囲設定（線形ラドン変換を想定）
slope_max = 1.0  # s/km
npx = 100
px = np.linspace(-slope_max, slope_max, npx)



# ③ Radon2D オペレータの構築（正しい引数名）
radonop = Radon2D(t, h, px, centeredh=False, kind='linear', interp=True,engine='numpy')

# ④ ラドン変換（時間空間 → ラドン空間）
d_in = Z.flatten()
m = np.fft.ifft(d_in)              # Forward transform (Adjoint)
d_GT = truth["GT"].flatten()

#print(Z.shape) #(500, 40) (100, 40)
#print(d_in.shape) #(20000,) (4000,)
#print(radonop.shape) #(20000, 100000) (4000, 20000)
#radon_dense = radonop.todense()
#np.save('radon_dense', radon_dense)

#print(radon_dense.shape) #(20000, 100000)
#rank = np.linalg.matrix_rank(radon_dense) 
#print(rank) #16679
#print(m.shape) #(100000,)

S_max = 57.757149818729566 #最大特異値 (slope_max = 0.0003)


epsilon_alpha = 0.95

epsilon = epsilon_alpha * sigma * np.sqrt(np.size(d_in))

maxiter = 50001
a = m
d = d_in
y = radonop.H * d_in
obj = 1
obj_a = 1
obj_d = 1
subject = 1
residual = 1

lam_1 = 0.2
lam_2 = 0.1

obj_aarr = []
obj_darr = []
obj_arr = []
sub_arr = []
res_arr = []

gamma_1 = 0.01
gamma_2 = 0.01
# gamma_2 = 0.999/(gamma_1 * S_max * S_max)
# gamma_2 = 0.99/(gamma_1 * (S_max_i ** 2)) - (((lam_2 ** 2) *  (S_max_l ** 2))/ (2 * (S_max_i ** 2)))

print(f"gamma_1 : {gamma_1}")
print(f"gamma_2 : {gamma_2}")
# print(f"gamma_1 * (beta / 2 + gamma_2 * S_max^2) : {gamma_1 * (((lam_2 ** 2 * (S_max_l ** 2)) / 2) + (gamma_2 * (S_max_i ** 2)))}")
print(f"epsilon : {epsilon}")

for i in range(maxiter):
    #終了条件
    if residual < 1e-5:
        d_in = d_in.real
        d = d.real
        print(f"i:{i:04d} a:{obj_a:.5f} d:{obj_d:.5f} obj:{obj:.5f} fidelity:{subject:.5f} residual:{residual:.5f}")
        print(f"psnr : {compute_psnr(d_GT, d, data_range=1)} snr : {compute_snr(d_GT, d)} cc : {compute_cc(d_GT, d)} rmse : {compute_rmse(d_GT, d)}")
        break

    #PDSアルゴリズム
    d_bef = d
    d = ProjL2ball(d_bef - gamma_1 * (radonop * y),d_in,epsilon)
    y_tmp = y + gamma_2 * (radonop.H * (2 * d - d_bef))
    y = y_tmp - gamma_2 * Prox_l1norm(y_tmp/gamma_2, 1/gamma_2)

    # print(f"proj : {np.linalg.norm(ProjL2ball(y_tmp/gamma_2,beta,epsilon))} ベータとの差 : {np.linalg.norm(ProjL2ball(y_tmp/gamma_2,beta,epsilon) - beta)}")

    #表示する値たち

    obj = np.linalg.norm(radonop.H * d, ord = 1)
    obj_arr.append(obj)
    subject = np.linalg.norm(d - d_in)
    sub_arr.append(subject)
    residual = np.linalg.norm(d - d_bef)/np.linalg.norm(d_bef)
    res_arr.append(residual)


    if i%100 == 0:
        print(f"i:{i:04d} obj:{obj:.5f} fidelity:{subject:.5f} residual:{residual:.5f}")


end = time.time()  # 現在時刻（処理完了後）を取得

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print(time_diff)  

np.save('d_f2_10_22', d)

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


#print(alpha.shape) #(20000,)
#print(y.shape) #(100000,)


#print(M.shape) #(500, 200)

# ⑤ 逆ラドン変換
dp = d.reshape(nt, nx).real


# # ③ Radon2D オペレータの構築（正しい引数名）
# radonop = Radon2D(t_GT, h_GT, px, centeredh=False, kind='linear', interp=True,engine='numpy')

# ④ ラドン変換（時間空間 → ラドン空間）

# m_GT = np.fft.ifft(d_GT)              # Forward transform (Adjoint)
# M_GT = m_GT.reshape(nt, nx)   # 時間 × スローネス に整形


# # ⑤ 逆ラドン変換（ノイズ除去）
# dp_GT = (np.fft.fft(m_GT)).reshape(nt_GT, nx_GT).real



# # ⑥ 可視化
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
scale = 0.5

# (a) 元データ wiggle plot
im = axs[0].imshow(Z,
                   extent=[h[0], h[-1], t[0], t[-1]],
                   aspect='auto', cmap='gray')
axs[0].set_title('Observe CMP Section(raw)')
axs[0].set_xlabel('Distance (km)')
axs[0].set_ylabel('Time (sec)')



# (c) ノイズ除去後 wiggle plot
im = axs[1].imshow(dp,
                   extent=[h[0], h[-1], t[0], t[-1]],
                   aspect='auto', cmap='gray')
axs[1].set_title('CMP Section(raw)')
axs[1].set_xlabel('Distance (km)')
axs[1].set_ylabel('Time (sec)')


plt.tight_layout()
plt.show()

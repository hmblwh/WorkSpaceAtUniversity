import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
"""\
Chương trình làm tròn tin hiệu để khử nhiễu
dùng bộ lọc trung bình cộng 3 điểm
có PTSP y[n] = 1/3(x[n]+x[n-1]+x[n-2])
"""

if __name__ == "__main__":
    # độ dài tín hiệu
    L = 100
    # các thời gian rời rạc lấy mẫu
    n = np.linspace(0, 3, L)
    # tín hiệu gốc s[n]
    original_signal = 4*n + 4*np.sin(n*10)+1
    # tin hiệu nhiễu
    noise_signal = np.random.randn(L)
    # tín hiệu lẫn nhiễu x[n]
    mixed_noise_signal = original_signal + noise_signal

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, sharey=True)

    # vẽ đồ thị các tín hiệu gốc, nhiễu , lẫn nhiễu
    axes[0].plot(noise_signal, color="red",
                 linewidth=0.5, linestyle="-", label="noise")
    axes[0].plot(original_signal, color="blue",
                 linewidth=0.5, linestyle="--", label="original")
    axes[0].plot(mixed_noise_signal, color="purple",
                 linewidth=0.5, linestyle=(0, (3, 1, 1, 1)), label="mixed")
    axes[0].legend()
    axes[0].set_title("Noise, Origin, Mixed")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Amplitude")
    # sinh tín hiệu x[n]
    x0 = mixed_noise_signal
    # sinh tín hiệu x[n-1]
    x1 = np.hstack((0, mixed_noise_signal[:-1]))
    # sinh tín hiệu x[n-2]
    x2 = np.hstack((0, 0, mixed_noise_signal[:-2]))
    # vẽ đồ thị x[n],x[n-1],x[n-2]
    axes[1].plot(x0, color="red", linestyle=(
        0, (1, 0)), linewidth=0.5, label="x[n]")
    axes[1].plot(x1, color="blue", linestyle=(
        0, (1, 0)), linewidth=0.5, label="x[n-1]")
    axes[1].plot(x2, color="purple", linestyle=(
        0, (1, 0)), linewidth=0.5, label="x[n-2]")
    axes[1].legend()
    axes[1].set_title("x[n], x[n-1], x[n-2]")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Amplitude")

    # cài dăt lệnh cách 1
    y1 = 1/3*(x0 + x1 + x2)
    # vẽ đồ thị y1[n] vs s[n]
    axes[2].plot(original_signal, color="red", linestyle=(
        0, (1, 0)), linewidth=0.5, label="x[n]")
    axes[2].plot(y1, color="blue", linestyle=(
        0, (1, 0)), linewidth=0.5, label="y1[n]")

    axes[2].legend()
    axes[2].set_title("3-points smoothed y1[n] vs original signal s[n]")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("Amplitude")

    # cách 2:  cài đặt hệ thống bằng cách dùng hàm tính convolution
    fig, axes = plt.subplots(figsize=(16, 9), sharex=True, sharey=True)
    # đáp ứng xung của hệ
    h = 1/3 * np.ones(3)

    y2 = np.convolve(mixed_noise_signal, h)
    axes.plot(original_signal, color="red", linestyle=(
        0, (1, 0)), linewidth=0.5, label="x[n]")
    axes.plot(y2, color="blue", linestyle=(
        0, (1, 0)), linewidth=0.5, label="y2[n]")
    axes.legend()
    axes.set_title("3-points smoothed y2[n] vs original signal s[n]")
    axes.set_xlabel("Index")
    axes.set_ylabel("Amplitude")

    fig, axes = plt.subplots(figsize=(16, 9), sharex=True, sharey=True)
    axes.plot(y1, color="red", linestyle=(
        0, (1, 0)), linewidth=0.5, label="y1[n]")
    axes.plot(y2, color="blue", linestyle=(
        0, (1, 0)), linewidth=0.5, label="y2[n]")
    axes.legend()
    axes.set_title("3-points smoothed y1[n] vs 3-points smoothed y2[n]")
    axes.set_xlabel("Index")
    axes.set_ylabel("Amplitude")
    # cách 3: dung for loop để keó trượt theo tín hiệu
    def my_colvolution(a,b):
        a = np.array(a)
        b = np.array(b)
        # đệ quy dùng tính chất a*b = b*a
        if a.size < b.size:
            return my_colvolution(b,a)
        x = np.hstack((np.zeros(b.size-1),a,np.zeros(b.size-1)))
        # dùng vòng for trược tín hiệu b trên x
        return np.array([np.sum(x[k:k+b.size]*b)
                  for k in range(0, x.size-(b.size-1))])
    h = 1/3 * np.ones(3)
    x = mixed_noise_signal
    y3 = my_colvolution(x,h)


    # cách 4: dùng hàm trong scipy
    numerator = [1/3, 1/3, 1/3]
    denominator = [1]
    y4 = signal.lfilter(numerator, denominator, mixed_noise_signal)

    # vẽ kết quả 4 cách
    fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True, sharey=True)
    axes[0].plot(y1, color="red",
                 linewidth=0.5, linestyle="-", label="y1[n]")
    axes[1].plot(y2, color="blue",
                 linewidth=0.5, linestyle="-", label="y2[n]")
    axes[2].plot(y3, color="purple",
                 linewidth=0.5, linestyle="-", label="y3[n]")
    axes[3].plot(y4, color="green",
                 linewidth=0.5, linestyle="-", label="y3[n]")
    axes[0].legend()
    axes[0].set_title("Y1[n] cách dùng trung bình")
    axes[0].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].set_title("Y2[n] dùng tổng chập hàm np.convolve") 
    axes[1].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].set_title("Y3[n] dùng vòng for để trượt")
    axes[2].set_ylabel("Amplitude") 
    axes[3].legend()
    axes[3].set_title("Y4[n] dùng hàm trong scipy ")
    axes[3].set_ylabel("Amplitude") 

    plt.show()

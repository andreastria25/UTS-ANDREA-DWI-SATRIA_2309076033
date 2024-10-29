import numpy as np
import matplotlib.pyplot as plt

# Parameter yang Diketahui
L = 0.5  # Induktansi dalam Henry
C = 10e-6  # Kapasitansi dalam Farad
fd = 1000  # Frekuensi dalam Hz

# Fungsi untuk menghitung frekuensi resonansi berdasarkan R
def hitung_frekuensi_resonansi(R):
    return (1 / (2 * np.pi)) * np.sqrt((1 / (L * C)) - (R*2 / (4 * L*2)))

# Fungsi yang menghitung selisih antara frekuensi resonansi dan fd
def selisih_frekuensi(R):
    return hitung_frekuensi_resonansi(R) - fd

# Mengecek nilai selisih_frekuensi pada batas tertentu
print("selisih_frekuensi(10):", selisih_frekuensi(10))
print("selisih_frekuensi(100):", selisih_frekuensi(100))

# Visualisasi selisih_frekuensi untuk menentukan interval yang sesuai
nilai_R = np.linspace(0, 200, 1000)
nilai_selisih_F_R = selisih_frekuensi(nilai_R)

plt.plot(nilai_R, nilai_selisih_F_R)
plt.axhline(0, color='red', lw=0.5)  # Garis horizontal di y=0
plt.xlabel('Nilai R')
plt.ylabel('Selisih Frekuensi F(R)')
plt.title('Plot Selisih Frekuensi F(R)')
plt.grid(True)
plt.show()

# Implementasi metode bisection untuk menemukan akar
def metode_bisection(fungsi, a, b, toleransi=1e-6):
    if fungsi(a) * fungsi(b) > 0:
        raise ValueError("Fungsi tidak mengalami perubahan tanda pada interval ini.")
    
    tengah = (a + b) / 2.0
    nilai_R_bisection = []
    while (b - a) / 2.0 > toleransi:
        tengah = (a + b) / 2.0
        nilai_R_bisection.append(tengah)
        if fungsi(tengah) == 0:
            return tengah, nilai_R_bisection
        elif fungsi(a) * fungsi(tengah) < 0:
            b = tengah
        else:
            a = tengah
    return tengah, nilai_R_bisection

# Mencari nilai R menggunakan metode bisection
R_bisection, nilai_R_bisection = metode_bisection(selisih_frekuensi, 0, 200)

# Menghitung turunan dari F(R)
def turunan_F_R(R):
    return -(R / (2 * np.pi * L*2 * np.sqrt((1 / (L * C)) - (R2 / (4 * L*2)))))

# Metode Newton-Raphson untuk mencari akar
def metode_newton_raphson(fungsi, turunan_fungsi, x0, toleransi=1e-6, iterasi_maks=100):
    x = x0
    nilai_R_newton = []
    for i in range(iterasi_maks):
        x_baru = x - fungsi(x) / turunan_fungsi(x)
        nilai_R_newton.append(x_baru)
        if abs(x_baru - x) < toleransi:
            return x_baru, nilai_R_newton
        x = x_baru
    raise ValueError("Metode Newton-Raphson gagal untuk konvergen.")

# Mencari nilai R menggunakan metode Newton-Raphson
R_newton, nilai_R_newton = metode_newton_raphson(selisih_frekuensi, turunan_F_R, 50)

# Visualisasi perbandingan konvergensi
def plot_perbandingan_konvergensi():
    plt.plot(nilai_R_bisection, label="Metode Bisection")
    plt.plot(nilai_R_newton, label="Metode Newton-Raphson")
    plt.xlabel('Iterasi')
    plt.ylabel('Nilai R')
    plt.title('Perbandingan Konvergensi')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_perbandingan_konvergensi()

# Fungsi eliminasi Gauss untuk sistem persamaan linear
def eliminasi_gauss(A, B):
    n = len(B)
    for i in range(n):
        for j in range(i+1, n):
            rasio = A[j][i] / A[i][i]
            for k in range(n):
                A[j][k] = A[j][k] - rasio * A[i][k]
            B[j] = B[j] - rasio * B[i]
    
    X = [0 for i in range(n)]
    X[n-1] = B[n-1] / A[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        X[i] = B[i]
        for j in range(i+1, n):
            X[i] = X[i] - A[i][j] * X[j]
        X[i] = X[i] / A[i][i]
    
    return X

# Sistem persamaan linear
A = [[4, -1, -1], [-1, 3, -1], [-1, -1, 5]]
B = [5, 3, 4]

# Penyelesaian sistem menggunakan eliminasi Gauss
solusi = eliminasi_gauss(A, B)
print(solusi)
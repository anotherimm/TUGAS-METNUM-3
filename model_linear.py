import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import unittest

# Membaca data dari file CSV
data = pd.read_csv('F:/COOLYEAH/SEM4/METODE NUMERIK/TUGAS METNUM 3/Student_Performance.csv')

# Menampilkan sebagian data
print(data.head())

# Definisi model linear
def linear_model(TB, a, b):
    return a * TB + b

# Fungsi untuk estimasi parameter model linear
def estimate_parameters(TB, NT):
    params, _ = curve_fit(linear_model, TB, NT)
    return params

# Fungsi untuk menghitung galat RMS
def calculate_rms_error(NT_actual, NT_pred):
    return np.sqrt(mean_squared_error(NT_actual, NT_pred))

# Variabel independen (TB) dan dependen (NT)
TB = data['Hours Studied'].values
NT = data['Performance Index'].values

# Estimasi parameter model menggunakan curve_fit
a, b = estimate_parameters(TB, NT)
print(f'Parameter a: {a}, Parameter b: {b}')

# Prediksi nilai NT menggunakan model yang diestimasi
NT_pred = linear_model(TB, a, b)

# Menghitung galat RMS
rms_error = calculate_rms_error(NT, NT_pred)
print(f'Galat RMS: {rms_error}')

# Membuat plot grafik titik data dan hasil regresi
plt.figure(figsize=(10, 6))
plt.scatter(TB, NT, label='Data Asli', color='blue')
plt.plot(TB, NT_pred, label='Hasil Regresi (Model Linear)', color='red')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Model Linear: NT = a * TB + b')
plt.legend()
plt.grid(True)
plt.show()

# Kelas untuk melakukan pengujian menggunakan unittest
class TestLinearModel(unittest.TestCase):
    def test_estimate_parameters(self):
        # Test data
        TB_test = np.array([1, 2, 3, 4, 5])
        NT_test = np.array([2, 4, 6, 8, 10])
        # Estimasi parameter
        a, b = estimate_parameters(TB_test, NT_test)
        # Memastikan parameter yang diestimasi mendekati nilai yang diharapkan
        self.assertAlmostEqual(a, 2.0, places=5)
        self.assertAlmostEqual(b, 0.0, places=5)

    def test_calculate_rms_error(self):
        # Test data
        NT_actual = np.array([2, 4, 6, 8, 10])
        NT_pred = np.array([2, 4, 6, 8, 10])
        # Menghitung galat RMS
        rms_error = calculate_rms_error(NT_actual, NT_pred)
        # Memastikan galat RMS mendekati nol
        self.assertAlmostEqual(rms_error, 0.0, places=5)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

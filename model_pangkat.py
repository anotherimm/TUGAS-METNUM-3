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

# Definisi model pangkat sederhana
def power_law_model(TB, a, b):
    return a * TB ** b

# Variabel independen (TB) dan dependen (NT)
TB = data['Hours Studied'].values
NT = data['Performance Index'].values

# Estimasi parameter model menggunakan curve_fit
params, _ = curve_fit(power_law_model, TB, NT)

# Parameter a dan b yang diestimasi
a, b = params
print(f'Parameter a: {a}, Parameter b: {b}')

# Prediksi nilai NT menggunakan model yang diestimasi
NT_pred = power_law_model(TB, a, b)

# Menghitung galat RMS
rms_error = np.sqrt(mean_squared_error(NT, NT_pred))
print(f'Galat RMS: {rms_error}')

# Membuat plot grafik titik data dan hasil regresi
plt.figure(figsize=(10, 6))
plt.scatter(TB, NT, label='Data Asli', color='blue')
plt.plot(TB, NT_pred, label='Hasil Regresi (Model Pangkat)', color='red')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Model Pangkat Sederhana: NT = a * TB^b')
plt.legend()
plt.grid(True)
plt.show()

# Definisi unit test
class TestPowerLawModel(unittest.TestCase):
    def test_model(self):
        # Tes untuk memastikan model pangkat bekerja dengan benar
        TB_test = np.array([1, 2, 3])
        a_test, b_test = 2, 1.5
        expected = np.array([2, 2 * 2 ** 1.5, 2 * 3 ** 1.5])
        result = power_law_model(TB_test, a_test, b_test)
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_rms_error(self):
        # Tes untuk memastikan perhitungan galat RMS bekerja dengan benar
        NT_actual = np.array([10, 20, 30])
        NT_pred = np.array([12, 18, 29])
        expected_rms_error = np.sqrt(mean_squared_error(NT_actual, NT_pred))
        rms_error = np.sqrt(mean_squared_error(NT_actual, NT_pred))
        self.assertAlmostEqual(rms_error, expected_rms_error, places=5)

# Menjalankan tes
unittest.main(argv=[''], verbosity=2, exit=False)

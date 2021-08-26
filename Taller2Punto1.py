"""
TALLER 2 Punto1- Paula Castro Aguilar
Este archivo contiene la definición de la clase thetaFilter y sus respectivos métodos
Corresponde al punto 1 del taller, donde el usuario ingresa los valores de teta y delta de teta y se obtiene el filtro de la imagen
"""

import cv2
import sys
import os
import numpy as np

class thetaFilter:
    # Constructor que recibe la ruta de una imagen la carga y convierte la imagen en escala de grises
    def __init__(self):
        path = sys.argv[1]
        image_name = sys.argv[2]
        path_file = os.path.join(path, image_name)
        self.image = cv2.imread(path_file)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


    # Método que recibe los valores de teta y delta de teta ingresados por el usuario
    def set_theta(self,theta,theta_delta):
        self.theta = theta
        #print(self.theta)
        self.theta_delta = theta_delta
        #print(self.theta_delta)

    # Método que aplica la trasnformada de Fourier a la imagen y aplica el filtro pasabandas teniendo en cuenta los valores de teta y delta ingresados
    def filtering(self):
        # Apply fft
        image_gray_fft = np.fft.fft2(self.image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        # fft visualization
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)
        # pre-computations
        num_rows, num_cols = (self.image_gray.shape[0], self.image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        # Filtro
        # band pass filter mask
        band_pass_mask1 = np.zeros_like(self.image_gray)
        idx_low = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi +180 > (self.theta - self.theta_delta)
        idx_high = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi +180 < (self.theta + self.theta_delta)
        idx_bp = np.bitwise_and(idx_low, idx_high)
        idx_low1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi +180 > (self.theta + 180 - self.theta_delta)
        idx_high1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi +180 < (self.theta + 180 + self.theta_delta)
        idx_bp1 = np.bitwise_and(idx_low1, idx_high1)
        idx_bpf = np.bitwise_or(idx_bp, idx_bp1)
        band_pass_mask1[idx_bpf] = 1
        band_pass_mask1[int(half_size),int(half_size)] = 1

        # filtering via FFT
        mask = band_pass_mask1  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered1 = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered1 = np.absolute(image_filtered1)
        image_filtered1 /= np.max(image_filtered1)

        cv2.imshow("Original image", self.image)
        cv2.imshow("Respuesta del filtro°", 255 * mask)
        cv2.imshow("Imagen filtrada°", image_filtered1)
        cv2.waitKey(0)

if __name__ == '__main__':
  Filter = thetaFilter()
  theta = int(input("Ingrese valor de teta: "))
  theta_delta = int(input("Ingrese valor de delta de teta: "))
  Filter.set_theta(theta,theta_delta)
  Filter.filtering()



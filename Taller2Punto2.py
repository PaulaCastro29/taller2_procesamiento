"""
TALLER 2 Punto 2- Paula Castro Aguilar
Este archivo contiene la definición de la clase thetaFilter y sus respectivos métodos
Corresponde al punto 2 del taller, donde a partir de 4 filtros se promedia una imagen y se obtiene una nueva
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
        cv2.imshow("Original imagen", self.image)


    # Método que recibe los valores de teta de cada filtro y delta de teta
    def set_theta(self,theta1,theta2,theta3,theta4,theta_delta):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4
        self.theta_delta = theta_delta

    # Método que aplica la trasnformada de Fourier a la imagen y aplica los filtros y obtiene una imagen nueva con el promedio
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
        half_size = num_rows / 2   # here we assume num_rows = num_columns

        # Filtro 0 grados
        # band pass filter mask
        band_pass_mask1 = np.zeros_like(self.image_gray)
        idx_low = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta1 - self.theta_delta)
        idx_high = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta1 + self.theta_delta)
        idx_bp = np.bitwise_and(idx_low, idx_high)
        idx_low1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta1 + 180 - self.theta_delta)
        idx_high1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta1 + 180 + self.theta_delta)
        idx_bp1 = np.bitwise_and(idx_low1, idx_high1)
        idx_bpf = np.bitwise_or(idx_bp, idx_bp1)
        band_pass_mask1[idx_bpf] = 1
        band_pass_mask1[int(half_size), int(half_size)] = 1

        # filtering via FFT
        mask = band_pass_mask1  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered1 = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered1 = np.absolute(image_filtered1)
        image_filtered1 /= np.max(image_filtered1)

        cv2.imshow("Respuesta en frecuencia del filtro a 0 grados", 255 * mask)
        cv2.imshow("Imagen filtrada a 0 grados", image_filtered1)


        #Filtro 45 grados
        # band pass filter mask
        band_pass_mask1 = np.zeros_like(self.image_gray)
        idx_low = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta2 - self.theta_delta)
        idx_high = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta2 + self.theta_delta)
        idx_bp = np.bitwise_and(idx_low, idx_high)
        idx_low1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta2 + 180 - self.theta_delta)
        idx_high1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta2 + 180 + self.theta_delta)
        idx_bp1 = np.bitwise_and(idx_low1, idx_high1)
        idx_bpf = np.bitwise_or(idx_bp, idx_bp1)
        band_pass_mask1[idx_bpf] = 1
        band_pass_mask1[int(half_size), int(half_size)] = 1

        # filtering via FFT
        mask = band_pass_mask1  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered2 = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered2 = np.absolute(image_filtered2)
        image_filtered2 /= np.max(image_filtered2)

        cv2.imshow("Respuesta en frecuencia del filtro a 45 grados", 255 * mask)
        cv2.imshow("Imagen filtrada a 45 grados", image_filtered2)


        #Filtro  90 grados
        # band pass filter mask
        band_pass_mask1 = np.zeros_like(self.image_gray)
        idx_low = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta3 - self.theta_delta)
        idx_high = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta3 + self.theta_delta)
        idx_bp = np.bitwise_and(idx_low, idx_high)
        idx_low1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta3 + 180 - self.theta_delta)
        idx_high1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta3 + 180 + self.theta_delta)
        idx_bp1 = np.bitwise_and(idx_low1, idx_high1)
        idx_bpf = np.bitwise_or(idx_bp, idx_bp1)
        band_pass_mask1[idx_bpf] = 1
        band_pass_mask1[int(half_size), int(half_size)] = 1

        # filtering via FFT
        mask = band_pass_mask1  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered3 = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered3 = np.absolute(image_filtered3)
        image_filtered3 /= np.max(image_filtered3)

        cv2.imshow("Respuesta en frecuencia del filtro a 90 grados", 255 * mask)
        cv2.imshow("Imagen filtrada a 90 grados", image_filtered3)


        # Filtro 135 grados
        # band pass filter mask
        band_pass_mask1 = np.zeros_like(self.image_gray)
        idx_low = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta4 - self.theta_delta)
        idx_high = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta4 + self.theta_delta)
        idx_bp = np.bitwise_and(idx_low, idx_high)
        idx_low1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 > (self.theta4 + 180 - self.theta_delta)
        idx_high1 = 180 * (np.arctan2(row_iter - half_size, col_iter - half_size)) / np.pi + 180 < (self.theta4 + 180 + self.theta_delta)
        idx_bp1 = np.bitwise_and(idx_low1, idx_high1)
        idx_bpf = np.bitwise_or(idx_bp, idx_bp1)
        band_pass_mask1[idx_bpf] = 1
        band_pass_mask1[int(half_size), int(half_size)] = 1
        # filtering via FFT
        mask = band_pass_mask1  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        image_filtered4 = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered4 = np.absolute(image_filtered4)
        image_filtered4 /= np.max(image_filtered4)

        cv2.imshow("Respuesta en frecuencia del filtro a 135 grados", 255 * mask)
        cv2.imshow("Imagen filtrada a 135 grados", image_filtered4)


        cv2.imshow("Original imagen", self.image)
        promedio = (image_filtered1+image_filtered2+image_filtered3+image_filtered4)/4
        promedio = promedio-promedio.min()
        promedio = promedio/promedio.max()
        cv2.imshow("Promedio imagenes", promedio)
        cv2.waitKey(0)
        cv2.imwrite('promedio.png', promedio)

if __name__ == '__main__':
  Filter = thetaFilter()
  theta1 = 0
  theta2 = 45
  theta3 = 90
  theta4 = 135
  theta_delta = 30
  Filter.set_theta(theta1,theta2,theta3,theta4,theta_delta)
  Filter.filtering()


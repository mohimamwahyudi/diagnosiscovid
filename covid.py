from PyQt5 import QtWidgets, uic, QtSql, QtCore,QtGui,QtPrintSupport
from PyQt5.QtWidgets import QDialog,QFileDialog
import sys
from PyQt5.QtGui import QPixmap 
from PyQt5.QtPrintSupport import * 
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
from PIL import Image

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('covid.ui', self)
        self.show()
        self.tambah.clicked.connect(self.add)
       


    def add(self):
        fname = QFileDialog.getOpenFileName(self,"open file", "G:", "All Files (*);;PNG(*.png;;Jpg File(*.jpg)")
        cek = QtGui.QPixmap(fname[0])
        self.gambar.setPixmap(cek)
        LABEL_NAMES = ["COVID","Normal","Pneumonia"]
        mod = load_model("modelfold3.h5")
        img = image.load_img(fname[0], target_size = (240, 240))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        images = np.vstack([x])
        classes = mod.predict(images)
        print(classes)
        y_pred = np.argmax(classes)
        print(y_pred)
        hasil = LABEL_NAMES[y_pred]
        self.label_4.setText(hasil)
                        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()

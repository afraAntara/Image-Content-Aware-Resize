# import numpy as np
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QScrollArea, QPushButton, \
    QFileDialog, QHBoxLayout, QSizePolicy, QLineEdit, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import warnings
from image_resize import algo1
from enlarge_image import enlarge_algo


class ImageGalleryViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = ''
        self.row_remove = 0
        self.col_remove = 0
        self.setWindowTitle("Image Content Aware Resize")
        self.setGeometry(100, 30, 1550, 1020)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Image areas for input and output
        self.image_area_in_out_layout = QHBoxLayout()

        input_image_layout = QVBoxLayout()
        self.input_image_container = QLabel()
        self.input_image_label = QLabel()
        self.input_image_label.setStyleSheet("border: 2px solid #aaa; background-color: #f0f0f0;")
        input_image_layout.addWidget(self.input_image_label)

        output_image_layout = QVBoxLayout()
        self.output_image_container = QLabel()
        self.output_image_label = QLabel()
        self.output_image_label.setStyleSheet("border: 2px solid #aaa; background-color: #f0f0f0;")
        output_image_layout.addWidget(self.output_image_label)

        self.image_area_in_out_layout.addLayout(input_image_layout)
        self.image_area_in_out_layout.addLayout(output_image_layout)

        # Navigation and action
        navigation_widget_layout = QHBoxLayout()

        load_image_button = QPushButton("Load New Image", self)
        load_image_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 10px; text-align: center; text-decoration: none; display: inline-block; font-size: 14px; margin: 4px 2px; cursor: pointer;")
        load_image_button.clicked.connect(self.open_file)

        col_input_layout = QVBoxLayout()
        self.col_remove_input = QLineEdit(self)
        self.col_remove_input.setPlaceholderText("Enter cols to remove")
        col_input_layout.addWidget(self.col_remove_input)

        row_input_layout = QVBoxLayout()
        self.row_remove_input = QLineEdit(self)
        self.row_remove_input.setPlaceholderText("Enter rows to remove")
        row_input_layout.addWidget(self.row_remove_input)

        algorithm_label = QLabel("Choose Algorithm")
        self.algorithm_combo = QComboBox(self)
        self.algorithm_combo.addItem("Forward")
        self.algorithm_combo.addItem("Backward")
        self.algorithm_combo.setStyleSheet("QComboBox {background-color: #f0f0f0; font-size: 14px; border: 1px solid #aaa; padding: 3px;}")

        self.operation_combo = QComboBox(self)
        self.operation_combo.addItem("Reduce")
        self.operation_combo.addItem("Enlarge")
        self.operation_combo.setStyleSheet("QComboBox {background-color: #f0f0f0; font-size: 14px; border: 1px solid #aaa; padding: 3px;}")

        navigation_widget_layout.addWidget(load_image_button)
        navigation_widget_layout.addLayout(row_input_layout)
        navigation_widget_layout.addLayout(col_input_layout)
        navigation_widget_layout.addWidget(algorithm_label)
        navigation_widget_layout.addWidget(self.algorithm_combo)
        navigation_widget_layout.addWidget(self.operation_combo)

        action_1_button = QPushButton("Resize", self)
        action_1_button.setStyleSheet("background-color: #008CBA; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;")
        action_1_button.clicked.connect(self.action1)

        navigation_widget_layout.addWidget(action_1_button)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(120)
        self.scroll_area.setContentsMargins(0, 0, 0, 0)
        self.scroll_content = QWidget(self)
        self.scroll_layout = QHBoxLayout(self.scroll_content)

        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)

        self.image_labels = []

        self.layout.addLayout(self.image_area_in_out_layout)
        self.layout.addLayout(navigation_widget_layout)
        self.layout.addWidget(self.scroll_area)


    def open_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.jpg *.png *.bmp *.jpeg *.gif)")
        if file_path:
            self.current_image_path = file_path
            if file_path:
                image_pixmap = QPixmap(file_path)
                image_pixmap.save(os.path.join(os.path.join(os.getcwd(), 'Input'), os.path.basename(file_path)))
            # save
            self.display_image(self.current_image_path)
            self.input_image_label.pixmap().save(self.current_image_path)
            self.load_images_to_scroll()

    def algorithm_changed(self, index):
        selected_algorithm = self.algorithm_combo.currentText()
        print(f"Selected Algorithm: {selected_algorithm}")

    def operation_changed(self, index):
        selected_operation = self.operation_combo.currentText()
        print(f"Selected Operation: {selected_operation}")


    def action1(self):

        #load image as a pixelMatrix and put into current_image_path
        pixmap = QPixmap(self.current_image_path)

        original_image = pixmap.toImage()

        # Get the number of rows to remove from the input
        rows_input = self.row_remove_input.text()
        if rows_input.isdigit():
            self.row_remove = int(rows_input)

        cols_input = self.col_remove_input.text()
        if cols_input.isdigit():
            self.col_remove = int(cols_input)

        selected_algorithm = self.algorithm_combo.currentText()
        is_forward_energy_flag = True if selected_algorithm == "Forward" else False

        selected_operation = self.operation_combo.currentText()

        if selected_operation == "Enlarge":
            resized_image = enlarge_algo(original_image, self.row_remove, self.col_remove)
        else:
            resized_image = algo1(original_image, self.row_remove, self.col_remove, is_forward_energy_flag)

        # Convert the QImage to a QPixmap for display
        resized_pixmap = QPixmap.fromImage(resized_image)
        print("Image Received")
        # Save the resized image
        output_path = os.path.join(os.getcwd(), 'Output', os.path.basename(self.current_image_path))
        resized_pixmap.save(output_path)
        print("Image saved")

        # Display the resized image
        self.display_image(output_path, output=True)
        print("Image displayed")

    def display_image(self, file_path, output=False):
            pixmap = QPixmap(file_path)

            if output == False:
                # Display the input image in the input QLabel
                self.input_image_label.setPixmap(pixmap)
                self.input_image_label.setAlignment(Qt.AlignCenter)
                self.input_image_label.setMaximumSize(700, 700)
                self.input_image_label.setMinimumSize(0, 0)
                self.input_image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            else:
                # Display the output image in the output QLabel
                # output_pixmap = scaled_pixmap.scaledToHeight(700, Qt.SmoothTransformation)
                self.output_image_label.setPixmap(pixmap)
                self.output_image_label.setAlignment(Qt.AlignCenter)
                self.output_image_label.setMaximumSize(700, 700)
                self.output_image_label.setMinimumSize(0, 0)
                self.output_image_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def load_images_to_scroll(self):
        directory = os.getcwd() + "\\Input"

        for label in self.image_labels:
            label.deleteLater()
        self.image_labels = []

        image_paths = [file for file in sorted(os.listdir(directory)) if
                       file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        for image_path in image_paths:
            # Handle left mouse button click here
            label = QLabel(self)
            img = QImage(os.path.join(directory, image_path))
            img = img.scaledToHeight(100, mode=0x00)
            pixmap = QPixmap.fromImage(img)
            label.setPixmap(pixmap)
            label.mousePressEvent = lambda event, path=image_path: self.labelClicked(event, path)

            self.image_labels.append(label)
            self.scroll_layout.addWidget(label)

    def labelClicked(self, event, path):
        if event.button() == Qt.LeftButton:
            self.current_image_path = os.path.join(os.path.join(os.getcwd(), 'Input'), os.path.basename(path))
            self.display_image(self.current_image_path)
            self.load_images_to_scroll()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    gallery_viewer = ImageGalleryViewer()
    gallery_viewer.show()
    gallery_viewer.load_images_to_scroll()
    sys.exit(app.exec_())




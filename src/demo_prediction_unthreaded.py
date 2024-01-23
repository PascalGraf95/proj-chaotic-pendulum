import os.path
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal
from PyQt5.QtGui import QMovie, QFont
from timeseries_forecasting.animation.generate_animation import GenerateAnimation
from camera_data_processing import run_data_acquisition
import shutil

abs_dir = os.path.abspath(os.path.dirname(__file__))

input_sequence = 50
output_sequence = 200
sequence = input_sequence + output_sequence


class ChaoticPendulumPredictionApp(QMainWindow):
    def __init__(self, data_length: int = sequence):
        super(ChaoticPendulumPredictionApp, self).__init__()
        self.worker_thread = None
        self.gif_label = None
        self.status_label = None
        self.prediction_button = None
        self.setWindowTitle("Chaotisches Pendel Vorhersage")
        self.setGeometry(100, 100, 800, 600)
        self.data_length = data_length
        self.work_dir = os.path.join(abs_dir, "work_dir")
        self.pendulum_data = os.path.join(self.work_dir, "data.csv")
        self.real_images_path = os.path.join(self.work_dir, "real_images")
        self.gif_path = os.path.join(self.work_dir, "output.gif")
        self.prediction_path = os.path.join(self.work_dir, "prediction_data")

        self.init_workdir()
        self.init_ui()

    def init_ui(self):
        self.create_instruction_label()
        self.create_prediction_button()
        self.create_status_label()
        self.create_gif_label()

    def init_workdir(self):
        """Initialize the work directory, removing existing content if necessary."""
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir)
        os.makedirs(self.prediction_path)
        os.makedirs(self.real_images_path)

    def create_instruction_label(self):
        instruction_label = QLabel(
            "Um die Vorhersage zu starten, bringe das Pendel bitte in eine möglichst hohe Position und drücke 'Vorhersage starten':",
            self)

        font = QFont()
        font.setPointSize(16)
        instruction_label.setFont(font)

        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setGeometry(0, 50, 2000, 50)

    def create_prediction_button(self):
        self.prediction_button = QPushButton("Vorhersage starten", self)
        button_font = QFont()
        button_font.setPointSize(14)
        self.prediction_button.setFont(button_font)
        self.prediction_button.clicked.connect(self.start_prediction)
        self.prediction_button.setGeometry(850, 150, 200, 50)

    def create_status_label(self):
        self.status_label = QLabel("", self)
        status_font = QFont()
        status_font.setPointSize(14)
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setGeometry(600, 250, 700, 50)

    def create_gif_label(self):
        self.gif_label = QLabel(self)
        self.gif_label.setGeometry(600, 350, 1400, 700)

    def start_prediction(self):
        self.status_label.setText("Vorhersage gestartet")
        QCoreApplication.processEvents()
        run_data_acquisition(pendulum_data=self.pendulum_data, data_length=self.data_length, image_path=self.real_images_path, video_path=None)
        generate_animation = GenerateAnimation(pendulum_path=self.pendulum_data,
                                               prediction_path=self.prediction_path,
                                               gif_path=self.gif_path)
        generate_animation.main()

        self.status_label.setText("Fertig!")
        self.set_gif()

    def set_gif(self):
        movie = QMovie(self.gif_path)

        # Set the scaled size to crop the gif
        movie.setScaledSize(self.gif_label.size())

        self.gif_label.setMovie(movie)
        self.gif_label.setGeometry(250, 300, 1400, 700)  # Adjust the size of the label for cropping
        movie.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChaoticPendulumPredictionApp()
    window.move(QApplication.desktop().screen().rect().center() - window.rect().center())
    window.showMaximized()
    sys.exit(app.exec_())

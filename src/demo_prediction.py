import os.path
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal
from PyQt5.QtGui import QMovie, QFont
from timeseries_forecasting.animation.generate_animation import GenerateAnimation
from camera_data_processing import run_data_acquisition
import shutil

abs_dir = os.path.abspath(os.path.dirname(__file__))


class WorkerThread(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, pendulum_data, prediction_path, gif_path, output_length, sequence_length, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.pendulum_data = pendulum_data
        self.prediction_path = prediction_path
        self.gif_path = gif_path
        self.output_length = output_length
        self.sequence_length = sequence_length

    def run(self):
        generate_animation = GenerateAnimation(pendulum_path=self.pendulum_data,
                                               prediction_path=self.prediction_path,
                                               gif_path=self.gif_path,
                                               output_length=self.output_length,
                                               sequence_length=self.sequence_length)
        generate_animation.main()
        self.finished_signal.emit()


class ChaoticPendulumPredictionApp(QMainWindow):
    def __init__(self, input_sequence: int = 50, output_sequence: int = 200):
        super(ChaoticPendulumPredictionApp, self).__init__()
        self.worker_thread = None
        self.gif_label = None
        self.status_label = None
        self.prediction_button = None
        self.setWindowTitle("Chaotisches Pendel Vorhersage")
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence
        self.data_length = self.input_sequence + self.output_sequence
        self.setGeometry(100, 100, 800, 600)
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
        # if os.path.isdir(self.work_dir):
        #     shutil.rmtree(self.work_dir)
        # os.makedirs(self.work_dir)
        # os.makedirs(self.prediction_path)
        # os.makedirs(self.real_images_path)

    def create_instruction_label(self):
        instruction_label = QLabel(
            "Um die Vorhersage zu starten, bringe das Pendel bitte in eine möglichst hohe Position und drücke 'Vorhersage starten':",
            self)

        font = QFont()
        font.setPointSize(16)
        instruction_label.setFont(font)

        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setGeometry(0, 40, 2000, 50)

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
        self.status_label.setGeometry(450, 250, 1000, 50)

    def create_gif_label(self):
        self.gif_label = QLabel(self)
        self.gif_label.setGeometry(600, 350, 1400, 700)

    def start_prediction(self):
        self.status_label.setText("Zeitdaten werden eingelesen")
        QCoreApplication.processEvents()
        # run_data_acquisition(pendulum_data=self.pendulum_data, data_length=self.data_length,
        #                      image_path=self.real_images_path, video_path=None)
        self.status_label.setText("Vorhersage und Animation wird generiert")
        QCoreApplication.processEvents()

        self.worker_thread = WorkerThread(pendulum_data=self.pendulum_data,
                                          prediction_path=self.prediction_path,
                                          gif_path=self.gif_path,
                                          output_length=self.output_sequence,
                                          sequence_length=self.input_sequence)
        self.worker_thread.finished_signal.connect(self.on_animation_finished)
        self.worker_thread.start()

    def on_animation_finished(self):
        self.status_label.setText(
            "Auf der linken Seite ist die reale Bewegung zu sehen, auf der rechten Seite die vorhergesagte.")
        QCoreApplication.processEvents()
        self.set_gif()

    def set_gif(self):
        movie = QMovie(self.gif_path)

        # Adjust the size of the movie
        scaled_size = self.gif_label.size() * 0.6
        movie.setScaledSize(scaled_size)

        self.gif_label.setMovie(movie)

        # Adjust the geometry of the label
        self.gif_label.setGeometry(550, 300, 800, 560)

        movie.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChaoticPendulumPredictionApp()
    window.move(QApplication.desktop().screen().rect().center() - window.rect().center())
    window.showMaximized()
    sys.exit(app.exec_())

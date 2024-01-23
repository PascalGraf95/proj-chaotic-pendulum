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


class WorkerThread(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.gif_path = None
        self.prediction_path = None
        self.data_length = None
        self.pendulum_data = None

    def run(self):
        """
        Run the GenerateAnimation process in a separate thread.
        Emits the finished_signal when the process is complete.
        """
        run_data_acquisition(pendulum_data=self.pendulum_data, data_length=self.data_length, video_path=None)
        generate_animation = GenerateAnimation(pendulum_path=self.pendulum_data,
                                               prediction_path=self.prediction_path,
                                               gif_path=self.gif_path)
        generate_animation.main()
        self.finished_signal.emit()


class ChaoticPendulumPredictionApp(QMainWindow):
    def __init__(self, data_length: int = sequence):
        """
        Initialize the ChaoticPendulumPredictionApp.

        Parameters
        ----------
        data_length : int, optional
            Number of data points to be collected, by default 350.
        """
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
        self.gif_path = os.path.join(self.work_dir, "output.gif")
        self.prediction_path = os.path.join(self.work_dir, "prediction_data")

        self.init_workdir()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface components."""
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

    def create_instruction_label(self):
        """Create and set up the instruction label."""
        instruction_label = QLabel(
            "Um die Vorhersage zu starten, bringe das Pendel bitte in eine möglichst hohe Position und drücke 'Vorhersage starten':",
            self)

        font = QFont()
        font.setPointSize(16)
        instruction_label.setFont(font)

        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setGeometry(0, 50, 2000, 50)

    def create_prediction_button(self):
        """Create and set up the prediction button."""
        self.prediction_button = QPushButton("Vorhersage starten", self)
        button_font = QFont()
        button_font.setPointSize(14)
        self.prediction_button.setFont(button_font)
        self.prediction_button.clicked.connect(self.start_prediction)
        self.prediction_button.setGeometry(850, 150, 200, 50)

    def create_status_label(self):
        """Create and set up the status label."""
        self.status_label = QLabel("", self)
        status_font = QFont()
        status_font.setPointSize(14)
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setGeometry(600, 250, 700, 50)

    def create_gif_label(self):
        """Create and set up the GIF label."""
        self.gif_label = QLabel(self)
        self.gif_label.setGeometry(600, 350, 1400, 700)

    def start_prediction(self):
        """Start the prediction process."""
        if self.worker_thread is not None and self.worker_thread.isRunning():
            # If the thread is still running, wait for it to finish
            self.status_label.setText("Warte auf Fertigstellung...")
            return

        # Clear the GIF label
        self.clear_gif_label()

        self.status_label.setText("Vorhersage gestartet")
        QCoreApplication.processEvents()

        self.worker_thread = WorkerThread()
        self.worker_thread.pendulum_data = self.pendulum_data
        self.worker_thread.gif_path = self.gif_path
        self.worker_thread.data_length = self.data_length
        self.worker_thread.prediction_path = self.prediction_path
        self.worker_thread.finished_signal.connect(self.update_gui)
        self.worker_thread.start()

    def update_gui(self):
        """Update the GUI after the prediction process is complete."""
        self.status_label.setText("Fertig!")
        self.set_gif()

    def set_gif(self):
        """Set up the GIF label to display the generated animation."""
        movie = QMovie(self.gif_path)
        movie.setScaledSize(self.gif_label.size())
        self.gif_label.setMovie(movie)
        self.gif_label.setGeometry(250, 300, 1400, 700)
        movie.start()

    def clear_gif_label(self):
        """Clear the GIF label."""
        self.gif_label.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChaoticPendulumPredictionApp()
    window.move(QApplication.desktop().screen().rect().center() - window.rect().center())
    window.showMaximized()
    sys.exit(app.exec_())

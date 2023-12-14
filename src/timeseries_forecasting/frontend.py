import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5.QtGui import QMovie, QFont
from animation.plot_predicted_trajectories import generate_animation_parallel


class ChaoticPendulumPredictionApp(QMainWindow):
    def __init__(self, gif_path, pendulum_data):
        super(ChaoticPendulumPredictionApp, self).__init__()
        self.timer = None
        self.gif_label = None
        self.status_label = None
        self.prediction_button = None
        self.gif_path: str = gif_path
        self.pendulum_data: str = pendulum_data
        self.setWindowTitle("Chaotisches Pendel Vorhersage")
        self.setGeometry(100, 100, 800, 600)

        self.init_ui()

    def init_ui(self):
        self.create_instruction_label()
        self.create_prediction_button()
        self.create_status_label()
        self.create_gif_label()

    def create_timer(self):
        self.timer = QTimer(self)

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
        generate_animation_parallel(data_path=self.pendulum_data)
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
    window = ChaoticPendulumPredictionApp(gif_path=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\animation\prediction_output\output.gif",
                                          pendulum_data=r"C:\Users\Marco\dev\git\proj-chaotic-pendulum\src\timeseries_forecasting\data\processed\12.csv")
    window.move(QApplication.desktop().screen().rect().center() - window.rect().center())
    window.showMaximized()
    sys.exit(app.exec_())

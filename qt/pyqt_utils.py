import time
import logging
from PyQt5.QtCore import QFile, QTextStream, QThread, pyqtSignal


def use_theme(app, path):
    file = QFile(path)
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


def set_logger(plain_text_edit):
    logTextBox = QTextEditLogger(plain_text_edit)
    # You can format what is printed to text box
    logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(logTextBox)
    # You can control the logging level
    logging.getLogger().setLevel(logging.DEBUG)

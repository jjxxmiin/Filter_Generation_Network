import os
import sys
import logging
import torchvision.transforms as transforms
from src.benchmark import get_flops
from src.models.vgg import load_model, get_layer_info
from src.search import Search
from PyQt5.QtCore import QFile, QTextStream, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic


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


class Main_Form(QDialog, QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.isModel = False
        self.isPath = False
        self.data_path = None
        self.model = None
        self.ui = uic.loadUi("./ui/main.ui", self)
        self.initUI()

    def initUI(self):
        """
        Init Event
        """
        set_logger(self.plainTextEdit)

        self.ui.setWindowTitle('Pruner')
        self.ui.show()

        self.model_btn.clicked.connect(self.set_model)
        self.data_btn.clicked.connect(self.set_dataset)
        self.start_btn.clicked.connect(self.start)

    def set_model(self):
        """
        Init Model
        Init Ori Benchmark
        """
        model_path, i = QFileDialog.getOpenFileName(self, 'Open file', '.', "Model files (*.pth)")

        if not model_path:
            logging.info("\nNot Selected Model")
            self.isModel = False

        else:
            logging.info("\nInput Model")
            self.isModel = True
            self.model_label.setText(model_path.split("/")[-1])

            # TODO : Dynamic for Any Model
            self.model = load_model(model_path,
                                    type='VGG16',
                                    mode='eval')

            self.set_ori_benchmark()
            self.set_ori_layer_view()

    def set_dataset(self):
        """
        Init Dataset Path
        Init Dataset Class List
        """
        self.data_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if not self.data_path:
            logging.info("\nNot Selected Dataset Path")
            self.isPath = False

        else:
            logging.info("\nInput Dataset Path")
            self.isPath = True

            self.data_label.setText(self.data_path.split("/")[-1])
            self.set_label_view()

    def set_label_view(self):
        self.label_list.clear()

        for label in os.listdir(self.data_path):
            item = QListWidgetItem(label)
            item.setCheckState(Qt.Unchecked)  # Unchecked
            item.setText(f'{str(label)}')
            self.label_list.addItem(item)  # listWidgetPDFlist

    def set_ori_layer_view(self):
        self.ori_layer_list.clear()

        layer = get_layer_info(self.model)

        for l in layer:
            item = QListWidgetItem(l)
            item.setText(f'{str(l)}')
            self.ori_layer_list.addItem(item)  # listWidgetPDFlist

    def set_prune_layer_view(self):
        self.prune_layer_list.clear()

        layer = get_layer_info(self.model)

        for l in layer:
            item = QListWidgetItem(l)
            item.setText(f'{str(l)}')
            self.prune_layer_list.addItem(item)  # listWidgetPDFlist

    def set_ori_benchmark(self):
        flops, params = get_flops(self.model)

        self.ori_flops_label.setText(str(flops))
        self.ori_params_label.setText(str(params))

    def set_prune_benchmark(self):
        flops, params = get_flops(self.model)

        self.prune_flops_label.setText(str(flops))
        self.prune_params_label.setText(str(params))

    def get_check_label(self):
        check_cls = []
        uncheck_cls = []

        for index in range(self.label_list.count()):
            if self.label_list.item(index).checkState() == Qt.Checked:
                label = self.label_list.item(index).text()
                check_cls.append(label)
                logging.info(f"checked label : {label}")
            else:
                label = self.label_list.item(index).text()
                uncheck_cls.append(label)
                logging.info(f"unchecked label : {label}")

        return check_cls, uncheck_cls

    def start(self):
        # if self.isModel and self.isPath:
        if self.isPath:
            # get check class
            check_cls, uncheck_cls = self.get_check_label()

            # TODO : Dynamic for pytorch transforms
            transformer = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010))])

            search = Search(self.model,
                            self.data_path,
                            check_cls,
                            transformer=transformer)

        else:
            logging.error(f"Input Model or Path")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    use_theme(app, "theme/style.qss")

    w = Main_Form()
    sys.exit(app.exec())

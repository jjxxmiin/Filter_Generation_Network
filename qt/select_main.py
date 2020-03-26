import os
import sys
from src.models.vgg import load_model
from qt.pyqt_utils import *

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import uic
from qt.status_main import Status_Form


class Selc_Form(QDialog):
    def __init__(self, parent=None):
        super().__init__()

        self.isModel = False
        self.isPath = False

        self.data_path = None
        self.model = None

        self.check_cls = None
        self.check_idx = None
        self.uncheck_cls = []

        self.ui = uic.loadUi("./ui/Selec_Form.ui", self)
        self.initUI()

    def initUI(self):
        """
        Init Event
        """

        self.ui.setWindowTitle('Prune Select')
        self.ui.show()

        self.model_btn.clicked.connect(self.set_model)
        self.data_btn.clicked.connect(self.set_dataset)
        self.open_btn.clicked.connect(self.open)

    def set_model(self):
        model_path, i = QFileDialog.getOpenFileName(self, 'Open file', '.', "Model files (*.pth)")

        if not model_path:
            self.isModel = False

        else:
            self.isModel = True
            self.model_label.setText(model_path.split("/")[-1])

            # TODO : Dynamic for Any Model
            self.model = load_model(model_path,
                                    type='VGG16',
                                    mode='eval')

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

        for label in os.listdir(os.path.join(self.data_path, 'train')):
            item = QListWidgetItem(label)
            item.setCheckState(Qt.Unchecked)  # Unchecked
            item.setText(label)
            self.label_list.addItem(item)  # listWidgetPDFlist

    def get_check_label(self):
        for index in range(self.label_list.count()):
            if self.label_list.item(index).checkState() == Qt.Checked:
                label = self.label_list.item(index).text()
                self.check_cls = label
                self.check_idx = index
                logging.info(f"checked label : {label}")
            else:
                label = self.label_list.item(index).text()
                self.uncheck_cls.append(label)
                logging.info(f"unchecked label : {label}")

    def open(self):
        self.get_check_label()
        self.hide()
        status_view = Status_Form(self)
        status_view.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    use_theme(app, "theme/style.qss")

    w = Selc_Form()
    sys.exit(app.exec())

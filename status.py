import os
import sys
import torchvision.transforms as transforms

from src.benchmark import get_flops
from src.models.vgg import load_model, get_layer_info
from src.prune import *
from src.loader import get_train_test_loader
from src.search import Search
from src.pyqt_utils import *

from PyQt5.QtCore import Qt, QObject
from PyQt5.QtWidgets import *
from PyQt5 import uic


class Update_List_View(QThread):
    update_view = pyqtSignal(str)

    def __init__(self, model):
        QThread.__init__(self)
        self.model = model

    def run(self):
        self.update_view.emit("1")

class Status_Form(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = uic.loadUi("./ui/Status_Form.ui", self)
        self.selc = self.parent()

        self.initUI()

    def initUI(self):
        """
        Init Event
        """
        self.ui.setWindowTitle('Prune Select')
        self.ui.show()

        self.set_ori_layer_view()

        ul = Update_List_View(self.selc.model)
        ul.update_view.connect(self.ori_flops_label.setText)
        ul.start()

    def set_ori_layer_view(self):
        self.ori_layer_list.clear()

        layer = get_layer_info(self.selc.model)

        for l in layer:
            item = QListWidgetItem(l)
            item.setText(l)
            self.ori_layer_list.addItem(item)  # listWidgetPDFlist

    def set_ori_benchmark(self):
        flops, params = get_flops(self.model)

        self.ori_flops_label.setText(str(flops))
        self.ori_params_label.setText(str(params))


if __name__ == '__main__':
    app = QApplication(sys.argv)

    use_theme(app, "theme/style.qss")

    w = Status_Form()
    sys.exit(app.exec())

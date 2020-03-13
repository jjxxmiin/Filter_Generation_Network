import sys
from src.models.vgg import get_layer_info
from src.pyqt_utils import *

from PyQt5.QtWidgets import *
from PyQt5 import uic


class Status_Form(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = uic.loadUi("./ui/Status_Form.ui", self)
        self.root = self.parent()

        self.initUI()

    def initUI(self):
        """
        Init Event
        """
        self.ui.setWindowTitle('Prune Select')
        self.ui.show()

        self.set_ori_layer_view()
        self.set_ori_benchmark()

    def set_ori_layer_view(self):
        self.ori_layer_list.clear()

        layer = get_layer_info(self.root.model)

        for l in layer:
            item = QListWidgetItem(l)
            item.setText(l)
            self.ori_layer_list.addItem(item)  # listWidgetPDFlist


if __name__ == '__main__':
    app = QApplication(sys.argv)

    use_theme(app, "theme/style.qss")

    w = Status_Form()
    sys.exit(app.exec())

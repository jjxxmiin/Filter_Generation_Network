import os
import sys
import torchvision.transforms as transforms

from src.benchmark import get_flops
from src.models.vgg import load_model, get_layer_info
from src.prune import *
from src.loader import get_train_test_loader
from src.search import Search
from qt.pyqt_utils import *

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import uic


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

            self.set_ori_layer_view()
            self.set_ori_benchmark()

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

    def set_ori_layer_view(self):
        self.ori_layer_list.clear()

        layer = get_layer_info(self.model)

        for l in layer:
            item = QListWidgetItem(l)
            item.setText(l)
            self.ori_layer_list.addItem(item)  # listWidgetPDFlist

    def set_prune_layer_view(self):
        self.prune_layer_list.clear()

        layer = get_layer_info(self.model)

        for l in layer:
            item = QListWidgetItem(l)
            item.setText(l)
            self.prune_layer_list.addItem(item)  # listWidgetPDFlist

    def set_ori_benchmark(self):
        flops, params = get_flops(self.model)

        self.ori_flops_label.setText(str(flops))
        self.ori_params_label.setText(str(params))

    def set_prune_benchmark(self):
        flops, params = get_flops(self.model)

        self.prune_flops_label.setText(str(flops))
        self.prune_params_label.setText(str(params))
        print(f"FLOPs : {flops} / Params : {params}")

    def get_check_label(self):
        check_cls = None
        check_idx = None
        uncheck_cls = []

        for index in range(self.label_list.count()):
            if self.label_list.item(index).checkState() == Qt.Checked:
                label = self.label_list.item(index).text()
                check_cls = label
                check_idx = index
                logging.info(f"checked label : {label}")
            else:
                label = self.label_list.item(index).text()
                uncheck_cls.append(label)
                logging.info(f"unchecked label : {label}")

        return check_cls, uncheck_cls, check_idx

    def search_prune(self, check_cls, transformer, stage=1):
        if stage == 1:
            search = Search(self.model, self.data_path, check_cls, transformer=transformer, prog=self.search_bar)
        else:
            search = Search(self.model, self.data_path, check_cls, transformer=transformer, prog=self.search_bar_2)

        filters = search.get_filter_idx()
        self.model = prune(self.model, filters)
        self.set_prune_benchmark()

    def start(self):
        batch_size = 32
        lr = 0.001

        if self.isModel and self.isPath:
            # get check class
            check_cls, uncheck_cls, check_idx = self.get_check_label()

            # TODO : Dynamic for pytorch transforms
            train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.RandomCrop(size=(32, 32), padding=4),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010))])

            test_transformer = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2010))])

            train_loader, test_loader = get_train_test_loader(self.data_path,
                                                              batch_size=batch_size,
                                                              train_transformer=train_transformer,
                                                              test_transformer=test_transformer)

            # 1 stage
            logging.info("Start 1 Stage")

            self.search_prune(check_cls, transformer=test_transformer, stage=1)

            for _ in range(0, 10):
                self.model, train_acc = train(self.model, train_loader, batch_size, lr, self.finetune_bar)
                self.train_acc_label.setText(str(train_acc))

                test_acc = test(self.model, train_loader, batch_size, self.test_bar)
                self.test_acc_label.setText(str(test_acc))

            logging.info("Convert Multi -> Binary")
            self.model = to_binary(self.model, check_idx)

            for _ in range(0, 5):
                self.search_prune(check_cls, transformer=test_transformer, stage=2)

                binary_train_loader, binary_test_loader = get_train_test_loader(self.data_path,
                                                                                batch_size=batch_size,
                                                                                train_transformer=train_transformer,
                                                                                test_transformer=test_transformer,
                                                                                true_name=check_cls)

                for _ in range(0, 5):
                    self.model, train_acc = binary_sigmoid_train(self.model, binary_train_loader,
                                                                 batch_size, lr,
                                                                 self.finetune_bar_2)
                    self.train_acc_label_2.setText(str(train_acc))
                    test_acc = binary_sigmoid_test(self.model, binary_test_loader, batch_size, self.test_bar_2)
                    self.test_acc_label_2.setText(str(test_acc))

        else:
            logging.error(f"Input Model or Path")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    use_theme(app, "theme/style.qss")

    w = Main_Form()
    sys.exit(app.exec())

"""
This file is part of Fish Tracker.
Copyright 2021, VTT Technical research centre of Finland Ltd.
Developed by: Mikael Uimonen.

Fish Tracker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Fish Tracker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Fish Tracker.  If not, see <https://www.gnu.org/licenses/>.
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from dropdown_delegate import DropdownDelegate
from detector import Detector
from log_object import LogObject

class DetectionDataModel(QtCore.QAbstractTableModel):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.detector.data_changed_signal.connect(self.checkLayout)
        self.row_count = 0

    def rowCount(self, index=None):
        dets = self.detector.getCurrentDetection()
        if dets is None:
            self.row_count = 0
        else:
            self.row_count = len(dets)

        return self.row_count

    def columnCount(self, index=None):
        return 4

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            dets = self.detector.getCurrentDetection()
            if dets is None:
                return 0

            d = dets[row]
            if d.center is None:
                return 0

            if col == 0:
                return round(d.distance, 1)
            elif col == 1:
                return round(d.angle, 1)
            elif col == 2:
                return round(d.length, 3)
            elif col == 3:
                return round(d.aspect, 1)
            else:
                return ""

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return "Distance (m)"
                elif section == 1:
                    return "Angle (°)"
                elif section == 2:
                    return "Length (m)"
                elif section == 3:
                    return "Aspect (°)"
            else:
                return '{: >4}'.format(section)

    def checkLayout(self, count):
        if self.row_count != count:
            self.layoutChanged.emit()
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled


class DetectionList(QtWidgets.QWidget):
    def __init__(self, data_model):
        super().__init__()
        self.data_model = data_model

        #self.scroll = QtWidgets.QScrollArea(self)
        #self.scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.table = QtWidgets.QTableView(self)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.table.setModel(data_model)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, QtCore.Qt.AscendingOrder);
        self.table.setStyleSheet("QTableView\n"
                                 "{\n"
                                 "border: none;\n"
                                 "}\n"
                                 "")

        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.vertical_layout.setObjectName("verticalLayout")
        self.vertical_layout.addWidget(self.table)
        self.vertical_layout.setSpacing(0)
        self.vertical_layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.vertical_layout)
            



if __name__ == "__main__":
    import sys
    from playback_manager import PlaybackManager
    from fish_manager import FishManager

    def startDetector(playback_manager, detector):
        detector.initMOG()
        detector.setShowDetections(True)
        playback_manager.play()

    def handleFrame(detector, tuple):
        ind, frame = tuple
        detector.compute(ind, frame)

    def defaultTest():
        app = QtWidgets.QApplication(sys.argv)
        main_window = QtWidgets.QMainWindow()
        playback_manager = PlaybackManager(app, main_window)
        playback_manager.fps = 1
        playback_manager.openTestFile()

        detector = Detector(playback_manager)
        detector.mog_parameters.nof_bg_frames = 100

        playback_manager.mapping_done.connect(lambda: startDetector(playback_manager, detector))
        playback_manager.frame_available.connect(lambda t: handleFrame(detector, t))

        data_model = DetectionDataModel(detector)
        detection_list = DetectionList(data_model)
        main_window.setCentralWidget(detection_list)
        main_window.show()
        sys.exit(app.exec_())

    def loadTest():
        app = QtWidgets.QApplication(sys.argv)
        main_window = QtWidgets.QMainWindow()
        playback_manager = PlaybackManager(app, main_window)
        playback_manager.fps = 1

        detector = Detector(playback_manager)
        detector.mog_parameters.nof_bg_frames = 100

        file = playback_manager.selectLoadFile()

        playback_manager.openTestFile()
        playback_manager.mapping_done.connect(lambda: startDetector(playback_manager, detector))

        detector.loadDetectionsFromFile(file)
        LogObject().print([d for d in detector.detections if d is not None])
       

        data_model = DetectionDataModel(detector)
        detection_list = DetectionList(data_model)
        main_window.setCentralWidget(detection_list)
        main_window.show()
        sys.exit(app.exec_())

    #defaultTest()
    loadTest()
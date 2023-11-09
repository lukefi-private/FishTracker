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

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt

import numpy as np
import cv2
import seaborn as sns
from bisect import insort
from enum import IntEnum
import file_handler as fh
from tracker import Tracker
from tracker_parameters import TrackerParameters
from log_object import LogObject
from filter_parameters import FilterParameters

fish_headers = ["", "ID", "Length", "Direction", "Frame in", "Frame out", "Duration", "Detections", "MAD",
                "Tortuosity", "Speed"]

fish_sort_keys = [lambda f: f.color_ind, lambda f: f.id, lambda f: -f.length, lambda f: f.dirSortValue(),
                  lambda f: f.frame_in, lambda f: f.frame_out, lambda f: f.duration, lambda f: f.detection_count,
                  lambda f: f.mad, lambda f: f.tortuosity, lambda f: f.speed]

data_lambda_list = [lambda f: f.color_ind, lambda f: f.id, lambda f: f.length, lambda f: f.direction.name,
                    lambda f: f.frame_in, lambda f: f.frame_out, lambda f: f.duration, lambda f: f.detection_count,
                    lambda f: f.mad, lambda f: f.tortuosity, lambda f: f.speed]
COLUMN_COUNT = 11

N_COLORS = 16
#color_palette = sns.color_palette('bright', N_COLORS)
color_palette = [[c[2], c[1], c[0]] for c in sns.color_palette('bright', N_COLORS)]
pyqt_palette = [QtGui.QColor.fromRgbF(c[2], c[1], c[0]) for c in color_palette]

#color_palette_deep = sns.color_palette('deep', N_COLORS)
color_palette_deep = [[c[2], c[1], c[0]] for c in sns.color_palette('deep', N_COLORS)]
pyqt_palette_deep = [QtGui.QColor.fromRgbF(c[2], c[1], c[0]) for c in color_palette_deep]

# Stores and manages tracked fish items.
# Items can be edited with the functions defined here through e.g. fish_list.py.
class FishManager(QtCore.QAbstractTableModel):
    updateContentsSignal = QtCore.pyqtSignal()
    updateSelectionSignal = QtCore.pyqtSignal(QtCore.QItemSelection, QtCore.QItemSelectionModel.SelectionFlags)

    def __init__(self, playback_manager, tracker):
        super().__init__()
        self.playback_manager = playback_manager
        if self.playback_manager is not None:
            self.playback_manager.file_opened.connect(self.onFileOpened)
            self.playback_manager.file_closed.connect(self.onFileClosed)

        self.tracker = tracker
        if tracker is not None:
            self.tracker.init_signal.connect(self.onTrackingInitialized)
            self.tracker.all_computed_signal.connect(self.updateDataFromTracker)

        # All fish items currently stored.
        self.all_fish = {}

        # Fish items that are currently displayed.
        self.fish_list = []

        self.selected_rows = set()

        # Index for fish_sort_keys array, that contains lambda functions to sort the currently shown array.
        # Default: ID
        self.sort_ind = 1

        # Sort direction, ascending or descending
        self.sort_order = QtCore.Qt.DescendingOrder

        # Min number of detections required for a fish to be included in fish_list
        self.min_detections = 2

        # Major axis distance, i.e. delta angle (degrees) between the first and the last associated detection
        self.mad_limit = 0

        # Percentile with which the shown length is determined
        self.length_percentile = 50

        # Clear previous fish entries when new data is acquired.
        self.clear_old_data = True

        # If fish (tracks) are shown.
        self.show_fish = True

        # If fish (tracks) are shown in Echogram.
        self.show_echogram_fish = True

        self.frame_rate = None
        self.frame_time = None
        self.update_fish_colors = False

        # Inverted upstream / downstream.
        self.up_down_inverted = False

    def testPopulate(self, frame_count):
        """
        Simple test function.
        """

        self.all_fish = {}
        self.fish_list.clear()
        for i in range(10):
            f = FishEntry(i + 1)
            f.length = round(np.random.normal(1.2, 0.1), 3)
            f.direction = SwimDirection(np.random.randint(low=0, high=2))
            f.frame_in = np.random.randint(frame_count)
            f.frame_out = min(f.frame_in + np.random.randint(100), frame_count)
            f.duration = f.frame_out - f.frame_in + 1
            f.mad = np.random.randint(30)
            f.tortuosity = np.random.uniform(1,2)
            self.all_fish[f.id] = f

        self.trimFishList()

    def refreshAllFishData(self):
        for f in self.all_fish.values():
            self.refreshData(f)

    def trimFishList(self, force_color_update=False):
        """
        Updates shown table (fish_list) from all instances containing dictionary (all_fish).
        fish_list is trimmed based on the minimum duration.
        """

        fl = [fish for fish in self.all_fish.values() if fish.checkConditions(self.min_detections, self.mad_limit)]

        reverse = self.sort_order != QtCore.Qt.AscendingOrder
        fl.sort(key=fish_sort_keys[self.sort_ind], reverse=reverse)

        len_new = len(fl)
        len_old = len(self.fish_list)

        if len_new > len_old:
            self.beginInsertRows(QtCore.QModelIndex(), len_old, max(0, len_new-1))
            self.fish_list = fl
            self.endInsertRows()
        elif len_new < len_old:
            self.beginRemoveRows(QtCore.QModelIndex(), max(0, len_new-1), max(0, len_old-1))
            self.fish_list = fl
            self.endRemoveRows()
        else:
            self.fish_list = fl

        if self.update_fish_colors or force_color_update:
            self.updateFishColors()
        self.refreshLayout()

    def onFileOpened(self):
        self.clear()
        self.frame_rate = self.playback_manager.getRecordFrameRate()
        self.frame_time = (1.0 / self.frame_rate) if self.frame_rate is not None else None

    def onFileClosed(self):
        self.clear()
        self.frame_rate = None
        self.frame_time = None

    def onTrackingInitialized(self, clearData):
        if clearData:
            self.clear()

    def clear(self):
        self.all_fish = {}
        self.trimFishList()

    def refreshLayout(self):
        self.layoutChanged.emit()
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())
        self.updateContentsSignal.emit()

    def data(self, index, role):
        """
        Return data for TableView based on row and column.
        Row: Fish 
        Column: Some parameter of the fish
        """
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()

            try:
                return data_lambda_list[col](self.fish_list[row])
            except IndexError:
                if row >= len(self.fish_list):
                    LogObject().print("Bad index {}/{}".format(row, len(self.fish_list) - 1))
                return QtCore.QVariant()
        else:
            return QtCore.QVariant()
    
    def rowCount(self, index=None):
        return len(self.fish_list)

    def columnCount(self, index=None):
        return COLUMN_COUNT;

    def allDirectionCounts(self):
        """
        Returns direction counts (Total, Up, Down, None) of all fish.
        """
        return self.directionCountsHelp(self.all_fish.values())

    def directionCounts(self):
        """
        Returns direction counts (Total, Up, Down, None) of currently displayed fish.
        """
        return self.directionCountsHelp(self.fish_list)

    def directionCountsHelp(self, f_list):
        total_count = 0
        up_count = 0
        down_count = 0
        none_count = 0
        for f in f_list:
            total_count += 1
            if f.direction == SwimDirection.UP:
                up_count += 1
            elif f.direction == SwimDirection.DOWN:
                down_count += 1
            else:
                none_count += 1
        return total_count, up_count, down_count, none_count

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return fish_headers[section]

    def sort(self, col, order=QtCore.Qt.AscendingOrder):
        #self.layoutAboutToBeChanged.emit()

        self.sort_ind = col
        self.sort_order = order

        reverse = order != QtCore.Qt.AscendingOrder
        self.fish_list.sort(key = fish_sort_keys[col], reverse = reverse)

        #self.layoutChanged.emit()
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def getShownFish(self, row):
        if row < len(self.fish_list):
            return self.fish_list[row]
        else:
            return None

    def addFish(self):
        """
        Manual addition of fish.
        Currently not supported. Manual fish detection from frames is required,
        i.e. user should be able to select fish location from SonarView for each frame.
        """
        f = FishEntry(self.getNewID())
        self.all_fish[f.id] = f
        self.trimFishList()

    def getNewID(self, ind=1):
        keys = self.all_fish.keys()
        while ind in keys:
            ind += 1
        return ind

    def removeFish(self, rows, update=True):
        if(len(rows) > 0):
            for row in sorted(rows, reverse=True):
                if row >= len(self.fish_list):
                    continue

                fish_id = self.fish_list[row].id
                try:
                    del_f = self.all_fish.pop(fish_id)
                    del del_f
                except KeyError:
                    LogObject().print("KeyError occured when removing entry with id:", fish_id)

            if update:
                self.trimFishList()

    def mergeFish(self, rows):
        if rows == None or len(rows) == 0:
            return

        sorted_rows = sorted(rows)
        new_fish = self.fish_list[sorted_rows[0]].copy()

        for i in range(1, len(sorted_rows)):
            row = sorted_rows[i]
            fish = self.fish_list[row]
            new_fish.merge(fish)

        self.refreshData(new_fish)
        self.removeFish(rows, False)
        self.all_fish[new_fish.id] = new_fish
        self.trimFishList()

    def splitFish(self, rows, frame):
        if rows == None or len(rows) == 0:
            return

        for row in sorted(rows):
            fish = self.fish_list[row]
            frame_inds = fish.tracks.keys()
            if fish.frame_in < frame and fish.frame_out > frame:
                id = self.getNewID(fish.id)
                new_fish = fish.split(frame, id)
                fish.forceLengthByPercentile(self.length_percentile)
                new_fish.forceLengthByPercentile(self.length_percentile)
                self.all_fish[id] = new_fish

                self.refreshData(fish)
                self.refreshData(new_fish)

        self.trimFishList()

    def clearMeasurements(self, rows):
        if rows == None or len(rows) == 0:
            return
        for row in rows:
            if row >= len(self.fish_list):
                    continue
            self.fish_list[row].forceLengthByPercentile(self.length_percentile)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def selectFromEchogram(self, frame_min, frame_max, height_min, height_max):
        """
        Finds fish that are within given frame and height limits and sends a signal
        to select the corresponding rows in table view.
        """
        new_selection = set()
        polar_transform = self.playback_manager.playback_thread.polar_transform
        min_d, max_d = self.playback_manager.getRadiusLimits()

        for ind, fish in enumerate(self.fish_list):
            # Skip fish outside the given range of frames
            if fish.frame_out < frame_min or fish.frame_in > frame_max:
                continue

            for frame, track in fish.tracks.items():
                if frame >= frame_min and frame <= frame_max:
                    track, det = fish.tracks[frame]
                    center = FishEntry.trackCenter(track)
                    distance, angle = polar_transform.cart2polMetric(center[0], center[1], True)

                    if distance >= height_min and distance <= height_max:
                        new_selection.add(ind)
                        break

        self.setSelection(new_selection)

    def setSelection(self, rows):
        """
        Creates a QItemSelection based on given rows (shown fish items)
        and emits updateSelectionSignal with the selection.
        """
        selection = QtCore.QItemSelection()
        for row in rows:
            ind_1 = self.index(row, 0)
            ind_2 = self.index(row, self.columnCount() - 1)
            range = QtCore.QItemSelectionRange(ind_1, ind_2)
            selection.append(range)

        self.updateSelectionSignal.emit(selection, QtCore.QItemSelectionModel.ClearAndSelect)

    def onSelectionChanged(self, selected):
        self.selected_rows = selected
        self.updateContentsSignal.emit()


    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        return Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled

    def setData(self, index, value, role):
        if index.isValid() and role == Qt.EditRole:
            col = index.column()
            row = index.row()
            fish = self.fish_list[row]

            if col == 1:
                id, success = intTryParse(value)
                if success:
                    if id not in self.all_fish:
                        self.all_fish[id] = self.all_fish.pop(fish.id)
                        fish.id = id
                        self.trimFishList()
                        return True
            elif col == 2:
                length, success = floatTryParse(value)
                if success:
                    fish.length = length
                    self.dataChanged.emit(index, index)
                    return True
            elif col == 3:
                try:
                    fish.direction = SwimDirection[value]
                    self.dataChanged.emit(index, index)
                    return True
                except KeyError:
                    pass

        return False

    def secondaryTrack(self, filter_parameters):
        """
        Applies filters to fish data and starts tracker's secondaryTrack process with used detections.
        Used detections are excluded from tracking.
        """
        self.clear_old_data = False

        min_dets = filter_parameters.getParameter(FilterParameters.ParametersEnum.min_duration)
        mad_limit = filter_parameters.getParameter(FilterParameters.ParametersEnum.mad_limit)
        LogObject().print1(f"Filter Parameters: {min_dets} {mad_limit}")

        used_dets = self.applyFiltersAndGetUsedDetections(min_dets, mad_limit)
        self.playback_manager.runInThread(lambda: self.tracker.secondaryTrack(used_dets, self.tracker.secondary_parameters))

    def updateDataFromTracker(self):
        """
        Iterates through the results of the tracker and updates the data in FishManager.
        If clear_old_data is set, the old data is first removed.
        """

        if self.clear_old_data:
            self.clear()

        self.clear_old_data = True

        # Iterate through all frames.
        for frame, tracks in self.tracker.tracks_by_frame.items():
            try:
                # Iterate through all tracks in a frame.
                for tr, det in tracks:
                    id = tr[4]
                    if id in self.all_fish:
                        f = self.all_fish[id]
                        f.addTrack(tr, det, frame)
                    else:
                        f = FishEntryFromTrack(tr, det, frame)
                        self.all_fish[id] = f
            except ValueError as e:
                print(tracks)
                raise e

        # Trim tails, i.e. remove last tracks with no corresponding detection.
        if self.tracker.parameters.getParameter(TrackerParameters.ParametersEnum.trim_tails):
            for id, fish in self.all_fish.items():
                fish.trimTail()

        # Refresh values
        for fish in self.all_fish.values():
            self.refreshData(fish)
            fish.setLengthByPercentile(self.length_percentile)

        self.printDirectionCounts()
        self.trimFishList(force_color_update=True)

    def applyFilters(self):
        """
        Applies the current filters by replacing the contents of all_fish
        with the contents of fish_list. fish_list is the "filtered version"
        of all_fish.
        """

        self.trimFishList()
        self.all_fish = {}
        for fish in self.fish_list:
            self.all_fish[fish.id] = fish

    def getDetectionsInFish(self):
        """
        Returns detections that have been associated with the current fish.
        """
        detections = {}
        for fish in self.all_fish.values():
            for frame, (_, det) in fish.tracks.items():
                if det is None:
                    continue

                if frame not in detections:
                    detections[frame] = [det]
                else:
                    detections[frame].append(det)

        return detections

    def applyFiltersAndGetUsedDetections(self, min_detections=None, mad_limit=None):
        """
        Applies filters and returns the detections associated to the remaining fish.
        If optional parameters are not given, the current values in FishManager are used.
        """
        temp_min_detections = self.min_detections
        temp_mad_limit = self.mad_limit

        if min_detections is not None:
            self.min_detections = min_detections

        if mad_limit is not None:
            self.mad_limit = mad_limit

        LogObject().print1(f"Fish before applying filters: {len(self.all_fish)}")
        self.applyFilters()
        LogObject().print1(f"Fish after applying filters: {len(self.all_fish)}")

        used_dets = self.getDetectionsInFish()
        count = 0
        for frame, dets in used_dets.items():
            count += len(dets)
        LogObject().print1(f"Total detections used in filtered results: {count}")

        self.min_detections = temp_min_detections
        self.mad_limit = temp_mad_limit

        return used_dets

    def refreshData(self, fish):
        """
        Refresh calculated variables of the given fish.
        """
        fish.setFrames()
        fish.setPathVariables(self.up_down_inverted, self.frame_time, 1.0/self.playback_manager.getPixelsPerMeter())
        fish.setLengths()

    def updateFishColors(self):
        color_ind = 0
        for id, fish in self.all_fish.items():
            fish.color_ind = color_ind
            color_ind = (color_ind + 1) % N_COLORS

    def isColor(self, index):
        return index.column() == 0

    def isDropdown(self, index):
        return index.column() == 3

    def dropdown_options(self):
        return [sd.name for sd in list(SwimDirection)]

    def getDropdownIndex(self, index):
        try:
            return self.fish_list[index.row()].direction
        except IndexError:
            return SwimDirection.NONE

    def setMinDetections(self, value):
        self.min_detections = value
        self.trimFishList()

    def setMAD(self, value):
        self.mad_limit = value
        self.trimFishList()

    def setLengthPercentile(self, value):
        self.length_percentile = value
        for fish in self.all_fish.values():
            fish.setLengthByPercentile(self.length_percentile)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def toggleUpDownInversion(self):
        self.setUpDownInversion(not self.up_down_inverted)

    def setUpDownInversion(self, value):
        self.up_down_inverted = value
        pixels_per_meter = self.playback_manager.getPixelsPerMeter()
        if pixels_per_meter is not None:
            meters_per_pixel = 1.0 / pixels_per_meter
            for fish in self.all_fish.values():
                fish.setPathVariables(self.up_down_inverted, self.frame_time, meters_per_pixel)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def setShowEchogramFish(self, value):
        self.show_echogram_fish = value

    def visualize(self, image, frame_ind, show_size=True, show_id=True, show_bounding_box=True):
        """
        Draws the tracked fish to the full sized image using opencv.
        Returns the modified image.
        """

        fish_by_frame = self.getFishInFrame(frame_ind)
        if len(fish_by_frame) == 0:
            return image
        
        colors = sns.color_palette('deep', max(0, len(fish_by_frame)))
        for fish in fish_by_frame:
            tr, det = fish.tracks[frame_ind]
            if show_id:
                center = FishEntry.trackCenter(tr)
                image = cv2.putText(image, f"ID: {fish.id}", (int(center[1])-20, int(center[0])+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            if show_size and det is not None:
                det.visualize(image, colors, True, False)

            if show_bounding_box:
                corners = np.array([[tr[0], tr[1]], [tr[2], tr[1]], [tr[2], tr[3]], [tr[0], tr[3]]]) #, [tr[0], tr[1]]

                for i in range(0,3):
                    cv2.line(image, (int(corners[i,1]),int(corners[i,0])), (int(corners[i+1,1]),int(corners[i+1,0])),  (255,255,255), 1)
                cv2.line(image, (int(corners[3,1]),int(corners[3,0])), (int(corners[0,1]),int(corners[0,0])),  (255,255,255), 1)

        return image

    def getFishInFrame(self, ind):
        return [f for f in self.fish_list if ind in f.tracks.keys()]

    def getSavedList(self):
        return self.fish_list if fh.getConfValue(fh.ConfKeys.filter_tracks_on_save) else self.all_fish.values()

    def saveToFile(self, path):
        """
        Tries to save all fish information (from all_fish dictionary) to a file.
        """
        if(self.playback_manager.playback_thread is None):
            LogObject().print("No file open, cannot save.")
            return

        try:
            with open(path, "w") as file:
                file.write("id;frame;length;distance;angle;aspect;direction;corner1 x;corner1 y;corner2 x;corner2 y;corner3 x;corner3 y;corner4 x;corner4 y; detection\n")

                lines = self.getSaveLines()
                lines.sort(key = lambda l: (l[0].id, l[1]))
                for _, _, line in lines:
                    file.write(line)

                LogObject().print("Tracks saved to path:", path)
        except PermissionError as e:
            LogObject().print("Cannot open file {}. Permission denied.".format(path))

    def getSaveLines(self):
        """
        Iterates through all the fish and returns a list containing the fish objects, frames the fish appear in, and the following information:
        ID, Frame, Length, Angle, Aspect, Direction, Corner coordinates and wether the values are from a detection or a track.
        Detection information are preferred over tracks.
        """
        lines = []
        polar_transform = self.playback_manager.playback_thread.polar_transform

        f1 = "{:.5f}"
        lineBase1 = "{};{};" + "{};{};{};".format(f1,f1,f1) + "{};"
        lineBase2 = "{};{};" + "{};{};{};".format(f1,f1,f1) + "{};"

        for fish in self.getSavedList():
            for frame, td in fish.tracks.items():
                track, detection = td

                # Values calculated from detection
                if detection is not None:
                    length = fish.length if fish.length_overwritten else detection.length
                    line = lineBase1.format(fish.id, frame, length, detection.distance, detection.angle, detection.aspect, fish.direction.name)
                    if detection.corners is not None:
                        line += self.cornersToString(detection.corners, ";")
                    else:
                        line += ";".join(8 * [" "])
                    line += ";1"

                # Values calculated from track
                else:
                    if fish.length_overwritten:
                        length = fish.length
                    else:
                        length, _ = polar_transform.getMetricDistance(*track[:4])
                    #center = [(track[2]+track[0])/2, (track[3]+track[1])/2]
                    center = FishEntry.trackCenter(track)
                    distance, angle = polar_transform.cart2polMetric(center[0], center[1], True)
                    angle = float(angle / np.pi * 180 + 90)
                    aspect = np.nan # not possible to extract this from track or detection files if not stored in the files

                    line = lineBase1.format(fish.id, frame, length, distance, angle, aspect, fish.direction.name)
                    line += self.cornersToString([[track[0], track[1]], [track[2], track[1]], [track[2], track[3]], [track[0], track[3]]], ";")
                    line += ";0"

                lines.append((fish, frame, line + "\n"))

        return lines

    def cornersToString(self, corners, delim):
        """
        Formats the corner information in a saveable format.
        """
        base = "{:.2f}" + delim + "{:.2f}"
        return delim.join(base.format(cx,cy) for cy, cx in corners[0:4])

    def loadFromFile(self, path):
        try:
            with open(path, 'r') as file:
                self.clear()
                header = file.readline()

                for line in file:
                    split_line = line.split(';')
                    id = int(split_line[0])
                    frame = int(split_line[1])
                    length = float(split_line[2])
                    aspect = float(split_line[5])
                    direction = SwimDirection[split_line[6]]
                    track = [float(split_line[8]), float(split_line[7]), float(split_line[12]), float(split_line[11]), id]

                    if id in self.all_fish:
                        f = self.all_fish[id]
                        f.addTrack(track, None, frame)
                    else:
                        f = FishEntryFromTrack(track, None, frame)
                        f.length = length
                        f.direction = direction
                        self.all_fish[id] = f

                self.refreshAllFishData()
                self.trimFishList(force_color_update=True)
        except PermissionError as e:
            LogObject().print(f"Cannot open file {path}. Permission denied.")
        except ValueError as e:
            LogObject().print(f"Invalid values encountered in {path}, when trying to import tracks. {e}")

    def convertToWritable(self, frame, label, track):
        return [frame, label, list(map(float, track))]


    def getSaveDictionary(self):
        """
		Returns a dictionary of fish to be saved in SaveManager.
		"""
        fish = {}

        for f in self.getSavedList():
            fish_tracks = [self.convertToWritable(frame, int(det.label), track[0:4]) if det is not None else
                           self.convertToWritable(frame, None, track[0:4])
                           for frame, (track, det) in f.tracks.items()]

            fish[str(f.id)] = fish_tracks

        return fish

    def applySaveDictionary(self, data, dets):
        """
		Load fish entries from data provided by SaveManager.
		"""
        self.clear()
        for _id, f_data in data.items():
            id = int(_id)
            f = None
            for frame, det_label, track in f_data:
                if f is None:
                    f = FishEntry(id, frame, frame)

                if det_label is not None:
                    # Goes through detections in the same frame and tries to assign the
                    # corresponding detection based on the label.

                    frame_dets = dets[frame]
                    match_found = False
                    for fd in frame_dets:
                        if fd.label == det_label:
                            # Adds track with a matching detection to the FishEntry

                            f.addTrack(track, fd, frame)
                            match_found = True
                            break

                    if not match_found:
                        LogObject().print("Warning: Match not found in frame {} for label {}".format(frame, det_label))
                else:
                    f.addTrack(track, None, frame)

            if f is not None:
                self.all_fish[id] = f

        self.refreshAllFishData()
        self.printDirectionCounts()
        self.trimFishList(force_color_update=True)

    def printDirectionCounts(self):
        tc, uc, dc, nc = self.allDirectionCounts()
        LogObject().print1(f"Direction counts: Total {tc}, Up {uc}, Down {dc}, None {nc}")



class SwimDirection(IntEnum):
    UP = 0
    DOWN = 1
    NONE = 2

def FishEntryFromTrack(track, detection, frame):
    fish = FishEntry(track[4], frame, frame)
    fish.addTrack(track, detection, frame)
    return fish

class FishEntry():
    def __init__(self, id, frame_in=0, frame_out=0):
        self.id = int(id)
        self.length = 0
        self.aspect = 0
        self.direction = SwimDirection.NONE
        self.frame_in = frame_in
        self.frame_out = frame_out
        self.duration = frame_out - frame_in + 1
        self.mad = 0
        self.tortuosity = 1
        self.speed = 0

        # tracks: Dictionary {frame index : (track, detection)}
        self.tracks = {}
        self.detection_count = 0

        # lengths: Sorted list [lengths of detections]
        self.lengths = []
        self.length_overwritten = False

        self.color_ind = 0

    def __repr__(self):
        return "FishEntry {}: {:.1f} {}".format(self.id, self.length, self.aspect, self.direction.name)

    def dirSortValue(self):
        return self.direction.value * 10**8 + self.id

    def setAspect(self, value):
        self.aspect = value

    def setMeanAspect(self):
        if not self.length_overwritten:
            if len(self.aspects) > 0:
                self.aspect = round(float(np.mean(self.lengths)),1)

    def setLength(self, value):
        self.length = value
        self.length_overwritten = True

    def setLengthByPercentile(self, percentile):
        if not self.length_overwritten:
            if len(self.lengths) > 0:
                self.length = round(float(np.percentile(self.lengths, percentile)),3)

    def forceLengthByPercentile(self, percentile):
        self.length_overwritten = False
        self.setLengthByPercentile(percentile)

    def checkConditions(self, duration, mad):
        return self.duration >= duration and self.mad >= mad

    def addTrack(self, track, detection, frame):
        self.tracks[frame] = (track[0:4], detection)
        if detection is not None:
            insort(self.lengths, detection.length)
        #self.setFrames()

    def copy(self):
        f = FishEntry(self.id, self.frame_in, self.frame_out)
        f.length = self.length
        f.aspect = self.aspect
        f.direction = self.direction
        f.tracks = self.tracks.copy()
        f.lengths = self.lengths.copy()
        f.length_overwritten = self.length_overwritten
        f.color_ind = self.color_ind
        f.mad = self.mad
        f.tortuosity = self.tortuosity
        f.speed = self.speed
        return f

    def merge(self, other):
        self.frame_in = min(self.frame_in, other.frame_in)
        self.frame_out = max(self.frame_out, other.frame_out)
        self.duration = self.frame_out - self.frame_in + 1
        
        for l in other.lengths:
            insort(self.lengths, l)

        for frame, track in other.tracks.items():
            if frame not in self.tracks:
                self.tracks[frame] = track
            else:
                # TODO: Overlapping tracks.
                # Currently only self.tracks are kept when tracks overlap each other.
                pass

    def split(self, frame, new_id):
        """
        Splits the fish in two at the argument "frame".
        frame: the first included frame of the second fish
        new_id: the new id of the second fish
        """
        f = FishEntry(new_id, frame, self.frame_out)
        for tr_frame in list(self.tracks.keys()):
            if tr_frame >= frame:
                tr, det = self.tracks.pop(tr_frame)
                f.addTrack(tr, det, tr_frame)

        #self.setLengths()
        #self.setFrames()
        return f

    def trimTail(self):
        """
        Removes the tail of the tracks, i.e. the last tracks with no corresponding detections (det == None).
        """
        for frame in self.getTail():
            self.tracks.pop(frame)

        #self.setLengths()
        #self.setFrames()

    def getTail(self):
        tail = []
        for frame, (tr, det) in self.tracks.items():
            if det is None:
                tail.append(frame)
            else:
                tail = []
        return tail

    def setFrames(self):
        inds = self.tracks.keys()
        if len(inds) > 0:
            self.frame_in = min(inds)
            self.frame_out = max(inds)
            self.duration = self.frame_out - self.frame_in + 1
        self.detection_count = len([det for _, det in self.tracks.values() if det is not None])

    def setPathVariables(self, inverted, frame_time, meters_per_pixel):
        """
        Calculates variables from the path of the fish,
        i.e. swim direction, mad, tortuosity and speed.
        """
        valid_dets = [d for _, d in self.tracks.values() if d is not None]
        if len(valid_dets) <= 1:
            self.direction = SwimDirection.NONE
            self.mad = 0
            self.tortuosity = 1
            self.speed = 0
        else:
            end_point_distance = valid_dets[-1].center - valid_dets[0].center
            if inverted:
                self.direction = SwimDirection.UP if end_point_distance[1] <= 0 else SwimDirection.DOWN
            else:
                self.direction = SwimDirection.UP if end_point_distance[1] > 0 else SwimDirection.DOWN

            self.aspect = float(np.mean(valid_dets.aspect))
            self.mad = abs(valid_dets[-1].angle - valid_dets[0].angle)
            path_length = self.calculatePathLength(valid_dets)
            norm_dist = np.linalg.norm(end_point_distance)
            self.tortuosity = float(path_length / norm_dist) if norm_dist > 0 else 1

            if frame_time is not None:
                self.speed = float(path_length * meters_per_pixel / ((self.frame_out - self.frame_in) * frame_time))
            else:
                self.speed = 0

    def calculatePathLength(self, dets):
        dist_sum = 0
        prev_point = dets[0].center
        for i in range(1, len(dets)):
            new_point = dets[i].center
            dist_sum += np.linalg.norm(new_point - prev_point)
            prev_point = new_point
        return float(dist_sum)

    def setLengths(self):
        self.lengths = sorted([det.length for _, det in self.tracks.values() if det is not None])

    def setAspects(self):
        self.aspects = sorted([det.aspects for _, det in self.tracks.values() if det is not None])

    @staticmethod
    def trackCenter(track):
        return [(track[2]+track[0])/2, (track[3]+track[1])/2]

def floatTryParse(value):
    try:
        return float(value), True
    except ValueError:
        return value, False

def intTryParse(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

if __name__ == "__main__":
    fish_manager = FishManager(None, None)
    fish_manager.testPopulate(500)
    for fish in fish_manager.fish_list:
        print(fish)

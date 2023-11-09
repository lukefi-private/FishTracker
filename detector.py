"""
This file is part of Fish Tracker.
Copyright 2021, VTT Technical research centre of Finland Ltd.
Developed by: Otto Korkalo and Mikael Uimonen.

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

import time
import glob
import numpy as np
import cv2
import time
import seaborn as sns
import sklearn.cluster as cluster
import random as rng
import collections
import os
import sys
import traceback

from PyQt5 import QtCore, QtGui, QtWidgets
from playback_manager import PlaybackManager, Event, TestFigure
from log_object import LogObject
from mog_parameters import MOGParameters
from detector_parameters import DetectorParameters
from background_subtractor import BackgroundSubtractor

def nothing(x):
    pass

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

class Detector(QtCore.QObject):

	# When detector parameters change.
	parameters_changed_signal = QtCore.pyqtSignal()

	# When detector state changes.
	state_changed_signal = QtCore.pyqtSignal()

	# When displayed results change (e.g. when a new frame is displayed). Parameter: number of detections displayed.
	data_changed_signal = QtCore.pyqtSignal(int)

	# When detector has computed all available frames.
	all_computed_signal = QtCore.pyqtSignal()

	def __init__(self, image_provider):
		super().__init__()
		self.image_provider = image_provider
		self.bg_subtractor = BackgroundSubtractor(image_provider)
		self.parameters = None
		self.applied_parameters = None

		self.resetParameters()
		self.detections = [None]
		self.vertical_detections = []

		self.current_ind = 0
		self.current_len = 0

		self.bg_subtractor.state_changed_signal.connect(self.state_changed_signal)
		self.bg_subtractor.parameters_changed_signal.connect(self.parameters_changed_signal)

		self._show_echogram_detections = False
		self.show_bgsub = False

		# [flag] Whether detector is computing all the frames.
		self.computing = False

		# [trigger] Terminate computing process.
		self.stop_computing = False

		# [flag] Calculate detections on event. Otherwise uses precalculated results.
		self.compute_on_event = False

	def resetParameters(self):
		self.setMOGParameters(MOGParameters())
		self.setDetectorParameters(DetectorParameters())

	def setDetectorParameters(self, parameters: DetectorParameters):
		if self.parameters is not None:
			self.parameters.values_changed_signal.disconnect(self.parameters_changed_signal)

		#self.bg_subtractor.setParameters(parameters.mog_parameters)
		self.parameters = parameters # DetectorParameters(mog_parameters=self.bg_subtractor.mog_parameters)
		self.parameters.values_changed_signal.connect(self.parameters_changed_signal)
		self.parameters_changed_signal.emit()

	def setMOGParameters(self, parameters: MOGParameters):
		self.bg_subtractor.setParameters(parameters)

	def initMOG(self, clear_detections=True):
		if clear_detections:
			self.clearDetections()
		self.bg_subtractor.initMOG()

	def compute_from_event(self, tuple):
		if tuple is None:
			return

		ind, img = tuple
		if self.compute_on_event:
			self.compute(ind, img)
		else:
			self.data_changed(ind)

	def compute(self, ind, image, get_images=False):
		images = self.computeBase(ind, image, get_images)
		self.data_changed(ind)
		if get_images:
			return images

	def data_changed(self, ind):
		dets = self.detections[ind]
		self.current_ind = ind
		self.current_len = 0 if dets is None else len(self.detections[ind])
		self.data_changed_signal.emit(self.current_len)

	def getPolarTransform(self):
		if hasattr(self.image_provider, "playback_thread"):
			return self.image_provider.playback_thread.polar_transform
		else:
			return None

	def computeBase(self, ind, image, get_images=False, show_size=True):
		params = self.parameters

		image_o = image_o_gray = image
		fg_mask_mog = self.bg_subtractor.subtractBG(image_o)
		if fg_mask_mog is None:
			return

		fg_mask_cpy = fg_mask_mog
		fg_mask_filt = cv2.medianBlur(fg_mask_cpy, params.getParameter(DetectorParameters.ParametersEnum.median_size))

		data_tr = np.nonzero(np.asarray(fg_mask_filt))
		data = np.asarray(data_tr).T
		detections = []

		if get_images:
			image_o_rgb = cv2.applyColorMap(image_o, cv2.COLORMAP_OCEAN)

		if data.shape[0] >=params.getParameter(DetectorParameters.ParametersEnum.min_fg_pixels):

			# DBSCAN clusterer, NOTE: parameters should be in UI / read from file
			clusterer = cluster.DBSCAN(eps=params.getParameter(DetectorParameters.ParametersEnum.dbscan_eps),
							  min_samples=params.getParameter(DetectorParameters.ParametersEnum.dbscan_min_samples))
			labels = clusterer.fit_predict(data)
		
			data = data[labels != -1]
			labels = labels[labels != -1]

			if labels.shape[0] > 0:

				polar_transform = self.getPolarTransform()

				for label in np.unique(labels):
					foo = data[labels == label]
					if foo.shape[0] < 2:
						continue
			
					d = Detection(label)
					d.init_from_data(foo, params.getParameter(DetectorParameters.ParametersEnum.detection_size), polar_transform)
					detections.append(d)

				if get_images:
					colors = sns.color_palette('deep', np.unique(labels).max() + 1)
					for d in detections:
						image_o_rgb = d.visualize(image_o_rgb, colors[d.label], show_size)

		self.detections[ind] = detections

		if get_images:
			return (fg_mask_mog, image_o_gray, image_o_rgb, fg_mask_filt)

	def computeAll(self):
		self.computing = True
		self.stop_computing = False
		self.compute_on_event = False
		self.state_changed_signal.emit()

		LogObject().print1(self.bg_subtractor.mog_parameters)
		LogObject().print1(self.parameters)

		if self.bg_subtractor.parametersDirty():
			self.initMOG()
			if self.bg_subtractor.parametersDirty():
				LogObject().print("Stopped before detecting.")
				self.abortComputing(True)
				return

		count = self.image_provider.getFrameCount()
		ten_perc = 0.1 * count
		print_limit = 0
		for ind in range(count):
			if ind > print_limit:
				LogObject().print("Detecting:", int(float(ind) / count * 100), "%")
				print_limit += ten_perc

			if self.stop_computing:
				LogObject().print("Stopped detecting at", ind)
				self.abortComputing(False)
				return

			img = self.image_provider.getFrame(ind)
			self.computeBase(ind, img)

		LogObject().print("Detecting: 100 %")
		self.computing = False
		#self.detections_clearable = True
		self.applied_parameters = self.parameters.copy()

		self.updateVerticalDetections()

		self.state_changed_signal.emit()
		self.all_computed_signal.emit()

	def updateVerticalDetections(self):
		self.vertical_detections = [[d.distance for d in dets if d.center is not None] if dets is not None else [] for dets in self.detections]

	def abortComputing(self, mog_aborted):
		self.stop_computing = False
		self.computing = False
		self.compute_on_event = True

		self.state_changed_signal.emit()
		self.applied_parameters = None

		if mog_aborted:
			self.bg_subtractor.abortComputing()

	def clearDetections(self):
		LogObject().print2("Cleared detections")
		nof_frames = self.image_provider.getFrameCount()
		self.detections = [None] * nof_frames
		self.vertical_detections = []
		#self.detections_clearable = False
		self.applied_parameters = None
		self.compute_on_event = True
		self.state_changed_signal.emit()
		self.data_changed_signal.emit(0)
		
	def overlayDetections(self, image, detections=None, show_size=True):
		if detections is None:
			detections = self.getCurrentDetection()

		colors = sns.color_palette('deep', max([0] + [d.label + 1 for d in detections]))
		for d in detections:
			image = d.visualize(image, colors[d.label], show_size)

		return image

	def overlayDetectionColors(self, image, detections=None):
		if detections is None:
			detections = self.getCurrentDetection()

		colors = sns.color_palette('deep', max([0] + [d.label + 1 for d in detections]))
		for d in detections:
			image = d.visualizeArea(image, colors[d.label])

		return image

	def setParameter(self, key, value):
		self.parameters.setKeyValuePair(key, value)

	def setShowEchogramDetections(self, value):
		self._show_echogram_detections = value
		if not self._show_echogram_detections:
			self.data_changed_signal.emit(0)

	def toggleShowBGSubtraction(self):
		self.show_bgsub = not self.show_bgsub

	def setShowBGSubtraction(self, value):
		self.show_bgsub = value

	def getDetection(self, ind):
		try:
			dets = self.detections[ind]
			if dets is None:
				return []

			return [d for d in dets if d.center is not None]
		except IndexError:
			LogObject().print2(traceback.format_exc())

	def getDetections(self):
		return [[d for d in dets if d.center is not None] if dets is not None else [] for dets in self.detections]

	def getCurrentDetection(self):
		return self.getDetection(self.current_ind)

	def getParameterDict(self):
		if self.parameters is not None:
			detector_params = self.parameters.getParameterDict()
			bg_sub_params = self.bg_subtractor.mog_parameters.getParameterDict()

			return { "bg_subtractor": bg_sub_params, "detector": detector_params }
		else:
			return None

	def setParameterDict(self, param_dict, set_as_applied=False):
		if "detector" in param_dict.keys():
			self.parameters.setParameterDict(param_dict["detector"])
			if set_as_applied:
				self.applied_parameters = self.parameters.copy()
		else:
			LogObject().print2("Detector parameters not found.")

		if "bg_subtractor" in param_dict.keys():
			self.bg_subtractor.mog_parameters.setParameterDict(param_dict["bg_subtractor"])
			if set_as_applied:
				self.bg_subtractor.applyParameters()
		else:
			LogObject().print2("Background subtractor parameters not found.")

	def bgSubtraction(self, image):
		median_size = self.parameters.getParameter(DetectorParameters.ParametersEnum.median_size)
		return self.bg_subtractor.subtractBGFiltered(image, median_size)

	def parametersDirty(self):
		return self.parameters != self.applied_parameters or self.bg_subtractor.parametersDirty()

	def allCalculationAvailable(self):
		return self.parametersDirty() and not self.bg_subtractor.initializing

	def saveDetectionsToFile(self, path):
		"""
		Writes current detections to a file at path. Values are separated by ';'.
		"""

		# Default formatting
		f1 = "{:.5f}"
		lineBase1 = "{};" + "{};{};{};".format(f1,f1,f1)

		try:
			with open(path, "w") as file:
				file.write("frame;length;distance;angle;aspect;corner1 x;corner1 y;corner2 x;corner2 y;corner3 x;corner3 y;corner4 x;corner4 y\n")
				for frame, dets in enumerate(self.detections):
					if dets is not None:
						for d in dets:
							if d.corners is not None:
								file.write(lineBase1.format(frame, d.length, d.distance, d.angle, d.aspect))
								file.write(d.cornersToString(";"))
								file.write("\n")
				LogObject().print("Detections saved to path:", path)

		except PermissionError as e:
			LogObject().print("Cannot open file {}. Permission denied.".format(path))

	def loadDetectionsFromFile(self, path):
		"""
		Loads a file from path. Values are expected to be separated by ';'.
		"""
		try:
			with open(path, 'r') as file:
				self.clearDetections()
				nof_frames = self.image_provider.getFrameCount()
				ignored_dets = 0

				header = file.readline()

				for line in file:
					split_line = line.split(';')
					frame = int(split_line[0])

					if frame >= nof_frames:
						ignored_dets += 1
						continue

					length = float(split_line[1])
					distance = float(split_line[2])
					angle = float(split_line[3])
					aspect = float(split_line[4])

					c1 = [float(split_line[6]), float(split_line[5])]
					c2 = [float(split_line[8]), float(split_line[7])]
					c3 = [float(split_line[10]), float(split_line[9])]
					c4 = [float(split_line[12]), float(split_line[11])]
					corners = np.array([c1, c2, c3, c4])

					det = Detection(0)
					det.init_from_file(corners, length, distance, angle, aspect)

					if self.detections[frame] is None:
						self.detections[frame] = [det]
					else:
						self.detections[frame].append(det)

				self.updateVerticalDetections()
				self.compute_on_event = False
				if ignored_dets > 0:
					LogObject().print("Encountered {} detections that were out of range {}.".format(ignored_dets, nof_frames))


		except PermissionError as e:
			LogObject().print(f"Cannot open file {path}. Permission denied.")
		except ValueError as e:
			LogObject().print(f"Invalid values encountered in {path}, when trying to import detections. {e}")

	def getSaveDictionary(self):
		"""
		Returns a dictionary of detection data to be saved in SaveManager.
		"""
		detections = {}
		for frame, dets in enumerate(self.detections):
			if dets is None:
				continue

			dets_in_frame = [d.convertToWritable() for d in dets if d.corners is not None]
			if len(dets_in_frame) > 0:
				detections[str(frame)] = dets_in_frame

		return detections

	def applySaveDictionary(self, parameters, data):
		"""
		Load detections from data provided by SaveManager.
		"""
		self.clearDetections()

		self.setParameterDict(parameters, True)

		polar_transform = self.getPolarTransform()

		for frame in range(len(self.detections)):
			frame_dets = []
			str_frame = str(frame)
			if str_frame in data.keys():
				for det_data in data[str_frame]:
					label = det_data[0]
					det_data = det_data[1]
					det = Detection(int(label))
					detection_size = self.parameters.getParameter(DetectorParameters.ParametersEnum.detection_size)
					det.init_from_data(det_data, detection_size, polar_transform)
					frame_dets.append(det)

			try:
				self.detections[frame] = frame_dets
			except IndexError as e:
				print(frame, len(self.detections))
				raise e

		self.updateVerticalDetections()
		self.compute_on_event = False
		self.state_changed_signal.emit()
		self.all_computed_signal.emit()


class Detection:
	def __init__(self, label):
		self.label = label
		self.data = None
		self.diff = None
		self.center = None
		self.corners = None

		self.length = 0
		self.distance = 0
		self.angle = 0
		self.aspect = 0

	def __repr__(self):
		return "Detection \"{}\" d:{:.1f}, a:{:.1f}".format(self.label, self.distance, self.angle, self.aspect)

	def init_from_data(self, data, detection_size, polar_transform):
		"""
		Initialize detection parameters from the pixel data from the clusterer / detection algorithm. Saved pixel data
		can also be used to (re)initialize the detection.
		"""
		self.data = np.asarray(data)

		ca = np.cov(data, y=None, rowvar=0, bias=1)
		v, vect = np.linalg.eig(ca)
		tvect = np.transpose(vect)
		ar = np.dot(data, np.linalg.inv(tvect))

		# NOTE a fixed parameter --> to UI / file.
		if ar.shape[0] > detection_size: #10:

			mina = np.min(ar, axis=0)
			maxa = np.max(ar, axis=0)
			diff = (maxa - mina) * 0.5
					
			center = mina + diff

			# Get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
			corners = np.array([center+[-diff[0],-diff[1]], \
								center+[diff[0],-diff[1]], \
								center+[diff[0],diff[1]], \
								center+[-diff[0],diff[1]]])#, \
								#center+[-diff[0],-diff[1]]])

			self.diff = diff

			# Use the the eigenvectors as a rotation matrix and rotate the corners and the center back
			self.corners = np.dot(corners, tvect)
			self.center = np.dot(center, tvect)

			if polar_transform is not None:
				metric_diff = polar_transform.pix2metCI(diff[0], diff[1])
				self.length = float(2 * metric_diff[1])
				self.distance, self.angle = polar_transform.cart2polMetric(self.center[0], self.center[1], True)
				self.distance = float(self.distance)
				self.angle = float(self.angle / np.pi * 180 + 90)
				self.aspect = float(np.arcsin(tvect[0,0]) / np.pi * 180 + 90) # aspect angle in degrees, 0 means that the length axis of the fish is perpendicular to the sound axis.  

	def init_from_file(self, corners, length, distance, angle, aspect):
		"""
		Initialize detection parameters from a csv file. Data is not stored when exporting a csv file,
		which means it cannot be recovered here. This mainly affects the visualization of the detection.
		"""
		self.corners = np.array(corners)
		self.center = np.average(self.corners, axis=0)
		self.diff = self.center - self.corners[0]
		self.length = length
		self.distance = distance
		self.angle = angle
		self.aspect = aspect

	def visualize(self, image, color, show_text, show_detection=True):
		if self.corners is None:
			return image

		# Draw size text
		if show_text:
			if self.length > 0:
				size_txt = self.getSizeText()
				image = cv2.putText(image, size_txt, (int(self.center[1])-20, int(self.center[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

		# Draw detection area and bounding box
		if show_detection:
			self.visualizeArea(image, color)

			for i in range(0,3):
				cv2.line(image, (int(self.corners[i,1]),int(self.corners[i,0])), (int(self.corners[i+1,1]),int(self.corners[i+1,0])),  (255,255,255), 1)
			cv2.line(image, (int(self.corners[3,1]),int(self.corners[3,0])), (int(self.corners[0,1]),int(self.corners[0,0])),  (255,255,255), 1)

		return image

	def visualizeArea(self, image, color):
		if self.data is not None:
			_color = (int(255*color[0]), int(255*color[1]), int(255*color[2]))
			for i in range(self.data.shape[0]):
				cv2.line(image, (self.data[i,1], self.data[i,0]), (self.data[i,1], self.data[i,0]), _color, 2)

		return image

	def getSizeText(self):
		return 'Size: ' + str(int(100*self.length))
	
	def getMessage(self):
		# This message was used to send data to tracker process (C++) or save detections to file
		if self.diff is None:
			return ""
		
		return str(int(self.center[1]*10)) + ' ' + str(int(self.center[0]*10)) + ' ' + str(int(self.diff[1]*2))

	def cornersToString(self, delim):
		if self.corners is None:
			return ""

		base = "{:.2f}" + delim + "{:.2f}"
		return delim.join(base.format(cx,cy) for cy, cx in self.corners[0:4])

	def convertToWritable(self):
		"""
		Returns data in applicable format to be used by SaveManager
		"""
		return [int(self.label), list(map(lambda x: [int(x[0]), int(x[1])], self.data))]


class DetectorDisplay:
	def __init__(self):
		self.array = [cv2.imread(file, 0) for file in sorted(glob.glob("out/*.png"))]
		self.detector = Detector(self)
		self.detector._show_detections = True
		self._show_echogram_detections = True

	def run(self):
		self.showWindow()
		self.detector.initMOG()

		for i in range(self.getFrameCount()):
			self.readParameters()
			images = self.detector.compute(i, self.getFrame(i), True)
			LogObject().print(images)
			self.updateWindows(*images)

	def showWindow(self):
		cv2.namedWindow('fg_mask_mog', cv2.WINDOW_NORMAL)
		cv2.namedWindow('image_o_rgb', cv2.WINDOW_NORMAL)
		cv2.namedWindow('image_o_gray', cv2.WINDOW_NORMAL)
		cv2.namedWindow('fg_mask_filt', cv2.WINDOW_NORMAL)

		cv2.createTrackbar('mog_var_thresh', 'image_o_rgb', 5, 30, nothing)
		cv2.createTrackbar('median_size', 'image_o_rgb', 1, 21, nothing)
		cv2.createTrackbar('min_fg_pixels', 'image_o_rgb', 10, 100, nothing)

		mog_var_thresh = self.detector.bg_subtractor.mog_parameters.getParameter(MOGParameters.ParametersEnum.mog_var_thresh)
		min_fg_pixels = self.detector.parameters.getParameter(DetectorParameters.ParametersEnum.min_fg_pixels)
		median_size = self.detector.parameters.getParameter(DetectorParameters.ParametersEnum.median_size)

		cv2.setTrackbarPos('mog_var_thresh','image_o_rgb', mog_var_thresh)
		cv2.setTrackbarPos('min_fg_pixels','image_o_rgb', min_fg_pixels)
		cv2.setTrackbarPos('median_size','image_o_rgb', median_size)

	def updateWindows(self, fg_mask_mog, image_o_gray, image_o_rgb, fg_mask_filt):
		pos_step = 600

		cv2.moveWindow("fg_mask_mog", pos_step*0, 20);
		cv2.moveWindow("image_o_gray", pos_step*1, 20);
		cv2.moveWindow("image_o_rgb", pos_step*2, 20);
		cv2.moveWindow("fg_mask_filt", pos_step*3, 20);

		cv2.imshow("fg_mask_mog", fg_mask_mog)
		cv2.imshow("image_o_gray", image_o_gray)
		cv2.imshow("image_o_rgb", image_o_rgb)
		cv2.imshow("fg_mask_filt", fg_mask_filt)

		sleep = cv2.getTrackbarPos('sleep','image_o_rgb')
		key = cv2.waitKey(sleep)
		if key == 32:
			sleep = -sleep

	def readParameters(self):
		# Read parameter values from trackbars
		self.detector.bg_subtractor.mog_parameters.mog_var_thresh = cv2.getTrackbarPos('mog_var_thresh','image_o_rgb')
		min_fg_pixels = cv2.getTrackbarPos('min_fg_pixels','image_o_rgb')
		self.detector.parameters.setParameter(DetectorParameters.ParametersEnum.min_fg_pixels, min_fg_pixels)
		median_size = int(round_up_to_odd(cv2.getTrackbarPos('median_size','image_o_rgb')))
		self.detector.parameters.setParameter(DetectorParameters.ParametersEnum.median_size, median_size)

	def getFrameCount(self):
		return len(self.array)

	def getFrame(self, ind):
		return self.array[ind]

def simpleTest():
	dd = DetectorDisplay()
	dd.run()

def playbackTest():

	def forwardImage(tuple):
		ind, frame = tuple
		detections = detector.getCurrentDetection()

		image = cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
		image = detector.overlayDetections(image, detections)

		figure.displayImage((ind, image))

	def startDetector():
		detector.initMOG()
		playback_manager.play()

	app = QtWidgets.QApplication(sys.argv)
	main_window = QtWidgets.QMainWindow()
	playback_manager = PlaybackManager(app, main_window)
	playback_manager.fps = 10
	playback_manager.openTestFile()
	playback_manager.frame_available.connect(forwardImage)
	detector = Detector(playback_manager)
	detector.bg_subtractor.mog_parameters.nof_bg_frames = 100
	detector.setShowEchogramDetections(True)
	playback_manager.mapping_done.connect(startDetector)
	playback_manager.frame_available_immediate.append(detector.compute_from_event)

	figure = TestFigure(playback_manager.togglePlay)
	main_window.setCentralWidget(figure)

	main_window.show()
	sys.exit(app.exec_())

def benchmark():
	def runDetector():
		detector.computeAll()
		print("All done.")
		main_window.close()

	app = QtWidgets.QApplication(sys.argv)
	main_window = QtWidgets.QMainWindow()
	playback_manager = PlaybackManager(app, main_window)
	detector = Detector(playback_manager)
	detector.setNofBGFrames(1000)
	playback_manager.mapping_done.connect(runDetector)
	main_window.show()
	playback_manager.openTestFile()
	sys.exit(app.exec_())


if __name__ == "__main__":
	#simpleTest()
	playbackTest()
	#benchmark()
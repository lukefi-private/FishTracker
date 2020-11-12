﻿import time
import glob
import numpy as np
import cv2
import time
import seaborn as sns
import sklearn.cluster as cluster
import random as rng
import collections
import math
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from playback_manager import PlaybackManager, Event, TestFigure

def nothing(x):
    pass

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

class Detector:

	def __init__(self, image_provider):
		# Test data, a folder with multiple png frames
		#directory = '/data/ttekoo/data/luke/2_vaihe_kaikki/frames/Teno1_test_6000-6999'
		#body = '*.png'
		#path = directory + '/' + body

		self.resetParameters()
		self.detections = [None]
		self.vertical_detections = []
		self.image_height = 0

		self.current_ind = 0
		self.current_len = 0

		self.fbgb_mog = None

		self.ts = 0

		self.image_provider = image_provider

		# When detector parameters change.
		self.state_changed_event = Event()

		# When results change (e.g. when a new frame is displayed).
		self.data_changed_event = Event()

		# When detector has computed all available frames.
		self.all_computed_event = Event()

		self._show_detections = False
		self._show_detection_size = True
		self.show_bgsub = False

		# [flag] Whether MOG is initializing
		self.initializing = False

		# [trigger] Terminate initializing process.
		self.stop_initializing = False

		# [flag] Whether MOG has been initialized
		self.mog_ready = False

		# [flag] Whether detector is computing all the frames.
		self.computing = False

		# [trigger] Terminate computing process.
		self.stop_computing = False

		# [flag] Whether detection array can be cleared.
		self.detections_clearable = True

		## [flag] Whether all frames should be calculated.
		#self.detections_dirty = True

		## ([flag] Whether MOG should be reinitialized.
		#self.mog_dirty = False

		self.applied_parameters = None
		self.applied_mog_parameters = None

	def resetParameters(self):
		self.mog_parameters = MOGParameters()
		self.parameters = DetectorParameters(mog_parameters=self.mog_parameters)
		#self.detection_size = 10
		#self.mog_var_thresh = 11
		#self.min_fg_pixels = 25
		#self.median_size = 3
		#self.nof_bg_frames = 1000
		#self.learning_rate = 0.01

		#self.dbscan_eps = 10
		#self.dbscan_min_samples = 10


	def initMOG(self):
		if hasattr(self.image_provider, "pausePolarLoading"):
			self.image_provider.pausePolarLoading(True)

		self.mog_ready = False
		self.initializing = True
		self.stop_initializing = False
		self.state_changed_event()

		self.fgbg_mog = cv2.createBackgroundSubtractorMOG2()
		self.fgbg_mog.setNMixtures(5)
		self.fgbg_mog.setVarThreshold(self.mog_parameters.mog_var_thresh)
		self.fgbg_mog.setShadowValue(0)

		nof_frames = self.image_provider.getFrameCount()
		nof_bg_frames = min(nof_frames, self.mog_parameters.nof_bg_frames)
		self.clearDetections(True)

		# Create background model from fixed number of frames.
		# Count step based on number of frames
		# NOTE: nof_bg_frames to UI / file
		#bg_counter = 0
		#step = np.ceil(nof_frames/self.nof_bg_frames)
		step = nof_frames / nof_bg_frames

		ten_perc = 0.1 * nof_bg_frames
		print_limit = 0
		for i in range(nof_bg_frames):
			if i > print_limit:
				print("Initializing:", int(float(print_limit) / nof_bg_frames * 100), "%")
				print_limit += ten_perc

			ind = math.floor(i * step)

			if self.stop_initializing:
				print("Stopped initializing at", ind)
				self.stop_initializing = False
				self.mog_ready = False
				self.initializing = False
				self.applied_mog_parameters = None
				#self.parameters.mog_parameters = None
				self.state_changed_event()
				return

			image_o = self.image_provider.getFrame(ind)

			# NOTE: now all frames are resized to (500,100).
			# Should be other way. This is now for keping parameter values same.
			# image_o = cv2.resize(image_o, (500,1000), interpolation = cv2.INTER_AREA)

			# NOTE: learningRate to UI / file / compute from nof_bg_frames (learningRate = 1/nof_bg_frames)
			self.fgbg_mog.apply(image_o, learningRate=self.mog_parameters.learning_rate)
			#bg_counter = bg_counter + step
			#if bg_counter == self.nof_bg_frames:
			#	break;

		self.image_height = image_o.shape[0]
		self.mog_ready = True
		self.initializing = False;
		self.applied_mog_parameters = self.mog_parameters.copy()

		self.state_changed_event()
		print("Initializing: 100 %")

		if hasattr(self.image_provider, "pausePolarLoading"):
			self.image_provider.pausePolarLoading(False)

		if hasattr(self.image_provider, "refreshFrame"):
			self.image_provider.refreshFrame()

	def compute_from_event(self, tuple):
		if tuple is None:
			return
		ind, frame = tuple
		self.compute(ind, frame)

	def compute(self, ind, image, get_images=False):
		if not self._show_detections:
			return

		images = self.computeBase(ind, image, get_images)

		self.current_ind = ind
		self.current_len = len(self.detections[ind])
		self.data_changed_event(self.current_len)

		if get_images:
			return images

	def computeBase(self, ind, image, get_images=False):
		# Update timestamp, TODO read from data
		self.ts += 0.1
		
		# NOTE: now all frames are resized to (500,100).
		# Should be other way. This is now for keping parameter values same.
		#image_o = cv2.resize(image, (500,1000), interpolation = cv2.INTER_AREA)
		#image_o_gray = image_o
		image_o = image_o_gray = image

		# Get foreground mask, without updating the  model (learningRate = 0)
		fg_mask_mog = self.fgbg_mog.apply(image_o, learningRate=0)

		msg = str(self.ts) + ' 9999999999 '

		# self.fgbg_mog.setVarThreshold(self.mog_parameters.mog_var_thresh)
		fg_mask_cpy = fg_mask_mog
		fg_mask_filt = cv2.medianBlur(fg_mask_cpy, self.parameters.median_size)
		image_o_rgb = cv2.applyColorMap(image_o, cv2.COLORMAP_OCEAN)

		data_tr = np.nonzero(np.asarray(fg_mask_filt))
		data = np.asarray(data_tr).T
		detections = []

		if data.shape[0] >= self.parameters.min_fg_pixels:

			# DBSCAN clusterer, NOTE: parameters should be in UI / read from file
			clusterer = cluster.DBSCAN(eps=self.parameters.dbscan_eps, min_samples=self.parameters.dbscan_min_samples)		
			labels = clusterer.fit_predict(data)
		
			data = data[labels != -1]
			labels = labels[labels != -1]

			if labels.shape[0] > 0:

				if hasattr(self.image_provider, "playback_thread"):
					polar_transform = self.image_provider.playback_thread.polar_transform
				else:
					polar_transform = None

				for label in np.unique(labels):
					foo = data[labels == label]
					if foo.shape[0] < 2:
						continue
			
					d = Detection(label, foo, self.parameters.detection_size, polar_transform)
					msg += " " + d.getMessage()
					detections.append(d)

				if get_images:
					colors = sns.color_palette('deep', np.unique(labels).max() + 1)
					for d in detections:
						image_o_rgb = d.visualize(image_o_rgb, colors)

				
			
		# Print message: [timestamp camera_id position_x position_y length]
		# Position now in pixel coordinates (times 10) for tracker, hould be in metric coordinates
		# print(msg)

		self.detections[ind] = detections
		self.detections_clearable = True

		if get_images:
			return (fg_mask_mog, image_o_gray, image_o_rgb, fg_mask_filt)

	def computeAll(self):
		self.computing = True
		self.stop_computing = False
		self.state_changed_event()

		#if self.mog_dirty:
		#	self.initMOG()
		#	if self.mog_dirty:
		#		self.abortComputing(True)
		#		return

		if self.mogParametersDirty():
			self.initMOG()
			if self.mogParametersDirty():
				print("Stopped before detecting.")
				self.abortComputing(True)
				return

		count = self.image_provider.getFrameCount()
		ten_perc = 0.1 * count
		print_limit = 0
		for ind in range(count):
			if ind > print_limit:
				print("Detecting:", int(float(ind) / count * 100), "%")
				print_limit += ten_perc

			if self.stop_computing:
				print("Stopped detecting at", ind)
				self.abortComputing(False)
				return

			img = self.image_provider.getFrame(ind)
			self.computeBase(ind, img)

		print("Detecting: 100 %")
		self.computing = False
		#self.detections_dirty = False
		self.detections_clearable = True
		self.applied_parameters = self.parameters.copy()

		self.vertical_detections = [[d.center[0] for d in dets if d.center is not None] if dets is not None else [] for dets in self.detections]

		self.state_changed_event()
		self.all_computed_event()

	def abortComputing(self, mog_aborted):
		self.stop_computing = False
		self.computing = False
		#self.detections_dirty = True
		self.detections_clearable = True
		self.state_changed_event()
		self.applied_parameters = None
		if mog_aborted:
			self.applied_mog_parameters = None

	def clearDetections(self, force=False):
		nof_frames = self.image_provider.getFrameCount()
		self.detections = [None] * nof_frames
		self.vertical_detections = []
		self.detections_clearable = False
		self.applied_parameters = None
		self.state_changed_event()
		
	def overlayDetections(self, image, detections=None):
		# labels = [d.label for d in labels]
		if detections is None:
			detections = self.getCurrentDetection()

		colors = sns.color_palette('deep', max([0] + [d.label + 1 for d in detections]))
		for d in detections:
			image = d.visualize(image, colors, self._show_detection_size)

		return image

	def setDetectionSize(self, value):
		try:
			self.parameters.detection_size = int(value)
			# self.clearDetections()
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setMOGVarThresh(self, value):
		try:
			self.mog_parameters.mog_var_thresh = int(value)
			# self.mog_dirty = True
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setMinFGPixels(self, value):
		try:
			self.parameters.min_fg_pixels = int(value)
			# self.clearDetections()
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setMedianSize(self, value):
		try:
			self.parameters.median_size = int(value)
			# self.clearDetections()
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setNofBGFrames(self, value):
		try:
			self.mog_parameters.nof_bg_frames = int(value)
			# self.mog_dirty = True
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setLearningRate(self, value):
		try:
			self.mog_parameters.learning_rate = float(value)
			# self.mog_dirty = True
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setDBScanEps(self, value):
		try:
			self.parameters.dbscan_eps = int(value)
			# self.clearDetections()
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setDBScanMinSamples(self, value):
		try:
			self.parameters.dbscan_min_samples = int(value)
			# self.clearDetections()
			self.state_changed_event()
		except ValueError as e:
			print(e)

	def setShowDetections(self, value):
		self._show_detections = value # and self.mog_ready
		#print("{} && {}: {}".format(self.mog_ready, value, self._show_detections))
		if not self._show_detections:
			self.data_changed_event(0)

	def setShowSize(self, value):
		self._show_detection_size = value

	def toggleShowBGSubtraction(self):
		self.show_bgsub = not self.show_bgsub

	def setShowBGSubtraction(self, value):
		self.show_bgsub = value

	def getDetection(self, ind):
		dets = self.detections[ind]
		if dets is None:
			return []

		return [d for d in dets if d.center is not None]

	def getDetections(self):
		return [[d for d in dets if d.center is not None] if dets is not None else [] for dets in self.detections]

	def getCurrentDetection(self):
		return self.getDetection(self.current_ind)

	def bgSubtraction(self, image):
		fg_mask_mog = self.fgbg_mog.apply(image, learningRate=0)
		fg_mask_cpy = fg_mask_mog
		fg_mask_filt = cv2.medianBlur(fg_mask_cpy, self.parameters.median_size)
		return fg_mask_filt

	def parametersDirty(self):
		return self.parameters != self.applied_parameters or self.mogParametersDirty()

	def mogParametersDirty(self):
		return self.mog_parameters != self.applied_mog_parameters

	def allCalculationAvailable(self):
		return self.parametersDirty() and not self.initializing

	def saveDetectionsToFile(self, path):
		# Default formatting
		f1 = "{:.5f}"
		lineBase1 = "{};" + "{};{};{};".format(f1,f1,f1)

		try:
			with open(path, "w") as file:
				file.write("frame;length;distance;angle;corner1 x;corner1 y;corner2 x;corner2 y;corner3 x;corner3 y;corner4 x;corner4 y\n")
				for frame, dets in enumerate(self.detections):
					if dets is not None:
						for d in dets:
							if d.corners is not None:
								file.write(lineBase1.format(frame, d.length, d.distance, d.angle))
								file.write(d.cornersToString(";"))
								file.write("\n")
				print("Detections saved to path:", path)

		except PermissionError as e:
			print("Cannot open file {}. Permission denied.".format(path))
			

class MOGParameters:
	def __init__(self, mog_var_thresh=11, nof_bg_frames=1000, learning_rate=0.01,):
		self.mog_var_thresh = mog_var_thresh
		self.nof_bg_frames = nof_bg_frames
		self.learning_rate = learning_rate

	def __eq__(self, other):
		if not isinstance(other, MOGParameters):
			return False

		return self.mog_var_thresh == other.mog_var_thresh \
			and self.nof_bg_frames == other.nof_bg_frames \
			and self.learning_rate == other.learning_rate

	def __repr__(self):
		return "MOG Parameters: {} {} {}".format(self.mog_var_thresh, self.nof_bg_frames, self.learning_rate)

	def copy(self):
		return MOGParameters( self.mog_var_thresh, self.nof_bg_frames, self.learning_rate )

class DetectorParameters:
	def __init__(self, mog_parameters=None, detection_size=10, min_fg_pixels=25, median_size=3, dbscan_eps=10, dbscan_min_samples=10):

		if mog_parameters is None:
			self.mog_parameters = MOGParameters()
		else:
			self.mog_parameters = mog_parameters

		self.detection_size = detection_size
		self.min_fg_pixels = min_fg_pixels
		self.median_size = median_size
		self.dbscan_eps = dbscan_eps
		self.dbscan_min_samples = dbscan_min_samples

	def __eq__(self, other):
		if not isinstance(other, DetectorParameters):
			return False

		value = self.mog_parameters == other.mog_parameters \
			and self.detection_size == other.detection_size \
			and self.min_fg_pixels == other.min_fg_pixels \
			and self.median_size == other.median_size \
			and self.dbscan_eps == other.dbscan_eps \
			and self.dbscan_min_samples == other.dbscan_min_samples

		#print(self, other, value)
		return value

	def __repr__(self):
		return "Parameters: {} {} {} {} {}".format(self.detection_size, self.min_fg_pixels, self.median_size, self.dbscan_eps, self.dbscan_min_samples)

	def copy(self):
		mog_parameters = self.mog_parameters.copy()
		return DetectorParameters( mog_parameters, self.detection_size, self.min_fg_pixels, self.median_size, self.dbscan_eps, self.dbscan_min_samples )

	def getParameterDict(self):
		return {
            "detection_size": self.detection_size,
	        "mog_var_thresh": self.mog_parameters.mog_var_thresh,
	        "min_fg_pixels": self.min_fg_pixels,
	        "median_size": self.median_size,
	        "nof_bg_frames": self.mog_parameters.nof_bg_frames,
	        "learning_rate": self.mog_parameters.learning_rate,
	        "dbscan_eps": self.dbscan_eps,
	        "dbscan_min_samples": self.dbscan_min_samples
        }



class Detection:
	def __init__(self, label, data, detection_size, polar_transform):
		self.label = label
		self.data = data
		self.diff = None
		self.center = None
		self.corners = None

		self.length = 0
		self.distance = 0
		self.angle = 0
		#self.metric_corners = None

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
				center+[-diff[0],diff[1]], \
				center+[-diff[0],-diff[1]]])

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
				#self.metric_corners = np.asarray([polar_transform.pix2metCI(corner[0], corner[1]) for corner in self.corners])
					

	def visualize(self, image, colors, show_text):
		if self.diff is None:
			return image

		# Visualize results	
		if show_text:
			if self.length > 0:
				size_txt = 'Size (cm): ' + str(int(100*self.length))
			else:
				size_txt = 'Size (pix): ' + str(int(self.diff[1]*2))
			image = cv2.putText(image, size_txt, (int(self.center[1])-50, int(self.center[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

		for i in range(self.data.shape[0]):
			cv2.line(image, (self.data[i,1], self.data[i,0]), (self.data[i,1], self.data[i,0]), \
				(int(255*colors[self.label][0]), int(255*colors[self.label][1]), int(255*colors[self.label][2])), 2)						
		for i in range(0,3):
			cv2.line(image, (int(self.corners[i,1]),int(self.corners[i,0])), (int(self.corners[i+1,1]),int(self.corners[i+1,0])),  (255,255,255), 1)
		cv2.line(image, (int(self.corners[3,1]),int(self.corners[3,0])), (int(self.corners[0,1]),int(self.corners[0,0])),  (255,255,255), 1)

		return image
	
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


class DetectorDisplay:
	def __init__(self):
		self.array = [cv2.imread(file, 0) for file in sorted(glob.glob("out/*.png"))]
		self.detector = Detector(self)
		self.detector._show_detections = True

	def run(self):
		self.showWindow()
		self.detector.initMOG()

		for i in range(self.getFrameCount()):
			self.readParameters()
			images = self.detector.compute(i, self.getFrame(i), True)
			print(images)
			self.updateWindows(*images)

	def showWindow(self):
		cv2.namedWindow('fg_mask_mog', 1)
		cv2.namedWindow('image_o_rgb', 1)
		cv2.namedWindow('image_o_gray', 1)
		cv2.namedWindow('fg_mask_filt', 1)

		cv2.createTrackbar('mog_var_thresh', 'image_o_rgb', 5, 30, nothing)
		cv2.createTrackbar('median_size', 'image_o_rgb', 1, 21, nothing)
		cv2.createTrackbar('min_fg_pixels', 'image_o_rgb', 10, 100, nothing)

		cv2.setTrackbarPos('mog_var_thresh','image_o_rgb', self.detector.mog_parameters.mog_var_thresh)
		cv2.setTrackbarPos('min_fg_pixels','image_o_rgb', self.detector.parameters.min_fg_pixels)
		cv2.setTrackbarPos('median_size','image_o_rgb', self.detector.parameters.median_size)

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
		self.detector.mog_parameters.mog_var_thresh = cv2.getTrackbarPos('mog_var_thresh','image_o_rgb')
		self.detector.parameters.min_fg_pixels = cv2.getTrackbarPos('min_fg_pixels','image_o_rgb')
		self.detector.parameters.median_size = int(round_up_to_odd(cv2.getTrackbarPos('median_size','image_o_rgb')))

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
		# detections = detector.compute(ind, frame)
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
	playback_manager.fps = 1000
	playback_manager.openTestFile()
	playback_manager.frame_available.append(forwardImage)
	detector = Detector(playback_manager)
	detector.mog_parameters.nof_bg_frames = 100
	detector._show_detections = True
	playback_manager.mapping_done.append(startDetector)
	playback_manager.frame_available.insert(0, detector.compute_from_event)

	figure = TestFigure(playback_manager.togglePlay)
	main_window.setCentralWidget(figure)

	main_window.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	#simpleTest()
	playbackTest()
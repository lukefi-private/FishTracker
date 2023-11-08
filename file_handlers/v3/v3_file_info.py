"""
Python3 module
provided by the University of Oulu in collaboration with
LUKE-OY. The software is intended to be an open-source.

author: Mina Ghobrial.
date:   April 19th, 2018.

References: 
#   https://github.com/SoundMetrics
#   https://github.com/EminentCodfish/pyARIS

"""
import struct
import file_handlers.v3.v3_frame_info as frame
import os
import json
import file_handlers.utils as utils
import file_handlers.beamLookUp as beamLookUp
import numpy as np
import re
import cv2
import array

cwd = os.getcwd()
JSON_FILE_PATH = cwd + "/file_handlers/v3/v3_file_headers_info.json"

class v3_File:
    """
    Abstraction of the ARIS file format.

    The following class contains all the tools needed
    to read, write and modify ARIS file formats. It also
    provides tools to export files and data in several
    file formats

    Example:
    >>> file = v3_File("sample.aris")

    Note:
        Naming Convention:
            - header values follow camel case naming convention.
            - calculated file values are in upper case
    """
    # File related calculated variables
    FILE_PATH = None
    FILE_SIZE = None
    FILE_HEADER_SIZE = None
    FILE_HEADER_NUM = 41

    # Sanity check variable
    sanity = None
    
    # Frame related calculated variables
    FRAME_SIZE = None
    ALL_FRAMES_SIZE = None
    FRAME_COUNT = None

    # ARIS File class initializer
    def __init__(self,  filename):
        try:
            with open(filename, 'rb') as fhand:
                self.FILE_PATH = filename
                self.version = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.frameCount = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.frameRate = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.highResolution = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.numRawBeams = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.sampleRate = struct.unpack(
                    utils.cType["float"], fhand.read(utils.c("float")))[0]
                self.samplesPerChannel = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.receiverGain = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.windowStart = struct.unpack(
                    utils.cType["float"], fhand.read(utils.c("float")))[0]
                self.windowLength = struct.unpack(
                    utils.cType["float"], fhand.read(utils.c("float")))[0]
                self.reverse = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.serialNumber = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.strDate = struct.unpack(
                    utils.cType["char[32]"], fhand.read(utils.c("char[32]")))[0]
                self.strHeaderID = struct.unpack(
                    utils.cType["char[256]"], fhand.read(utils.c("char[256]")))[0]
                self.userID1 = struct.unpack(
                    utils.cType["int32_t"], fhand.read(utils.c("int32_t")))[0]
                self.userID2 = struct.unpack(
                    utils.cType["int32_t"], fhand.read(utils.c("int32_t")))[0]
                self.userID3 = struct.unpack(
                    utils.cType["int32_t"], fhand.read(utils.c("int32_t")))[0]
                self.userID4 = struct.unpack(
                    utils.cType["int32_t"], fhand.read(utils.c("int32_t")))[0]
                self.startFrame = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.endFrame = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.timelapse = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.recordInterval = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.radioSecond = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.frameInterval = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.flags = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.auxFlags = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.soundVelocity = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.flags3D = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.softwareVersion = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.waterTemp = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.salinity = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.pulseLength = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.TxMode = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.versionFPGA = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.versionPSuC = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.thumbnailFI = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.fileSize = struct.unpack(
                    utils.cType["uint64_t"], fhand.read(utils.c("uint64_t")))[0]
                self.optionalHeaderSize = struct.unpack(
                    utils.cType["uint64_t"], fhand.read(utils.c("uint64_t")))[0]
                self.optionalTailSize = struct.unpack(
                    utils.cType["uint64_t"], fhand.read(utils.c("uint64_t")))[0]
                self.versionMinor = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]
                self.largeLens = struct.unpack(
                    utils.cType["uint32_t"], fhand.read(utils.c("uint32_t")))[0]

        except FileNotFoundError as e:
            raise FileNotFoundError(e.errno, e.strerror, filename)

        
        self.FRAME_SIZE = self.getFrameSize()
        self.FILE_SIZE = self.getFileSize()
        self.FILE_HEADER_SIZE = self.getFileHeaderSize()
        self.ALL_FRAMES_SIZE = self.getAllFramesSize()
        self.sanity = self.sanityChecks()

        
    ##############################################################
    #       Usable user Functions
    ##############################################################

    def __len__(self):
        """
        Returns number of frames inside the file
        """
        return self.frameCount

    def __repr__(self):
        fileName = re.search("/([a-zA-Z0-9]+).aris", self.FILE_PATH)
        return fileName.group(0)
    
    def readFrame(self, frameIndex):
        """This function reads a frame given the frame index
        of that frame and returns a class handle of the read
        frame.
        
        Arguments:
            frameIndex {[integer]} -- [specifies a frame to be read from
                                    a given file]
        
        Returns:
            [class handle] -- [returns a class handle pointing
                                to the data]
        """

        return frame.ARIS_Frame(self.FILE_PATH, frameIndex, self.FRAME_SIZE)

    def getImages(self):
        images = list()
        for frameIndex in range(self.frameCount):
            wholeFrame = frame.ARIS_Frame(self.FILE_PATH, frameIndex, self.FRAME_SIZE)
            images.append(wholeFrame.IMAGE)
        return images
    
    def printFileHeader(self):
    
        try:
            with open(JSON_FILE_PATH) as json_fhand:
                orderedSet = dict()
                file_headers = json_fhand.read()
                data = json.loads(file_headers)
                checkList = data.get("file").keys()
                headerFields = self.__dict__
                for headerField in headerFields:
                    if(headerField in checkList):
                        headerValue = headerField + " = " + str(headerFields[headerField])
                        index = str(data["file"][headerField]["order"])
                        orderedSet[index] = headerValue
                    else:
                        continue
                for i in range(self.FILE_HEADER_NUM):
                    print(orderedSet[str(i)])

        except FileNotFoundError as e:
            raise FileNotFoundError(e.errno, e.strerror, JSON_FILE_PATH)
        
        return

    def getInfo(self):
        Info = {
            "File Name": self.FILE_PATH,
            "Software Version": self.softwareVersion,
            "ARIS SN": self.serialNumber,
            "File Size": self.FILE_SIZE,
            "Number Of Frames": self.frameCount,
            "Beam Count": self.numRawBeams,
            "Samples per Beam": self.samplesPerChannel
        }
        return Info

    def fileName(self):
        return self.FILE_PATH

    def fileVersion(self):
        return self.sanity

    def exportFrameHeaders(self, format = "JSON", outputFilePath = None):
        if(format == "JSON"):
            print("Exporting JSON file...")

        elif(format == "CSV"):
            print("Exporting CSV file...")
        pass


    # def formImage(self, frameIndex):
    #     output = self.readFrame(frameIndex)
    #     createLUP(self, output)
    #     # self.warpFrame(frameIndex)
    #     remapARIS(self, output)
    #     return

    ##############################################################
    #       Functions called when initializing ARIS File Class
    ##############################################################
    
    def getFrameSize(self):
        """a function that takes an instant of the class and returns
        an integer containing the frame size in the given file.
        
        Returns:
            [integer] -- [number of beams * number of samples per channel]
        """

        return self.numRawBeams*self.samplesPerChannel

    def getFileSize(self):
        """Returns the file size on disk
        
        Returns:
            [integer] -- [Given file size]
        """

        return os.path.getsize(self.FILE_PATH)

    def getFileHeaderSize(self):
        """Returns the default header size written in the
        `file_headers_info.json`
        
        Returns:
            [integer] -- [Number of bytes that the header occupies]
        """

        size = int()
        try:
            with open(JSON_FILE_PATH) as json_fhand:
                file_headers = json_fhand.read()
                data = json.loads(file_headers)
                size = data['headerSize']['size']
        except FileNotFoundError as e:
            raise FileNotFoundError(e.errno, e.strerror, JSON_FILE_PATH)
        
        return size

    def getAllFramesSize(self):
        """Returns the size of all frames with their headers
        
        Returns:
            [integer] -- [bytes that the header file occupy]
        """

        return self.FILE_SIZE - self.FILE_HEADER_SIZE

    def sanityChecks(self):
        """
        Checking for file's sanity.
        
        returns:
            [bool] -- [True if everything working, otherwise False]

        """
        if ((self.version == 88491076)):
            # check number of frames == self.frameCount
            # for i in range(self.frameCount):
            #     x = frame.ARIS_Frame(self.FILE_PATH, i, self.FRAME_SIZE)
            #     if(x.sanityCheck() != True):
            #         return False
            return True
        return False


    def play(self):
        pass
    

def setWindowStart(configFlags, highResolution,cls):
    if((configFlags == 1) or (configFlags == 3)):
        cls.windowStart = cls.windowStart*(0.375 + (highResolution == 0)*0.375)
    if((configFlags == 0) or (configFlags == 2)):
        cls.windowStart = cls.windowStart*(0.419 + (highResolution == 0)*0.419)
    else:
        raise ValueError("configuration flag has irrelevant value.")
    return

def setWindowLength(configFlags, index, cls):
    winLengthsLists = {
        0 : [0.83, 2.5, 5, 10, 20, 40],
        1 : [1.125, 2.25, 4.5, 9, 18, 36],
        2 : [2.5, 5, 10, 20, 40, 70],
        3 : [2.25, 4.5, 9, 18, 36, 72]
    }
    cls.windowLength = winLengthsLists[configFlags][index]
    return

def setFirstBeamAngle(highResolution, cls):
    cls.firstBeamAngle = beamLookUp.BeamLookUp(cls.BEAM_COUNT, 0)[-1]

    return    

def v3_getAllFramesData(fhand, version, cls):
    """Opens a .ddf file and extracts all bytes for all frames and returns a
    list containing all frames data, to be used in drawing the images.
    For images to be drawn from frames, the following attributes are needed
    from this function:
        - SONAR Sample Data     --> `allFrames`
        - Number of Beams [fl]  --> `numRawBeams`
        - Samples Per Beam [fl] --> `samplesPerChannel`
        - Type of Lens  [fr]    --> `largeLens`
        - Sample Start Delay[fr]--> `sampleStartDelay`
        - Sound Velocity[fr]    --> `soundSpeed`
        - Sample Period[fr]     --> `samplePeriod`

    """
    
    ## TODO _
    
    cls.version = "DDF_03"
    fileAttributesList = ["numRawBeams", "samplesPerChannel", "frameCount", "highResolution", "serialNumber"]
    frameAttributesList = ["configFlags", "windowStart", "windowLengthIndex"]

    fileHeader = utils.getFileHeaderValue(version, fileAttributesList)
    frameHeader = utils.getFrameHeaderValue(version, frameAttributesList)
    print("inside v3_getAllFramesData(fhand)")
    #   Reading Number of frames in the file [from file header]
    fhand.seek(fileHeader["frameCount"]["location"], 0)
    cls.frameCount = struct.unpack(
            utils.cType[fileHeader["frameCount"]["size"]],
            fhand.read(utils.c(fileHeader["frameCount"]["size"])))[0]

    #   Reading highResolution value to detect whether Hi/Lo frequency
    #   [from file header]
    fhand.seek(fileHeader["highResolution"]["location"], 0)
    highResolution = struct.unpack(
            utils.cType[fileHeader["highResolution"]["size"]],
            fhand.read(utils.c(fileHeader["highResolution"]["size"])))[0]

    #   Reading number of beams in each frame [from file header]
    fhand.seek(fileHeader["numRawBeams"]["location"], 0)
    cls.BEAM_COUNT = struct.unpack(
            utils.cType[fileHeader["numRawBeams"]["size"]],
            fhand.read(utils.c(fileHeader["numRawBeams"]["size"])))[0]

    #   Reading number of samples in each beam [from file header]
    fhand.seek(fileHeader["samplesPerChannel"]["location"], 0)
    cls.samplesPerBeam = struct.unpack(
            utils.cType[fileHeader["samplesPerChannel"]["size"]],
            fhand.read(utils.c(fileHeader["samplesPerChannel"]["size"])))[0]

    #   Reading Serial number of file format to decide configuration flags later
    #   [from file header]
    fhand.seek(fileHeader["serialNumber"]["location"], 0)
    serialNumber = struct.unpack(
            utils.cType[fileHeader["serialNumber"]["size"]],
            fhand.read(utils.c(fileHeader["serialNumber"]["size"])))[0]

    frameoffset = cls.FILE_HEADER_SIZE
    #   Reading Sample Period [from frame header]
    # fhand.seek(frameoffset + fhand.seek(frameHeader["samplePeriod"]["location"], 0))
    # cls.samplePeriod = struct.unpack(
    #         utils.cType[frameHeader["samplePeriod"]["size"]],
    #         fhand.read(utils.c(frameHeader["samplePeriod"]["size"])))[0]

    #   Reading window start length [from frame header]
    fhand.seek(frameoffset + fhand.seek(frameHeader["windowStart"]["location"], 0))
    cls.windowStart = struct.unpack(
            utils.cType[frameHeader["windowStart"]["size"]],
            fhand.read(utils.c(frameHeader["windowStart"]["size"])))[0]
    
    #   Reading window length index [from frame header]
    #   will be modified and used to determine window length later
    fhand.seek(frameoffset + fhand.seek(frameHeader["windowLengthIndex"]["location"], 0))
    windowLengthIndex = struct.unpack(
            utils.cType[frameHeader["windowLengthIndex"]["size"]],
            fhand.read(utils.c(frameHeader["windowLengthIndex"]["size"])))[0]

    #   Reading configuration flags [from frame header]
    if (serialNumber < 19):
        configFlags = 1
    elif (serialNumber == 15):
        configFlags = 3
    else:
        fhand.seek(frameoffset + fhand.seek(frameHeader["configFlags"]["location"], 0))
        configFlags = struct.unpack(
                utils.cType[frameHeader["configFlags"]["size"]],
                fhand.read(utils.c(frameHeader["configFlags"]["size"])))[0]
        ## bit0: 1=classic, 0=extended windows; bit1: 0=Standard, 1=LR

        # configFlagsStr = bin(configFlags) # Doesn't include 0's at the front
        configFlagsStr = '{:032b}'.format(configFlags) # 32bit has zeros at the front
        # Maybe irrelevant. No testing with configflags 1 or 3 
        configFlags = 2 * int(configFlagsStr[-2]) + int(configFlagsStr[-1]) 

        ## TODO _ : this needs to be completed
    
    windowLengthIndex = windowLengthIndex + 2 * (highResolution == 0)

    setWindowLength(configFlags, windowLengthIndex, cls)
    setWindowStart(configFlags, highResolution, cls)
    setFirstBeamAngle(not highResolution, cls)

    frameSize = cls.BEAM_COUNT * cls.samplesPerBeam
    frameoffset = cls.FILE_HEADER_SIZE + cls.FRAME_HEADER_SIZE
    fhand.seek(frameoffset, 0)
    strCat = frameSize*"B"
    
    cls.FRAMES = np.array(struct.unpack(strCat, fhand.read(frameSize)), dtype=np.uint8)
    cls.FRAMES = cv2.flip(cls.FRAMES.reshape((cls.samplesPerBeam, cls.BEAM_COUNT)), 0)
    cls.DATA_SHAPE = cls.FRAMES.shape
    
    cls.FRAMES = cls.constructImages(cls.FRAMES)
    return

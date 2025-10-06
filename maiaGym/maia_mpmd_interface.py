#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np

mapCommandToTag = {
    "timeStep" : 0,
    "lbRequest" : 1,
    "lbContinue" : 2,
    "lbControlActions" : 3,
    "lbProbePoint" : 4,
    "lbReinit" : 5,
    "lbForce" : 6
    }

class MaiaInterface:
  """
  This class provides an interface to m-AIA via the MPI multiple program
  multiple data (MPMD) execution model.
  Therefore, the m-AIA binary <maia> and python programming using an instance
  of this class <controller> needs to be called e.g. as following
    mpirun -np <#ranksController> <controller> : -np <#ranksMaia> <maia>
  This way both programs share the MPI_COMM_WORLD being able to communicate via
  usual MPI function calls.
  """
  def __init__(self, nDim):
    """
    Initialize member variables.
    """
    self.worldComm = None
    self.appComm = None
    self.appRank = None
    self.appRoot = 0
    self.appRootInWorld = None # in MPI_COMM_WORLD !
    self.appNoRanks = None
    self.appGroup = None
    self.appnum = None
    self.remoteRoot = None
    self.nDim = nDim

  def init_comm(self, commWorld):
    """
    Initialize the communication with m-AIA.
    commWorld : MPI communicator
      Usualy just provide mpi4py.MPI.COMM_WORLD
    """
    self.appnum = commWorld.Get_attr(MPI.APPNUM)
    rankWorld = commWorld.Get_rank()
    self.worldComm = commWorld
    self.appComm = commWorld.Split(self.appnum, rankWorld)
    self.appRank = self.appComm.Get_rank()
    self.appNoRanks = self.appComm.Get_size()
    self.appGroup = self.appComm.Get_group()
    # get root of other application (here, always neccessary)
    groupWorld = commWorld.Get_group()
    self.appRootInWorld = groupWorld.Translate_ranks([self.appRoot], self.appGroup)[0]
    noApp = 2
    buffSend = np.zeros(noApp, dtype='i')
    appRootsInWorld = np.empty_like(buffSend)
    buffSend.fill(-1)
    buffSend[self.appnum] = self.appRootInWorld
    self.worldComm.Allreduce(buffSend, appRootsInWorld, op=MPI.MAX)
    self.remoteRoot = appRootsInWorld[1 - self.appnum]
    print(f"python: appRootsInWorld: {appRootsInWorld[0]}, {appRootsInWorld[1]}")

  def __commSend(self, name, data, sendTag = True):
    """
    Communication between this root and m-AIA root process. Send to m-AIA.
    name : str
      Name of the communcation
    data : list,array,.. of doubles
      Data to be communicated
    sendTag : bool
      Whether the info of the 'name' shall be send in advance of data send
    """
    if self.appRank == self.appRoot:
      mpiTag = mapCommandToTag[name]
      if sendTag:
        sendBuf = np.zeros(1, dtype='i')
        sendBuf[0] = mpiTag
        self.worldComm.Ssend(sendBuf, dest=self.remoteRoot, tag=mapCommandToTag["lbRequest"])
      sendBuf = np.zeros(len(data), dtype=np.float64)
      for i, value in enumerate(data):
        sendBuf[i] = value
      self.worldComm.Ssend(sendBuf, dest=self.remoteRoot, tag=mpiTag)

  def __commSendI(self, name, data):
    """
    Communication between this root and m-AIA root process. Send to m-AIA.
    name : str
      Name of the communcation
    data : list,array,.. of integers
      Data to be communicated
    """
    if self.appRank == self.appRoot:
      mpiTag = mapCommandToTag[name]
      sendBuf = np.zeros(1, dtype='i')
      sendBuf[0] = mpiTag
      self.worldComm.Ssend(sendBuf, dest=self.remoteRoot, tag=mapCommandToTag["lbRequest"])
      sendBuf = np.zeros(len(data), dtype='i')
      for i, value in enumerate(data):
        sendBuf[i] = value
      self.worldComm.Ssend(sendBuf, dest=self.remoteRoot, tag=mpiTag)

  def __commRecv(self, name):
    """
    Communication between this root and m-AIA root process. Receive from m-AIA.
    name : str
      Name of the communcation
    return : list,array,.. of doubles
      Data to be communicated
    """
    if self.appRank == self.appRoot:
      mpiTag = mapCommandToTag[name]
      # probe msg from any
      status = MPI.Status()
      self.worldComm.Probe(tag=mpiTag, status=status)
      size = status.Get_elements(MPI.DOUBLE)
      recvBuf = np.zeros(size, dtype=np.float64)
      self.worldComm.Recv(recvBuf, source=self.remoteRoot, tag=mpiTag)
      return recvBuf
    else:
      return None

  def continueRun(self):
    """
    """
    if self.appRank == self.appRoot:
      mpiTag = mapCommandToTag["lbContinue"]
      sendBuf = np.zeros(1, dtype='i')
      sendBuf[0] = mpiTag
      self.worldComm.Ssend(sendBuf, dest=self.remoteRoot, tag=mapCommandToTag["lbRequest"])

  def runTimeSteps(self, timeSteps:int=1):
    """
    timeSteps : int
      Number of time steps m-AIA shall progress.
    """
    if self.appRank == self.appRoot:
      sendBuf = np.zeros(1, dtype='i')
      sendBuf[0] = timeSteps
      self.worldComm.Ssend(sendBuf, dest=self.remoteRoot, tag=mapCommandToTag["timeStep"])

  def finishRun(self):
    """
    """
    self.runTimeSteps(0)

  def setControlProperties(self, control_actions):
    """
    Setting LB properties for controlling the BC
    jetVelocities : list,array,.. of doubles
      e.g. for 2D and n noJets [u0,v0, u1,v1, .., un, vn]
    """
    self.__commSend("lbControlActions", control_actions)

  def getProbeData(self, probePointCoords):
    """
    probePointCoords: list,array,.. of doubles
      e.g. for 2D and three probe points [x0,y0, x1,y1, x2,y2]
    Return:
      np.array with format 2D(3D) [time, rho, u, v (, w)]
    """
    noProbes = int(len(probePointCoords) / self.nDim)
    self.__commSendI("lbProbePoint", [noProbes])
    self.__commSend("lbProbePoint", probePointCoords, False)
    probeStates = self.__commRecv("lbProbePoint")
    return probeStates

  def getForce(self, bcSegmentId):
    """
    Getting force on a boundary with index bcSegmentId.
    bcSegmentId: integer
      Specify the BC for which the force is calculated.
    Return: np.array of length nDim containing doubles
    """
    self.__commSendI("lbForce", [bcSegmentId])
    force = self.__commRecv("lbForce")
    return force

  def reinit(self):
    """
    TODO
    At the moment just triggering to recall the initialization routine.
    """
    if self.appRank == self.appRoot:
      mpiTag = mapCommandToTag["lbReinit"]
      sendBuf = np.zeros(1, dtype='i')
      sendBuf[0] = mpiTag
      self.worldComm.Ssend(sendBuf, dest=self.remoteRoot, tag=mapCommandToTag["lbRequest"])

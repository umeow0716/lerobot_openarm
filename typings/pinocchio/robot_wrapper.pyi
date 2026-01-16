from __future__ import annotations
from pinocchio import pinocchio_pywrap_default as pin
from pinocchio.shortcuts import buildModelsFromMJCF
from pinocchio.shortcuts import buildModelsFromSdf
from pinocchio.shortcuts import buildModelsFromUrdf
from pinocchio.shortcuts import createDatas
from pinocchio import utils
__all__: list = ['RobotWrapper']
class RobotWrapper:
    @staticmethod
    def BuildFromMJCF(filename, *args, **kwargs):
        ...
    @staticmethod
    def BuildFromSDF(filename, *args, **kwargs):
        ...
    @staticmethod
    def BuildFromURDF(filename, *args, **kwargs):
        ...
    def Jcom(self, q):
        ...
    def __init__(self, model = ..., collision_model = None, visual_model = None, verbose = False):
        ...
    def acceleration(self, q, v, a, index, update_kinematics = True, reference_frame = ...):
        ...
    def acom(self, q, v, a):
        ...
    def buildReducedRobot(self, list_of_joints_to_lock, reference_configuration = None):
        """
        
                Build a reduced robot model given a list of joints to lock.
                Parameters:
                	list_of_joints_to_lock: list of joint indexes/names to lock.
                	reference_configuration: reference configuration to compute the
                placement of the lock joints. If not provided, reference_configuration
                defaults to the robot's neutral configuration.
        
                Returns: a new robot model.
                
        """
    def centroidal(self, q, v):
        """
        
                Computes all the quantities related to the centroidal dynamics (hg, Ag and Ig),
                corresponding to the centroidal momentum, the centroidal map and the centroidal
                rigid inertia.
                
        """
    def centroidalMap(self, q):
        """
        
                Computes the centroidal momentum matrix which maps from the joint velocity
                vector to the centroidal momentum expressed around the center of mass.
                
        """
    def centroidalMomentum(self, q, v):
        ...
    def centroidalMomentumVariation(self, q, v, a):
        ...
    def classicalAcceleration(self, q, v, a, index, update_kinematics = True, reference_frame = ...):
        ...
    def com(self, q = None, v = None, a = None):
        ...
    def computeFrameJacobian(self, q, frame_id):
        """
        
                Similar to getFrameJacobian but does not need pin.computeJointJacobians and
                pin.updateFramePlacements to update internal value of self.data related to
                frames.
                
        """
    def computeJointJacobian(self, q, index):
        ...
    def computeJointJacobians(self, q):
        ...
    def display(self, q):
        """
        
                Display the robot at configuration q in the viewer by placing all the bodies.
                
        """
    def displayCollisions(self, visibility):
        """
        Set whether to diplay collision objects or not
        """
    def displayVisuals(self, visibility):
        """
        Set whether to diplay visual objects or not
        """
    def forwardKinematics(self, q, v = None, a = None):
        ...
    def frameAcceleration(self, q, v, a, index, update_kinematics = True, reference_frame = ...):
        ...
    def frameClassicalAcceleration(self, q, v, a, index, update_kinematics = True, reference_frame = ...):
        ...
    def framePlacement(self, q, index, update_kinematics = True):
        ...
    def frameVelocity(self, q, v, index, update_kinematics = True, reference_frame = ...):
        ...
    def framesForwardKinematics(self, q):
        ...
    def getFrameJacobian(self, frame_id, rf_frame = ...):
        """
        
                It computes the Jacobian of frame given by its id (frame_id) either expressed in
                the local coordinate frame or in the world coordinate frame.
                
        """
    def getJointJacobian(self, index, rf_frame = ...):
        ...
    def getViewerNodeName(self, geometry_object, geometry_type):
        """
        
                For each geometry object, returns the corresponding name of the node in the
                display.
                
        """
    def gravity(self, q):
        ...
    def index(self, name):
        ...
    def initFromMJCF(self, filename, *args, **kwargs):
        ...
    def initFromSDF(self, filename, *args, **kwargs):
        ...
    def initFromURDF(self, filename, *args, **kwargs):
        ...
    def initViewer(self, share_data = True, *args, **kwargs):
        """
        Init the viewer
        """
    def loadViewerModel(self, *args, **kwargs):
        """
        Create the scene displaying the robot meshes in MeshCat
        """
    def mass(self, q):
        ...
    def nle(self, q, v):
        ...
    def placement(self, q, index, update_kinematics = True):
        ...
    def play(self, q_trajectory, dt):
        """
        Play a trajectory with given time step
        """
    def rebuildData(self):
        """
        Re-build the data objects. Needed if the models were modified.
                Warning: this will delete any information stored in all data objects.
        """
    def setVisualizer(self, visualizer, init = True, copy_models = False):
        """
        
                Set the visualizer. If init is True, the visualizer is initialized with this
                wrapper's models.  If copy_models is also True, the models are copied.
                Otherwise, they are simply kept as a reference.
                
        """
    def updateGeometryPlacements(self, q = None, visual = False):
        ...
    def vcom(self, q, v):
        ...
    def velocity(self, q, v, index, update_kinematics = True, reference_frame = ...):
        ...
    @property
    def nq(self):
        ...
    @property
    def nv(self):
        ...
    @property
    def viewer(self):
        ...

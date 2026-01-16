from __future__ import annotations
from builtins import float as ScalarType
from coal.coal_pywrap import AngleAxis
from coal.coal_pywrap import CachedMeshLoader
from coal.coal_pywrap import CollisionGeometry
from coal.coal_pywrap import CollisionResult
from coal.coal_pywrap import Contact
from coal.coal_pywrap import DistanceResult
from coal.coal_pywrap import MeshLoader
from coal.coal_pywrap import Quaternion
from coal.coal_pywrap import StdVec_CollisionResult
from coal.coal_pywrap import StdVec_Contact
from coal.coal_pywrap import StdVec_DistanceResult
import hppfcl as hppfcl
import inspect as inspect
import numpy as numpy
from pinocchio.explog import exp
from pinocchio.explog import log
from pinocchio.pinocchio_pywrap_default import *
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.shortcuts import buildModelsFromMJCF
from pinocchio.shortcuts import buildModelsFromSdf
from pinocchio.shortcuts import buildModelsFromUrdf
from pinocchio.shortcuts import createDatas
import sys as sys
from . import deprecated
from . import explog
from . import pinocchio_pywrap_default
from . import robot_wrapper
from . import shortcuts
from . import utils
__all__: list[str] = ['ACCELERATION', 'ADMMContactSolver', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'AngleAxis', 'ArgumentPosition', 'BODY', 'BaumgarteCorrectorParameters', 'BroadPhaseManager_DynamicAABBTreeArrayCollisionManager', 'BroadPhaseManager_DynamicAABBTreeCollisionManager', 'BroadPhaseManager_IntervalTreeCollisionManager', 'BroadPhaseManager_NaiveCollisionManager', 'BroadPhaseManager_SSaPCollisionManager', 'BroadPhaseManager_SaPCollisionManager', 'COLLISION', 'CachedMeshLoader', 'CollisionCallBackBase', 'CollisionCallBackDefault', 'CollisionGeometry', 'CollisionObject', 'CollisionPair', 'CollisionResult', 'ComputeCollision', 'ComputeDistance', 'Contact', 'ContactCholeskyDecomposition', 'ContactType', 'Convention', 'CoulombFrictionCone', 'Data', 'DelassusCholeskyExpression', 'DelassusOperatorDense', 'DelassusOperatorSparse', 'DistanceResult', 'DualCoulombFrictionCone', 'Exception', 'FIXED_JOINT', 'Force', 'Frame', 'FrameType', 'GeometryData', 'GeometryModel', 'GeometryNoMaterial', 'GeometryObject', 'GeometryPhongMaterial', 'GeometryType', 'Inertia', 'JOINT', 'JointData', 'JointDataComposite', 'JointDataFreeFlyer', 'JointDataHX', 'JointDataHY', 'JointDataHZ', 'JointDataHelicalUnaligned', 'JointDataMimic_JointDataRX', 'JointDataMimic_JointDataRY', 'JointDataMimic_JointDataRZ', 'JointDataPX', 'JointDataPY', 'JointDataPZ', 'JointDataPlanar', 'JointDataPrismaticUnaligned', 'JointDataRUBX', 'JointDataRUBY', 'JointDataRUBZ', 'JointDataRX', 'JointDataRY', 'JointDataRZ', 'JointDataRevoluteUnaligned', 'JointDataRevoluteUnboundedUnalignedTpl', 'JointDataSpherical', 'JointDataSphericalZYX', 'JointDataTranslation', 'JointDataUniversal', 'JointModel', 'JointModelComposite', 'JointModelFreeFlyer', 'JointModelHX', 'JointModelHY', 'JointModelHZ', 'JointModelHelicalUnaligned', 'JointModelMimic_JointModelRX', 'JointModelMimic_JointModelRY', 'JointModelMimic_JointModelRZ', 'JointModelPX', 'JointModelPY', 'JointModelPZ', 'JointModelPlanar', 'JointModelPrismaticUnaligned', 'JointModelRUBX', 'JointModelRUBY', 'JointModelRUBZ', 'JointModelRX', 'JointModelRY', 'JointModelRZ', 'JointModelRevoluteUnaligned', 'JointModelRevoluteUnboundedUnaligned', 'JointModelSpherical', 'JointModelSphericalZYX', 'JointModelTranslation', 'JointModelUniversal', 'KinematicLevel', 'LOCAL', 'LOCAL_WORLD_ALIGNED', 'LanczosDecomposition', 'LieGroup', 'LogCholeskyParameters', 'LogLevel', 'MeshLoader', 'Model', 'Motion', 'OP_FRAME', 'PGSContactSolver', 'PINOCCHIO_MAJOR_VERSION', 'PINOCCHIO_MINOR_VERSION', 'PINOCCHIO_PATCH_VERSION', 'POSITION', 'PowerIterationAlgo', 'ProximalSettings', 'PseudoInertia', 'Quaternion', 'ReferenceFrame', 'RigidConstraintData', 'RigidConstraintModel', 'RobotWrapper', 'SE3', 'SENSOR', 'ScalarType', 'SolverStats', 'StdMap_String_VectorXd', 'StdVec_Bool', 'StdVec_CollisionObject', 'StdVec_CollisionPair', 'StdVec_CollisionResult', 'StdVec_ComputeCollision', 'StdVec_ComputeDistance', 'StdVec_Contact', 'StdVec_CoulombFrictionCone', 'StdVec_DistanceResult', 'StdVec_Double', 'StdVec_DualCoulombFrictionCone', 'StdVec_FCL_CollisionObjectPointer', 'StdVec_Force', 'StdVec_Frame', 'StdVec_GeometryModel', 'StdVec_GeometryObject', 'StdVec_Index', 'StdVec_IndexVector', 'StdVec_Inertia', 'StdVec_JointDataVector', 'StdVec_JointModelVector', 'StdVec_Matrix6', 'StdVec_Matrix6x', 'StdVec_MatrixXs', 'StdVec_Motion', 'StdVec_RigidConstraintData', 'StdVec_RigidConstraintModel', 'StdVec_SE3', 'StdVec_Scalar', 'StdVec_StdString', 'StdVec_Symmetric3', 'StdVec_Vector3', 'StdVec_VectorXb', 'StdVec_int', 'Symmetric3', 'TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager', 'TreeBroadPhaseManager_DynamicAABBTreeCollisionManager', 'TreeBroadPhaseManager_IntervalTreeCollisionManager', 'TreeBroadPhaseManager_NaiveCollisionManager', 'TreeBroadPhaseManager_SSaPCollisionManager', 'TreeBroadPhaseManager_SaPCollisionManager', 'TridiagonalSymmetricMatrix', 'VELOCITY', 'VISUAL', 'WITH_CPPAD', 'WITH_HPP_FCL', 'WITH_HPP_FCL_BINDINGS', 'WITH_OPENMP', 'WITH_SDFORMAT', 'WITH_URDFDOM', 'WORLD', 'XAxis', 'YAxis', 'ZAxis', 'boost_type_index', 'buildModelsFromMJCF', 'buildModelsFromSdf', 'buildModelsFromUrdf', 'cholesky', 'createDatas', 'deprecated', 'exp', 'explog', 'hppfcl', 'inspect', 'liegroups', 'linalg', 'log', 'map_indexing_suite_StdMap_String_VectorXd_entry', 'module_info', 'numpy', 'pin', 'pinocchio_pywrap_default', 'robot_wrapper', 'rpy', 'serialization', 'shortcuts', 'std_type_index', 'submodules', 'sys', 'utils']
ACCELERATION: pinocchio_pywrap_default.KinematicLevel  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.ACCELERATION
ARG0: pinocchio_pywrap_default.ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG0
ARG1: pinocchio_pywrap_default.ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG1
ARG2: pinocchio_pywrap_default.ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG2
ARG3: pinocchio_pywrap_default.ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG3
ARG4: pinocchio_pywrap_default.ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG4
BODY: pinocchio_pywrap_default.FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.BODY
COLLISION: pinocchio_pywrap_default.GeometryType  # value = pinocchio.pinocchio_pywrap_default.GeometryType.COLLISION
FIXED_JOINT: pinocchio_pywrap_default.FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.FIXED_JOINT
JOINT: pinocchio_pywrap_default.FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.JOINT
LOCAL: pinocchio_pywrap_default.ReferenceFrame  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL
LOCAL_WORLD_ALIGNED: pinocchio_pywrap_default.ReferenceFrame  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL_WORLD_ALIGNED
OP_FRAME: pinocchio_pywrap_default.FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.OP_FRAME
PINOCCHIO_MAJOR_VERSION: int = 3
PINOCCHIO_MINOR_VERSION: int = 4
PINOCCHIO_PATCH_VERSION: int = 0
POSITION: pinocchio_pywrap_default.KinematicLevel  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.POSITION
SENSOR: pinocchio_pywrap_default.FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.SENSOR
VELOCITY: pinocchio_pywrap_default.KinematicLevel  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.VELOCITY
VISUAL: pinocchio_pywrap_default.GeometryType  # value = pinocchio.pinocchio_pywrap_default.GeometryType.VISUAL
WITH_CPPAD: bool = False
WITH_HPP_FCL: bool = True
WITH_HPP_FCL_BINDINGS: bool = True
WITH_OPENMP: bool = False
WITH_SDFORMAT: bool = False
WITH_URDFDOM: bool = True
WORLD: pinocchio_pywrap_default.ReferenceFrame  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.WORLD
XAxis: numpy.ndarray  # value = array([1., 0., 0.])
YAxis: numpy.ndarray  # value = array([0., 1., 0.])
ZAxis: numpy.ndarray  # value = array([0., 0., 1.])
__raw_version__: str = '3.4.0'
__version__: str = '3.4.0'
module_info: tuple = ('serialization', pinocchio.pinocchio_pywrap_default.serialization)
submodules: list = [('cholesky', pinocchio.pinocchio_pywrap_default.cholesky), ('liegroups', pinocchio.pinocchio_pywrap_default.liegroups), ('linalg', pinocchio.pinocchio_pywrap_default.linalg), ('rpy', pinocchio.pinocchio_pywrap_default.rpy), ('serialization', pinocchio.pinocchio_pywrap_default.serialization)]
pin = pinocchio_pywrap_default

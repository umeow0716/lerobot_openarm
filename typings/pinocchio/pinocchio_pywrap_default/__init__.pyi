from __future__ import annotations
from builtins import float as ScalarType
import coal.coal_pywrap
from coal.coal_pywrap import AngleAxis
from coal.coal_pywrap import Quaternion
import numpy
import typing
from . import cholesky
from . import liegroups
from . import linalg
from . import rpy
from . import serialization
__all__: list[str] = ['ACCELERATION', 'ADMMContactSolver', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'AngleAxis', 'ArgumentPosition', 'BODY', 'BaumgarteCorrectorParameters', 'BroadPhaseManager_DynamicAABBTreeArrayCollisionManager', 'BroadPhaseManager_DynamicAABBTreeCollisionManager', 'BroadPhaseManager_IntervalTreeCollisionManager', 'BroadPhaseManager_NaiveCollisionManager', 'BroadPhaseManager_SSaPCollisionManager', 'BroadPhaseManager_SaPCollisionManager', 'COLLISION', 'CollisionCallBackBase', 'CollisionCallBackDefault', 'CollisionObject', 'CollisionPair', 'ComputeCollision', 'ComputeDistance', 'ContactCholeskyDecomposition', 'ContactType', 'Convention', 'CoulombFrictionCone', 'Data', 'DelassusCholeskyExpression', 'DelassusOperatorDense', 'DelassusOperatorSparse', 'DualCoulombFrictionCone', 'Exception', 'FIXED_JOINT', 'Force', 'Frame', 'FrameType', 'GeometryData', 'GeometryModel', 'GeometryNoMaterial', 'GeometryObject', 'GeometryPhongMaterial', 'GeometryType', 'Hlog3', 'Inertia', 'JOINT', 'Jexp3', 'Jexp6', 'Jlog3', 'Jlog6', 'JointData', 'JointDataComposite', 'JointDataFreeFlyer', 'JointDataHX', 'JointDataHY', 'JointDataHZ', 'JointDataHelicalUnaligned', 'JointDataMimic_JointDataRX', 'JointDataMimic_JointDataRY', 'JointDataMimic_JointDataRZ', 'JointDataPX', 'JointDataPY', 'JointDataPZ', 'JointDataPlanar', 'JointDataPrismaticUnaligned', 'JointDataRUBX', 'JointDataRUBY', 'JointDataRUBZ', 'JointDataRX', 'JointDataRY', 'JointDataRZ', 'JointDataRevoluteUnaligned', 'JointDataRevoluteUnboundedUnalignedTpl', 'JointDataSpherical', 'JointDataSphericalZYX', 'JointDataTranslation', 'JointDataUniversal', 'JointModel', 'JointModelComposite', 'JointModelFreeFlyer', 'JointModelHX', 'JointModelHY', 'JointModelHZ', 'JointModelHelicalUnaligned', 'JointModelMimic_JointModelRX', 'JointModelMimic_JointModelRY', 'JointModelMimic_JointModelRZ', 'JointModelPX', 'JointModelPY', 'JointModelPZ', 'JointModelPlanar', 'JointModelPrismaticUnaligned', 'JointModelRUBX', 'JointModelRUBY', 'JointModelRUBZ', 'JointModelRX', 'JointModelRY', 'JointModelRZ', 'JointModelRevoluteUnaligned', 'JointModelRevoluteUnboundedUnaligned', 'JointModelSpherical', 'JointModelSphericalZYX', 'JointModelTranslation', 'JointModelUniversal', 'KinematicLevel', 'LOCAL', 'LOCAL_WORLD_ALIGNED', 'LanczosDecomposition', 'LieGroup', 'LogCholeskyParameters', 'LogLevel', 'Model', 'Motion', 'OP_FRAME', 'PGSContactSolver', 'PINOCCHIO_MAJOR_VERSION', 'PINOCCHIO_MINOR_VERSION', 'PINOCCHIO_PATCH_VERSION', 'POSITION', 'PowerIterationAlgo', 'ProximalSettings', 'PseudoInertia', 'Quaternion', 'ReferenceFrame', 'RigidConstraintData', 'RigidConstraintModel', 'SE3', 'SE3ToXYZQUAT', 'SE3ToXYZQUATtuple', 'SENSOR', 'ScalarType', 'SolverStats', 'StdMap_String_VectorXd', 'StdVec_Bool', 'StdVec_CollisionObject', 'StdVec_CollisionPair', 'StdVec_ComputeCollision', 'StdVec_ComputeDistance', 'StdVec_CoulombFrictionCone', 'StdVec_Double', 'StdVec_DualCoulombFrictionCone', 'StdVec_FCL_CollisionObjectPointer', 'StdVec_Force', 'StdVec_Frame', 'StdVec_GeometryModel', 'StdVec_GeometryObject', 'StdVec_Index', 'StdVec_IndexVector', 'StdVec_Inertia', 'StdVec_JointDataVector', 'StdVec_JointModelVector', 'StdVec_Matrix6', 'StdVec_Matrix6x', 'StdVec_MatrixXs', 'StdVec_Motion', 'StdVec_RigidConstraintData', 'StdVec_RigidConstraintModel', 'StdVec_SE3', 'StdVec_Scalar', 'StdVec_StdString', 'StdVec_Symmetric3', 'StdVec_Vector3', 'StdVec_VectorXb', 'StdVec_int', 'Symmetric3', 'TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager', 'TreeBroadPhaseManager_DynamicAABBTreeCollisionManager', 'TreeBroadPhaseManager_IntervalTreeCollisionManager', 'TreeBroadPhaseManager_NaiveCollisionManager', 'TreeBroadPhaseManager_SSaPCollisionManager', 'TreeBroadPhaseManager_SaPCollisionManager', 'TridiagonalSymmetricMatrix', 'VELOCITY', 'VISUAL', 'WITH_CPPAD', 'WITH_HPP_FCL', 'WITH_OPENMP', 'WITH_SDFORMAT', 'WITH_URDFDOM', 'WORLD', 'XAxis', 'XYZQUATToSE3', 'YAxis', 'ZAxis', 'aba', 'appendModel', 'bodyRegressor', 'boost_type_index', 'buildGeomFromMJCF', 'buildGeomFromUrdf', 'buildGeomFromUrdfString', 'buildModelFromMJCF', 'buildModelFromUrdf', 'buildModelFromXML', 'buildReducedModel', 'buildSampleGeometryModelHumanoid', 'buildSampleGeometryModelManipulator', 'buildSampleModelHumanoid', 'buildSampleModelHumanoidRandom', 'buildSampleModelManipulator', 'ccrba', 'centerOfMass', 'checkVersionAtLeast', 'cholesky', 'classicAcceleration', 'computeABADerivatives', 'computeAllTerms', 'computeBodyRadius', 'computeCentroidalDynamicsDerivatives', 'computeCentroidalMap', 'computeCentroidalMapTimeVariation', 'computeCentroidalMomentum', 'computeCentroidalMomentumTimeVariation', 'computeCollision', 'computeCollisions', 'computeComplementarityShift', 'computeConeProjection', 'computeConstraintDynamicsDerivatives', 'computeContactForces', 'computeCoriolisMatrix', 'computeDampedDelassusMatrixInverse', 'computeDelassusMatrix', 'computeDistance', 'computeDistances', 'computeDualConeProjection', 'computeForwardKinematicsDerivatives', 'computeFrameJacobian', 'computeFrameKinematicRegressor', 'computeGeneralizedGravity', 'computeGeneralizedGravityDerivatives', 'computeImpulseDynamicsDerivatives', 'computeJointJacobian', 'computeJointJacobians', 'computeJointJacobiansTimeVariation', 'computeJointKinematicRegressor', 'computeJointTorqueRegressor', 'computeKKTContactDynamicMatrixInverse', 'computeKineticEnergy', 'computeKineticEnergyRegressor', 'computeMechanicalEnergy', 'computeMinverse', 'computePotentialEnergy', 'computePotentialEnergyRegressor', 'computePrimalFeasibility', 'computeRNEADerivatives', 'computeReprojectionError', 'computeStaticRegressor', 'computeStaticTorque', 'computeStaticTorqueDerivatives', 'computeSubtreeMasses', 'computeSupportedForceByFrame', 'computeSupportedInertiaByFrame', 'computeTotalMass', 'constraintDynamics', 'contactInverseDynamics', 'crba', 'dDifference', 'dIntegrate', 'dIntegrateTransport', 'dccrba', 'difference', 'distance', 'exp3', 'exp3_quat', 'exp6', 'exp6_quat', 'findCommonAncestor', 'forwardDynamics', 'forwardKinematics', 'frameBodyRegressor', 'frameJacobianTimeVariation', 'framesForwardKinematics', 'getAcceleration', 'getCenterOfMassVelocityDerivatives', 'getCentroidalDynamicsDerivatives', 'getClassicalAcceleration', 'getConstraintJacobian', 'getConstraintsJacobian', 'getCoriolisMatrix', 'getFrameAcceleration', 'getFrameAccelerationDerivatives', 'getFrameClassicalAcceleration', 'getFrameJacobian', 'getFrameJacobianTimeVariation', 'getFrameVelocity', 'getFrameVelocityDerivatives', 'getJacobianSubtreeCenterOfMass', 'getJointAccelerationDerivatives', 'getJointJacobian', 'getJointJacobianTimeVariation', 'getJointVelocityDerivatives', 'getKKTContactDynamicMatrixInverse', 'getPointClassicAccelerationDerivatives', 'getPointVelocityDerivatives', 'getVelocity', 'impulseDynamics', 'initConstraintDynamics', 'integrate', 'interpolate', 'isNormalized', 'isSameConfiguration', 'jacobianCenterOfMass', 'jacobianSubtreeCenterOfMass', 'jointBodyRegressor', 'liegroups', 'linalg', 'loadReferenceConfigurations', 'loadReferenceConfigurationsFromXML', 'loadRotorParameters', 'log3', 'log6', 'log6_quat', 'map_indexing_suite_StdMap_String_VectorXd_entry', 'neutral', 'nonLinearEffects', 'normalize', 'printVersion', 'randomConfiguration', 'removeCollisionPairs', 'removeCollisionPairsFromXML', 'rnea', 'rpy', 'seed', 'serialization', 'sharedMemory', 'skew', 'skewSquare', 'squaredDistance', 'std_type_index', 'unSkew', 'updateFramePlacement', 'updateFramePlacements', 'updateGeometryPlacements', 'updateGlobalPlacements']
class ADMMContactSolver(Boost.Python.instance):
    """
    Alternating Direction Method of Multi-pliers solver for contact dynamics.
    """
    __instance_size__: typing.ClassVar[int] = 568
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (int)problem_dim [, (float)mu_prox=1e-06 [, (float)tau=0.5 [, (float)rho_power=0.2 [, (float)rho_power_factor=0.05 [, (float)ratio_primal_dual=10.0 [, (int)max_it_largest_eigen_value_solver=20]]]]]]) -> None :
            Default constructor.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeRho(*args, **kwargs):
        """
        
        computeRho( (float)L, (float)m, (float)rho_power) -> float :
            Compute the penalty ADMM value from the current largest and lowest eigenvalues and the scaling spectral factor.
        """
    @staticmethod
    def computeRhoPower(*args, **kwargs):
        """
        
        computeRhoPower( (float)L, (float)m, (float)rho) -> float :
            Compute the  scaling spectral factor of the ADMM penalty term from the current largest and lowest eigenvalues and the ADMM penalty term.
        """
    @staticmethod
    def getAbsoluteConvergenceResidual(*args, **kwargs):
        """
        
        getAbsoluteConvergenceResidual( (ADMMContactSolver)self) -> float :
            Returns the value of the absolute residual value corresponding to the contact complementary conditions.
        """
    @staticmethod
    def getAbsolutePrecision(*args, **kwargs):
        """
        
        getAbsolutePrecision( (ADMMContactSolver)self) -> float :
            Get the absolute precision requested.
        """
    @staticmethod
    def getCPUTimes(*args, **kwargs):
        """
        
        getCPUTimes( (ADMMContactSolver)self) -> coal.coal_pywrap.CPUTimes
        """
    @staticmethod
    def getCholeskyUpdateCount(*args, **kwargs):
        """
        
        getCholeskyUpdateCount( (ADMMContactSolver)self) -> int :
            Returns the number of updates of the Cholesky factorization due to rho updates.
        """
    @staticmethod
    def getDualSolution(*args, **kwargs):
        """
        
        getDualSolution( (ADMMContactSolver)self) -> object :
            Returns the dual solution of the problem.
        """
    @staticmethod
    def getIterationCount(*args, **kwargs):
        """
        
        getIterationCount( (ADMMContactSolver)self) -> int :
            Get the number of iterations achieved by the PGS algorithm.
        """
    @staticmethod
    def getMaxIterations(*args, **kwargs):
        """
        
        getMaxIterations( (ADMMContactSolver)self) -> int :
            Get the maximum number of iterations allowed.
        """
    @staticmethod
    def getPowerIterationAlgo(*args, **kwargs):
        """
        
        getPowerIterationAlgo( (ADMMContactSolver)self) -> PowerIterationAlgo
        """
    @staticmethod
    def getPrimalSolution(*args, **kwargs):
        """
        
        getPrimalSolution( (ADMMContactSolver)self) -> object :
            Returns the primal solution of the problem.
        """
    @staticmethod
    def getProximalValue(*args, **kwargs):
        """
        
        getProximalValue( (ADMMContactSolver)self) -> float :
            Get the proximal value.
        """
    @staticmethod
    def getRatioPrimalDual(*args, **kwargs):
        """
        
        getRatioPrimalDual( (ADMMContactSolver)self) -> float :
            Get the primal/dual ratio.
        """
    @staticmethod
    def getRelativeConvergenceResidual(*args, **kwargs):
        """
        
        getRelativeConvergenceResidual( (ADMMContactSolver)self) -> float :
            Returns the value of the relative residual value corresponding to the difference between two successive iterates (infinity norms).
        """
    @staticmethod
    def getRelativePrecision(*args, **kwargs):
        """
        
        getRelativePrecision( (ADMMContactSolver)self) -> float :
            Get the relative precision requested.
        """
    @staticmethod
    def getRho(*args, **kwargs):
        """
        
        getRho( (ADMMContactSolver)self) -> float :
            Get the ADMM penalty value.
        """
    @staticmethod
    def getRhoPower(*args, **kwargs):
        """
        
        getRhoPower( (ADMMContactSolver)self) -> float :
            Get the power associated to the problem conditionning.
        """
    @staticmethod
    def getRhoPowerFactor(*args, **kwargs):
        """
        
        getRhoPowerFactor( (ADMMContactSolver)self) -> float :
            Get the power factor associated to the problem conditionning.
        """
    @staticmethod
    def getStats(*args, **kwargs):
        """
        
        getStats( (ADMMContactSolver)self) -> SolverStats
        """
    @staticmethod
    def getTau(*args, **kwargs):
        """
        
        getTau( (ADMMContactSolver)self) -> float :
            Get the tau linear scaling factor.
        """
    @staticmethod
    def setAbsolutePrecision(*args, **kwargs):
        """
        
        setAbsolutePrecision( (ADMMContactSolver)self, (float)absolute_precision) -> None :
            Set the absolute precision for the problem.
        """
    @staticmethod
    def setMaxIterations(*args, **kwargs):
        """
        
        setMaxIterations( (ADMMContactSolver)self, (int)max_it) -> None :
            Set the maximum number of iterations.
        """
    @staticmethod
    def setProximalValue(*args, **kwargs):
        """
        
        setProximalValue( (ADMMContactSolver)self, (float)mu) -> None :
            Set the proximal value.
        """
    @staticmethod
    def setRatioPrimalDual(*args, **kwargs):
        """
        
        setRatioPrimalDual( (ADMMContactSolver)self, (float)ratio_primal_dual) -> None :
            Set the primal/dual ratio.
        """
    @staticmethod
    def setRelativePrecision(*args, **kwargs):
        """
        
        setRelativePrecision( (ADMMContactSolver)self, (float)relative_precision) -> None :
            Set the relative precision for the problem.
        """
    @staticmethod
    def setRho(*args, **kwargs):
        """
        
        setRho( (ADMMContactSolver)self, (float)rho) -> None :
            Set the ADMM penalty value.
        """
    @staticmethod
    def setRhoPower(*args, **kwargs):
        """
        
        setRhoPower( (ADMMContactSolver)self, (float)rho_power) -> None :
            Set the power associated to the problem conditionning.
        """
    @staticmethod
    def setRhoPowerFactor(*args, **kwargs):
        """
        
        setRhoPowerFactor( (ADMMContactSolver)self, (float)rho_power_factor) -> None :
            Set the power factor associated to the problem conditionning.
        """
    @staticmethod
    def setTau(*args, **kwargs):
        """
        
        setTau( (ADMMContactSolver)self, (float)tau) -> None :
            Set the tau linear scaling factor.
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (ADMMContactSolver)self, (DelassusCholeskyExpression)delassus, (numpy.ndarray)g, (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)R [, (numpy.ndarray)primal_solution=None [, (numpy.ndarray)dual_solution=None [, (bool)compute_largest_eigen_values=True [, (bool)stat_record=False]]]]) -> bool :
            Solve the constrained conic problem, starting from the optional initial guess.
        
        solve( (ADMMContactSolver)self, (DelassusOperatorDense)delassus, (numpy.ndarray)g, (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)R [, (numpy.ndarray)primal_solution=None [, (numpy.ndarray)dual_solution=None [, (bool)compute_largest_eigen_values=True [, (bool)stat_record=False]]]]) -> bool :
            Solve the constrained conic problem, starting from the optional initial guess.
        
        solve( (ADMMContactSolver)self, (DelassusOperatorSparse)delassus, (numpy.ndarray)g, (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)R [, (numpy.ndarray)primal_solution=None [, (numpy.ndarray)dual_solution=None [, (bool)compute_largest_eigen_values=True [, (bool)stat_record=False]]]]) -> bool :
            Solve the constrained conic problem, starting from the optional initial guess.
        """
class ArgumentPosition(Boost.Python.enum):
    ARG0: typing.ClassVar[ArgumentPosition]  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG0
    ARG1: typing.ClassVar[ArgumentPosition]  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG1
    ARG2: typing.ClassVar[ArgumentPosition]  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG2
    ARG3: typing.ClassVar[ArgumentPosition]  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG3
    ARG4: typing.ClassVar[ArgumentPosition]  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG4
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'ARG0': pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG0, 'ARG1': pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG1, 'ARG2': pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG2, 'ARG3': pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG3, 'ARG4': pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG4}
    values: typing.ClassVar[dict]  # value = {0: pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG0, 1: pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG1, 2: pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG2, 3: pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG3, 4: pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG4}
class BaumgarteCorrectorParameters(Boost.Python.instance):
    """
    Paramaters of the Baumgarte Corrector.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (BaumgarteCorrectorParameters)arg1, (BaumgarteCorrectorParameters)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (int)size) -> None :
            Default constructor.
        
        __init__( (object)arg1, (BaumgarteCorrectorParameters)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.BaumgarteCorrectorParameters -> pinocchio.pinocchio_pywrap_default.BaumgarteCorrectorParameters
        
        __init__( (object)arg1, (BaumgarteCorrectorParameters)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.BaumgarteCorrectorParameters -> pinocchio.pinocchio_pywrap_default.BaumgarteCorrectorParameters
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (BaumgarteCorrectorParameters)arg1, (BaumgarteCorrectorParameters)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (BaumgarteCorrectorParameters)arg1) -> BaumgarteCorrectorParameters :
            Returns a cast of *this.
        """
    @property
    def Kd(*args, **kwargs):
        """
        Damping corrector value.
        """
    @Kd.setter
    def Kd(*args, **kwargs):
        ...
    @property
    def Kp(*args, **kwargs):
        """
        Proportional corrector value.
        """
    @Kp.setter
    def Kp(*args, **kwargs):
        ...
class BroadPhaseManager_DynamicAABBTreeArrayCollisionManager(Boost.Python.instance):
    """
    Broad phase manager associated to coal::DynamicAABBTreeArrayCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getCollisionObjectInflation(*args, **kwargs):
        """
        
        getCollisionObjectInflation( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> object :
            Returns the inflation value related to each collision object.
        """
    @staticmethod
    def getCollisionObjectStatus(*args, **kwargs):
        """
        
        getCollisionObjectStatus( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> StdVec_Bool :
            Returns the status of the collision object.
        """
    @staticmethod
    def getCollisionObjects(*args, **kwargs):
        """
        
        getCollisionObjects( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> StdVec_CollisionObject
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getManager(*args, **kwargs):
        """
        
        getManager( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> coal.coal_pywrap.DynamicAABBTreeArrayCollisionManager :
            Returns the internal FCL manager
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (BroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class BroadPhaseManager_DynamicAABBTreeCollisionManager(Boost.Python.instance):
    """
    Broad phase manager associated to coal::DynamicAABBTreeCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (BroadPhaseManager_DynamicAABBTreeCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self, (BroadPhaseManager_DynamicAABBTreeCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getCollisionObjectInflation(*args, **kwargs):
        """
        
        getCollisionObjectInflation( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> object :
            Returns the inflation value related to each collision object.
        """
    @staticmethod
    def getCollisionObjectStatus(*args, **kwargs):
        """
        
        getCollisionObjectStatus( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> StdVec_Bool :
            Returns the status of the collision object.
        """
    @staticmethod
    def getCollisionObjects(*args, **kwargs):
        """
        
        getCollisionObjects( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> StdVec_CollisionObject
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getManager(*args, **kwargs):
        """
        
        getManager( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> coal.coal_pywrap.DynamicAABBTreeCollisionManager :
            Returns the internal FCL manager
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (BroadPhaseManager_DynamicAABBTreeCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class BroadPhaseManager_IntervalTreeCollisionManager(Boost.Python.instance):
    """
    Broad phase manager associated to coal::IntervalTreeCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (BroadPhaseManager_IntervalTreeCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (BroadPhaseManager_IntervalTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (BroadPhaseManager_IntervalTreeCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (BroadPhaseManager_IntervalTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (BroadPhaseManager_IntervalTreeCollisionManager)self, (BroadPhaseManager_IntervalTreeCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getCollisionObjectInflation(*args, **kwargs):
        """
        
        getCollisionObjectInflation( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> object :
            Returns the inflation value related to each collision object.
        """
    @staticmethod
    def getCollisionObjectStatus(*args, **kwargs):
        """
        
        getCollisionObjectStatus( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> StdVec_Bool :
            Returns the status of the collision object.
        """
    @staticmethod
    def getCollisionObjects(*args, **kwargs):
        """
        
        getCollisionObjects( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> StdVec_CollisionObject
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getManager(*args, **kwargs):
        """
        
        getManager( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> coal.coal_pywrap.IntervalTreeCollisionManager :
            Returns the internal FCL manager
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (BroadPhaseManager_IntervalTreeCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (BroadPhaseManager_IntervalTreeCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (BroadPhaseManager_IntervalTreeCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class BroadPhaseManager_NaiveCollisionManager(Boost.Python.instance):
    """
    Broad phase manager associated to coal::NaiveCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (BroadPhaseManager_NaiveCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (BroadPhaseManager_NaiveCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (BroadPhaseManager_NaiveCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (BroadPhaseManager_NaiveCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (BroadPhaseManager_NaiveCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (BroadPhaseManager_NaiveCollisionManager)self, (BroadPhaseManager_NaiveCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getCollisionObjectInflation(*args, **kwargs):
        """
        
        getCollisionObjectInflation( (BroadPhaseManager_NaiveCollisionManager)self) -> object :
            Returns the inflation value related to each collision object.
        """
    @staticmethod
    def getCollisionObjectStatus(*args, **kwargs):
        """
        
        getCollisionObjectStatus( (BroadPhaseManager_NaiveCollisionManager)self) -> StdVec_Bool :
            Returns the status of the collision object.
        """
    @staticmethod
    def getCollisionObjects(*args, **kwargs):
        """
        
        getCollisionObjects( (BroadPhaseManager_NaiveCollisionManager)self) -> StdVec_CollisionObject
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (BroadPhaseManager_NaiveCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (BroadPhaseManager_NaiveCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getManager(*args, **kwargs):
        """
        
        getManager( (BroadPhaseManager_NaiveCollisionManager)self) -> coal.coal_pywrap.NaiveCollisionManager :
            Returns the internal FCL manager
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (BroadPhaseManager_NaiveCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (BroadPhaseManager_NaiveCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (BroadPhaseManager_NaiveCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class BroadPhaseManager_SSaPCollisionManager(Boost.Python.instance):
    """
    Broad phase manager associated to coal::SSaPCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (BroadPhaseManager_SSaPCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (BroadPhaseManager_SSaPCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (BroadPhaseManager_SSaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (BroadPhaseManager_SSaPCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (BroadPhaseManager_SSaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (BroadPhaseManager_SSaPCollisionManager)self, (BroadPhaseManager_SSaPCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getCollisionObjectInflation(*args, **kwargs):
        """
        
        getCollisionObjectInflation( (BroadPhaseManager_SSaPCollisionManager)self) -> object :
            Returns the inflation value related to each collision object.
        """
    @staticmethod
    def getCollisionObjectStatus(*args, **kwargs):
        """
        
        getCollisionObjectStatus( (BroadPhaseManager_SSaPCollisionManager)self) -> StdVec_Bool :
            Returns the status of the collision object.
        """
    @staticmethod
    def getCollisionObjects(*args, **kwargs):
        """
        
        getCollisionObjects( (BroadPhaseManager_SSaPCollisionManager)self) -> StdVec_CollisionObject
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (BroadPhaseManager_SSaPCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (BroadPhaseManager_SSaPCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getManager(*args, **kwargs):
        """
        
        getManager( (BroadPhaseManager_SSaPCollisionManager)self) -> coal.coal_pywrap.SSaPCollisionManager :
            Returns the internal FCL manager
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (BroadPhaseManager_SSaPCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (BroadPhaseManager_SSaPCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (BroadPhaseManager_SSaPCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class BroadPhaseManager_SaPCollisionManager(Boost.Python.instance):
    """
    Broad phase manager associated to coal::SaPCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (BroadPhaseManager_SaPCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (BroadPhaseManager_SaPCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (BroadPhaseManager_SaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (BroadPhaseManager_SaPCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (BroadPhaseManager_SaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (BroadPhaseManager_SaPCollisionManager)self, (BroadPhaseManager_SaPCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getCollisionObjectInflation(*args, **kwargs):
        """
        
        getCollisionObjectInflation( (BroadPhaseManager_SaPCollisionManager)self) -> object :
            Returns the inflation value related to each collision object.
        """
    @staticmethod
    def getCollisionObjectStatus(*args, **kwargs):
        """
        
        getCollisionObjectStatus( (BroadPhaseManager_SaPCollisionManager)self) -> StdVec_Bool :
            Returns the status of the collision object.
        """
    @staticmethod
    def getCollisionObjects(*args, **kwargs):
        """
        
        getCollisionObjects( (BroadPhaseManager_SaPCollisionManager)self) -> StdVec_CollisionObject
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (BroadPhaseManager_SaPCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (BroadPhaseManager_SaPCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getManager(*args, **kwargs):
        """
        
        getManager( (BroadPhaseManager_SaPCollisionManager)self) -> coal.coal_pywrap.SaPCollisionManager :
            Returns the internal FCL manager
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (BroadPhaseManager_SaPCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (BroadPhaseManager_SaPCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (BroadPhaseManager_SaPCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class CollisionCallBackBase(coal.coal_pywrap.CollisionCallBackBase):
    @staticmethod
    def __init__(*args, **kwargs):
        """
        Raises an exception
        This class cannot be instantiated from Python
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def done(*args, **kwargs):
        """
        
        done( (CollisionCallBackBase)arg1) -> None :
            Callback method called after the termination of a collisition detection algorithm.
        
        done( (CollisionCallBackBase)arg1) -> None
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (CollisionCallBackDefault)self) -> GeometryData
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (CollisionCallBackBase)self) -> GeometryModel
        """
    @staticmethod
    def stop(*args, **kwargs):
        """
        
        stop( (CollisionCallBackBase)self) -> bool :
            If true, the stopping criteria related to the collision callback has been met and one can stop.
        
        stop( (CollisionCallBackBase)self) -> None
        """
    @property
    def accumulate(*args, **kwargs):
        """
        Whether the callback is used in an accumulate mode where several collide methods are called successively.
        """
    @property
    def collision(*args, **kwargs):
        """
        Whether there is a collision or not.
        """
class CollisionCallBackDefault(CollisionCallBackBase):
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (GeometryModel)geometry_model, (GeometryData)geometry_data [, (bool)stopAtFirstCollision]) -> None :
            Default constructor from a given GeometryModel and a GeometryData
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def collisionPairIndex(*args, **kwargs):
        """
        The collision index of the first pair in collision
        """
    @property
    def count(*args, **kwargs):
        """
        Number of visits of the collide method
        """
    @property
    def stopAtFirstCollision(*args, **kwargs):
        """
        Whether to stop or not when localizing a first collision
        """
    @stopAtFirstCollision.setter
    def stopAtFirstCollision(*args, **kwargs):
        ...
class CollisionObject(coal.coal_pywrap.CollisionObject):
    """
    A Pinocchio collision object derived from FCL CollisionObject.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (coal.coal_pywrap.CollisionGeometry)collision_geometry [, (int)geometryObjectIndex=18446744073709551615 [, (bool)compute_local_aabb=True]]) -> None :
            Constructor
        
        __init__( (object)self, (coal.coal_pywrap.CollisionGeometry)collision_geometry, (SE3)placement [, (int)geometryObjectIndex=18446744073709551615 [, (bool)compute_local_aabb=True]]) -> None :
            Constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
class CollisionPair(Boost.Python.instance):
    """
    Pair of ordered index defining a pair of collisions
    """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (CollisionPair)self) -> CollisionPair :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (CollisionPair)self, (dict)memo) -> CollisionPair :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (CollisionPair)arg1, (CollisionPair)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Empty constructor.
        
        __init__( (object)self, (int)index1, (int)index2) -> None :
            Initializer of collision pair.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (CollisionPair)arg1, (CollisionPair)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (CollisionPair)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (CollisionPair)arg1) -> object
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (CollisionPair)self) -> CollisionPair :
            Returns a copy of *this.
        """
    @property
    def first(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.CollisionPair)arg1) -> int
        """
    @first.setter
    def first(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.CollisionPair)arg1, (int)arg2) -> None
        """
    @property
    def second(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.CollisionPair)arg1) -> int
        """
    @second.setter
    def second(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.CollisionPair)arg1, (int)arg2) -> None
        """
class ComputeCollision(Boost.Python.instance):
    """
    Collision function between two geometry objects.
    
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (GeometryObject)geometry_object1, (GeometryObject)geometry_object2) -> None :
            Constructor of a pinocchio::ComputeCollision
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def getGeometryObject1(*args, **kwargs):
        """
        
        getGeometryObject1( (ComputeCollision)arg1) -> GeometryObject
        """
    @staticmethod
    def getGeometryObject2(*args, **kwargs):
        """
        
        getGeometryObject2( (ComputeCollision)arg1) -> GeometryObject
        """
    @staticmethod
    def print(*args, **kwargs):
        """
        
        print( (ComputeCollision)arg1) -> None
        """
    @staticmethod
    def run(*args, **kwargs):
        """
        
        run( (ComputeCollision)self, (coal.coal_pywrap.Transform3s)tf1, (coal.coal_pywrap.Transform3s)tf2, (coal.coal_pywrap.CollisionRequest)request, (coal.coal_pywrap.CollisionResult)result) -> int :
            Call the function and return the result
        """
class ComputeDistance(Boost.Python.instance):
    """
    Distance function between two geometry objects.
    
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (GeometryObject)geometry_object1, (GeometryObject)geometry_object2) -> None :
            Constructor of a pinocchio::ComputeDistance
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def getGeometryObject1(*args, **kwargs):
        """
        
        getGeometryObject1( (ComputeDistance)arg1) -> GeometryObject
        """
    @staticmethod
    def getGeometryObject2(*args, **kwargs):
        """
        
        getGeometryObject2( (ComputeDistance)arg1) -> GeometryObject
        """
    @staticmethod
    def print(*args, **kwargs):
        """
        
        print( (ComputeDistance)arg1) -> None
        """
    @staticmethod
    def run(*args, **kwargs):
        """
        
        run( (ComputeDistance)self, (coal.coal_pywrap.Transform3s)tf1, (coal.coal_pywrap.Transform3s)tf2, (coal.coal_pywrap.DistanceRequest)request, (coal.coal_pywrap.DistanceResult)result) -> float :
            Call the function and return the result
        """
class ContactCholeskyDecomposition(Boost.Python.instance):
    """
    Contact information container for contact dynamic algorithms.
    """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (ContactCholeskyDecomposition)self) -> ContactCholeskyDecomposition :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (ContactCholeskyDecomposition)self, (dict)memo) -> ContactCholeskyDecomposition :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (ContactCholeskyDecomposition)arg1, (ContactCholeskyDecomposition)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor.
        
        __init__( (object)self, (Model)model) -> None :
            Constructor from a model.
        
        __init__( (object)self, (Model)model, (StdVec_RigidConstraintModel)contact_models) -> None :
            Constructor from a model and a collection of RigidConstraintModels.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (ContactCholeskyDecomposition)arg1, (ContactCholeskyDecomposition)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (ContactCholeskyDecomposition)self, (Model)model, (Data)data, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas [, (float)mu=0]) -> None :
            Computes the Cholesky decompostion of the augmented matrix containing the KKT matrix
            related to the system mass matrix and the Jacobians of the contact patches contained in
            the vector of RigidConstraintModel named contact_models. The decomposition is regularized with a factor mu.
        
        compute( (ContactCholeskyDecomposition)self, (Model)model, (Data)data, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas, (numpy.ndarray)mus) -> None :
            Computes the Cholesky decompostion of the augmented matrix containing the KKT matrix
            related to the system mass matrix and the Jacobians of the contact patches contained in
            the vector of RigidConstraintModel named contact_models. The decomposition is regularized with a factor mu.
        """
    @staticmethod
    def constraintDim(*args, **kwargs):
        """
        
        constraintDim( (ContactCholeskyDecomposition)self) -> int :
            Returns the total dimension of the constraints contained in the Cholesky factorization.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (ContactCholeskyDecomposition)self) -> ContactCholeskyDecomposition :
            Returns a copy of *this.
        """
    @staticmethod
    def getDelassusCholeskyExpression(*args, **kwargs):
        """
        
        getDelassusCholeskyExpression( (ContactCholeskyDecomposition)self) -> DelassusCholeskyExpression :
            Returns the Cholesky decomposition expression associated to the underlying Delassus matrix.
        """
    @staticmethod
    def getInverseMassMatrix(*args, **kwargs):
        """
        
        getInverseMassMatrix( (ContactCholeskyDecomposition)self) -> numpy.ndarray :
            Returns the inverse of the Joint Space Inertia Matrix or "mass matrix".
        """
    @staticmethod
    def getInverseOperationalSpaceInertiaMatrix(*args, **kwargs):
        """
        
        getInverseOperationalSpaceInertiaMatrix( (ContactCholeskyDecomposition)self) -> numpy.ndarray :
            Returns the Inverse of the Operational Space Inertia Matrix resulting from the decomposition.
        """
    @staticmethod
    def getMassMatrixChoeslkyDecomposition(*args, **kwargs):
        """
        
        getMassMatrixChoeslkyDecomposition( (ContactCholeskyDecomposition)arg1, (Model)self) -> ContactCholeskyDecomposition :
            Retrieves the Cholesky decomposition of the Mass Matrix contained in the current decomposition.
        """
    @staticmethod
    def getOperationalSpaceInertiaMatrix(*args, **kwargs):
        """
        
        getOperationalSpaceInertiaMatrix( (ContactCholeskyDecomposition)self) -> numpy.ndarray :
            Returns the Operational Space Inertia Matrix resulting from the decomposition.
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (ContactCholeskyDecomposition)self) -> numpy.ndarray :
            Returns the inverse matrix resulting from the decomposition.
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (ContactCholeskyDecomposition)self) -> numpy.ndarray :
            Returns the matrix resulting from the decomposition.
        """
    @staticmethod
    def numContacts(*args, **kwargs):
        """
        
        numContacts( (ContactCholeskyDecomposition)self) -> int :
            Returns the number of contacts associated to this decomposition.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (ContactCholeskyDecomposition)self) -> int :
            Size of the decomposition.
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (ContactCholeskyDecomposition)self, (numpy.ndarray)matrix) -> numpy.ndarray :
            Computes the solution of 
        $ A x = b 
        $ where self corresponds to the Cholesky decomposition of A.
        """
    @staticmethod
    def updateDamping(*args, **kwargs):
        """
        
        updateDamping( (ContactCholeskyDecomposition)self, (float)mu) -> None :
            Update the damping term on the upper left block part of the KKT matrix. The damping term should be positive.
        
        updateDamping( (ContactCholeskyDecomposition)self, (numpy.ndarray)mus) -> None :
            Update the damping terms on the upper left block part of the KKT matrix. The damping terms should be all positives.
        """
    @property
    def D(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.ContactCholeskyDecomposition)arg1) -> numpy.ndarray
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.ContactCholeskyDecomposition)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.ContactCholeskyDecomposition)arg1) -> numpy.ndarray
        """
class ContactType(Boost.Python.enum):
    CONTACT_3D: typing.ClassVar[ContactType]  # value = pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_3D
    CONTACT_6D: typing.ClassVar[ContactType]  # value = pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_6D
    CONTACT_UNDEFINED: typing.ClassVar[ContactType]  # value = pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_UNDEFINED
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'CONTACT_3D': pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_3D, 'CONTACT_6D': pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_6D, 'CONTACT_UNDEFINED': pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_UNDEFINED}
    values: typing.ClassVar[dict]  # value = {0: pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_3D, 1: pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_6D, 2: pinocchio.pinocchio_pywrap_default.ContactType.CONTACT_UNDEFINED}
class Convention(Boost.Python.enum):
    LOCAL: typing.ClassVar[Convention]  # value = pinocchio.pinocchio_pywrap_default.Convention.LOCAL
    WORLD: typing.ClassVar[Convention]  # value = pinocchio.pinocchio_pywrap_default.Convention.WORLD
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'WORLD': pinocchio.pinocchio_pywrap_default.Convention.WORLD, 'LOCAL': pinocchio.pinocchio_pywrap_default.Convention.LOCAL}
    values: typing.ClassVar[dict]  # value = {0: pinocchio.pinocchio_pywrap_default.Convention.WORLD, 1: pinocchio.pinocchio_pywrap_default.Convention.LOCAL}
class CoulombFrictionCone(Boost.Python.instance):
    """
    3d Coulomb friction cone.
    """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (CoulombFrictionCone)self) -> CoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (CoulombFrictionCone)self, (dict)memo) -> CoulombFrictionCone :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (CoulombFrictionCone)arg1, (CoulombFrictionCone)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (float)mu) -> None :
            Default constructor
        
        __init__( (object)self, (CoulombFrictionCone)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (CoulombFrictionCone)arg1, (CoulombFrictionCone)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def computeNormalCorrection(*args, **kwargs):
        """
        
        computeNormalCorrection( (CoulombFrictionCone)self, (numpy.ndarray)v) -> numpy.ndarray :
            Compute the complementary shift associted to the Coulomb friction cone for complementarity satisfaction in complementary problems.
        """
    @staticmethod
    def computeRadialProjection(*args, **kwargs):
        """
        
        computeRadialProjection( (CoulombFrictionCone)self, (numpy.ndarray)f) -> numpy.ndarray :
            Compute the radial projection associted to the Coulomb friction cone.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (CoulombFrictionCone)self) -> CoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def dim(*args, **kwargs):
        """
        
        dim() -> int :
            Returns the dimension of the cone.
        """
    @staticmethod
    def dual(*args, **kwargs):
        """
        
        dual( (CoulombFrictionCone)self) -> DualCoulombFrictionCone :
            Returns the dual cone associated to this
        """
    @staticmethod
    def isInside(*args, **kwargs):
        """
        
        isInside( (CoulombFrictionCone)arg1, (numpy.ndarray)self, (float)f) -> bool :
            Check whether a vector x lies within the cone.
        """
    @staticmethod
    def project(*args, **kwargs):
        """
        
        project( (CoulombFrictionCone)self, (numpy.ndarray)f) -> numpy.ndarray :
            Normal projection of a vector f onto the cone.
        """
    @staticmethod
    def weightedProject(*args, **kwargs):
        """
        
        weightedProject( (CoulombFrictionCone)self, (numpy.ndarray)f, (numpy.ndarray)R) -> numpy.ndarray :
            Weighted projection of a vector f onto the cone.
        """
    @property
    def mu(*args, **kwargs):
        """
        Friction coefficient.
        """
    @mu.setter
    def mu(*args, **kwargs):
        ...
class Data(Boost.Python.instance):
    """
    Articulated rigid body data related to a Model.
    It contains all the data that can be modified by the Pinocchio algorithms.
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (Data)self) -> Data :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (Data)self, (dict)memo) -> Data :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Data)arg1, (Data)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (Data)arg1) -> tuple
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (Data)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor.
        
        __init__( (object)self, (Model)model) -> None :
            Constructs a data structure from a given model.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Data)arg1, (Data)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (Data)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (Data)self) -> Data :
            Returns a copy of *this.
        """
    @staticmethod
    def loadFromBinary(*args, **kwargs):
        """
        
        loadFromBinary( (Data)self, (str)filename) -> None :
            Loads *this from a binary file.
        
        loadFromBinary( (Data)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Loads *this from a binary buffer.
        
        loadFromBinary( (Data)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Loads *this from a static binary buffer.
        """
    @staticmethod
    def loadFromString(*args, **kwargs):
        """
        
        loadFromString( (Data)self, (str)string) -> None :
            Parses from the input string the content of the current object.
        """
    @staticmethod
    def loadFromText(*args, **kwargs):
        """
        
        loadFromText( (Data)self, (str)filename) -> None :
            Loads *this from a text file.
        """
    @staticmethod
    def loadFromXML(*args, **kwargs):
        """
        
        loadFromXML( (Data)self, (str)filename, (str)tag_name) -> None :
            Loads *this from a XML file.
        """
    @staticmethod
    def saveToBinary(*args, **kwargs):
        """
        
        saveToBinary( (Data)self, (str)filename) -> None :
            Saves *this inside a binary file.
        
        saveToBinary( (Data)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Saves *this inside a binary buffer.
        
        saveToBinary( (Data)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Saves *this inside a static binary buffer.
        """
    @staticmethod
    def saveToString(*args, **kwargs):
        """
        
        saveToString( (Data)self) -> str :
            Parses the current object to a string.
        """
    @staticmethod
    def saveToText(*args, **kwargs):
        """
        
        saveToText( (Data)self, (str)filename) -> None :
            Saves *this inside a text file.
        """
    @staticmethod
    def saveToXML(*args, **kwargs):
        """
        
        saveToXML( (Data)arg1, (str)filename, (str)tag_name) -> None :
            Saves *this inside a XML file.
        """
    @property
    def Ag(*args, **kwargs):
        """
        Centroidal matrix which maps from joint velocity to the centroidal momentum.
        """
    @Ag.setter
    def Ag(*args, **kwargs):
        ...
    @property
    def B(*args, **kwargs):
        """
        Combined variations of the inertia matrix consistent with Christoffel symbols.
        """
    @B.setter
    def B(*args, **kwargs):
        ...
    @property
    def C(*args, **kwargs):
        """
        The Coriolis C(q,v) matrix such that the Coriolis effects are given by c(q,v) = C(q,v)v
        """
    @C.setter
    def C(*args, **kwargs):
        ...
    @property
    def D(*args, **kwargs):
        """
        Diagonal of UDUT inertia decomposition
        """
    @D.setter
    def D(*args, **kwargs):
        ...
    @property
    def Fcrb(*args, **kwargs):
        """
        Spatial forces set, used in CRBA
        """
    @Fcrb.setter
    def Fcrb(*args, **kwargs):
        ...
    @property
    def Ig(*args, **kwargs):
        """
        Centroidal Composite Rigid Body Inertia.
        """
    @Ig.setter
    def Ig(*args, **kwargs):
        ...
    @property
    def Ivx(*args, **kwargs):
        """
        Right variation of the inertia matrix.
        """
    @Ivx.setter
    def Ivx(*args, **kwargs):
        ...
    @property
    def J(*args, **kwargs):
        """
        Jacobian of joint placement
        """
    @J.setter
    def J(*args, **kwargs):
        ...
    @property
    def Jcom(*args, **kwargs):
        """
        Jacobian of center of mass.
        """
    @Jcom.setter
    def Jcom(*args, **kwargs):
        ...
    @property
    def M(*args, **kwargs):
        """
        The joint space inertia matrix
        """
    @M.setter
    def M(*args, **kwargs):
        ...
    @property
    def Minv(*args, **kwargs):
        """
        The inverse of the joint space inertia matrix
        """
    @Minv.setter
    def Minv(*args, **kwargs):
        ...
    @property
    def U(*args, **kwargs):
        """
        Joint Inertia square root (upper triangle)
        """
    @U.setter
    def U(*args, **kwargs):
        ...
    @property
    def Yaba(*args, **kwargs):
        """
        Articulated Body Inertia of the sub-tree
        """
    @Yaba.setter
    def Yaba(*args, **kwargs):
        ...
    @property
    def Ycrb(*args, **kwargs):
        """
        Inertia of the sub-tree composit rigid body
        """
    @Ycrb.setter
    def Ycrb(*args, **kwargs):
        ...
    @property
    def a(*args, **kwargs):
        """
        Vector of joint accelerations expressed in the local frame of the joint.
        """
    @a.setter
    def a(*args, **kwargs):
        ...
    @property
    def a_gf(*args, **kwargs):
        """
        Joint spatial acceleration containing also the contribution of the gravity acceleration
        """
    @a_gf.setter
    def a_gf(*args, **kwargs):
        ...
    @property
    def acom(*args, **kwargs):
        """
        CoM acceleration of the subtree starting at joint index i.
        """
    @acom.setter
    def acom(*args, **kwargs):
        ...
    @property
    def com(*args, **kwargs):
        """
        CoM position of the subtree starting at joint index i.
        """
    @com.setter
    def com(*args, **kwargs):
        ...
    @property
    def contact_chol(*args, **kwargs):
        """
        Contact Cholesky decomposition.
        """
    @contact_chol.setter
    def contact_chol(*args, **kwargs):
        ...
    @property
    def dAdq(*args, **kwargs):
        """
        Variation of the spatial acceleration set with respect to the joint configuration.
        """
    @dAdq.setter
    def dAdq(*args, **kwargs):
        ...
    @property
    def dAdv(*args, **kwargs):
        """
        Variation of the spatial acceleration set with respect to the joint velocity.
        """
    @dAdv.setter
    def dAdv(*args, **kwargs):
        ...
    @property
    def dAg(*args, **kwargs):
        """
        Time derivative of the centroidal momentum matrix Ag.
        """
    @dAg.setter
    def dAg(*args, **kwargs):
        ...
    @property
    def dFda(*args, **kwargs):
        """
        Variation of the force set with respect to the joint acceleration.
        """
    @dFda.setter
    def dFda(*args, **kwargs):
        ...
    @property
    def dFdq(*args, **kwargs):
        """
        Variation of the force set with respect to the joint configuration.
        """
    @dFdq.setter
    def dFdq(*args, **kwargs):
        ...
    @property
    def dFdv(*args, **kwargs):
        """
        Variation of the force set with respect to the joint velocity.
        """
    @dFdv.setter
    def dFdv(*args, **kwargs):
        ...
    @property
    def dHdq(*args, **kwargs):
        """
        Variation of the spatial momenta set with respect to the joint configuration.
        """
    @dHdq.setter
    def dHdq(*args, **kwargs):
        ...
    @property
    def dJ(*args, **kwargs):
        """
        Time variation of the Jacobian of joint placement (data.J).
        """
    @dJ.setter
    def dJ(*args, **kwargs):
        ...
    @property
    def dac_da(*args, **kwargs):
        """
        Partial derivative of the contact acceleration vector vector with respect to the joint acceleration.
        """
    @dac_da.setter
    def dac_da(*args, **kwargs):
        ...
    @property
    def dac_dq(*args, **kwargs):
        """
        Partial derivative of the contact acceleration vector with respect to the joint configuration.
        """
    @dac_dq.setter
    def dac_dq(*args, **kwargs):
        ...
    @property
    def dac_dv(*args, **kwargs):
        """
        Partial derivative of the contact acceleration vector vector with respect to the joint velocity.
        """
    @dac_dv.setter
    def dac_dv(*args, **kwargs):
        ...
    @property
    def ddq(*args, **kwargs):
        """
        Joint accelerations (output of ABA)
        """
    @ddq.setter
    def ddq(*args, **kwargs):
        ...
    @property
    def ddq_dq(*args, **kwargs):
        """
        Partial derivative of the joint acceleration vector with respect to the joint configuration.
        """
    @ddq_dq.setter
    def ddq_dq(*args, **kwargs):
        ...
    @property
    def ddq_dtau(*args, **kwargs):
        """
        Partial derivative of the joint acceleration vector with respect to the joint torque.
        """
    @ddq_dtau.setter
    def ddq_dtau(*args, **kwargs):
        ...
    @property
    def ddq_dv(*args, **kwargs):
        """
        Partial derivative of the joint acceleration vector with respect to the joint velocity.
        """
    @ddq_dv.setter
    def ddq_dv(*args, **kwargs):
        ...
    @property
    def dhg(*args, **kwargs):
        """
        Centroidal momentum time derivative (expressed in the frame centered at the CoM and aligned with the world frame).
        """
    @dhg.setter
    def dhg(*args, **kwargs):
        ...
    @property
    def dlambda_dq(*args, **kwargs):
        """
        Partial derivative of the contact force vector with respect to the joint configuration.
        """
    @property
    def dlambda_dtau(*args, **kwargs):
        """
        Partial derivative of the contact force vector with respect to the torque.
        """
    @property
    def dlambda_dv(*args, **kwargs):
        """
        Partial derivative of the contact force vector with respect to the joint velocity.
        """
    @property
    def dq_after(*args, **kwargs):
        """
        Generalized velocity after the impact.
        """
    @dq_after.setter
    def dq_after(*args, **kwargs):
        ...
    @property
    def dtau_dq(*args, **kwargs):
        """
        Partial derivative of the joint torque vector with respect to the joint configuration.
        """
    @dtau_dq.setter
    def dtau_dq(*args, **kwargs):
        ...
    @property
    def dtau_dv(*args, **kwargs):
        """
        Partial derivative of the joint torque vector with respect to the joint velocity.
        """
    @dtau_dv.setter
    def dtau_dv(*args, **kwargs):
        ...
    @property
    def dvc_dq(*args, **kwargs):
        """
        Partial derivative of the constraint velocity vector with respect to the joint configuration.
        """
    @dvc_dq.setter
    def dvc_dq(*args, **kwargs):
        ...
    @property
    def f(*args, **kwargs):
        """
        Vector of body forces expressed in the local frame of the joint.
        """
    @f.setter
    def f(*args, **kwargs):
        ...
    @property
    def g(*args, **kwargs):
        """
        Vector of generalized gravity (dim model.nv).
        """
    @g.setter
    def g(*args, **kwargs):
        ...
    @property
    def h(*args, **kwargs):
        """
        Vector of spatial momenta expressed in the local frame of the joint.
        """
    @h.setter
    def h(*args, **kwargs):
        ...
    @property
    def hg(*args, **kwargs):
        """
        Centroidal momentum (expressed in the frame centered at the CoM and aligned with the world frame).
        """
    @hg.setter
    def hg(*args, **kwargs):
        ...
    @property
    def iMf(*args, **kwargs):
        """
        Body placement wrt to algorithm end effector.
        """
    @iMf.setter
    def iMf(*args, **kwargs):
        ...
    @property
    def impulse_c(*args, **kwargs):
        """
        Lagrange Multipliers linked to contact impulses
        """
    @impulse_c.setter
    def impulse_c(*args, **kwargs):
        ...
    @property
    def jointTorqueRegressor(*args, **kwargs):
        """
        Joint torque regressor.
        """
    @jointTorqueRegressor.setter
    def jointTorqueRegressor(*args, **kwargs):
        ...
    @property
    def joints(*args, **kwargs):
        """
        Vector of JointData associated to each JointModel stored in the related model.
        """
    @joints.setter
    def joints(*args, **kwargs):
        ...
    @property
    def kineticEnergyRegressor(*args, **kwargs):
        """
        Kinetic energy regressor.
        """
    @kineticEnergyRegressor.setter
    def kineticEnergyRegressor(*args, **kwargs):
        ...
    @property
    def kinetic_energy(*args, **kwargs):
        """
        Kinetic energy in [J] computed by computeKineticEnergy
        """
    @kinetic_energy.setter
    def kinetic_energy(*args, **kwargs):
        ...
    @property
    def lambda_c(*args, **kwargs):
        """
        Lagrange Multipliers linked to contact forces
        """
    @lambda_c.setter
    def lambda_c(*args, **kwargs):
        ...
    @property
    def lambda_c_prox(*args, **kwargs):
        """
        Proximal Lagrange Multipliers used in the computation of the Forward Dynamics computations.
        """
    @lambda_c_prox.setter
    def lambda_c_prox(*args, **kwargs):
        ...
    @property
    def lastChild(*args, **kwargs):
        """
        Index of the last child (for CRBA)
        """
    @lastChild.setter
    def lastChild(*args, **kwargs):
        ...
    @property
    def liMi(*args, **kwargs):
        """
        Body relative placement (wrt parent)
        """
    @liMi.setter
    def liMi(*args, **kwargs):
        ...
    @property
    def mass(*args, **kwargs):
        """
        Mass of the subtree starting at joint index i.
        """
    @mass.setter
    def mass(*args, **kwargs):
        ...
    @property
    def mechanical_energy(*args, **kwargs):
        """
        Mechanical energy in [J] of the system computed by computeMechanicalEnergy
        """
    @mechanical_energy.setter
    def mechanical_energy(*args, **kwargs):
        ...
    @property
    def nle(*args, **kwargs):
        """
        Non Linear Effects (output of nle algorithm)
        """
    @nle.setter
    def nle(*args, **kwargs):
        ...
    @property
    def nvSubtree(*args, **kwargs):
        """
        Dimension of the subtree motion space (for CRBA)
        """
    @nvSubtree.setter
    def nvSubtree(*args, **kwargs):
        ...
    @property
    def nvSubtree_fromRow(*args, **kwargs):
        """
        Subtree of the current row index (used in Cholesky)
        """
    @nvSubtree_fromRow.setter
    def nvSubtree_fromRow(*args, **kwargs):
        ...
    @property
    def oK(*args, **kwargs):
        """
        Inverse articulated inertia.
        """
    @oK.setter
    def oK(*args, **kwargs):
        ...
    @property
    def oL(*args, **kwargs):
        """
        Acceleration propagator.
        """
    @oL.setter
    def oL(*args, **kwargs):
        ...
    @property
    def oMf(*args, **kwargs):
        """
        frames absolute placement (wrt world)
        """
    @oMf.setter
    def oMf(*args, **kwargs):
        ...
    @property
    def oMi(*args, **kwargs):
        """
        Body absolute placement (wrt world)
        """
    @oMi.setter
    def oMi(*args, **kwargs):
        ...
    @property
    def oYaba(*args, **kwargs):
        """
        Articulated Body Inertia of the sub-tree expressed in the WORLD coordinate system.
        """
    @oYaba.setter
    def oYaba(*args, **kwargs):
        ...
    @property
    def oYcrb(*args, **kwargs):
        """
        Composite Rigid Body Inertia of the sub-tree expressed in the WORLD coordinate system.
        """
    @oYcrb.setter
    def oYcrb(*args, **kwargs):
        ...
    @property
    def oa(*args, **kwargs):
        """
        Joint spatial acceleration expressed at the origin of the world frame.
        """
    @oa.setter
    def oa(*args, **kwargs):
        ...
    @property
    def oa_gf(*args, **kwargs):
        """
        Joint spatial acceleration containing also the contribution of the gravity acceleration, but expressed at the origin of the world frame.
        """
    @oa_gf.setter
    def oa_gf(*args, **kwargs):
        ...
    @property
    def of(*args, **kwargs):
        """
        Vector of body forces expressed at the origin of the world.
        """
    @of.setter
    def of(*args, **kwargs):
        ...
    @property
    def of_augmented(*args, **kwargs):
        """
        Vector of body forces expressed at the origin of the world, in the context of lagrangian formulation
        """
    @of_augmented.setter
    def of_augmented(*args, **kwargs):
        ...
    @property
    def oh(*args, **kwargs):
        """
        Vector of spatial momenta expressed at the origin of the world.
        """
    @oh.setter
    def oh(*args, **kwargs):
        ...
    @property
    def osim(*args, **kwargs):
        """
        Operational space inertia matrix.
        """
    @osim.setter
    def osim(*args, **kwargs):
        ...
    @property
    def ov(*args, **kwargs):
        """
        Vector of joint velocities expressed at the origin of the world.
        """
    @ov.setter
    def ov(*args, **kwargs):
        ...
    @property
    def parents_fromRow(*args, **kwargs):
        """
        First previous non-zero row in M (used in Cholesky)
        """
    @parents_fromRow.setter
    def parents_fromRow(*args, **kwargs):
        ...
    @property
    def potentialEnergyRegressor(*args, **kwargs):
        """
        Potential energy regressor.
        """
    @potentialEnergyRegressor.setter
    def potentialEnergyRegressor(*args, **kwargs):
        ...
    @property
    def potential_energy(*args, **kwargs):
        """
        Potential energy in [J] computed by computePotentialEnergy
        """
    @potential_energy.setter
    def potential_energy(*args, **kwargs):
        ...
    @property
    def primal_dual_contact_solution(*args, **kwargs):
        """
        Right hand side vector when solving the contact dynamics KKT problem.
        """
    @primal_dual_contact_solution.setter
    def primal_dual_contact_solution(*args, **kwargs):
        ...
    @property
    def primal_rhs_contact(*args, **kwargs):
        """
        Primal RHS in contact dynamic equations.
        """
    @primal_rhs_contact.setter
    def primal_rhs_contact(*args, **kwargs):
        ...
    @property
    def staticRegressor(*args, **kwargs):
        """
        Static regressor.
        """
    @staticRegressor.setter
    def staticRegressor(*args, **kwargs):
        ...
    @property
    def tau(*args, **kwargs):
        """
        Joint torques (output of RNEA)
        """
    @tau.setter
    def tau(*args, **kwargs):
        ...
    @property
    def v(*args, **kwargs):
        """
        Vector of joint velocities expressed in the local frame of the joint.
        """
    @v.setter
    def v(*args, **kwargs):
        ...
    @property
    def vcom(*args, **kwargs):
        """
        CoM velocity of the subtree starting at joint index i.
        """
    @vcom.setter
    def vcom(*args, **kwargs):
        ...
    @property
    def vxI(*args, **kwargs):
        """
        Left variation of the inertia matrix.
        """
    @vxI.setter
    def vxI(*args, **kwargs):
        ...
class DelassusCholeskyExpression(Boost.Python.instance):
    """
    Delassus Cholesky expression associated to a given ContactCholeskyDecomposition object.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (ContactCholeskyDecomposition)cholesky_decomposition) -> None :
            Build from a given ContactCholeskyDecomposition object.
        """
    @staticmethod
    def __matmul__(*args, **kwargs):
        """
        
        __matmul__( (DelassusCholeskyExpression)self, (numpy.ndarray)other) -> numpy.ndarray :
            Matrix multiplication between self and another matrix. Returns the result of Delassus * matrix.
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (DelassusCholeskyExpression)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def cholesky(*args, **kwargs):
        """
        
        cholesky( (DelassusCholeskyExpression)self) -> ContactCholeskyDecomposition :
            Returns the Constraint Cholesky decomposition associated to this DelassusCholeskyExpression.
        """
    @staticmethod
    def cols(*args, **kwargs):
        """
        
        cols( (DelassusCholeskyExpression)self) -> int :
            Returns the number of columns.
        """
    @staticmethod
    def computeLargestEigenValue(*args, **kwargs):
        """
        
        computeLargestEigenValue( (DelassusCholeskyExpression)self [, (bool)reset=True [, (int)max_it=10 [, (float)prec=1e-08]]]) -> float :
            Compute the largest eigenvalue associated to the underlying Delassus matrix.
        """
    @staticmethod
    def computeLowestEigenValue(*args, **kwargs):
        """
        
        computeLowestEigenValue( (DelassusCholeskyExpression)self [, (bool)reset=True [, (bool)compute_largest=True [, (int)max_it=10 [, (float)prec=1e-08]]]]) -> float :
            Compute the lowest eigenvalue associated to the underlying Delassus matrix.
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (DelassusCholeskyExpression)self) -> numpy.ndarray :
            Returns the inverse of the Delassus expression as a dense matrix.
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (DelassusCholeskyExpression)self) -> numpy.ndarray :
            Returns the Delassus expression as a dense matrix.
        """
    @staticmethod
    def rows(*args, **kwargs):
        """
        
        rows( (DelassusCholeskyExpression)self) -> int :
            Returns the number of rows.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (DelassusCholeskyExpression)self) -> int :
            Returns the size of the decomposition.
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (DelassusCholeskyExpression)self, (numpy.ndarray)mat) -> numpy.ndarray :
            Returns the solution x of Delassus * x = mat using the current decomposition of the Delassus matrix.
        """
    @staticmethod
    def updateDamping(*args, **kwargs):
        """
        
        updateDamping( (DelassusCholeskyExpression)self, (float)mu) -> None :
            Add a damping term to the diagonal of the Delassus matrix. The damping term should be positive.
        
        updateDamping( (DelassusCholeskyExpression)self, (numpy.ndarray)mus) -> None :
            Add a damping term to the diagonal of the Delassus matrix. The damping terms should be all positive.
        """
class DelassusOperatorDense(Boost.Python.instance):
    """
    Delassus Cholesky dense operator from a dense matrix.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (numpy.ndarray)matrix) -> None :
            Build from a given dense matrix
        """
    @staticmethod
    def __matmul__(*args, **kwargs):
        """
        
        __matmul__( (DelassusOperatorDense)self, (numpy.ndarray)other) -> numpy.ndarray :
            Matrix multiplication between self and another matrix. Returns the result of Delassus * matrix.
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (DelassusOperatorDense)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def cols(*args, **kwargs):
        """
        
        cols( (DelassusOperatorDense)self) -> int :
            Returns the number of columns.
        """
    @staticmethod
    def computeLargestEigenValue(*args, **kwargs):
        """
        
        computeLargestEigenValue( (DelassusOperatorDense)self [, (bool)reset=True [, (int)max_it=10 [, (float)prec=1e-08]]]) -> float :
            Compute the largest eigenvalue associated to the underlying Delassus matrix.
        """
    @staticmethod
    def computeLowestEigenValue(*args, **kwargs):
        """
        
        computeLowestEigenValue( (DelassusOperatorDense)self [, (bool)reset=True [, (bool)compute_largest=True [, (int)max_it=10 [, (float)prec=1e-08]]]]) -> float :
            Compute the lowest eigenvalue associated to the underlying Delassus matrix.
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (DelassusOperatorDense)self) -> numpy.ndarray :
            Returns the inverse of the Delassus expression as a dense matrix.
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (DelassusOperatorDense)self) -> numpy.ndarray :
            Returns the Delassus expression as a dense matrix.
        """
    @staticmethod
    def rows(*args, **kwargs):
        """
        
        rows( (DelassusOperatorDense)self) -> int :
            Returns the number of rows.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (DelassusOperatorDense)self) -> int :
            Returns the size of the decomposition.
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (DelassusOperatorDense)self, (numpy.ndarray)mat) -> numpy.ndarray :
            Returns the solution x of Delassus * x = mat using the current decomposition of the Delassus matrix.
        """
    @staticmethod
    def updateDamping(*args, **kwargs):
        """
        
        updateDamping( (DelassusOperatorDense)self, (float)mu) -> None :
            Add a damping term to the diagonal of the Delassus matrix. The damping term should be positive.
        
        updateDamping( (DelassusOperatorDense)self, (numpy.ndarray)mus) -> None :
            Add a damping term to the diagonal of the Delassus matrix. The damping terms should be all positive.
        """
class DelassusOperatorSparse(Boost.Python.instance):
    """
    Delassus Cholesky sparse operator from a sparse matrix.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (scipy.sparse._csc.csc_matrix)matrix) -> None :
            Build from a given sparse matrix
        """
    @staticmethod
    def __matmul__(*args, **kwargs):
        """
        
        __matmul__( (DelassusOperatorSparse)self, (numpy.ndarray)other) -> numpy.ndarray :
            Matrix multiplication between self and another matrix. Returns the result of Delassus * matrix.
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (DelassusOperatorSparse)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def cols(*args, **kwargs):
        """
        
        cols( (DelassusOperatorSparse)self) -> int :
            Returns the number of columns.
        """
    @staticmethod
    def computeLargestEigenValue(*args, **kwargs):
        """
        
        computeLargestEigenValue( (DelassusOperatorSparse)self [, (bool)reset=True [, (int)max_it=10 [, (float)prec=1e-08]]]) -> float :
            Compute the largest eigenvalue associated to the underlying Delassus matrix.
        """
    @staticmethod
    def computeLowestEigenValue(*args, **kwargs):
        """
        
        computeLowestEigenValue( (DelassusOperatorSparse)self [, (bool)reset=True [, (bool)compute_largest=True [, (int)max_it=10 [, (float)prec=1e-08]]]]) -> float :
            Compute the lowest eigenvalue associated to the underlying Delassus matrix.
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (DelassusOperatorSparse)self) -> scipy.sparse._csc.csc_matrix :
            Returns the inverse of the Delassus expression as a dense matrix.
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (DelassusOperatorSparse)self) -> scipy.sparse._csc.csc_matrix :
            Returns the Delassus expression as a dense matrix.
        """
    @staticmethod
    def rows(*args, **kwargs):
        """
        
        rows( (DelassusOperatorSparse)self) -> int :
            Returns the number of rows.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (DelassusOperatorSparse)self) -> int :
            Returns the size of the decomposition.
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (DelassusOperatorSparse)self, (numpy.ndarray)mat) -> numpy.ndarray :
            Returns the solution x of Delassus * x = mat using the current decomposition of the Delassus matrix.
        """
    @staticmethod
    def updateDamping(*args, **kwargs):
        """
        
        updateDamping( (DelassusOperatorSparse)self, (float)mu) -> None :
            Add a damping term to the diagonal of the Delassus matrix. The damping term should be positive.
        
        updateDamping( (DelassusOperatorSparse)self, (numpy.ndarray)mus) -> None :
            Add a damping term to the diagonal of the Delassus matrix. The damping terms should be all positive.
        """
class DualCoulombFrictionCone(Boost.Python.instance):
    """
    Dual cone of the 3d Coulomb friction cone.
    """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (DualCoulombFrictionCone)self) -> DualCoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (DualCoulombFrictionCone)self, (dict)memo) -> DualCoulombFrictionCone :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (DualCoulombFrictionCone)arg1, (DualCoulombFrictionCone)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (float)mu) -> None :
            Default constructor
        
        __init__( (object)self, (DualCoulombFrictionCone)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (DualCoulombFrictionCone)arg1, (DualCoulombFrictionCone)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (DualCoulombFrictionCone)self) -> DualCoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def dim(*args, **kwargs):
        """
        
        dim() -> int :
            Returns the dimension of the cone.
        """
    @staticmethod
    def dual(*args, **kwargs):
        """
        
        dual( (DualCoulombFrictionCone)self) -> CoulombFrictionCone :
            Returns the dual cone associated to this.
        """
    @staticmethod
    def isInside(*args, **kwargs):
        """
        
        isInside( (DualCoulombFrictionCone)arg1, (numpy.ndarray)self, (float)v) -> bool :
            Check whether a vector x lies within the cone.
        """
    @staticmethod
    def project(*args, **kwargs):
        """
        
        project( (DualCoulombFrictionCone)self, (numpy.ndarray)v) -> numpy.ndarray :
            Project a vector v onto the cone.
        """
    @property
    def mu(*args, **kwargs):
        """
        Friction coefficient.
        """
    @mu.setter
    def mu(*args, **kwargs):
        ...
class Exception(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 72
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (str)arg2) -> None
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def message(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.Exception)arg1) -> str
        """
class Force(Boost.Python.instance):
    """
    Force vectors, in se3* == F^6.
    
    Supported operations ...
    """
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def Random(*args, **kwargs):
        """
        
        Random() -> Force :
            Returns a random Force.
        """
    @staticmethod
    def Zero(*args, **kwargs):
        """
        
        Zero() -> Force :
            Returns a zero Force.
        """
    @staticmethod
    def __add__(*args, **kwargs):
        """
        
        __add__( (Force)arg1, (Force)arg2) -> object
        """
    @staticmethod
    def __array__(*args, **kwargs):
        """
        
        __array__( (Force)arg1) -> object
        
        __array__( (Force)self [, (object)dtype=None [, (object)copy=None]]) -> object
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (Force)self) -> Force :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (Force)self, (dict)memo) -> Force :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Force)arg1, (Force)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (Force)arg1) -> tuple
        """
    @staticmethod
    def __iadd__(*args, **kwargs):
        """
        
        __iadd__( (Force)arg1, (Force)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor
        
        __init__( (object)self, (numpy.ndarray)linear, (numpy.ndarray)angular) -> None :
            Initialize from linear and angular components of a Wrench vector (don't mix the order).
        
        __init__( (object)self, (numpy.ndarray)array) -> None :
            Init from a vector 6 [force,torque]
        
        __init__( (object)self, (Force)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (Force)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Force -> pinocchio.pinocchio_pywrap_default.Force
        
        __init__( (object)arg1, (Force)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Force -> pinocchio.pinocchio_pywrap_default.Force
        """
    @staticmethod
    def __isub__(*args, **kwargs):
        """
        
        __isub__( (Force)arg1, (Force)arg2) -> object
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (Force)arg1, (float)arg2) -> object
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Force)arg1, (Force)arg2) -> object
        """
    @staticmethod
    def __neg__(*args, **kwargs):
        """
        
        __neg__( (Force)arg1) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (Force)arg1) -> object
        """
    @staticmethod
    def __rmul__(*args, **kwargs):
        """
        
        __rmul__( (Force)arg1, (float)arg2) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (Force)arg1) -> object
        """
    @staticmethod
    def __sub__(*args, **kwargs):
        """
        
        __sub__( (Force)arg1, (Force)arg2) -> object
        """
    @staticmethod
    def __truediv__(*args, **kwargs):
        """
        
        __truediv__( (Force)arg1, (float)arg2) -> object
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (Force)arg1) -> Force :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (Force)self) -> Force :
            Returns a copy of *this.
        """
    @staticmethod
    def dot(*args, **kwargs):
        """
        
        dot( (Force)self, (object)m) -> float :
            Dot product between *this and a Motion m.
        """
    @staticmethod
    def isApprox(*args, **kwargs):
        """
        
        isApprox( (Force)self, (Force)other [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to other, within the precision given by prec.
        """
    @staticmethod
    def isZero(*args, **kwargs):
        """
        
        isZero( (Force)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the zero Force, within the precision given by prec.
        """
    @staticmethod
    def se3Action(*args, **kwargs):
        """
        
        se3Action( (Force)self, (SE3)M) -> Force :
            Returns the result of the dual action of M on *this.
        """
    @staticmethod
    def se3ActionInverse(*args, **kwargs):
        """
        
        se3ActionInverse( (Force)self, (SE3)M) -> Force :
            Returns the result of the dual action of the inverse of M on *this.
        """
    @staticmethod
    def setRandom(*args, **kwargs):
        """
        
        setRandom( (Force)self) -> None :
            Set the linear and angular components of *this to random values.
        """
    @staticmethod
    def setZero(*args, **kwargs):
        """
        
        setZero( (Force)self) -> None :
            Set the linear and angular components of *this to zero.
        """
    @property
    def angular(*args, **kwargs):
        """
        Angular part of a *this, corresponding to the angular velocity in case of a Spatial velocity.
        """
    @angular.setter
    def angular(*args, **kwargs):
        ...
    @property
    def linear(*args, **kwargs):
        """
        Linear part of a *this, corresponding to the linear velocity in case of a Spatial velocity.
        """
    @linear.setter
    def linear(*args, **kwargs):
        ...
    @property
    def np(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.Force)arg1) -> object
        """
    @property
    def vector(*args, **kwargs):
        """
        Returns the components of *this as a 6d vector.
        """
    @vector.setter
    def vector(*args, **kwargs):
        ...
class Frame(Boost.Python.instance):
    """
    A Plucker coordinate frame related to a parent joint inside a kinematic tree.
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (Frame)self) -> Frame :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (Frame)self, (dict)memo) -> Frame :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Frame)arg1, (Frame)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (Frame)arg1) -> tuple
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (Frame)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor
        
        __init__( (object)self, (Frame)other) -> None :
            Copy constructor
        
        __init__( (object)arg1, (str)name, (int)parent_joint, (SE3)placement, (FrameType)type [, (Inertia)inertia]) -> None :
            Initialize from a given name, type, parent frame index and placement wrt parent joint and an spatial inertia object.
        
        __init__( (object)arg1, (str)name, (int)parent_joint, (int)parent_frame, (SE3)placement, (FrameType)type [, (Inertia)inertia]) -> None :
            Initialize from a given name, type, parent joint index, parent frame index and placement wrt parent joint and an spatial inertia object.
        
        __init__( (object)self, (Frame)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (Frame)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Frame -> pinocchio.pinocchio_pywrap_default.Frame
        
        __init__( (object)arg1, (Frame)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Frame -> pinocchio.pinocchio_pywrap_default.Frame
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Frame)arg1, (Frame)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (Frame)arg1) -> object
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (Frame)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (Frame)arg1) -> object
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (Frame)arg1) -> Frame :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (Frame)self) -> Frame :
            Returns a copy of *this.
        """
    @property
    def inertia(*args, **kwargs):
        """
        Inertia information attached to the frame.
        """
    @inertia.setter
    def inertia(*args, **kwargs):
        ...
    @property
    def name(*args, **kwargs):
        """
        name of the frame
        """
    @name.setter
    def name(*args, **kwargs):
        ...
    @property
    def parent(*args, **kwargs):
        """
        See parentJoint property.
        """
    @parent.setter
    def parent(*args, **kwargs):
        ...
    @property
    def parentFrame(*args, **kwargs):
        """
        Index of the parent frame
        """
    @parentFrame.setter
    def parentFrame(*args, **kwargs):
        ...
    @property
    def parentJoint(*args, **kwargs):
        """
        Index of the parent joint
        """
    @parentJoint.setter
    def parentJoint(*args, **kwargs):
        ...
    @property
    def placement(*args, **kwargs):
        """
        placement in the parent joint local frame
        """
    @placement.setter
    def placement(*args, **kwargs):
        ...
    @property
    def previousFrame(*args, **kwargs):
        """
        See parentFrame property.
        """
    @previousFrame.setter
    def previousFrame(*args, **kwargs):
        ...
    @property
    def type(*args, **kwargs):
        """
        type of the frame
        """
    @type.setter
    def type(*args, **kwargs):
        ...
class FrameType(Boost.Python.enum):
    BODY: typing.ClassVar[FrameType]  # value = pinocchio.pinocchio_pywrap_default.FrameType.BODY
    FIXED_JOINT: typing.ClassVar[FrameType]  # value = pinocchio.pinocchio_pywrap_default.FrameType.FIXED_JOINT
    JOINT: typing.ClassVar[FrameType]  # value = pinocchio.pinocchio_pywrap_default.FrameType.JOINT
    OP_FRAME: typing.ClassVar[FrameType]  # value = pinocchio.pinocchio_pywrap_default.FrameType.OP_FRAME
    SENSOR: typing.ClassVar[FrameType]  # value = pinocchio.pinocchio_pywrap_default.FrameType.SENSOR
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'OP_FRAME': pinocchio.pinocchio_pywrap_default.FrameType.OP_FRAME, 'JOINT': pinocchio.pinocchio_pywrap_default.FrameType.JOINT, 'FIXED_JOINT': pinocchio.pinocchio_pywrap_default.FrameType.FIXED_JOINT, 'BODY': pinocchio.pinocchio_pywrap_default.FrameType.BODY, 'SENSOR': pinocchio.pinocchio_pywrap_default.FrameType.SENSOR}
    values: typing.ClassVar[dict]  # value = {1: pinocchio.pinocchio_pywrap_default.FrameType.OP_FRAME, 2: pinocchio.pinocchio_pywrap_default.FrameType.JOINT, 4: pinocchio.pinocchio_pywrap_default.FrameType.FIXED_JOINT, 8: pinocchio.pinocchio_pywrap_default.FrameType.BODY, 16: pinocchio.pinocchio_pywrap_default.FrameType.SENSOR}
class GeometryData(Boost.Python.instance):
    """
    Geometry data linked to a Geometry Model and a Data struct.
    """
    @staticmethod
    def __address__(*args, **kwargs):
        """
        
        __address__( (GeometryModel)self) -> int :
            Returns the address of the underlying C++ object.
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (GeometryData)self) -> GeometryData :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (GeometryData)self, (dict)memo) -> GeometryData :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (GeometryData)arg1, (GeometryData)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (GeometryModel)geometry_model) -> None :
            Default constructor from a given GeometryModel.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (GeometryData)arg1, (GeometryData)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (GeometryData)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (GeometryData)arg1) -> object
        """
    @staticmethod
    def activateCollisionPair(*args, **kwargs):
        """
        
        activateCollisionPair( (GeometryData)self, (int)pair_id) -> None :
            Activate the collsion pair pair_id in geomModel.collisionPairs if it exists.
            note: Only active pairs are check for collision and distance computations.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (GeometryData)self) -> GeometryData :
            Returns a copy of *this.
        """
    @staticmethod
    def deactivateAllCollisionPairs(*args, **kwargs):
        """
        
        deactivateAllCollisionPairs( (GeometryData)self) -> None :
            Deactivate all collision pairs.
        """
    @staticmethod
    def deactivateCollisionPair(*args, **kwargs):
        """
        
        deactivateCollisionPair( (GeometryData)self, (int)pair_id) -> None :
            Deactivate the collsion pair pair_id in geomModel.collisionPairs if it exists.
        """
    @staticmethod
    def fillInnerOuterObjectMaps(*args, **kwargs):
        """
        
        fillInnerOuterObjectMaps( (GeometryData)self, (GeometryModel)geometry_model) -> None :
            Fill inner and outer objects maps
        """
    @staticmethod
    def loadFromBinary(*args, **kwargs):
        """
        
        loadFromBinary( (GeometryData)self, (str)filename) -> None :
            Loads *this from a binary file.
        
        loadFromBinary( (GeometryData)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Loads *this from a binary buffer.
        
        loadFromBinary( (GeometryData)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Loads *this from a static binary buffer.
        """
    @staticmethod
    def loadFromString(*args, **kwargs):
        """
        
        loadFromString( (GeometryData)self, (str)string) -> None :
            Parses from the input string the content of the current object.
        """
    @staticmethod
    def loadFromText(*args, **kwargs):
        """
        
        loadFromText( (GeometryData)self, (str)filename) -> None :
            Loads *this from a text file.
        """
    @staticmethod
    def loadFromXML(*args, **kwargs):
        """
        
        loadFromXML( (GeometryData)self, (str)filename, (str)tag_name) -> None :
            Loads *this from a XML file.
        """
    @staticmethod
    def saveToBinary(*args, **kwargs):
        """
        
        saveToBinary( (GeometryData)self, (str)filename) -> None :
            Saves *this inside a binary file.
        
        saveToBinary( (GeometryData)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Saves *this inside a binary buffer.
        
        saveToBinary( (GeometryData)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Saves *this inside a static binary buffer.
        """
    @staticmethod
    def saveToString(*args, **kwargs):
        """
        
        saveToString( (GeometryData)self) -> str :
            Parses the current object to a string.
        """
    @staticmethod
    def saveToText(*args, **kwargs):
        """
        
        saveToText( (GeometryData)self, (str)filename) -> None :
            Saves *this inside a text file.
        """
    @staticmethod
    def saveToXML(*args, **kwargs):
        """
        
        saveToXML( (GeometryData)arg1, (str)filename, (str)tag_name) -> None :
            Saves *this inside a XML file.
        """
    @staticmethod
    def setActiveCollisionPairs(*args, **kwargs):
        """
        
        setActiveCollisionPairs( (GeometryData)self, (GeometryModel)geometry_model, (numpy.ndarray)collision_map [, (bool)upper=True]) -> None :
            Set the collision pair association from a given input array.
            Each entry of the input matrix defines the activation of a given collision pair.
        """
    @staticmethod
    def setGeometryCollisionStatus(*args, **kwargs):
        """
        
        setGeometryCollisionStatus( (GeometryData)self, (GeometryModel)geom_model, (int)geom_id, (bool)enable_collision) -> None :
            Enable or disable collision for the given geometry given by its geometry id with all the other geometries registered in the list of collision pairs.
        """
    @staticmethod
    def setSecurityMargins(*args, **kwargs):
        """
        
        setSecurityMargins( (GeometryData)self, (GeometryModel)geometry_model, (numpy.ndarray)security_margin_map [, (bool)upper=True [, (bool)sync_distance_upper_bound=True]]) -> None :
            Set the security margin of all the collision request in a row, according to the values stored in the associative map.
        """
    @property
    def activeCollisionPairs(*args, **kwargs):
        """
        Vector of active CollisionPairs
        """
    @activeCollisionPairs.setter
    def activeCollisionPairs(*args, **kwargs):
        ...
    @property
    def collisionRequests(*args, **kwargs):
        """
        Defines which information should be computed by FCL for collision computations.
        
        Note: it is possible to define a security_margin and a break_distance for a collision request.
        Most likely, for robotics application, these thresholds will be different for each collision pairs
        (e.g. the two hands can have a large security margin while the two hips cannot.)
        """
    @collisionRequests.setter
    def collisionRequests(*args, **kwargs):
        ...
    @property
    def collisionResults(*args, **kwargs):
        """
        Vector of collision results.
        """
    @collisionResults.setter
    def collisionResults(*args, **kwargs):
        ...
    @property
    def collision_functors(*args, **kwargs):
        """
        Vector of collision functors.
        """
    @collision_functors.setter
    def collision_functors(*args, **kwargs):
        ...
    @property
    def distanceRequests(*args, **kwargs):
        """
        Defines which information should be computed by FCL for distance computations
        """
    @distanceRequests.setter
    def distanceRequests(*args, **kwargs):
        ...
    @property
    def distanceResults(*args, **kwargs):
        """
        Vector of distance results.
        """
    @distanceResults.setter
    def distanceResults(*args, **kwargs):
        ...
    @property
    def distance_functors(*args, **kwargs):
        """
        Vector of distance functors.
        """
    @distance_functors.setter
    def distance_functors(*args, **kwargs):
        ...
    @property
    def oMg(*args, **kwargs):
        """
        Vector of collision objects placement relative to the world frame.
        note: These quantities have to be updated by calling updateGeometryPlacements.
        """
    @oMg.setter
    def oMg(*args, **kwargs):
        ...
    @property
    def radius(*args, **kwargs):
        """
        Vector of radius of bodies, i.e. the distance between the further point of the geometry object from the joint center.
        note: This radius information might be usuful in continuous collision checking
        """
    @radius.setter
    def radius(*args, **kwargs):
        ...
class GeometryModel(Boost.Python.instance):
    """
    Geometry model containing the collision or visual geometries associated to a model.
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __address__(*args, **kwargs):
        """
        
        __address__( (GeometryModel)self) -> int :
            Returns the address of the underlying C++ object.
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (GeometryModel)self) -> GeometryModel :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (GeometryModel)self, (dict)memo) -> GeometryModel :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (GeometryModel)arg1, (GeometryModel)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (GeometryModel)arg1) -> tuple
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (GeometryModel)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor
        
        __init__( (object)self, (GeometryModel)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (GeometryModel)arg1, (GeometryModel)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (GeometryModel)arg1) -> object
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (GeometryModel)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (GeometryModel)arg1) -> object
        """
    @staticmethod
    def addAllCollisionPairs(*args, **kwargs):
        """
        
        addAllCollisionPairs( (GeometryModel)arg1) -> None :
            Add all collision pairs.
            note : collision pairs between geometries having the same parent joint are not added.
        """
    @staticmethod
    def addCollisionPair(*args, **kwargs):
        """
        
        addCollisionPair( (GeometryModel)self, (CollisionPair)collision_pair) -> None :
            Add a collision pair given by the index of the two collision objects.
        """
    @staticmethod
    def addGeometryObject(*args, **kwargs):
        """
        
        addGeometryObject( (GeometryModel)self, (GeometryObject)geometry_object) -> int :
            Add a GeometryObject to a GeometryModel.
            Parameters
            	geometry_object : a GeometryObject
            
        
        addGeometryObject( (GeometryModel)self, (GeometryObject)geometry_object, (Model)model) -> int :
            Add a GeometryObject to a GeometryModel and set its parent joint by reading its value in the model.
            Parameters
            	geometry_object : a GeometryObject
            	model : a Model of the system
            
        """
    @staticmethod
    def clone(*args, **kwargs):
        """
        
        clone( (GeometryModel)self) -> GeometryModel :
            Create a deep copy of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (GeometryModel)self) -> GeometryModel :
            Returns a copy of *this.
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (GeometryModel)self) -> GeometryData :
            Create a GeometryData associated to the current model.
        """
    @staticmethod
    def existCollisionPair(*args, **kwargs):
        """
        
        existCollisionPair( (GeometryModel)self, (CollisionPair)collision_pair) -> bool :
            Check if a collision pair exists.
        """
    @staticmethod
    def existGeometryName(*args, **kwargs):
        """
        
        existGeometryName( (GeometryModel)self, (str)name) -> bool :
            Checks if a GeometryObject  given by its name exists.
        """
    @staticmethod
    def findCollisionPair(*args, **kwargs):
        """
        
        findCollisionPair( (GeometryModel)self, (CollisionPair)collision_pair) -> int :
            Return the index of a collision pair.
        """
    @staticmethod
    def getGeometryId(*args, **kwargs):
        """
        
        getGeometryId( (GeometryModel)self, (str)name) -> int :
            Returns the index of a GeometryObject given by its name.
        """
    @staticmethod
    def loadFromBinary(*args, **kwargs):
        """
        
        loadFromBinary( (GeometryModel)self, (str)filename) -> None :
            Loads *this from a binary file.
        
        loadFromBinary( (GeometryModel)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Loads *this from a binary buffer.
        
        loadFromBinary( (GeometryModel)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Loads *this from a static binary buffer.
        """
    @staticmethod
    def loadFromString(*args, **kwargs):
        """
        
        loadFromString( (GeometryModel)self, (str)string) -> None :
            Parses from the input string the content of the current object.
        """
    @staticmethod
    def loadFromText(*args, **kwargs):
        """
        
        loadFromText( (GeometryModel)self, (str)filename) -> None :
            Loads *this from a text file.
        """
    @staticmethod
    def loadFromXML(*args, **kwargs):
        """
        
        loadFromXML( (GeometryModel)self, (str)filename, (str)tag_name) -> None :
            Loads *this from a XML file.
        """
    @staticmethod
    def removeAllCollisionPairs(*args, **kwargs):
        """
        
        removeAllCollisionPairs( (GeometryModel)arg1) -> None :
            Remove all collision pairs.
        """
    @staticmethod
    def removeCollisionPair(*args, **kwargs):
        """
        
        removeCollisionPair( (GeometryModel)self, (CollisionPair)collision_pair) -> None :
            Remove a collision pair.
        """
    @staticmethod
    def removeGeometryObject(*args, **kwargs):
        """
        
        removeGeometryObject( (GeometryModel)self, (str)name) -> None :
            Remove a GeometryObject. Remove also the collision pairs that contain the object.
        """
    @staticmethod
    def saveToBinary(*args, **kwargs):
        """
        
        saveToBinary( (GeometryModel)self, (str)filename) -> None :
            Saves *this inside a binary file.
        
        saveToBinary( (GeometryModel)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Saves *this inside a binary buffer.
        
        saveToBinary( (GeometryModel)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Saves *this inside a static binary buffer.
        """
    @staticmethod
    def saveToString(*args, **kwargs):
        """
        
        saveToString( (GeometryModel)self) -> str :
            Parses the current object to a string.
        """
    @staticmethod
    def saveToText(*args, **kwargs):
        """
        
        saveToText( (GeometryModel)self, (str)filename) -> None :
            Saves *this inside a text file.
        """
    @staticmethod
    def saveToXML(*args, **kwargs):
        """
        
        saveToXML( (GeometryModel)arg1, (str)filename, (str)tag_name) -> None :
            Saves *this inside a XML file.
        """
    @staticmethod
    def setCollisionPairs(*args, **kwargs):
        """
        
        setCollisionPairs( (GeometryModel)self, (numpy.ndarray)collision_map [, (bool)upper=True]) -> None :
            Set the collision pairs from a given input array.
            Each entry of the input matrix defines the activation of a given collision pair(map[i,j] == True means that the pair (i,j) is active).
        """
    @property
    def collisionPairMapping(*args, **kwargs):
        """
        Matrix relating the collision pair ID to a pair of two GeometryObject indexes.
        """
    @property
    def collisionPairs(*args, **kwargs):
        """
        Vector of collision pairs.
        """
    @property
    def geometryObjects(*args, **kwargs):
        """
        Vector of geometries objects.
        """
    @property
    def ngeoms(*args, **kwargs):
        """
        Number of geometries contained in the Geometry Model.
        """
class GeometryNoMaterial(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 40
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)arg1, (GeometryNoMaterial)arg2) -> None
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
class GeometryObject(Boost.Python.instance):
    """
    A wrapper on a collision geometry including its parent joint, parent frame, placement in parent joint's frame.
    
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def CreateCapsule(*args, **kwargs):
        """
        
        CreateCapsule( (float)arg1, (float)arg2) -> GeometryObject
        """
    @staticmethod
    def __address__(*args, **kwargs):
        """
        
        __address__( (GeometryObject)self) -> int :
            Returns the address of the underlying C++ object.
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (GeometryObject)self) -> GeometryObject :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (GeometryObject)self, (dict)memo) -> GeometryObject :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (GeometryObject)arg1, (GeometryObject)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (GeometryObject)arg1) -> tuple
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (GeometryObject)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (str)name, (int)parent_joint, (int)parent_frame, (SE3)placement, (coal.coal_pywrap.CollisionGeometry)collision_geometry [, (str)mesh_path [, (numpy.ndarray)mesh_scale [, (bool)override_material [, (numpy.ndarray)mesh_color [, (str)mesh_texture_path [, (object)mesh_material]]]]]]) -> None :
            Full constructor of a GeometryObject.
        
        __init__( (object)self, (str)name, (int)parent_joint, (SE3)placement, (coal.coal_pywrap.CollisionGeometry)collision_geometry [, (str)mesh_path [, (numpy.ndarray)mesh_scale [, (bool)override_material [, (numpy.ndarray)mesh_color [, (str)mesh_texture_path [, (object)mesh_material]]]]]]) -> None :
            Reduced constructor of a GeometryObject. This constructor does not require to specify the parent frame index.
        
        __init__( (object)arg1, (str)self, (int)name, (int)parent_frame, (coal.coal_pywrap.CollisionGeometry)parent_joint, (SE3)collision_geometry [, (str)placement [, (numpy.ndarray)mesh_path [, (bool)mesh_scale [, (numpy.ndarray)override_material [, (str)mesh_color [, (object)mesh_texture_pathmesh_material]]]]]]) -> None :
            Deprecated. Full constructor of a GeometryObject.
        
        __init__( (object)self, (str)name, (int)parent_joint, (coal.coal_pywrap.CollisionGeometry)collision_geometry, (SE3)placement [, (str)mesh_path [, (numpy.ndarray)mesh_scale [, (bool)override_material [, (numpy.ndarray)mesh_color [, (str)mesh_texture_path [, (object)mesh_material]]]]]]) -> None :
            Deprecated. Reduced constructor of a GeometryObject. This constructor does not require to specify the parent frame index.
        
        __init__( (object)self, (GeometryObject)otherGeometryObject) -> None :
            Copy constructor
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (GeometryObject)arg1, (GeometryObject)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (GeometryObject)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def clone(*args, **kwargs):
        """
        
        clone( (GeometryObject)self) -> GeometryObject :
            Perform a deep copy of this. It will create a copy of the underlying FCL geometry.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (GeometryObject)self) -> GeometryObject :
            Returns a copy of *this.
        """
    @staticmethod
    def loadFromBinary(*args, **kwargs):
        """
        
        loadFromBinary( (GeometryObject)self, (str)filename) -> None :
            Loads *this from a binary file.
        
        loadFromBinary( (GeometryObject)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Loads *this from a binary buffer.
        
        loadFromBinary( (GeometryObject)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Loads *this from a static binary buffer.
        """
    @staticmethod
    def loadFromString(*args, **kwargs):
        """
        
        loadFromString( (GeometryObject)self, (str)string) -> None :
            Parses from the input string the content of the current object.
        """
    @staticmethod
    def loadFromText(*args, **kwargs):
        """
        
        loadFromText( (GeometryObject)self, (str)filename) -> None :
            Loads *this from a text file.
        """
    @staticmethod
    def loadFromXML(*args, **kwargs):
        """
        
        loadFromXML( (GeometryObject)self, (str)filename, (str)tag_name) -> None :
            Loads *this from a XML file.
        """
    @staticmethod
    def saveToBinary(*args, **kwargs):
        """
        
        saveToBinary( (GeometryObject)self, (str)filename) -> None :
            Saves *this inside a binary file.
        
        saveToBinary( (GeometryObject)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Saves *this inside a binary buffer.
        
        saveToBinary( (GeometryObject)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Saves *this inside a static binary buffer.
        """
    @staticmethod
    def saveToString(*args, **kwargs):
        """
        
        saveToString( (GeometryObject)self) -> str :
            Parses the current object to a string.
        """
    @staticmethod
    def saveToText(*args, **kwargs):
        """
        
        saveToText( (GeometryObject)self, (str)filename) -> None :
            Saves *this inside a text file.
        """
    @staticmethod
    def saveToXML(*args, **kwargs):
        """
        
        saveToXML( (GeometryObject)arg1, (str)filename, (str)tag_name) -> None :
            Saves *this inside a XML file.
        """
    @property
    def disableCollision(*args, **kwargs):
        """
        If true, no collision or distance check will be done between the Geometry and any other geometry.
        """
    @disableCollision.setter
    def disableCollision(*args, **kwargs):
        ...
    @property
    def geometry(*args, **kwargs):
        """
        The FCL CollisionGeometry associated to the given GeometryObject.
        """
    @geometry.setter
    def geometry(*args, **kwargs):
        ...
    @property
    def meshColor(*args, **kwargs):
        """
        Color rgba of the mesh.
        """
    @meshColor.setter
    def meshColor(*args, **kwargs):
        ...
    @property
    def meshMaterial(*args, **kwargs):
        """
        Material associated to the mesh (applied only if overrideMaterial is True)
        """
    @meshMaterial.setter
    def meshMaterial(*args, **kwargs):
        ...
    @property
    def meshPath(*args, **kwargs):
        """
        Path to the mesh file.
        """
    @meshPath.setter
    def meshPath(*args, **kwargs):
        ...
    @property
    def meshScale(*args, **kwargs):
        """
        Scaling parameter of the mesh.
        """
    @meshScale.setter
    def meshScale(*args, **kwargs):
        ...
    @property
    def meshTexturePath(*args, **kwargs):
        """
        Path to the mesh texture file.
        """
    @meshTexturePath.setter
    def meshTexturePath(*args, **kwargs):
        ...
    @property
    def name(*args, **kwargs):
        """
        Name associated to the given GeometryObject.
        """
    @name.setter
    def name(*args, **kwargs):
        ...
    @property
    def overrideMaterial(*args, **kwargs):
        """
        Boolean that tells whether material information is stored inside the given GeometryObject.
        """
    @overrideMaterial.setter
    def overrideMaterial(*args, **kwargs):
        ...
    @property
    def parentFrame(*args, **kwargs):
        """
        Index of the parent frame.
        """
    @parentFrame.setter
    def parentFrame(*args, **kwargs):
        ...
    @property
    def parentJoint(*args, **kwargs):
        """
        Index of the parent joint.
        """
    @parentJoint.setter
    def parentJoint(*args, **kwargs):
        ...
    @property
    def placement(*args, **kwargs):
        """
        Position of geometry object in parent joint's frame.
        """
    @placement.setter
    def placement(*args, **kwargs):
        ...
class GeometryPhongMaterial(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 112
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)arg1, (GeometryPhongMaterial)arg2) -> None
        
        __init__( (object)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (float)arg4) -> None
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def meshEmissionColor(*args, **kwargs):
        """
        RGBA emission (ambient) color value of the mesh
        """
    @meshEmissionColor.setter
    def meshEmissionColor(*args, **kwargs):
        ...
    @property
    def meshShininess(*args, **kwargs):
        """
        Shininess associated to the specular lighting model (between 0 and 1)
        """
    @meshShininess.setter
    def meshShininess(*args, **kwargs):
        ...
    @property
    def meshSpecularColor(*args, **kwargs):
        """
        RGBA specular value of the mesh
        """
    @meshSpecularColor.setter
    def meshSpecularColor(*args, **kwargs):
        ...
class GeometryType(Boost.Python.enum):
    COLLISION: typing.ClassVar[GeometryType]  # value = pinocchio.pinocchio_pywrap_default.GeometryType.COLLISION
    VISUAL: typing.ClassVar[GeometryType]  # value = pinocchio.pinocchio_pywrap_default.GeometryType.VISUAL
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'VISUAL': pinocchio.pinocchio_pywrap_default.GeometryType.VISUAL, 'COLLISION': pinocchio.pinocchio_pywrap_default.GeometryType.COLLISION}
    values: typing.ClassVar[dict]  # value = {0: pinocchio.pinocchio_pywrap_default.GeometryType.VISUAL, 1: pinocchio.pinocchio_pywrap_default.GeometryType.COLLISION}
class Inertia(Boost.Python.instance):
    """
    This class represenses a sparse version of a Spatial Inertia and its is defined by its mass, its center of mass location and the rotational inertia expressed around this center of mass.
    
    Supported operations ...
    """
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def FromBox(*args, **kwargs):
        """
        
        FromBox( (float)mass, (float)length_x, (float)length_y, (float)length_z) -> Inertia :
            Returns the Inertia of a box shape with a mass and of dimension the semi axis of length_{x,y,z}.
        """
    @staticmethod
    def FromCapsule(*args, **kwargs):
        """
        
        FromCapsule( (float)mass, (float)radius, (float)height) -> Inertia :
            Computes the Inertia of a capsule defined by its mass, radius and length along the Z axis. Assumes a uniform density.
        """
    @staticmethod
    def FromCylinder(*args, **kwargs):
        """
        
        FromCylinder( (float)mass, (float)radius, (float)length) -> Inertia :
            Returns the Inertia of a cylinder defined by its mass, radius and length along the Z axis.
        """
    @staticmethod
    def FromDynamicParameters(*args, **kwargs):
        """
        
        FromDynamicParameters( (numpy.ndarray)dynamic_parameters) -> Inertia :
            Builds and inertia matrix from a vector of dynamic parameters.
            The parameters are given as dynamic_parameters = [m, mc_x, mc_y, mc_z, I_{xx}, I_{xy}, I_{yy}, I_{xz}, I_{yz}, I_{zz}]^T where I = I_C + mS^T(c)S(c) and I_C has its origin at the barycenter.
        """
    @staticmethod
    def FromEllipsoid(*args, **kwargs):
        """
        
        FromEllipsoid( (float)mass, (float)length_x, (float)length_y, (float)length_z) -> Inertia :
            Returns the Inertia of an ellipsoid shape defined by a mass and given dimensions the semi-axis of values length_{x,y,z}.
        """
    @staticmethod
    def FromLogCholeskyParameters(*args, **kwargs):
        """
        
        FromLogCholeskyParameters( (LogCholeskyParameters)log_cholesky_parameters) -> Inertia :
            Returns the Inertia created from log Cholesky parameters.
        """
    @staticmethod
    def FromPseudoInertia(*args, **kwargs):
        """
        
        FromPseudoInertia( (PseudoInertia)pseudo_inertia) -> Inertia :
            Returns the Inertia created from a pseudo inertia object.
        """
    @staticmethod
    def FromSphere(*args, **kwargs):
        """
        
        FromSphere( (float)mass, (float)radius) -> Inertia :
            Returns the Inertia of a sphere defined by a given mass and radius.
        """
    @staticmethod
    def Identity(*args, **kwargs):
        """
        
        Identity() -> Inertia :
            Returns the identity Inertia.
        """
    @staticmethod
    def Random(*args, **kwargs):
        """
        
        Random() -> Inertia :
            Returns a random Inertia.
        """
    @staticmethod
    def Zero(*args, **kwargs):
        """
        
        Zero() -> Inertia :
            Returns the zero Inertia.
        """
    @staticmethod
    def __add__(*args, **kwargs):
        """
        
        __add__( (Inertia)arg1, (Inertia)arg2) -> object
        """
    @staticmethod
    def __array__(*args, **kwargs):
        """
        
        __array__( (Inertia)arg1) -> numpy.ndarray
        
        __array__( (Inertia)self [, (object)dtype=None [, (object)copy=None]]) -> numpy.ndarray
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (Inertia)self) -> Inertia :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (Inertia)self, (dict)memo) -> Inertia :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Inertia)arg1, (Inertia)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (Inertia)arg1) -> tuple
        """
    @staticmethod
    def __iadd__(*args, **kwargs):
        """
        
        __iadd__( (Inertia)arg1, (Inertia)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (float)mass, (numpy.ndarray)lever, (numpy.ndarray)inertia) -> object :
            Initialize from mass, lever and 3d inertia.
        
        __init__( (object)self) -> None :
            Default constructor.
        
        __init__( (object)self, (Inertia)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (Inertia)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Inertia -> pinocchio.pinocchio_pywrap_default.Inertia
        
        __init__( (object)arg1, (Inertia)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Inertia -> pinocchio.pinocchio_pywrap_default.Inertia
        """
    @staticmethod
    def __isub__(*args, **kwargs):
        """
        
        __isub__( (Inertia)arg1, (Inertia)arg2) -> object
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (Inertia)arg1, (Motion)arg2) -> object
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Inertia)arg1, (Inertia)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (Inertia)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (Inertia)arg1) -> object
        """
    @staticmethod
    def __sub__(*args, **kwargs):
        """
        
        __sub__( (Inertia)arg1, (Inertia)arg2) -> object
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (Inertia)arg1) -> Inertia :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (Inertia)self) -> Inertia :
            Returns a copy of *this.
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (Inertia)self) -> numpy.ndarray
        """
    @staticmethod
    def isApprox(*args, **kwargs):
        """
        
        isApprox( (Inertia)self, (Inertia)other [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to other, within the precision given by prec.
        """
    @staticmethod
    def isZero(*args, **kwargs):
        """
        
        isZero( (Inertia)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the zero Inertia, within the precision given by prec.
        """
    @staticmethod
    def ivx(*args, **kwargs):
        """
        
        ivx( (Inertia)self, (object)v) -> numpy.ndarray :
            Returns the result of I vx, a 6x6 matrix.
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (Inertia)self) -> numpy.ndarray
        """
    @staticmethod
    def se3Action(*args, **kwargs):
        """
        
        se3Action( (Inertia)self, (SE3)M) -> Inertia :
            Returns the result of the action of M on *this.
        """
    @staticmethod
    def se3ActionInverse(*args, **kwargs):
        """
        
        se3ActionInverse( (Inertia)self, (SE3)M) -> Inertia :
            Returns the result of the action of the inverse of M on *this.
        """
    @staticmethod
    def setIdentity(*args, **kwargs):
        """
        
        setIdentity( (Inertia)self) -> None :
            Set *this to be the Identity inertia.
        """
    @staticmethod
    def setRandom(*args, **kwargs):
        """
        
        setRandom( (Inertia)self) -> None :
            Set all the components of *this to random values.
        """
    @staticmethod
    def setZero(*args, **kwargs):
        """
        
        setZero( (Inertia)self) -> None :
            Set all the components of *this to zero.
        """
    @staticmethod
    def toDynamicParameters(*args, **kwargs):
        """
        
        toDynamicParameters( (Inertia)self) -> numpy.ndarray :
            Returns the representation of the matrix as a vector of dynamic parameters.
            The parameters are given as v = [m, mc_x, mc_y, mc_z, I_{xx}, I_{xy}, I_{yy}, I_{xz}, I_{yz}, I_{zz}]^T where I = I_C + mS^T(c)S(c) and I_C has its origin at the barycenter
        """
    @staticmethod
    def toPseudoInertia(*args, **kwargs):
        """
        
        toPseudoInertia( (Inertia)self) -> PseudoInertia :
            Returns the pseudo inertia representation of the inertia.
        """
    @staticmethod
    def variation(*args, **kwargs):
        """
        
        variation( (Inertia)self, (object)v) -> numpy.ndarray :
            Returns the time derivative of the inertia.
        """
    @staticmethod
    def vtiv(*args, **kwargs):
        """
        
        vtiv( (Inertia)self, (object)v) -> float :
            Returns the result of v.T * Iv.
        """
    @staticmethod
    def vxi(*args, **kwargs):
        """
        
        vxi( (Inertia)self, (object)v) -> numpy.ndarray :
            Returns the result of v x* I, a 6x6 matrix.
        """
    @staticmethod
    def vxiv(*args, **kwargs):
        """
        
        vxiv( (Inertia)self, (object)v) -> Force :
            Returns the result of v x Iv.
        """
    @property
    def inertia(*args, **kwargs):
        """
        Rotational part of the Spatial Inertia, i.e. a symmetric matrix representing the rotational inertia around the center of mass.
        """
    @inertia.setter
    def inertia(*args, **kwargs):
        ...
    @property
    def lever(*args, **kwargs):
        """
        Center of mass location of the Spatial Inertia. It corresponds to the location of the center of mass regarding to the frame where the Spatial Inertia is expressed.
        """
    @lever.setter
    def lever(*args, **kwargs):
        ...
    @property
    def mass(*args, **kwargs):
        """
        Mass of the Spatial Inertia.
        """
    @mass.setter
    def mass(*args, **kwargs):
        ...
    @property
    def np(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.Inertia)arg1) -> numpy.ndarray
        """
class JointData(Boost.Python.instance):
    """
    Generic Joint Data
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointData)arg1, (JointData)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (object)joint_data) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointData)arg1, (JointData)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointData)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointData)arg1) -> object
        """
    @staticmethod
    def extract(*args, **kwargs):
        """
        
        extract( (JointData)self) -> object :
            Returns a reference of the internal joint managed by the JointData
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointData)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> pinocchio.pinocchio_pywrap_default.Motion
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointData)arg1) -> pinocchio.pinocchio_pywrap_default.Motion
        """
class JointDataComposite(Boost.Python.instance):
    """
    JointDataComposite
    """
    __instance_size__: typing.ClassVar[int] = 432
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataComposite)arg1, (JointDataComposite)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)arg1, (StdVec_JointDataVector)joint_data_vectors, (int)nq, (int)nv) -> None :
            Init JointDataComposite from a given collection of joint data
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataComposite)arg1, (JointDataComposite)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataComposite)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataComposite)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataComposite)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> numpy.ndarray
        """
    @property
    def StU(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> object
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> pinocchio.pinocchio_pywrap_default.Motion
        """
    @property
    def iMlast(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_SE3
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> numpy.ndarray
        """
    @property
    def joints(*args, **kwargs):
        """
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> object
        """
    @property
    def pjMi(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_SE3
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataComposite)arg1) -> pinocchio.pinocchio_pywrap_default.Motion
        """
class JointDataFreeFlyer(Boost.Python.instance):
    """
    JointDataFreeFlyer
    """
    __instance_size__: typing.ClassVar[int] = 1472
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataFreeFlyer)arg1, (JointDataFreeFlyer)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataFreeFlyer)arg1, (JointDataFreeFlyer)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataFreeFlyer)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataFreeFlyer)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataFreeFlyer)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataFreeFlyer)arg1) -> pinocchio.pinocchio_pywrap_default.Motion
        """
class JointDataHX(Boost.Python.instance):
    """
    JointDataHX
    """
    __instance_size__: typing.ClassVar[int] = 240
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataHX)arg1, (JointDataHX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataHX)arg1, (JointDataHX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataHX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataHX)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataHX)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHX)arg1) -> object
        """
class JointDataHY(Boost.Python.instance):
    """
    JointDataHY
    """
    __instance_size__: typing.ClassVar[int] = 240
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataHY)arg1, (JointDataHY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataHY)arg1, (JointDataHY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataHY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataHY)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataHY)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHY)arg1) -> object
        """
class JointDataHZ(Boost.Python.instance):
    """
    JointDataHZ
    """
    __instance_size__: typing.ClassVar[int] = 240
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataHZ)arg1, (JointDataHZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataHZ)arg1, (JointDataHZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataHZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataHZ)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataHZ)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHZ)arg1) -> object
        """
class JointDataHelicalUnaligned(Boost.Python.instance):
    """
    JointDataHelicalUnaligned
    """
    __instance_size__: typing.ClassVar[int] = 352
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataHelicalUnaligned)arg1, (JointDataHelicalUnaligned)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)arg1, (numpy.ndarray)axis) -> None :
            Init JointDataHelicalUnaligned from an axis with x-y-z components
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataHelicalUnaligned)arg1, (JointDataHelicalUnaligned)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataHelicalUnaligned)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataHelicalUnaligned)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataHelicalUnaligned)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataHelicalUnaligned)arg1) -> object
        """
class JointDataMimic_JointDataRX(Boost.Python.instance):
    """
    JointDataMimic_JointDataRX
    """
    __instance_size__: typing.ClassVar[int] = 272
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataMimic_JointDataRX)arg1, (JointDataMimic_JointDataRX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataMimic_JointDataRX)arg1, (JointDataMimic_JointDataRX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataMimic_JointDataRX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataMimic_JointDataRX)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataMimic_JointDataRX)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRX)arg1) -> object
        """
class JointDataMimic_JointDataRY(Boost.Python.instance):
    """
    JointDataMimic_JointDataRY
    """
    __instance_size__: typing.ClassVar[int] = 272
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataMimic_JointDataRY)arg1, (JointDataMimic_JointDataRY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataMimic_JointDataRY)arg1, (JointDataMimic_JointDataRY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataMimic_JointDataRY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataMimic_JointDataRY)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataMimic_JointDataRY)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRY)arg1) -> object
        """
class JointDataMimic_JointDataRZ(Boost.Python.instance):
    """
    JointDataMimic_JointDataRZ
    """
    __instance_size__: typing.ClassVar[int] = 272
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataMimic_JointDataRZ)arg1, (JointDataMimic_JointDataRZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataMimic_JointDataRZ)arg1, (JointDataMimic_JointDataRZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataMimic_JointDataRZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataMimic_JointDataRZ)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataMimic_JointDataRZ)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataMimic_JointDataRZ)arg1) -> object
        """
class JointDataPX(Boost.Python.instance):
    """
    JointDataPX
    """
    __instance_size__: typing.ClassVar[int] = 208
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataPX)arg1, (JointDataPX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataPX)arg1, (JointDataPX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataPX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataPX)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataPX)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPX)arg1) -> object
        """
class JointDataPY(Boost.Python.instance):
    """
    JointDataPY
    """
    __instance_size__: typing.ClassVar[int] = 208
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataPY)arg1, (JointDataPY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataPY)arg1, (JointDataPY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataPY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataPY)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataPY)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPY)arg1) -> object
        """
class JointDataPZ(Boost.Python.instance):
    """
    JointDataPZ
    """
    __instance_size__: typing.ClassVar[int] = 208
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataPZ)arg1, (JointDataPZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataPZ)arg1, (JointDataPZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataPZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataPZ)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataPZ)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPZ)arg1) -> object
        """
class JointDataPlanar(Boost.Python.instance):
    """
    JointDataPlanar
    """
    __instance_size__: typing.ClassVar[int] = 672
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataPlanar)arg1, (JointDataPlanar)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataPlanar)arg1, (JointDataPlanar)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataPlanar)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataPlanar)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataPlanar)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> object
        """
    @property
    def StU(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> object
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> object
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> object
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPlanar)arg1) -> object
        """
class JointDataPrismaticUnaligned(Boost.Python.instance):
    """
    JointDataPrismaticUnaligned
    """
    __instance_size__: typing.ClassVar[int] = 272
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataPrismaticUnaligned)arg1, (JointDataPrismaticUnaligned)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)arg1, (numpy.ndarray)axis) -> None :
            Init JointDataPrismaticUnaligned from an axis with x-y-z components
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataPrismaticUnaligned)arg1, (JointDataPrismaticUnaligned)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataPrismaticUnaligned)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataPrismaticUnaligned)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataPrismaticUnaligned)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataPrismaticUnaligned)arg1) -> object
        """
class JointDataRUBX(Boost.Python.instance):
    """
    JointDataRUBX
    """
    __instance_size__: typing.ClassVar[int] = 224
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRUBX)arg1, (JointDataRUBX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRUBX)arg1, (JointDataRUBX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRUBX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRUBX)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRUBX)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBX)arg1) -> object
        """
class JointDataRUBY(Boost.Python.instance):
    """
    JointDataRUBY
    """
    __instance_size__: typing.ClassVar[int] = 224
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRUBY)arg1, (JointDataRUBY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRUBY)arg1, (JointDataRUBY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRUBY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRUBY)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRUBY)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBY)arg1) -> object
        """
class JointDataRUBZ(Boost.Python.instance):
    """
    JointDataRUBZ
    """
    __instance_size__: typing.ClassVar[int] = 224
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRUBZ)arg1, (JointDataRUBZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRUBZ)arg1, (JointDataRUBZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRUBZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRUBZ)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRUBZ)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRUBZ)arg1) -> object
        """
class JointDataRX(Boost.Python.instance):
    """
    JointDataRX
    """
    __instance_size__: typing.ClassVar[int] = 224
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRX)arg1, (JointDataRX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRX)arg1, (JointDataRX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRX)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRX)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRX)arg1) -> object
        """
class JointDataRY(Boost.Python.instance):
    """
    JointDataRY
    """
    __instance_size__: typing.ClassVar[int] = 224
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRY)arg1, (JointDataRY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRY)arg1, (JointDataRY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRY)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRY)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRY)arg1) -> object
        """
class JointDataRZ(Boost.Python.instance):
    """
    JointDataRZ
    """
    __instance_size__: typing.ClassVar[int] = 224
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRZ)arg1, (JointDataRZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRZ)arg1, (JointDataRZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRZ)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRZ)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRZ)arg1) -> object
        """
class JointDataRevoluteUnaligned(Boost.Python.instance):
    """
    JointDataRevoluteUnaligned
    """
    __instance_size__: typing.ClassVar[int] = 336
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRevoluteUnaligned)arg1, (JointDataRevoluteUnaligned)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)arg1, (numpy.ndarray)axis) -> None :
            Init JointDataRevoluteUnaligned from an axis with x-y-z components
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRevoluteUnaligned)arg1, (JointDataRevoluteUnaligned)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRevoluteUnaligned)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRevoluteUnaligned)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRevoluteUnaligned)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnaligned)arg1) -> object
        """
class JointDataRevoluteUnboundedUnalignedTpl(Boost.Python.instance):
    """
    JointDataRevoluteUnboundedUnalignedTpl
    """
    __instance_size__: typing.ClassVar[int] = 352
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataRevoluteUnboundedUnalignedTpl)arg1, (JointDataRevoluteUnboundedUnalignedTpl)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataRevoluteUnboundedUnalignedTpl)arg1, (JointDataRevoluteUnboundedUnalignedTpl)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataRevoluteUnboundedUnalignedTpl)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataRevoluteUnboundedUnalignedTpl)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataRevoluteUnboundedUnalignedTpl)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> numpy.ndarray
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> numpy.ndarray
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> numpy.ndarray
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataRevoluteUnboundedUnalignedTpl)arg1) -> object
        """
class JointDataSpherical(Boost.Python.instance):
    """
    JointDataSpherical
    """
    __instance_size__: typing.ClassVar[int] = 672
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataSpherical)arg1, (JointDataSpherical)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataSpherical)arg1, (JointDataSpherical)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataSpherical)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataSpherical)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataSpherical)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> object
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> object
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> object
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSpherical)arg1) -> object
        """
class JointDataSphericalZYX(Boost.Python.instance):
    """
    JointDataSphericalZYX
    """
    __instance_size__: typing.ClassVar[int] = 752
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataSphericalZYX)arg1, (JointDataSphericalZYX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataSphericalZYX)arg1, (JointDataSphericalZYX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataSphericalZYX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataSphericalZYX)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataSphericalZYX)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> object
        """
    @property
    def StU(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> object
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> object
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> object
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataSphericalZYX)arg1) -> object
        """
class JointDataTranslation(Boost.Python.instance):
    """
    JointDataTranslation
    """
    __instance_size__: typing.ClassVar[int] = 592
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataTranslation)arg1, (JointDataTranslation)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataTranslation)arg1, (JointDataTranslation)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataTranslation)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataTranslation)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataTranslation)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> object
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> object
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> object
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> object
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataTranslation)arg1) -> object
        """
class JointDataUniversal(Boost.Python.instance):
    """
    JointDataUniversal
    """
    __instance_size__: typing.ClassVar[int] = 512
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointDataUniversal)arg1, (JointDataUniversal)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointDataUniversal)arg1, (JointDataUniversal)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointDataUniversal)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointDataUniversal)arg1) -> object
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointDataUniversal)self) -> str
        """
    @property
    def Dinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> numpy.ndarray
        """
    @property
    def M(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @property
    def S(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> object
        """
    @property
    def U(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> object
        """
    @property
    def UDinv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> object
        """
    @property
    def c(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> object
        """
    @property
    def joint_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> numpy.ndarray
        """
    @property
    def joint_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> numpy.ndarray
        """
    @property
    def v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointDataUniversal)arg1) -> object
        """
class JointModel(Boost.Python.instance):
    """
    Generic Joint Model
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModel)arg1, (JointModel)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor
        
        __init__( (object)self, (JointModel)other) -> None :
            Copy constructor
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModel)arg1, (JointModel)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModel)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModel)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModel)self, (JointData)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModel)self, (JointData)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModel)self) -> JointData :
            Create data associated to the joint model.
        """
    @staticmethod
    def extract(*args, **kwargs):
        """
        
        extract( (JointModel)self) -> object :
            Returns a reference of the internal joint managed by the JointModel
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModel)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModel)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModel)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModel)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModel)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModel)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModel)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModel)arg1) -> int
        """
class JointModelComposite(Boost.Python.instance):
    """
    JointModelComposite
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelComposite)arg1, (JointModelComposite)arg2) -> object
        
        __eq__( (JointModelComposite)arg1, (JointModelComposite)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (int)size) -> None :
            Init JointModelComposite with a defined size
        
        __init__( (object)arg1, (JointModel)joint_model) -> object :
            Init JointModelComposite from a joint
        
        __init__( (object)arg1, (JointModel)joint_model, (SE3)joint_placement) -> object :
            Init JointModelComposite from a joint and a placement
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelComposite)arg1, (JointModelComposite)arg2) -> object
        
        __ne__( (JointModelComposite)arg1, (JointModelComposite)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelComposite)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelComposite)arg1) -> object
        """
    @staticmethod
    def addJoint(*args, **kwargs):
        """
        
        addJoint( (JointModelComposite)self, (JointModel)joint_model [, (SE3)joint_placement=SE3(array([[1., 0., 0., 0.],[0., 1., 0., 0.],[0., 0., 1., 0.],[0., 0., 0., 1.]]))]) -> JointModelComposite :
            Add a joint to the vector of joints.
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelComposite)self, (JointDataComposite)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelComposite)self, (JointDataComposite)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelComposite)self) -> JointDataComposite :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelComposite)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelComposite)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelComposite)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> int
        """
    @property
    def jointPlacements(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_SE3
        """
    @property
    def joints(*args, **kwargs):
        """
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> object
        """
    @property
    def njoints(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelComposite)arg1) -> int
        """
class JointModelFreeFlyer(Boost.Python.instance):
    """
    JointModelFreeFlyer
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelFreeFlyer)arg1, (JointModelFreeFlyer)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelFreeFlyer)arg1, (JointModelFreeFlyer)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelFreeFlyer)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelFreeFlyer)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelFreeFlyer)self, (JointDataFreeFlyer)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelFreeFlyer)self, (JointDataFreeFlyer)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelFreeFlyer)self) -> JointDataFreeFlyer :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelFreeFlyer)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelFreeFlyer)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelFreeFlyer)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelFreeFlyer)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelFreeFlyer)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelFreeFlyer)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelFreeFlyer)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelFreeFlyer)arg1) -> int
        """
class JointModelHX(Boost.Python.instance):
    """
    JointModelHX
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelHX)arg1, (JointModelHX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (float)pitch) -> None :
            Init JointModelHX with pitch value and the X axis ([1, 0, 0]) as a rotation axis.
        
        __init__( (object)self) -> None :
            Init JointModelHX with pitch 0.0 and the X axis ([1, 0, 0]) as a rotation axis.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelHX)arg1, (JointModelHX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelHX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelHX)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelHX)self, (JointDataHX)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelHX)self, (JointDataHX)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelHX)self) -> JointDataHX :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelHX)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelHX.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelHX)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelHX)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelHX)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHX)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHX)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHX)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHX)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHX)arg1) -> int
        """
    @property
    def pitch(*args, **kwargs):
        """
        Pitch h of the JointModelHX.
        """
    @pitch.setter
    def pitch(*args, **kwargs):
        ...
class JointModelHY(Boost.Python.instance):
    """
    JointModelHY
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelHY)arg1, (JointModelHY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (float)pitch) -> None :
            Init JointModelHY with pitch value and the Y axis ([0, 1, 0]) as a rotation axis.
        
        __init__( (object)self) -> None :
            Init JointModelHY with pitch 0.0 and the Y axis ([0, 1, 0]) as a rotation axis.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelHY)arg1, (JointModelHY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelHY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelHY)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelHY)self, (JointDataHY)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelHY)self, (JointDataHY)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelHY)self) -> JointDataHY :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelHY)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelHY.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelHY)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelHY)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelHY)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHY)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHY)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHY)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHY)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHY)arg1) -> int
        """
    @property
    def pitch(*args, **kwargs):
        """
        Pitch h of the JointModelHY.
        """
    @pitch.setter
    def pitch(*args, **kwargs):
        ...
class JointModelHZ(Boost.Python.instance):
    """
    JointModelHZ
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelHZ)arg1, (JointModelHZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (float)pitch) -> None :
            Init JointModelHZ with pitch value and the Z axis ([0, 0, 1]) as a rotation axis.
        
        __init__( (object)self) -> None :
            Init JointModelHZ with pitch 0.0 and the Z axis ([0, 0, 1]) as a rotation axis.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelHZ)arg1, (JointModelHZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelHZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelHZ)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelHZ)self, (JointDataHZ)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelHZ)self, (JointDataHZ)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelHZ)self) -> JointDataHZ :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelHZ)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelHZ.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelHZ)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelHZ)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelHZ)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHZ)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHZ)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHZ)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHZ)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHZ)arg1) -> int
        """
    @property
    def pitch(*args, **kwargs):
        """
        Pitch h of the JointModelHZ.
        """
    @pitch.setter
    def pitch(*args, **kwargs):
        ...
class JointModelHelicalUnaligned(Boost.Python.instance):
    """
    JointModelHelicalUnaligned
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelHelicalUnaligned)arg1, (JointModelHelicalUnaligned)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (float)x, (float)y, (float)z, (float)pitch) -> None :
            Init JointModelHelicalUnaligned from the components x, y, z of the axis and the pitch
        
        __init__( (object)self, (numpy.ndarray)axis, (float)pitch) -> None :
            Init JointModelHelicalUnaligned from an axis with x-y-z components and the pitch
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelHelicalUnaligned)arg1, (JointModelHelicalUnaligned)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelHelicalUnaligned)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelHelicalUnaligned)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelHelicalUnaligned)self, (JointDataHelicalUnaligned)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelHelicalUnaligned)self, (JointDataHelicalUnaligned)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelHelicalUnaligned)self) -> JointDataHelicalUnaligned :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelHelicalUnaligned)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelHelicalUnaligned)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelHelicalUnaligned)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def axis(*args, **kwargs):
        """
        Translation axis of the JointModelHelicalUnaligned.
        """
    @axis.setter
    def axis(*args, **kwargs):
        ...
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHelicalUnaligned)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHelicalUnaligned)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHelicalUnaligned)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHelicalUnaligned)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelHelicalUnaligned)arg1) -> int
        """
    @property
    def pitch(*args, **kwargs):
        """
        Pitch h of the JointModelHelicalUnaligned.
        """
    @pitch.setter
    def pitch(*args, **kwargs):
        ...
class JointModelMimic_JointModelRX(Boost.Python.instance):
    """
    JointModelMimic_JointModelRX
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelMimic_JointModelRX)arg1, (JointModelMimic_JointModelRX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelMimic_JointModelRX)arg1, (JointModelMimic_JointModelRX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelMimic_JointModelRX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelMimic_JointModelRX)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelMimic_JointModelRX)self, (JointDataMimic_JointDataRX)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelMimic_JointModelRX)self, (JointDataMimic_JointDataRX)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelMimic_JointModelRX)self) -> JointDataMimic_JointDataRX :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelMimic_JointModelRX)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelMimic_JointModelRX)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelMimic_JointModelRX)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRX)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRX)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRX)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRX)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRX)arg1) -> int
        """
class JointModelMimic_JointModelRY(Boost.Python.instance):
    """
    JointModelMimic_JointModelRY
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelMimic_JointModelRY)arg1, (JointModelMimic_JointModelRY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelMimic_JointModelRY)arg1, (JointModelMimic_JointModelRY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelMimic_JointModelRY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelMimic_JointModelRY)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelMimic_JointModelRY)self, (JointDataMimic_JointDataRY)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelMimic_JointModelRY)self, (JointDataMimic_JointDataRY)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelMimic_JointModelRY)self) -> JointDataMimic_JointDataRY :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelMimic_JointModelRY)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelMimic_JointModelRY)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelMimic_JointModelRY)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRY)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRY)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRY)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRY)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRY)arg1) -> int
        """
class JointModelMimic_JointModelRZ(Boost.Python.instance):
    """
    JointModelMimic_JointModelRZ
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelMimic_JointModelRZ)arg1, (JointModelMimic_JointModelRZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelMimic_JointModelRZ)arg1, (JointModelMimic_JointModelRZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelMimic_JointModelRZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelMimic_JointModelRZ)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelMimic_JointModelRZ)self, (JointDataMimic_JointDataRZ)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelMimic_JointModelRZ)self, (JointDataMimic_JointDataRZ)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelMimic_JointModelRZ)self) -> JointDataMimic_JointDataRZ :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelMimic_JointModelRZ)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelMimic_JointModelRZ)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelMimic_JointModelRZ)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRZ)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRZ)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRZ)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRZ)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelMimic_JointModelRZ)arg1) -> int
        """
class JointModelPX(Boost.Python.instance):
    """
    JointModelPX
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelPX)arg1, (JointModelPX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelPX with the X axis ([1, 0, 0]) as rotation axis
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelPX)arg1, (JointModelPX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelPX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelPX)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelPX)self, (JointDataPX)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelPX)self, (JointDataPX)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelPX)self) -> JointDataPX :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelPX)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelPX.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelPX)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelPX)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelPX)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPX)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPX)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPX)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPX)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPX)arg1) -> int
        """
class JointModelPY(Boost.Python.instance):
    """
    JointModelPY
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelPY)arg1, (JointModelPY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelPY with the Y axis ([0, 1, 0]) as rotation axis
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelPY)arg1, (JointModelPY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelPY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelPY)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelPY)self, (JointDataPY)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelPY)self, (JointDataPY)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelPY)self) -> JointDataPY :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelPY)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelPY.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelPY)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelPY)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelPY)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPY)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPY)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPY)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPY)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPY)arg1) -> int
        """
class JointModelPZ(Boost.Python.instance):
    """
    JointModelPZ
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelPZ)arg1, (JointModelPZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelPZ with the Z axis ([0, 0, 1]) as rotation axis
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelPZ)arg1, (JointModelPZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelPZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelPZ)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelPZ)self, (JointDataPZ)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelPZ)self, (JointDataPZ)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelPZ)self) -> JointDataPZ :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelPZ)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelPZ.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelPZ)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelPZ)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelPZ)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPZ)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPZ)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPZ)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPZ)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPZ)arg1) -> int
        """
class JointModelPlanar(Boost.Python.instance):
    """
    JointModelPlanar
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelPlanar)arg1, (JointModelPlanar)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelPlanar)arg1, (JointModelPlanar)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelPlanar)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelPlanar)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelPlanar)self, (JointDataPlanar)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelPlanar)self, (JointDataPlanar)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelPlanar)self) -> JointDataPlanar :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelPlanar)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelPlanar)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelPlanar)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPlanar)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPlanar)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPlanar)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPlanar)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPlanar)arg1) -> int
        """
class JointModelPrismaticUnaligned(Boost.Python.instance):
    """
    JointModelPrismaticUnaligned
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelPrismaticUnaligned)arg1, (JointModelPrismaticUnaligned)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (float)x, (float)y, (float)z) -> None :
            Init JointModelPrismaticUnaligned from the components x, y, z of the axis
        
        __init__( (object)self, (numpy.ndarray)axis) -> None :
            Init JointModelPrismaticUnaligned from an axis with x-y-z components
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelPrismaticUnaligned)arg1, (JointModelPrismaticUnaligned)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelPrismaticUnaligned)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelPrismaticUnaligned)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelPrismaticUnaligned)self, (JointDataPrismaticUnaligned)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelPrismaticUnaligned)self, (JointDataPrismaticUnaligned)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelPrismaticUnaligned)self) -> JointDataPrismaticUnaligned :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelPrismaticUnaligned)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelPrismaticUnaligned)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelPrismaticUnaligned)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def axis(*args, **kwargs):
        """
        Translation axis of the JointModelPrismaticUnaligned.
        """
    @axis.setter
    def axis(*args, **kwargs):
        ...
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPrismaticUnaligned)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPrismaticUnaligned)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPrismaticUnaligned)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPrismaticUnaligned)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelPrismaticUnaligned)arg1) -> int
        """
class JointModelRUBX(Boost.Python.instance):
    """
    JointModelRUBX
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRUBX)arg1, (JointModelRUBX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelRUBX with the X axis ([1, 0, 0]) as rotation axis
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRUBX)arg1, (JointModelRUBX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRUBX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRUBX)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRUBX)self, (JointDataRUBX)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRUBX)self, (JointDataRUBX)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRUBX)self) -> JointDataRUBX :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelRUBX)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelRUBX.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRUBX)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRUBX)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRUBX)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBX)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBX)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBX)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBX)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBX)arg1) -> int
        """
class JointModelRUBY(Boost.Python.instance):
    """
    JointModelRUBY
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRUBY)arg1, (JointModelRUBY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelRUBY with the Y axis ([0, 1, 0]) as rotation axis
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRUBY)arg1, (JointModelRUBY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRUBY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRUBY)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRUBY)self, (JointDataRUBY)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRUBY)self, (JointDataRUBY)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRUBY)self) -> JointDataRUBY :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelRUBY)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelRUBY.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRUBY)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRUBY)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRUBY)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBY)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBY)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBY)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBY)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBY)arg1) -> int
        """
class JointModelRUBZ(Boost.Python.instance):
    """
    JointModelRUBZ
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRUBZ)arg1, (JointModelRUBZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelRUBZ with the Z axis ([0, 0, 1]) as rotation axis
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRUBZ)arg1, (JointModelRUBZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRUBZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRUBZ)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRUBZ)self, (JointDataRUBZ)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRUBZ)self, (JointDataRUBZ)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRUBZ)self) -> JointDataRUBZ :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelRUBZ)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelRUBZ.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRUBZ)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRUBZ)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRUBZ)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBZ)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBZ)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBZ)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBZ)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRUBZ)arg1) -> int
        """
class JointModelRX(Boost.Python.instance):
    """
    JointModelRX
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRX)arg1, (JointModelRX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelRX with the X axis ([1, 0, 0]) as rotation axis.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRX)arg1, (JointModelRX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRX)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRX)self, (JointDataRX)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRX)self, (JointDataRX)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRX)self) -> JointDataRX :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelRX)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelRX.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRX)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRX)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRX)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRX)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRX)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRX)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRX)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRX)arg1) -> int
        """
class JointModelRY(Boost.Python.instance):
    """
    JointModelRY
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRY)arg1, (JointModelRY)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelRY with the Y axis ([0, 1, 0]) as rotation axis.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRY)arg1, (JointModelRY)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRY)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRY)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRY)self, (JointDataRY)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRY)self, (JointDataRY)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRY)self) -> JointDataRY :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelRY)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelRY.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRY)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRY)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRY)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRY)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRY)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRY)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRY)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRY)arg1) -> int
        """
class JointModelRZ(Boost.Python.instance):
    """
    JointModelRZ
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRZ)arg1, (JointModelRZ)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self) -> None :
            Init JointModelRZ with the Z axis ([0, 0, 1]) as rotation axis
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRZ)arg1, (JointModelRZ)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRZ)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRZ)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRZ)self, (JointDataRZ)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRZ)self, (JointDataRZ)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRZ)self) -> JointDataRZ :
            Create data associated to the joint model.
        """
    @staticmethod
    def getMotionAxis(*args, **kwargs):
        """
        
        getMotionAxis( (JointModelRZ)arg1) -> numpy.ndarray :
            Rotation axis of the JointModelRZ.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRZ)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRZ)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRZ)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRZ)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRZ)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRZ)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRZ)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRZ)arg1) -> int
        """
class JointModelRevoluteUnaligned(Boost.Python.instance):
    """
    JointModelRevoluteUnaligned
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRevoluteUnaligned)arg1, (JointModelRevoluteUnaligned)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (float)x, (float)y, (float)z) -> None :
            Init JointModelRevoluteUnaligned from the components x, y, z of the axis
        
        __init__( (object)self, (numpy.ndarray)axis) -> None :
            Init JointModelRevoluteUnaligned from an axis with x-y-z components
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRevoluteUnaligned)arg1, (JointModelRevoluteUnaligned)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRevoluteUnaligned)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRevoluteUnaligned)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRevoluteUnaligned)self, (JointDataRevoluteUnaligned)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRevoluteUnaligned)self, (JointDataRevoluteUnaligned)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRevoluteUnaligned)self) -> JointDataRevoluteUnaligned :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRevoluteUnaligned)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRevoluteUnaligned)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRevoluteUnaligned)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def axis(*args, **kwargs):
        """
        Rotation axis of the JointModelRevoluteUnaligned.
        """
    @axis.setter
    def axis(*args, **kwargs):
        ...
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnaligned)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnaligned)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnaligned)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnaligned)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnaligned)arg1) -> int
        """
class JointModelRevoluteUnboundedUnaligned(Boost.Python.instance):
    """
    JointModelRevoluteUnboundedUnaligned
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelRevoluteUnboundedUnaligned)arg1, (JointModelRevoluteUnboundedUnaligned)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelRevoluteUnboundedUnaligned)arg1, (JointModelRevoluteUnboundedUnaligned)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelRevoluteUnboundedUnaligned)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelRevoluteUnboundedUnaligned)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelRevoluteUnboundedUnaligned)self, (JointDataRevoluteUnboundedUnalignedTpl)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelRevoluteUnboundedUnaligned)self, (JointDataRevoluteUnboundedUnalignedTpl)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelRevoluteUnboundedUnaligned)self) -> JointDataRevoluteUnboundedUnalignedTpl :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelRevoluteUnboundedUnaligned)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelRevoluteUnboundedUnaligned)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelRevoluteUnboundedUnaligned)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnboundedUnaligned)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnboundedUnaligned)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnboundedUnaligned)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnboundedUnaligned)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelRevoluteUnboundedUnaligned)arg1) -> int
        """
class JointModelSpherical(Boost.Python.instance):
    """
    JointModelSpherical
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelSpherical)arg1, (JointModelSpherical)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelSpherical)arg1, (JointModelSpherical)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelSpherical)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelSpherical)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelSpherical)self, (JointDataSpherical)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelSpherical)self, (JointDataSpherical)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelSpherical)self) -> JointDataSpherical :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelSpherical)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelSpherical)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelSpherical)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSpherical)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSpherical)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSpherical)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSpherical)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSpherical)arg1) -> int
        """
class JointModelSphericalZYX(Boost.Python.instance):
    """
    JointModelSphericalZYX
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelSphericalZYX)arg1, (JointModelSphericalZYX)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelSphericalZYX)arg1, (JointModelSphericalZYX)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelSphericalZYX)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelSphericalZYX)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelSphericalZYX)self, (JointDataSphericalZYX)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelSphericalZYX)self, (JointDataSphericalZYX)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelSphericalZYX)self) -> JointDataSphericalZYX :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelSphericalZYX)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelSphericalZYX)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelSphericalZYX)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSphericalZYX)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSphericalZYX)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSphericalZYX)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSphericalZYX)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelSphericalZYX)arg1) -> int
        """
class JointModelTranslation(Boost.Python.instance):
    """
    JointModelTranslation
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelTranslation)arg1, (JointModelTranslation)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelTranslation)arg1, (JointModelTranslation)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelTranslation)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelTranslation)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelTranslation)self, (JointDataTranslation)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelTranslation)self, (JointDataTranslation)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelTranslation)self) -> JointDataTranslation :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelTranslation)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelTranslation)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelTranslation)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelTranslation)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelTranslation)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelTranslation)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelTranslation)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelTranslation)arg1) -> int
        """
class JointModelUniversal(Boost.Python.instance):
    """
    JointModelUniversal
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (JointModelUniversal)arg1, (JointModelUniversal)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None
        
        __init__( (object)self, (float)x1, (float)y1, (float)z1, (float)x2, (float)y2, (float)z2) -> None :
            Init JointModelUniversal from the components x, y, z of the axes
        
        __init__( (object)self, (numpy.ndarray)axis1, (numpy.ndarray)axis2) -> None :
            Init JointModelUniversal from two axes with x-y-z components
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (JointModelUniversal)arg1, (JointModelUniversal)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (JointModelUniversal)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (JointModelUniversal)arg1) -> object
        """
    @staticmethod
    def calc(*args, **kwargs):
        """
        
        calc( (JointModelUniversal)self, (JointDataUniversal)jdata, (numpy.ndarray)q) -> None
        
        calc( (JointModelUniversal)self, (JointDataUniversal)jdata, (numpy.ndarray)q, (numpy.ndarray)v) -> None
        """
    @staticmethod
    def classname(*args, **kwargs):
        """
        
        classname() -> str
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (JointModelUniversal)self) -> JointDataUniversal :
            Create data associated to the joint model.
        """
    @staticmethod
    def hasSameIndexes(*args, **kwargs):
        """
        
        hasSameIndexes( (JointModelUniversal)self, (object)other) -> bool :
            Check if this has same indexes than other.
        """
    @staticmethod
    def setIndexes(*args, **kwargs):
        """
        
        setIndexes( (JointModelUniversal)self, (int)joint_id, (int)idx_q, (int)idx_v) -> None
        """
    @staticmethod
    def shortname(*args, **kwargs):
        """
        
        shortname( (JointModelUniversal)self) -> str :
            Returns string indicating the joint type (class name):
            	- JointModelR[*]: Revolute Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnaligned: Revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelRUB[*]: Unbounded revolute Joint (without position limits), with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelRevoluteUnboundedUnaligned: Unbounded revolute Joint, with rotation axis not aligned with X, Y, nor Z
            	- JointModelP[*]: Prismatic Joint, with rotation axis [*] ∈ [X,Y,Z]
            	- JointModelPlanar: Planar joint
            	- JointModelPrismaticUnaligned: Prismatic joint, with translation axis not aligned with X, Y, nor Z
            	- JointModelSphericalZYX: Spherical joint (3D rotation)
            	- JointModelTranslation: Translation joint (3D translation)
            	- JointModelFreeFlyer: Joint enabling 3D rotation and translations.
        """
    @property
    def axis1(*args, **kwargs):
        """
        First rotation axis of the JointModelUniversal.
        """
    @axis1.setter
    def axis1(*args, **kwargs):
        ...
    @property
    def axis2(*args, **kwargs):
        """
        Second rotation axis of the JointModelUniversal.
        """
    @axis2.setter
    def axis2(*args, **kwargs):
        ...
    @property
    def hasConfigurationLimit(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits.
        """
    @property
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        Return vector of boolean if joint has configuration limits in tangent space.
        """
    @property
    def id(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelUniversal)arg1) -> int
        """
    @property
    def idx_q(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelUniversal)arg1) -> int
        """
    @property
    def idx_v(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelUniversal)arg1) -> int
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelUniversal)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.JointModelUniversal)arg1) -> int
        """
class KinematicLevel(Boost.Python.enum):
    ACCELERATION: typing.ClassVar[KinematicLevel]  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.ACCELERATION
    POSITION: typing.ClassVar[KinematicLevel]  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.POSITION
    VELOCITY: typing.ClassVar[KinematicLevel]  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.VELOCITY
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'POSITION': pinocchio.pinocchio_pywrap_default.KinematicLevel.POSITION, 'VELOCITY': pinocchio.pinocchio_pywrap_default.KinematicLevel.VELOCITY, 'ACCELERATION': pinocchio.pinocchio_pywrap_default.KinematicLevel.ACCELERATION}
    values: typing.ClassVar[dict]  # value = {0: pinocchio.pinocchio_pywrap_default.KinematicLevel.POSITION, 1: pinocchio.pinocchio_pywrap_default.KinematicLevel.VELOCITY, 2: pinocchio.pinocchio_pywrap_default.KinematicLevel.ACCELERATION}
class LanczosDecomposition(Boost.Python.instance):
    """
    Lanczos decomposition.
    """
    @staticmethod
    def Qs(*args, **kwargs):
        """
        
        Qs( (LanczosDecomposition)self) -> object :
            Returns the orthogonal basis associated with the Lanczos decomposition.
        """
    @staticmethod
    def Ts(*args, **kwargs):
        """
        
        Ts( (LanczosDecomposition)self) -> TridiagonalSymmetricMatrix :
            Returns the tridiagonal matrix associated with the Lanczos decomposition.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (LanczosDecomposition)arg1, (LanczosDecomposition)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (numpy.ndarray)mat, (int)decomposition_size) -> None :
            Default constructor from a given matrix and a given decomposition size.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (LanczosDecomposition)arg1, (LanczosDecomposition)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def compute(*args, **kwargs):
        """
        
        compute( (LanczosDecomposition)self, (numpy.ndarray)mat) -> None :
            Computes the Lanczos decomposition for the given input matrix.
        """
    @staticmethod
    def computeDecompositionResidual(*args, **kwargs):
        """
        
        computeDecompositionResidual( (LanczosDecomposition)self, (numpy.ndarray)mat) -> numpy.ndarray :
            Computes the residual associated with the decomposition, namely, the quantity 
        $ A Q_s - Q_s T_s 
        $
        """
    @staticmethod
    def rank(*args, **kwargs):
        """
        
        rank( (LanczosDecomposition)self) -> int :
            Returns the rank of the decomposition.
        """
class LieGroup(Boost.Python.instance):
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (LieGroup)arg1, (LieGroup)arg2) -> object
        """
    @staticmethod
    def __imul__(*args, **kwargs):
        """
        
        __imul__( (LieGroup)arg1, (LieGroup)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None :
            Default constructor
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (LieGroup)arg1, (LieGroup)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def dDifference(*args, **kwargs):
        """
        
        dDifference( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (ArgumentPosition)arg4) -> numpy.ndarray
        
        dDifference( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (ArgumentPosition)arg4, (numpy.ndarray)arg5, (int)arg6) -> numpy.ndarray
        
        dDifference( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (ArgumentPosition)arg4, (int)arg5, (numpy.ndarray)arg6) -> numpy.ndarray
        """
    @staticmethod
    def dIntegrate(*args, **kwargs):
        """
        
        dIntegrate( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (ArgumentPosition)arg4) -> numpy.ndarray
        """
    @staticmethod
    def dIntegrateTransport(*args, **kwargs):
        """
        
        dIntegrateTransport( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (numpy.ndarray)arg4, (ArgumentPosition)arg5) -> numpy.ndarray
        """
    @staticmethod
    def dIntegrate_dq(*args, **kwargs):
        """
        
        dIntegrate_dq( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3) -> numpy.ndarray
        
        dIntegrate_dq( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (numpy.ndarray)arg4, (int)arg5) -> numpy.ndarray
        
        dIntegrate_dq( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (int)arg4, (numpy.ndarray)arg5) -> numpy.ndarray
        """
    @staticmethod
    def dIntegrate_dv(*args, **kwargs):
        """
        
        dIntegrate_dv( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3) -> numpy.ndarray
        
        dIntegrate_dv( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (numpy.ndarray)arg4, (int)arg5) -> numpy.ndarray
        
        dIntegrate_dv( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (int)arg4, (numpy.ndarray)arg5) -> numpy.ndarray
        """
    @staticmethod
    def difference(*args, **kwargs):
        """
        
        difference( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3) -> numpy.ndarray
        """
    @staticmethod
    def distance(*args, **kwargs):
        """
        
        distance( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3) -> float
        """
    @staticmethod
    def integrate(*args, **kwargs):
        """
        
        integrate( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3) -> numpy.ndarray
        """
    @staticmethod
    def interpolate(*args, **kwargs):
        """
        
        interpolate( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3, (float)arg4) -> numpy.ndarray
        """
    @staticmethod
    def normalize(*args, **kwargs):
        """
        
        normalize( (LieGroup)arg1, (numpy.ndarray)arg2) -> None
        """
    @staticmethod
    def random(*args, **kwargs):
        """
        
        random( (LieGroup)arg1) -> numpy.ndarray
        """
    @staticmethod
    def randomConfiguration(*args, **kwargs):
        """
        
        randomConfiguration( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3) -> numpy.ndarray
        """
    @staticmethod
    def squaredDistance(*args, **kwargs):
        """
        
        squaredDistance( (LieGroup)arg1, (numpy.ndarray)arg2, (numpy.ndarray)arg3) -> float
        """
    @property
    def name(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.LieGroup)arg1) -> str
        """
    @property
    def neutral(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.LieGroup)arg1) -> numpy.ndarray
        """
    @property
    def nq(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.LieGroup)arg1) -> int
        """
    @property
    def nv(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.LieGroup)arg1) -> int
        """
class LogCholeskyParameters(Boost.Python.instance):
    """
    This class represents log Cholesky parameters.
    
    Supported operations ...
    """
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __array__(*args, **kwargs):
        """
        
        __array__( (LogCholeskyParameters)arg1) -> numpy.ndarray
        
        __array__( (LogCholeskyParameters)self [, (object)dtype=None [, (object)copy=None]]) -> numpy.ndarray
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (LogCholeskyParameters)self) -> LogCholeskyParameters :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (LogCholeskyParameters)self, (dict)memo) -> LogCholeskyParameters :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (LogCholeskyParameters)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1, (numpy.ndarray)log_cholesky_parameters) -> object :
            Initialize from log cholesky parameters.
        
        __init__( (object)self, (LogCholeskyParameters)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (LogCholeskyParameters)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.LogCholeskyParameters -> pinocchio.pinocchio_pywrap_default.LogCholeskyParameters
        
        __init__( (object)arg1, (LogCholeskyParameters)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.LogCholeskyParameters -> pinocchio.pinocchio_pywrap_default.LogCholeskyParameters
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (LogCholeskyParameters)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (LogCholeskyParameters)arg1) -> object
        """
    @staticmethod
    def calculateJacobian(*args, **kwargs):
        """
        
        calculateJacobian( (LogCholeskyParameters)self) -> numpy.ndarray :
            Calculates the Jacobian of the log Cholesky parameters.
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (LogCholeskyParameters)arg1) -> LogCholeskyParameters :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (LogCholeskyParameters)self) -> LogCholeskyParameters :
            Returns a copy of *this.
        """
    @staticmethod
    def toDynamicParameters(*args, **kwargs):
        """
        
        toDynamicParameters( (LogCholeskyParameters)self) -> numpy.ndarray :
            Returns the dynamic parameters representation.
        """
    @staticmethod
    def toInertia(*args, **kwargs):
        """
        
        toInertia( (LogCholeskyParameters)self) -> Inertia :
            Returns the Inertia representation.
        """
    @staticmethod
    def toPseudoInertia(*args, **kwargs):
        """
        
        toPseudoInertia( (LogCholeskyParameters)self) -> PseudoInertia :
            Returns the Pseudo Inertia representation.
        """
    @property
    def parameters(*args, **kwargs):
        """
        Log Cholesky parameters.
        """
    @parameters.setter
    def parameters(*args, **kwargs):
        ...
class LogLevel(Boost.Python.enum):
    CONSOLE_BRIDGE_LOG_DEBUG: typing.ClassVar[LogLevel]  # value = pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_DEBUG
    CONSOLE_BRIDGE_LOG_ERROR: typing.ClassVar[LogLevel]  # value = pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_ERROR
    CONSOLE_BRIDGE_LOG_INFO: typing.ClassVar[LogLevel]  # value = pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_INFO
    CONSOLE_BRIDGE_LOG_NONE: typing.ClassVar[LogLevel]  # value = pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_NONE
    CONSOLE_BRIDGE_LOG_WARN: typing.ClassVar[LogLevel]  # value = pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_WARN
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'CONSOLE_BRIDGE_LOG_DEBUG': pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_DEBUG, 'CONSOLE_BRIDGE_LOG_INFO': pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_INFO, 'CONSOLE_BRIDGE_LOG_WARN': pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_WARN, 'CONSOLE_BRIDGE_LOG_ERROR': pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_ERROR, 'CONSOLE_BRIDGE_LOG_NONE': pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_NONE}
    values: typing.ClassVar[dict]  # value = {0: pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_DEBUG, 1: pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_INFO, 2: pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_WARN, 3: pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_ERROR, 4: pinocchio.pinocchio_pywrap_default.LogLevel.CONSOLE_BRIDGE_LOG_NONE}
class Model(Boost.Python.instance):
    """
    Articulated Rigid Body model
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    gravity981: typing.ClassVar[numpy.ndarray]  # value = array([ 0.  ,  0.  , -9.81])
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (Model)self) -> Model :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (Model)self, (dict)memo) -> Model :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Model)arg1, (Model)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (Model)arg1) -> tuple
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (Model)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor. Constructs an empty model.
        
        __init__( (object)self, (Model)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (Model)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Model -> pinocchio.pinocchio_pywrap_default.Model
        
        __init__( (object)arg1, (Model)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Model -> pinocchio.pinocchio_pywrap_default.Model
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Model)arg1, (Model)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (Model)arg1) -> object
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (Model)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (Model)arg1) -> object
        """
    @staticmethod
    def addBodyFrame(*args, **kwargs):
        """
        
        addBodyFrame( (Model)self, (str)body_name, (int)parentJoint, (SE3)body_placement, (int)previous_frame) -> int :
            add a body to the frame tree
        """
    @staticmethod
    def addFrame(*args, **kwargs):
        """
        
        addFrame( (Model)self, (Frame)frame [, (bool)append_inertia=True]) -> int :
            Add a frame to the vector of frames. If append_inertia set to True, the inertia value contained in frame will be added to the inertia supported by the parent joint.
        """
    @staticmethod
    def addJoint(*args, **kwargs):
        """
        
        addJoint( (Model)self, (int)parent_id, (JointModel)joint_model, (SE3)joint_placement, (str)joint_name) -> int :
            Adds a joint to the kinematic tree. The joint is defined by its placement relative to its parent joint and its name.
        
        addJoint( (Model)self, (int)parent_id, (JointModel)joint_model, (SE3)joint_placement, (str)joint_name, (numpy.ndarray)max_effort, (numpy.ndarray)max_velocity, (numpy.ndarray)min_config, (numpy.ndarray)max_config) -> int :
            Adds a joint to the kinematic tree with given bounds. The joint is defined by its placement relative to its parent joint and its name.This signature also takes as input effort, velocity limits as well as the bounds on the joint configuration.
        
        addJoint( (Model)self, (int)parent_id, (JointModel)joint_model, (SE3)joint_placement, (str)joint_name, (numpy.ndarray)max_effort, (numpy.ndarray)max_velocity, (numpy.ndarray)min_config, (numpy.ndarray)max_config, (numpy.ndarray)friction, (numpy.ndarray)damping) -> int :
            Adds a joint to the kinematic tree with given bounds. The joint is defined by its placement relative to its parent joint and its name.
            This signature also takes as input effort, velocity limits as well as the bounds on the joint configuration.
            The user should also provide the friction and damping related to the joint.
        """
    @staticmethod
    def addJointFrame(*args, **kwargs):
        """
        
        addJointFrame( (Model)self, (int)joint_id [, (int)frame_id=0]) -> int :
            Add the joint provided by its joint_id as a frame to the frame tree.
            The frame_id may be optionally provided.
        """
    @staticmethod
    def appendBodyToJoint(*args, **kwargs):
        """
        
        appendBodyToJoint( (Model)self, (int)joint_id, (Inertia)body_inertia, (SE3)body_placement) -> None :
            Appends a body to the joint given by its index. The body is defined by its inertia, its relative placement regarding to the joint and its name.
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (Model)arg1) -> Model :
            Returns a cast of *this.
        """
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (Model)self, (Data)data) -> bool :
            Check consistency of data wrt model.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (Model)self) -> Model :
            Returns a copy of *this.
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (Model)self) -> Data :
            Create a Data object for the given model.
        """
    @staticmethod
    def existBodyName(*args, **kwargs):
        """
        
        existBodyName( (Model)self, (str)name) -> bool :
            Check if a frame of type BODY exists, given its name
        """
    @staticmethod
    def existFrame(*args, **kwargs):
        """
        
        existFrame( (Model)self, (str)name [, (FrameType)type=pinocchio.pinocchio_pywrap_default.FrameType(31)]) -> bool :
            Returns true if the frame given by its name exists inside the Model with the given type.
        """
    @staticmethod
    def existJointName(*args, **kwargs):
        """
        
        existJointName( (Model)self, (str)name) -> bool :
            Check if a joint given by its name exists
        """
    @staticmethod
    def getBodyId(*args, **kwargs):
        """
        
        getBodyId( (Model)self, (str)name) -> int :
            Return the index of a frame of type BODY given by its name
        """
    @staticmethod
    def getFrameId(*args, **kwargs):
        """
        
        getFrameId( (Model)self, (str)name [, (FrameType)type=pinocchio.pinocchio_pywrap_default.FrameType(31)]) -> int :
            Returns the index of the frame given by its name and its type.If the frame is not in the frames vector, it returns the current size of the frames vector.
        """
    @staticmethod
    def getJointId(*args, **kwargs):
        """
        
        getJointId( (Model)self, (str)name) -> int :
            Return the index of a joint given by its name
        """
    @staticmethod
    def hasConfigurationLimit(*args, **kwargs):
        """
        
        hasConfigurationLimit( (Model)self) -> StdVec_Bool :
            Returns list of boolean if joints have configuration limit.
        """
    @staticmethod
    def hasConfigurationLimitInTangent(*args, **kwargs):
        """
        
        hasConfigurationLimitInTangent( (Model)self) -> StdVec_Bool :
            Returns list of boolean if joints have configuration limit in tangent space  .
        """
    @staticmethod
    def loadFromBinary(*args, **kwargs):
        """
        
        loadFromBinary( (Model)self, (str)filename) -> None :
            Loads *this from a binary file.
        
        loadFromBinary( (Model)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Loads *this from a binary buffer.
        
        loadFromBinary( (Model)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Loads *this from a static binary buffer.
        """
    @staticmethod
    def loadFromString(*args, **kwargs):
        """
        
        loadFromString( (Model)self, (str)string) -> None :
            Parses from the input string the content of the current object.
        """
    @staticmethod
    def loadFromText(*args, **kwargs):
        """
        
        loadFromText( (Model)self, (str)filename) -> None :
            Loads *this from a text file.
        """
    @staticmethod
    def loadFromXML(*args, **kwargs):
        """
        
        loadFromXML( (Model)self, (str)filename, (str)tag_name) -> None :
            Loads *this from a XML file.
        """
    @staticmethod
    def saveToBinary(*args, **kwargs):
        """
        
        saveToBinary( (Model)self, (str)filename) -> None :
            Saves *this inside a binary file.
        
        saveToBinary( (Model)self, (pinocchio.pinocchio_pywrap_default.serialization.StreamBuffer)buffer) -> None :
            Saves *this inside a binary buffer.
        
        saveToBinary( (Model)self, (pinocchio.pinocchio_pywrap_default.serialization.StaticBuffer)buffer) -> None :
            Saves *this inside a static binary buffer.
        """
    @staticmethod
    def saveToString(*args, **kwargs):
        """
        
        saveToString( (Model)self) -> str :
            Parses the current object to a string.
        """
    @staticmethod
    def saveToText(*args, **kwargs):
        """
        
        saveToText( (Model)self, (str)filename) -> None :
            Saves *this inside a text file.
        """
    @staticmethod
    def saveToXML(*args, **kwargs):
        """
        
        saveToXML( (Model)arg1, (str)filename, (str)tag_name) -> None :
            Saves *this inside a XML file.
        """
    @property
    def armature(*args, **kwargs):
        """
        Armature vector.
        """
    @armature.setter
    def armature(*args, **kwargs):
        ...
    @property
    def children(*args, **kwargs):
        """
        Vector of children index. Chidren of the *i*th joint, denoted *mu(i)* corresponds to the set (i==parents[k] for k in mu(i)).
        """
    @property
    def damping(*args, **kwargs):
        """
        Vector of joint damping parameters.
        """
    @damping.setter
    def damping(*args, **kwargs):
        ...
    @property
    def effortLimit(*args, **kwargs):
        """
        Joint max effort.
        """
    @effortLimit.setter
    def effortLimit(*args, **kwargs):
        ...
    @property
    def frames(*args, **kwargs):
        """
        Vector of frames contained in the model.
        """
    @frames.setter
    def frames(*args, **kwargs):
        ...
    @property
    def friction(*args, **kwargs):
        """
        Vector of joint friction parameters.
        """
    @friction.setter
    def friction(*args, **kwargs):
        ...
    @property
    def gravity(*args, **kwargs):
        """
        Motion vector corresponding to the gravity field expressed in the world Frame.
        """
    @gravity.setter
    def gravity(*args, **kwargs):
        ...
    @property
    def idx_qs(*args, **kwargs):
        """
        Vector of starting index of the *i*th  joint in the configuration space.
        """
    @property
    def idx_vs(*args, **kwargs):
        """
        Starting index of the *i*th joint in the tangent configuration space.
        """
    @property
    def inertias(*args, **kwargs):
        """
        Vector of spatial inertias supported by each joint.
        """
    @property
    def jointPlacements(*args, **kwargs):
        """
        Vector of joint placements: placement of a joint *i* wrt its parent joint frame.
        """
    @jointPlacements.setter
    def jointPlacements(*args, **kwargs):
        ...
    @property
    def joints(*args, **kwargs):
        """
        Vector of joint models.
        """
    @property
    def lowerPositionLimit(*args, **kwargs):
        """
        Limit for joint lower position.
        """
    @lowerPositionLimit.setter
    def lowerPositionLimit(*args, **kwargs):
        ...
    @property
    def name(*args, **kwargs):
        """
        Name of the model.
        """
    @name.setter
    def name(*args, **kwargs):
        ...
    @property
    def names(*args, **kwargs):
        """
        Name of the joints.
        """
    @property
    def nbodies(*args, **kwargs):
        """
        Number of bodies.
        """
    @property
    def nframes(*args, **kwargs):
        """
        Number of frames.
        """
    @property
    def njoints(*args, **kwargs):
        """
        Number of joints.
        """
    @property
    def nq(*args, **kwargs):
        """
        Dimension of the configuration vector representation.
        """
    @property
    def nqs(*args, **kwargs):
        """
        Vector of dimension of the  joint configuration subspace.
        """
    @property
    def nv(*args, **kwargs):
        """
        Dimension of the velocity vector space.
        """
    @property
    def nvs(*args, **kwargs):
        """
        Dimension of the *i*th joint tangent subspace.
        """
    @property
    def parents(*args, **kwargs):
        """
        Vector of parent joint indexes. The parent of joint *i*, denoted *li*, corresponds to li==parents[i].
        """
    @property
    def referenceConfigurations(*args, **kwargs):
        """
        Map of reference configurations, indexed by user given names.
        """
    @referenceConfigurations.setter
    def referenceConfigurations(*args, **kwargs):
        ...
    @property
    def rotorGearRatio(*args, **kwargs):
        """
        Vector of rotor gear ratio parameters.
        """
    @rotorGearRatio.setter
    def rotorGearRatio(*args, **kwargs):
        ...
    @property
    def rotorInertia(*args, **kwargs):
        """
        Vector of rotor inertia parameters.
        """
    @rotorInertia.setter
    def rotorInertia(*args, **kwargs):
        ...
    @property
    def subtrees(*args, **kwargs):
        """
        Vector of subtrees. subtree[j] corresponds to the subtree supported by the joint j.
        """
    @subtrees.setter
    def subtrees(*args, **kwargs):
        ...
    @property
    def supports(*args, **kwargs):
        """
        Vector of supports. supports[j] corresponds to the list of joints on the path between
        the current *j* to the root of the kinematic tree.
        """
    @supports.setter
    def supports(*args, **kwargs):
        ...
    @property
    def upperPositionLimit(*args, **kwargs):
        """
        Limit for joint upper position.
        """
    @upperPositionLimit.setter
    def upperPositionLimit(*args, **kwargs):
        ...
    @property
    def velocityLimit(*args, **kwargs):
        """
        Joint max velocity.
        """
    @velocityLimit.setter
    def velocityLimit(*args, **kwargs):
        ...
class Motion(Boost.Python.instance):
    """
    Motion vectors, in se3 == M^6.
    
    Supported operations ...
    """
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def Random(*args, **kwargs):
        """
        
        Random() -> Motion :
            Returns a random Motion.
        """
    @staticmethod
    def Zero(*args, **kwargs):
        """
        
        Zero() -> Motion :
            Returns a zero Motion.
        """
    @staticmethod
    def __add__(*args, **kwargs):
        """
        
        __add__( (Motion)arg1, (Motion)arg2) -> object
        """
    @staticmethod
    def __array__(*args, **kwargs):
        """
        
        __array__( (Motion)arg1) -> object
        
        __array__( (Motion)self [, (object)dtype=None [, (object)copy=None]]) -> object
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (Motion)self) -> Motion :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (Motion)self, (dict)memo) -> Motion :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Motion)arg1, (Motion)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (Motion)arg1) -> tuple
        """
    @staticmethod
    def __iadd__(*args, **kwargs):
        """
        
        __iadd__( (Motion)arg1, (Motion)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor
        
        __init__( (object)self, (numpy.ndarray)linear, (numpy.ndarray)angular) -> None :
            Initialize from linear and angular components of a Motion vector (don't mix the order).
        
        __init__( (object)self, (numpy.ndarray)array) -> None :
            Init from a vector 6 [linear velocity, angular velocity]
        
        __init__( (object)self, (Motion)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (Motion)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Motion -> pinocchio.pinocchio_pywrap_default.Motion
        
        __init__( (object)arg1, (Motion)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Motion -> pinocchio.pinocchio_pywrap_default.Motion
        """
    @staticmethod
    def __isub__(*args, **kwargs):
        """
        
        __isub__( (Motion)arg1, (Motion)arg2) -> object
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (Motion)arg1, (float)arg2) -> object
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Motion)arg1, (Motion)arg2) -> object
        """
    @staticmethod
    def __neg__(*args, **kwargs):
        """
        
        __neg__( (Motion)arg1) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (Motion)arg1) -> object
        """
    @staticmethod
    def __rmul__(*args, **kwargs):
        """
        
        __rmul__( (Motion)arg1, (float)arg2) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (Motion)arg1) -> object
        """
    @staticmethod
    def __sub__(*args, **kwargs):
        """
        
        __sub__( (Motion)arg1, (Motion)arg2) -> object
        """
    @staticmethod
    def __truediv__(*args, **kwargs):
        """
        
        __truediv__( (Motion)arg1, (float)arg2) -> object
        """
    @staticmethod
    def __xor__(*args, **kwargs):
        """
        
        __xor__( (Motion)arg1, (Motion)arg2) -> object
        
        __xor__( (Motion)arg1, (Force)arg2) -> object
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (Motion)arg1) -> Motion :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (Motion)self) -> Motion :
            Returns a copy of *this.
        """
    @staticmethod
    def cross(*args, **kwargs):
        """
        
        cross( (Motion)self, (Motion)m) -> Motion :
            Action of *this onto another Motion m. Returns ¨*this x m.
        
        cross( (Motion)self, (Force)f) -> Force :
            Dual action of *this onto a Force f. Returns *this x* f.
        """
    @staticmethod
    def dot(*args, **kwargs):
        """
        
        dot( (Motion)self, (object)f) -> float :
            Dot product between *this and a Force f.
        """
    @staticmethod
    def isApprox(*args, **kwargs):
        """
        
        isApprox( (Motion)self, (Motion)other [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to other, within the precision given by prec.
        """
    @staticmethod
    def isZero(*args, **kwargs):
        """
        
        isZero( (Motion)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the zero Motion, within the precision given by prec.
        """
    @staticmethod
    def se3Action(*args, **kwargs):
        """
        
        se3Action( (Motion)self, (SE3)M) -> Motion :
            Returns the result of the action of M on *this.
        """
    @staticmethod
    def se3ActionInverse(*args, **kwargs):
        """
        
        se3ActionInverse( (Motion)self, (SE3)M) -> Motion :
            Returns the result of the action of the inverse of M on *this.
        """
    @staticmethod
    def setRandom(*args, **kwargs):
        """
        
        setRandom( (Motion)self) -> None :
            Set the linear and angular components of *this to random values.
        """
    @staticmethod
    def setZero(*args, **kwargs):
        """
        
        setZero( (Motion)self) -> None :
            Set the linear and angular components of *this to zero.
        """
    @property
    def action(*args, **kwargs):
        """
        Returns the action matrix of *this (acting on Motion).
        """
    @property
    def angular(*args, **kwargs):
        """
        Angular part of a *this, corresponding to the angular velocity in case of a Spatial velocity.
        """
    @angular.setter
    def angular(*args, **kwargs):
        ...
    @property
    def dualAction(*args, **kwargs):
        """
        Returns the dual action matrix of *this (acting on Force).
        """
    @property
    def homogeneous(*args, **kwargs):
        """
        Equivalent homogeneous representation of the Motion vector
        """
    @property
    def linear(*args, **kwargs):
        """
        Linear part of a *this, corresponding to the linear velocity in case of a Spatial velocity.
        """
    @linear.setter
    def linear(*args, **kwargs):
        ...
    @property
    def np(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.Motion)arg1) -> object
        """
    @property
    def vector(*args, **kwargs):
        """
        Returns the components of *this as a 6d vector.
        """
    @vector.setter
    def vector(*args, **kwargs):
        ...
class PGSContactSolver(Boost.Python.instance):
    """
    Projected Gauss Siedel solver for contact dynamics.
    """
    __instance_size__: typing.ClassVar[int] = 152
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (int)problem_dim) -> None :
            Default constructor.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def getAbsoluteConvergenceResidual(*args, **kwargs):
        """
        
        getAbsoluteConvergenceResidual( (PGSContactSolver)self) -> float :
            Returns the value of the absolute residual value corresponding to the contact complementary conditions.
        """
    @staticmethod
    def getAbsolutePrecision(*args, **kwargs):
        """
        
        getAbsolutePrecision( (PGSContactSolver)self) -> float :
            Get the absolute precision requested.
        """
    @staticmethod
    def getCPUTimes(*args, **kwargs):
        """
        
        getCPUTimes( (PGSContactSolver)self) -> coal.coal_pywrap.CPUTimes
        """
    @staticmethod
    def getIterationCount(*args, **kwargs):
        """
        
        getIterationCount( (PGSContactSolver)self) -> int :
            Get the number of iterations achieved by the PGS algorithm.
        """
    @staticmethod
    def getMaxIterations(*args, **kwargs):
        """
        
        getMaxIterations( (PGSContactSolver)self) -> int :
            Get the maximum number of iterations allowed.
        """
    @staticmethod
    def getRelativeConvergenceResidual(*args, **kwargs):
        """
        
        getRelativeConvergenceResidual( (PGSContactSolver)self) -> float :
            Returns the value of the relative residual value corresponding to the difference between two successive iterates (infinity norms).
        """
    @staticmethod
    def getRelativePrecision(*args, **kwargs):
        """
        
        getRelativePrecision( (PGSContactSolver)self) -> float :
            Get the relative precision requested.
        """
    @staticmethod
    def setAbsolutePrecision(*args, **kwargs):
        """
        
        setAbsolutePrecision( (PGSContactSolver)self, (float)absolute_precision) -> None :
            Set the absolute precision for the problem.
        """
    @staticmethod
    def setMaxIterations(*args, **kwargs):
        """
        
        setMaxIterations( (PGSContactSolver)self, (int)max_it) -> None :
            Set the maximum number of iterations.
        """
    @staticmethod
    def setRelativePrecision(*args, **kwargs):
        """
        
        setRelativePrecision( (PGSContactSolver)self, (float)relative_precision) -> None :
            Set the relative precision for the problem.
        """
    @staticmethod
    def solve(*args, **kwargs):
        """
        
        solve( (PGSContactSolver)self, (numpy.ndarray)G, (numpy.ndarray)g, (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)x [, (float)over_relax=1.0]) -> bool :
            Solve the constrained conic problem composed of problem data (G,g,cones) and starting from the initial guess.
        
        solve( (PGSContactSolver)self, (scipy.sparse._csc.csc_matrix)G, (numpy.ndarray)g, (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)x [, (float)over_relax=1.0]) -> bool :
            Solve the constrained conic problem composed of problem data (G,g,cones) and starting from the initial guess.
        """
class PowerIterationAlgo(Boost.Python.instance):
    """
    """
    __instance_size__: typing.ClassVar[int] = 120
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (int)size [, (int)max_it=10 [, (float)rel_tol=1e-08]]) -> None :
            Default constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def lowest(*args, **kwargs):
        """
        
        lowest( (PowerIterationAlgo)arg1, (numpy.ndarray)self [, (bool)compute_largest=True]) -> None
        """
    @staticmethod
    def reset(*args, **kwargs):
        """
        
        reset( (PowerIterationAlgo)self) -> None
        """
    @staticmethod
    def run(*args, **kwargs):
        """
        
        run( (PowerIterationAlgo)arg1, (numpy.ndarray)self) -> None
        """
    @property
    def convergence_criteria(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> float
        """
    @property
    def it(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> int
        """
    @property
    def largest_eigen_value(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> float
        """
    @property
    def lowest_eigen_value(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> float
        """
    @property
    def lowest_eigen_vector(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> object
        """
    @lowest_eigen_vector.setter
    def lowest_eigen_vector(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1, (numpy.ndarray)arg2) -> None
        """
    @property
    def max_it(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> int
        """
    @max_it.setter
    def max_it(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1, (int)arg2) -> None
        """
    @property
    def principal_eigen_vector(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> object
        """
    @principal_eigen_vector.setter
    def principal_eigen_vector(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1, (numpy.ndarray)arg2) -> None
        """
    @property
    def rel_tol(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1) -> float
        """
    @rel_tol.setter
    def rel_tol(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.PowerIterationAlgo)arg1, (float)arg2) -> None
        """
class ProximalSettings(Boost.Python.instance):
    """
    Structure containing all the settings parameters for proximal algorithms.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor.
        
        __init__( (object)self, (float)accuracy, (float)mu, (int)max_iter) -> None :
            Structure containing all the settings parameters for the proximal algorithms.
        
        __init__( (object)self, (float)absolute_accuracy, (float)relative_accuracy, (float)mu, (int)max_iter) -> None :
            Structure containing all the settings parameters for the proximal algorithms.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (ProximalSettings)arg1) -> str
        """
    @property
    def absolute_accuracy(*args, **kwargs):
        """
        Absolute proximal accuracy.
        """
    @property
    def absolute_residual(*args, **kwargs):
        """
        Absolute residual.
        """
    @property
    def iter(*args, **kwargs):
        """
        Final number of iteration of the algorithm when it has converged or reached the maximal number of allowed iterations.
        """
    @property
    def max_iter(*args, **kwargs):
        """
        Maximal number of iterations.
        """
    @property
    def mu(*args, **kwargs):
        """
        Regularization parameter of the Proximal algorithms.
        """
    @property
    def relative_accuracy(*args, **kwargs):
        """
        Relative proximal accuracy between two iterates.
        """
    @property
    def relative_residual(*args, **kwargs):
        """
        Relatice residual between two iterates.
        """
class PseudoInertia(Boost.Python.instance):
    """
    This class represents a pseudo inertia matrix and it is defined by its mass, vector part, and 3x3 matrix part.
    
    Supported operations ...
    """
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def FromDynamicParameters(*args, **kwargs):
        """
        
        FromDynamicParameters( (numpy.ndarray)dynamic_parameters) -> PseudoInertia :
            Builds a pseudo inertia matrix from a vector of dynamic parameters.
            The parameters are given as dynamic_parameters = [m, h_x, h_y, h_z, I_{xx}, I_{xy}, I_{yy}, I_{xz}, I_{yz}, I_{zz}]^T.
        """
    @staticmethod
    def FromInertia(*args, **kwargs):
        """
        
        FromInertia( (Inertia)inertia) -> PseudoInertia :
            Returns the Pseudo Inertia from an Inertia object.
        """
    @staticmethod
    def FromMatrix(*args, **kwargs):
        """
        
        FromMatrix( (numpy.ndarray)pseudo_inertia_matrix) -> PseudoInertia :
            Returns the Pseudo Inertia from a 4x4 matrix.
        """
    @staticmethod
    def __array__(*args, **kwargs):
        """
        
        __array__( (PseudoInertia)arg1) -> numpy.ndarray
        
        __array__( (PseudoInertia)self [, (object)dtype=None [, (object)copy=None]]) -> numpy.ndarray
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (PseudoInertia)self) -> PseudoInertia :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (PseudoInertia)self, (dict)memo) -> PseudoInertia :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (PseudoInertia)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (float)mass, (numpy.ndarray)h, (numpy.ndarray)sigma) -> None :
            Initialize from mass, vector part of the pseudo inertia and matrix part of the pseudo inertia.
        
        __init__( (object)self, (PseudoInertia)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (PseudoInertia)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.PseudoInertia -> pinocchio.pinocchio_pywrap_default.PseudoInertia
        
        __init__( (object)arg1, (PseudoInertia)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.PseudoInertia -> pinocchio.pinocchio_pywrap_default.PseudoInertia
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (PseudoInertia)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (PseudoInertia)arg1) -> object
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (PseudoInertia)arg1) -> PseudoInertia :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (PseudoInertia)self) -> PseudoInertia :
            Returns a copy of *this.
        """
    @staticmethod
    def toDynamicParameters(*args, **kwargs):
        """
        
        toDynamicParameters( (PseudoInertia)self) -> numpy.ndarray :
            Returns the dynamic parameters representation.
        """
    @staticmethod
    def toInertia(*args, **kwargs):
        """
        
        toInertia( (PseudoInertia)self) -> Inertia :
            Returns the inertia representation.
        """
    @staticmethod
    def toMatrix(*args, **kwargs):
        """
        
        toMatrix( (PseudoInertia)self) -> numpy.ndarray :
            Returns the pseudo inertia as a 4x4 matrix.
        """
    @property
    def h(*args, **kwargs):
        """
        Vector part of the Pseudo Inertia.
        """
    @h.setter
    def h(*args, **kwargs):
        ...
    @property
    def mass(*args, **kwargs):
        """
        Mass of the Pseudo Inertia.
        """
    @mass.setter
    def mass(*args, **kwargs):
        ...
    @property
    def sigma(*args, **kwargs):
        """
        Matrix part of the Pseudo Inertia.
        """
    @sigma.setter
    def sigma(*args, **kwargs):
        ...
class ReferenceFrame(Boost.Python.enum):
    LOCAL: typing.ClassVar[ReferenceFrame]  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL
    LOCAL_WORLD_ALIGNED: typing.ClassVar[ReferenceFrame]  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL_WORLD_ALIGNED
    WORLD: typing.ClassVar[ReferenceFrame]  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.WORLD
    __slots__: typing.ClassVar[tuple] = tuple()
    names: typing.ClassVar[dict]  # value = {'WORLD': pinocchio.pinocchio_pywrap_default.ReferenceFrame.WORLD, 'LOCAL': pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL, 'LOCAL_WORLD_ALIGNED': pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL_WORLD_ALIGNED}
    values: typing.ClassVar[dict]  # value = {0: pinocchio.pinocchio_pywrap_default.ReferenceFrame.WORLD, 1: pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL, 2: pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL_WORLD_ALIGNED}
class RigidConstraintData(Boost.Python.instance):
    """
    Rigid constraint data associated to a RigidConstraintModel for contact dynamic algorithms.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (RigidConstraintData)arg1, (RigidConstraintData)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (RigidConstraintModel)contact_model) -> None :
            Default constructor.
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (RigidConstraintData)arg1, (RigidConstraintData)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @property
    def c1Mc2(*args, **kwargs):
        """
        Relative displacement between the two frames.
        """
    @c1Mc2.setter
    def c1Mc2(*args, **kwargs):
        ...
    @property
    def contact1_acceleration_drift(*args, **kwargs):
        """
        Current contact drift acceleration (acceleration only due to the Coriolis and centrifugal effects) of the contact frame 1.
        """
    @contact1_acceleration_drift.setter
    def contact1_acceleration_drift(*args, **kwargs):
        ...
    @property
    def contact1_velocity(*args, **kwargs):
        """
        Current contact Spatial velocity of the constraint 1.
        """
    @contact1_velocity.setter
    def contact1_velocity(*args, **kwargs):
        ...
    @property
    def contact2_acceleration_drift(*args, **kwargs):
        """
        Current contact drift acceleration (acceleration only due to the Coriolis and centrifugal effects) of the contact frame 2.
        """
    @contact2_acceleration_drift.setter
    def contact2_acceleration_drift(*args, **kwargs):
        ...
    @property
    def contact2_velocity(*args, **kwargs):
        """
        Current contact Spatial velocity of the constraint 2.
        """
    @contact2_velocity.setter
    def contact2_velocity(*args, **kwargs):
        ...
    @property
    def contact_acceleration(*args, **kwargs):
        """
        Current contact Spatial acceleration.
        """
    @contact_acceleration.setter
    def contact_acceleration(*args, **kwargs):
        ...
    @property
    def contact_acceleration_desired(*args, **kwargs):
        """
        Desired contact acceleration.
        """
    @contact_acceleration_desired.setter
    def contact_acceleration_desired(*args, **kwargs):
        ...
    @property
    def contact_acceleration_deviation(*args, **kwargs):
        """
        Contact deviation from the reference acceleration (a.k.a the error).
        """
    @contact_acceleration_deviation.setter
    def contact_acceleration_deviation(*args, **kwargs):
        ...
    @property
    def contact_acceleration_error(*args, **kwargs):
        """
        Current contact spatial error (due to the integration step).
        """
    @contact_acceleration_error.setter
    def contact_acceleration_error(*args, **kwargs):
        ...
    @property
    def contact_force(*args, **kwargs):
        """
        Constraint force.
        """
    @contact_force.setter
    def contact_force(*args, **kwargs):
        ...
    @property
    def contact_placement_error(*args, **kwargs):
        """
        Current contact placement error between the two contact Frames.
        This corresponds to the relative placement between the two contact Frames seen as a Motion error.
        """
    @contact_placement_error.setter
    def contact_placement_error(*args, **kwargs):
        ...
    @property
    def contact_velocity_error(*args, **kwargs):
        """
        Current contact Spatial velocity error between the two contact Frames.
        This corresponds to the relative velocity between the two contact Frames.
        """
    @contact_velocity_error.setter
    def contact_velocity_error(*args, **kwargs):
        ...
    @property
    def extended_motion_propagators_joint1(*args, **kwargs):
        """
        Extended force/motion propagators for joint 1.
        """
    @extended_motion_propagators_joint1.setter
    def extended_motion_propagators_joint1(*args, **kwargs):
        ...
    @property
    def extended_motion_propagators_joint2(*args, **kwargs):
        """
        Extended force/motion propagators for joint 2.
        """
    @extended_motion_propagators_joint2.setter
    def extended_motion_propagators_joint2(*args, **kwargs):
        ...
    @property
    def lambdas_joint1(*args, **kwargs):
        """
        Extended force/motion propagators for joint 1.
        """
    @lambdas_joint1.setter
    def lambdas_joint1(*args, **kwargs):
        ...
    @property
    def oMc1(*args, **kwargs):
        """
        Placement of the constraint frame 1 with respect to the WORLD frame.
        """
    @oMc1.setter
    def oMc1(*args, **kwargs):
        ...
    @property
    def oMc2(*args, **kwargs):
        """
        Placement of the constraint frame 2 with respect to the WORLD frame.
        """
    @oMc2.setter
    def oMc2(*args, **kwargs):
        ...
class RigidConstraintModel(Boost.Python.instance):
    """
    Rigid contact model for contact dynamic algorithms.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (RigidConstraintModel)arg1, (RigidConstraintModel)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (ContactType)contact_type, (Model)model, (int)joint1_id, (SE3)joint1_placement, (int)joint2_id, (SE3)joint2_placement [, (ReferenceFrame)reference_frame]) -> None :
            Contructor from a given ContactType, joint index and placement for the two joints implied in the constraint.
        
        __init__( (object)self, (ContactType)contact_type, (Model)model, (int)joint1_id, (SE3)joint1_placement [, (ReferenceFrame)reference_frame]) -> None :
            Contructor from a given ContactType, joint index and placement only for the first joint implied in the constraint.
        
        __init__( (object)self, (ContactType)contact_type, (Model)model, (int)joint1_id [, (ReferenceFrame)reference_frame]) -> None :
            Contructor from a given ContactType and joint index. The base joint is taken as 0 in the constraint.
        
        __init__( (object)arg1, (RigidConstraintModel)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.RigidConstraintModel -> pinocchio.pinocchio_pywrap_default.RigidConstraintModel
        
        __init__( (object)arg1, (RigidConstraintModel)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.RigidConstraintModel -> pinocchio.pinocchio_pywrap_default.RigidConstraintModel
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (RigidConstraintModel)arg1, (RigidConstraintModel)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (RigidConstraintModel)arg1) -> RigidConstraintModel :
            Returns a cast of *this.
        """
    @staticmethod
    def createData(*args, **kwargs):
        """
        
        createData( (RigidConstraintModel)arg1) -> RigidConstraintData :
            Create a Data object for the given model.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (RigidConstraintModel)arg1) -> int :
            Size of the constraint
        """
    @property
    def colwise_joint1_sparsity(*args, **kwargs):
        """
        Sparsity pattern associated to joint 1.
        """
    @colwise_joint1_sparsity.setter
    def colwise_joint1_sparsity(*args, **kwargs):
        ...
    @property
    def colwise_joint2_sparsity(*args, **kwargs):
        """
        Sparsity pattern associated to joint 2.
        """
    @colwise_joint2_sparsity.setter
    def colwise_joint2_sparsity(*args, **kwargs):
        ...
    @property
    def colwise_span_indexes(*args, **kwargs):
        """
        Indexes of the columns spanned by the constraints.
        """
    @colwise_span_indexes.setter
    def colwise_span_indexes(*args, **kwargs):
        ...
    @property
    def corrector(*args, **kwargs):
        """
        Corrector parameters.
        """
    @corrector.setter
    def corrector(*args, **kwargs):
        ...
    @property
    def desired_contact_acceleration(*args, **kwargs):
        """
        Desired contact spatial acceleration.
        """
    @desired_contact_acceleration.setter
    def desired_contact_acceleration(*args, **kwargs):
        ...
    @property
    def desired_contact_placement(*args, **kwargs):
        """
        Desired contact placement.
        """
    @desired_contact_placement.setter
    def desired_contact_placement(*args, **kwargs):
        ...
    @property
    def desired_contact_velocity(*args, **kwargs):
        """
        Desired contact spatial velocity.
        """
    @desired_contact_velocity.setter
    def desired_contact_velocity(*args, **kwargs):
        ...
    @property
    def joint1_id(*args, **kwargs):
        """
        Index of first parent joint in the model tree.
        """
    @joint1_id.setter
    def joint1_id(*args, **kwargs):
        ...
    @property
    def joint1_placement(*args, **kwargs):
        """
        Relative placement with respect to the frame of joint1.
        """
    @joint1_placement.setter
    def joint1_placement(*args, **kwargs):
        ...
    @property
    def joint2_id(*args, **kwargs):
        """
        Index of second parent joint in the model tree.
        """
    @joint2_id.setter
    def joint2_id(*args, **kwargs):
        ...
    @property
    def joint2_placement(*args, **kwargs):
        """
        Relative placement with respect to the frame of joint2.
        """
    @joint2_placement.setter
    def joint2_placement(*args, **kwargs):
        ...
    @property
    def name(*args, **kwargs):
        """
        Name of the contact.
        """
    @name.setter
    def name(*args, **kwargs):
        ...
    @property
    def reference_frame(*args, **kwargs):
        """
        Reference frame where the constraint is expressed (WORLD, LOCAL_WORLD_ALIGNED or LOCAL).
        """
    @reference_frame.setter
    def reference_frame(*args, **kwargs):
        ...
    @property
    def type(*args, **kwargs):
        """
        Type of the contact.
        """
    @type.setter
    def type(*args, **kwargs):
        ...
class SE3(Boost.Python.instance):
    """
    SE3 transformation defined by a 3d vector and a rotation matrix.
    """
    __instance_size__: typing.ClassVar[int] = 120
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def Identity(*args, **kwargs):
        """
        
        Identity() -> SE3 :
            Returns the identity transformation.
        """
    @staticmethod
    def Interpolate(*args, **kwargs):
        """
        
        Interpolate( (SE3)A, (SE3)B, (float)alpha) -> SE3 :
            Linear interpolation on the SE3 manifold.
            
            This method computes the linear interpolation between A and B, such that the result C = A + (B-A)*t if it would be applied on classic Euclidian space.
            This operation is very similar to the SLERP operation on Rotations.
            Parameters:
            	A: Initial transformation
            	B: Target transformation
            	alpha: Interpolation factor
        """
    @staticmethod
    def Random(*args, **kwargs):
        """
        
        Random() -> SE3 :
            Returns a random transformation.
        """
    @staticmethod
    def __array__(*args, **kwargs):
        """
        
        __array__( (SE3)arg1) -> numpy.ndarray
        
        __array__( (SE3)self [, (object)dtype=None [, (object)copy=None]]) -> numpy.ndarray
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (SE3)self) -> SE3 :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (SE3)self, (dict)memo) -> SE3 :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (SE3)arg1, (SE3)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (SE3)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor.
        
        __init__( (object)self, (numpy.ndarray)rotation, (numpy.ndarray)translation) -> None :
            Initialize from a rotation matrix and a translation vector.
        
        __init__( (object)self, (coal.coal_pywrap.Quaternion)quat, (numpy.ndarray)translation) -> None :
            Initialize from a quaternion and a translation vector.
        
        __init__( (object)self, (int)int) -> None :
            Init to identity.
        
        __init__( (object)self, (numpy.ndarray)array) -> None :
            Initialize from an homogeneous matrix.
        
        __init__( (object)self, (SE3)clone) -> None :
            Copy constructor
        
        __init__( (object)arg1, (SE3)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.SE3 -> pinocchio.pinocchio_pywrap_default.SE3
        
        __init__( (object)arg1, (SE3)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.SE3 -> pinocchio.pinocchio_pywrap_default.SE3
        """
    @staticmethod
    def __invert__(*args, **kwargs):
        """
        
        __invert__( (SE3)arg1) -> SE3 :
            Returns the inverse of *this.
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (SE3)arg1, (SE3)arg2) -> object
        
        __mul__( (SE3)arg1, (Motion)arg2) -> Motion
        
        __mul__( (SE3)arg1, (Force)arg2) -> Force
        
        __mul__( (SE3)arg1, (Inertia)arg2) -> Inertia
        
        __mul__( (SE3)arg1, (numpy.ndarray)arg2) -> numpy.ndarray
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (SE3)arg1, (SE3)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (SE3)arg1) -> str
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (SE3)arg1) -> object
        """
    @staticmethod
    def act(*args, **kwargs):
        """
        
        act( (SE3)self, (numpy.ndarray)point) -> numpy.ndarray :
            Returns a point which is the result of the entry point transforms by *this.
        
        act( (SE3)self, (SE3)M) -> SE3 :
            Returns the result of *this * M.
        
        act( (SE3)self, (Motion)motion) -> Motion :
            Returns the result action of *this onto a Motion.
        
        act( (SE3)self, (Force)force) -> Force :
            Returns the result of *this onto a Force.
        
        act( (SE3)self, (Inertia)inertia) -> Inertia :
            Returns the result of *this onto a Force.
        """
    @staticmethod
    def actInv(*args, **kwargs):
        """
        
        actInv( (SE3)self, (numpy.ndarray)point) -> numpy.ndarray :
            Returns a point which is the result of the entry point by the inverse of *this.
        
        actInv( (SE3)self, (SE3)M) -> SE3 :
            Returns the result of the inverse of *this times M.
        
        actInv( (SE3)self, (Motion)motion) -> Motion :
            Returns the result of the inverse of *this onto a Motion.
        
        actInv( (SE3)self, (Force)force) -> Force :
            Returns the result of the inverse of *this onto an Inertia.
        
        actInv( (SE3)self, (Inertia)inertia) -> Inertia :
            Returns the result of the inverse of *this onto an Inertia.
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (SE3)arg1) -> SE3 :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (SE3)self) -> SE3 :
            Returns a copy of *this.
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (SE3)self) -> SE3 :
            Returns the inverse transform
        """
    @staticmethod
    def isApprox(*args, **kwargs):
        """
        
        isApprox( (SE3)self, (SE3)other [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to other, within the precision given by prec.
        """
    @staticmethod
    def isIdentity(*args, **kwargs):
        """
        
        isIdentity( (SE3)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the identity placement, within the precision given by prec.
        """
    @staticmethod
    def setIdentity(*args, **kwargs):
        """
        
        setIdentity( (SE3)self) -> None :
            Set *this to the identity placement.
        """
    @staticmethod
    def setRandom(*args, **kwargs):
        """
        
        setRandom( (SE3)self) -> None :
            Set *this to a random placement.
        """
    @staticmethod
    def toActionMatrix(*args, **kwargs):
        """
        
        toActionMatrix( (SE3)self) -> numpy.ndarray :
            Returns the related action matrix (acting on Motion).
        """
    @staticmethod
    def toActionMatrixInverse(*args, **kwargs):
        """
        
        toActionMatrixInverse( (SE3)self) -> numpy.ndarray :
            Returns the inverse of the action matrix (acting on Motion).
            This is equivalent to do m.inverse().toActionMatrix()
        """
    @staticmethod
    def toDualActionMatrix(*args, **kwargs):
        """
        
        toDualActionMatrix( (SE3)self) -> numpy.ndarray :
            Returns the related dual action matrix (acting on Force).
        """
    @property
    def action(*args, **kwargs):
        """
        Returns the related action matrix (acting on Motion).
        """
    @property
    def actionInverse(*args, **kwargs):
        """
        Returns the inverse of the action matrix (acting on Motion).
        This is equivalent to do m.inverse().action
        """
    @property
    def dualAction(*args, **kwargs):
        """
        Returns the related dual action matrix (acting on Force).
        """
    @property
    def homogeneous(*args, **kwargs):
        """
        Returns the equivalent homegeneous matrix (acting on SE3).
        """
    @property
    def np(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.SE3)arg1) -> numpy.ndarray
        """
    @property
    def rotation(*args, **kwargs):
        """
        The rotation part of the transformation.
        """
    @rotation.setter
    def rotation(*args, **kwargs):
        ...
    @property
    def translation(*args, **kwargs):
        """
        The translation part of the transformation.
        """
    @translation.setter
    def translation(*args, **kwargs):
        ...
class SolverStats(Boost.Python.instance):
    """
    """
    __instance_size__: typing.ClassVar[int] = 152
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (int)max_it) -> None :
            Default constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def reset(*args, **kwargs):
        """
        
        reset( (SolverStats)self) -> None :
            Reset the stasts.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (SolverStats)self) -> int :
            Size of the vectors stored in the structure.
        """
    @property
    def cholesky_update_count(*args, **kwargs):
        """
        Number of Cholesky updates performed by the algorithm.
        """
    @property
    def complementarity(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.SolverStats)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_Scalar
        """
    @property
    def dual_feasibility(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.SolverStats)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_Scalar
        """
    @property
    def dual_feasibility_ncp(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.SolverStats)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_Scalar
        """
    @property
    def it(*args, **kwargs):
        """
        Number of iterations performed by the algorithm.
        """
    @property
    def primal_feasibility(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.SolverStats)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_Scalar
        """
    @property
    def rho(*args, **kwargs):
        """
        
        None( (pinocchio.pinocchio_pywrap_default.SolverStats)arg1) -> pinocchio.pinocchio_pywrap_default.StdVec_Scalar
        """
class StdMap_String_VectorXd(Boost.Python.instance):
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 72
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdMap_String_VectorXd)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdMap_String_VectorXd)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdMap_String_VectorXd)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdMap_String_VectorXd)arg1, (object)arg2) -> object
        
        __getitem__( (StdMap_String_VectorXd)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdMap_String_VectorXd)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdMap_String_VectorXd)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdMap_String_VectorXd)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
class StdVec_Bool(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 72
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Bool)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Bool)self) -> StdVec_Bool :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Bool)self, (dict)memo) -> StdVec_Bool :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Bool)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Bool)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Bool)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (bool)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Bool)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Bool)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Bool)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Bool)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Bool)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Bool)self) -> StdVec_Bool :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Bool)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_Bool)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_Bool)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Bool)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_CollisionObject(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_CollisionObject)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_CollisionObject)self) -> StdVec_CollisionObject :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_CollisionObject)self, (dict)memo) -> StdVec_CollisionObject :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_CollisionObject)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_CollisionObject)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_CollisionObject)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (CollisionObject)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_CollisionObject)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_CollisionObject)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_CollisionObject)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_CollisionObject)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_CollisionObject)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_CollisionObject)self) -> StdVec_CollisionObject :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_CollisionObject)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_CollisionObject)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_CollisionObject)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_CollisionObject)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_CollisionPair(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_CollisionPair)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_CollisionPair)self) -> StdVec_CollisionPair :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_CollisionPair)self, (dict)memo) -> StdVec_CollisionPair :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_CollisionPair)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_CollisionPair)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_CollisionPair)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (CollisionPair)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_CollisionPair)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_CollisionPair)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_CollisionPair)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_CollisionPair)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_CollisionPair)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_CollisionPair)self) -> StdVec_CollisionPair :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_CollisionPair)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_CollisionPair)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_CollisionPair)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_CollisionPair)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_ComputeCollision(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_ComputeCollision)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_ComputeCollision)self) -> StdVec_ComputeCollision :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_ComputeCollision)self, (dict)memo) -> StdVec_ComputeCollision :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_ComputeCollision)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_ComputeCollision)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_ComputeCollision)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (ComputeCollision)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_ComputeCollision)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_ComputeCollision)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_ComputeCollision)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_ComputeCollision)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_ComputeCollision)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_ComputeCollision)self) -> StdVec_ComputeCollision :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_ComputeCollision)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_ComputeCollision)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_ComputeDistance(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_ComputeDistance)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_ComputeDistance)self) -> StdVec_ComputeDistance :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_ComputeDistance)self, (dict)memo) -> StdVec_ComputeDistance :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_ComputeDistance)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_ComputeDistance)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_ComputeDistance)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (ComputeDistance)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_ComputeDistance)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_ComputeDistance)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_ComputeDistance)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_ComputeDistance)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_ComputeDistance)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_ComputeDistance)self) -> StdVec_ComputeDistance :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_ComputeDistance)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_ComputeDistance)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_CoulombFrictionCone(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_CoulombFrictionCone)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_CoulombFrictionCone)self) -> StdVec_CoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_CoulombFrictionCone)self, (dict)memo) -> StdVec_CoulombFrictionCone :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_CoulombFrictionCone)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_CoulombFrictionCone)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_CoulombFrictionCone)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (CoulombFrictionCone)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_CoulombFrictionCone)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_CoulombFrictionCone)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_CoulombFrictionCone)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_CoulombFrictionCone)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_CoulombFrictionCone)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_CoulombFrictionCone)self) -> StdVec_CoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_CoulombFrictionCone)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_CoulombFrictionCone)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_CoulombFrictionCone)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_CoulombFrictionCone)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_DualCoulombFrictionCone(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_DualCoulombFrictionCone)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_DualCoulombFrictionCone)self) -> StdVec_DualCoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_DualCoulombFrictionCone)self, (dict)memo) -> StdVec_DualCoulombFrictionCone :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_DualCoulombFrictionCone)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_DualCoulombFrictionCone)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_DualCoulombFrictionCone)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (DualCoulombFrictionCone)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_DualCoulombFrictionCone)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_DualCoulombFrictionCone)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_DualCoulombFrictionCone)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_DualCoulombFrictionCone)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_DualCoulombFrictionCone)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_DualCoulombFrictionCone)self) -> StdVec_DualCoulombFrictionCone :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_DualCoulombFrictionCone)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_DualCoulombFrictionCone)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_DualCoulombFrictionCone)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_DualCoulombFrictionCone)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_FCL_CollisionObjectPointer(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_FCL_CollisionObjectPointer)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_FCL_CollisionObjectPointer)self) -> StdVec_FCL_CollisionObjectPointer :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_FCL_CollisionObjectPointer)self, (dict)memo) -> StdVec_FCL_CollisionObjectPointer :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_FCL_CollisionObjectPointer)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_FCL_CollisionObjectPointer)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_FCL_CollisionObjectPointer)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (coal.coal_pywrap.CollisionObject)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_FCL_CollisionObjectPointer)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_FCL_CollisionObjectPointer)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_FCL_CollisionObjectPointer)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_FCL_CollisionObjectPointer)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_FCL_CollisionObjectPointer)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_FCL_CollisionObjectPointer)self) -> StdVec_FCL_CollisionObjectPointer :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_FCL_CollisionObjectPointer)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_FCL_CollisionObjectPointer)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_FCL_CollisionObjectPointer)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_FCL_CollisionObjectPointer)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_Force(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Force)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Force)self) -> StdVec_Force :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Force)self, (dict)memo) -> StdVec_Force :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Force)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Force)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Force)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (Force)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Force)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Force)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Force)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Force)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Force)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Force)self) -> StdVec_Force :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Force)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Force)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_Frame(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Frame)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Frame)self) -> StdVec_Frame :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Frame)self, (dict)memo) -> StdVec_Frame :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Frame)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Frame)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Frame)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (Frame)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Frame)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Frame)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Frame)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Frame)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Frame)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Frame)self) -> StdVec_Frame :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Frame)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Frame)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_GeometryModel(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_GeometryModel)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_GeometryModel)self) -> StdVec_GeometryModel :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_GeometryModel)self, (dict)memo) -> StdVec_GeometryModel :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_GeometryModel)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_GeometryModel)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_GeometryModel)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (GeometryModel)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_GeometryModel)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_GeometryModel)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_GeometryModel)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_GeometryModel)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_GeometryModel)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_GeometryModel)self) -> StdVec_GeometryModel :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_GeometryModel)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_GeometryModel)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_GeometryModel)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_GeometryModel)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_GeometryObject(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_GeometryObject)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_GeometryObject)self) -> StdVec_GeometryObject :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_GeometryObject)self, (dict)memo) -> StdVec_GeometryObject :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_GeometryObject)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_GeometryObject)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_GeometryObject)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (GeometryObject)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_GeometryObject)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_GeometryObject)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_GeometryObject)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_GeometryObject)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_GeometryObject)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_GeometryObject)self) -> StdVec_GeometryObject :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_GeometryObject)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_GeometryObject)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_Index(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Index)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Index)self) -> StdVec_Index :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Index)self, (dict)memo) -> StdVec_Index :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Index)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Index)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Index)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (int)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Index)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Index)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Index)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Index)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Index)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Index)self) -> StdVec_Index :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Index)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_Index)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_Index)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Index)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_IndexVector(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_IndexVector)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_IndexVector)self) -> StdVec_IndexVector :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_IndexVector)self, (dict)memo) -> StdVec_IndexVector :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_IndexVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_IndexVector)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_IndexVector)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (StdVec_Index)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_IndexVector)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_IndexVector)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_IndexVector)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_IndexVector)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_IndexVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_IndexVector)self) -> StdVec_IndexVector :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_IndexVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_IndexVector)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_IndexVector)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_IndexVector)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_Inertia(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Inertia)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Inertia)self) -> StdVec_Inertia :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Inertia)self, (dict)memo) -> StdVec_Inertia :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Inertia)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Inertia)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Inertia)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (Inertia)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Inertia)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Inertia)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Inertia)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Inertia)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Inertia)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Inertia)self) -> StdVec_Inertia :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Inertia)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Inertia)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_JointDataVector(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_JointDataVector)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_JointDataVector)self) -> StdVec_JointDataVector :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_JointDataVector)self, (dict)memo) -> StdVec_JointDataVector :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_JointDataVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_JointDataVector)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_JointDataVector)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (JointData)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_JointDataVector)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_JointDataVector)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_JointDataVector)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_JointDataVector)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_JointDataVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_JointDataVector)self) -> StdVec_JointDataVector :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_JointDataVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_JointDataVector)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_JointModelVector(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_JointModelVector)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_JointModelVector)self) -> StdVec_JointModelVector :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_JointModelVector)self, (dict)memo) -> StdVec_JointModelVector :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_JointModelVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_JointModelVector)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_JointModelVector)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (JointModel)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_JointModelVector)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_JointModelVector)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_JointModelVector)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_JointModelVector)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_JointModelVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_JointModelVector)self) -> StdVec_JointModelVector :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_JointModelVector)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_JointModelVector)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_Matrix6(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Matrix6)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Matrix6)self) -> StdVec_Matrix6 :
            Returns a copy of *this.
        
        __copy__( (StdVec_Matrix6)self) -> StdVec_Matrix6 :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Matrix6)self, (dict)memo) -> StdVec_Matrix6 :
            Returns a deep copy of *this.
        
        __deepcopy__( (StdVec_Matrix6)self, (dict)memo) -> StdVec_Matrix6 :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Matrix6)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Matrix6)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Matrix6)arg1, (object)arg2) -> object
        
        __getitem__( (StdVec_Matrix6)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (numpy.ndarray)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Matrix6)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Matrix6)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Matrix6)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Matrix6)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Matrix6)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Matrix6)self) -> StdVec_Matrix6 :
            Returns a copy of *this.
        
        copy( (StdVec_Matrix6)self) -> StdVec_Matrix6 :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Matrix6)arg1, (object)arg2) -> None
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_Matrix6)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Matrix6)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        
        tolist( (StdVec_Matrix6)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_Matrix6x(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Matrix6x)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Matrix6x)self) -> StdVec_Matrix6x :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Matrix6x)self, (dict)memo) -> StdVec_Matrix6x :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Matrix6x)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Matrix6x)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Matrix6x)arg1, (object)arg2) -> object
        
        __getitem__( (StdVec_Matrix6x)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (numpy.ndarray)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Matrix6x)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Matrix6x)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Matrix6x)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Matrix6x)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Matrix6x)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Matrix6x)self) -> StdVec_Matrix6x :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Matrix6x)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Matrix6x)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_MatrixXs(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_MatrixXs)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_MatrixXs)self) -> StdVec_MatrixXs :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_MatrixXs)self, (dict)memo) -> StdVec_MatrixXs :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_MatrixXs)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_MatrixXs)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_MatrixXs)arg1, (object)arg2) -> object
        
        __getitem__( (StdVec_MatrixXs)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (numpy.ndarray)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_MatrixXs)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_MatrixXs)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_MatrixXs)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_MatrixXs)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_MatrixXs)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_MatrixXs)self) -> StdVec_MatrixXs :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_MatrixXs)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_MatrixXs)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_MatrixXs)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_MatrixXs)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_Motion(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Motion)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Motion)self) -> StdVec_Motion :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Motion)self, (dict)memo) -> StdVec_Motion :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Motion)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Motion)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Motion)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (Motion)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Motion)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Motion)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Motion)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Motion)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Motion)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Motion)self) -> StdVec_Motion :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Motion)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Motion)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_RigidConstraintData(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_RigidConstraintData)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_RigidConstraintData)self) -> StdVec_RigidConstraintData :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_RigidConstraintData)self, (dict)memo) -> StdVec_RigidConstraintData :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_RigidConstraintData)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_RigidConstraintData)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_RigidConstraintData)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (RigidConstraintData)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_RigidConstraintData)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_RigidConstraintData)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_RigidConstraintData)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_RigidConstraintData)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_RigidConstraintData)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_RigidConstraintData)self) -> StdVec_RigidConstraintData :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_RigidConstraintData)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_RigidConstraintData)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_RigidConstraintData)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_RigidConstraintData)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_RigidConstraintModel(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_RigidConstraintModel)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_RigidConstraintModel)self) -> StdVec_RigidConstraintModel :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_RigidConstraintModel)self, (dict)memo) -> StdVec_RigidConstraintModel :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_RigidConstraintModel)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_RigidConstraintModel)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_RigidConstraintModel)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (RigidConstraintModel)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_RigidConstraintModel)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_RigidConstraintModel)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_RigidConstraintModel)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_RigidConstraintModel)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_RigidConstraintModel)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_RigidConstraintModel)self) -> StdVec_RigidConstraintModel :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_RigidConstraintModel)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_RigidConstraintModel)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_RigidConstraintModel)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_RigidConstraintModel)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_SE3(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_SE3)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_SE3)self) -> StdVec_SE3 :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_SE3)self, (dict)memo) -> StdVec_SE3 :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_SE3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_SE3)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_SE3)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (SE3)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_SE3)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_SE3)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_SE3)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_SE3)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_SE3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_SE3)self) -> StdVec_SE3 :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_SE3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_SE3)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_Scalar(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Scalar)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Scalar)self) -> StdVec_Scalar :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Scalar)self, (dict)memo) -> StdVec_Scalar :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Scalar)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Scalar)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Scalar)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (float)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Scalar)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Scalar)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Scalar)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Scalar)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Scalar)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Scalar)self) -> StdVec_Scalar :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Scalar)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_Scalar)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_Scalar)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Scalar)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_StdString(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_StdString)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_StdString)self) -> StdVec_StdString :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_StdString)self, (dict)memo) -> StdVec_StdString :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_StdString)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_StdString)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_StdString)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (str)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_StdString)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_StdString)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_StdString)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_StdString)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_StdString)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_StdString)self) -> StdVec_StdString :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_StdString)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_StdString)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_StdString)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_StdString)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_Symmetric3(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Symmetric3)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Symmetric3)self) -> StdVec_Symmetric3 :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Symmetric3)self, (dict)memo) -> StdVec_Symmetric3 :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Symmetric3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Symmetric3)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Symmetric3)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (Symmetric3)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Symmetric3)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Symmetric3)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Symmetric3)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Symmetric3)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Symmetric3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Symmetric3)self) -> StdVec_Symmetric3 :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Symmetric3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Symmetric3)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_Vector3(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_Vector3)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_Vector3)self) -> StdVec_Vector3 :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_Vector3)self, (dict)memo) -> StdVec_Vector3 :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_Vector3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_Vector3)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_Vector3)arg1, (object)arg2) -> object
        
        __getitem__( (StdVec_Vector3)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (numpy.ndarray)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_Vector3)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_Vector3)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_Vector3)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_Vector3)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_Vector3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_Vector3)self) -> StdVec_Vector3 :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_Vector3)arg1, (object)arg2) -> None
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_Vector3)self [, (bool)deep_copy=False]) -> list :
            Returns the aligned_vector as a Python list.
        """
class StdVec_VectorXb(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_VectorXb)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_VectorXb)self) -> StdVec_VectorXb :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_VectorXb)self, (dict)memo) -> StdVec_VectorXb :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_VectorXb)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_VectorXb)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_VectorXb)arg1, (object)arg2) -> object
        
        __getitem__( (StdVec_VectorXb)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (numpy.ndarray)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_VectorXb)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_VectorXb)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_VectorXb)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_VectorXb)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_VectorXb)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_VectorXb)self) -> StdVec_VectorXb :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_VectorXb)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_VectorXb)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_VectorXb)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_VectorXb)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class StdVec_int(Boost.Python.instance):
    """
    """
    __getstate_manages_dict__: typing.ClassVar[bool] = True
    __instance_size__: typing.ClassVar[int] = 56
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def __contains__(*args, **kwargs):
        """
        
        __contains__( (StdVec_int)arg1, (object)arg2) -> bool
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (StdVec_int)self) -> StdVec_int :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (StdVec_int)self, (dict)memo) -> StdVec_int :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __delitem__(*args, **kwargs):
        """
        
        __delitem__( (StdVec_int)arg1, (object)arg2) -> None
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (StdVec_int)arg1) -> tuple
        """
    @staticmethod
    def __getitem__(*args, **kwargs):
        """
        
        __getitem__( (StdVec_int)arg1, (object)arg2) -> object
        """
    @staticmethod
    def __getstate__(*args, **kwargs):
        """
        
        __getstate__( (object)arg1) -> tuple
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        
        __init__( (object)self, (int)size, (int)value) -> None :
            Constructor from a given size and a given value.
        
        __init__( (object)self, (StdVec_int)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __iter__(*args, **kwargs):
        """
        
        __iter__( (StdVec_int)arg1) -> object
        """
    @staticmethod
    def __len__(*args, **kwargs):
        """
        
        __len__( (StdVec_int)arg1) -> int
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __setitem__(*args, **kwargs):
        """
        
        __setitem__( (StdVec_int)arg1, (object)arg2, (object)arg3) -> None
        """
    @staticmethod
    def __setstate__(*args, **kwargs):
        """
        
        __setstate__( (object)arg1, (tuple)arg2) -> None
        """
    @staticmethod
    def append(*args, **kwargs):
        """
        
        append( (StdVec_int)arg1, (object)arg2) -> None
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (StdVec_int)self) -> StdVec_int :
            Returns a copy of *this.
        """
    @staticmethod
    def extend(*args, **kwargs):
        """
        
        extend( (StdVec_int)arg1, (object)arg2) -> None
        """
    @staticmethod
    def id(*args, **kwargs):
        """
        
        id( (StdVec_int)self) -> int :
            Returns the unique identity of an object.
            For object held in C++, it corresponds to its memory address.
        """
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StdVec_int)self, (int)new_cap) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_cap.
        """
    @staticmethod
    def tolist(*args, **kwargs):
        """
        
        tolist( (StdVec_int)self [, (bool)deep_copy=False]) -> list :
            Returns the std::vector as a Python list.
        """
class Symmetric3(Boost.Python.instance):
    """
    This class represents symmetric 3x3 matrices.
    
    Supported operations ...
    """
    __safe_for_unpickling__: typing.ClassVar[bool] = True
    @staticmethod
    def Identity(*args, **kwargs):
        """
        
        Identity() -> Symmetric3 :
            Returns identity matrix.
        """
    @staticmethod
    def Random(*args, **kwargs):
        """
        
        Random() -> Symmetric3 :
            Returns a random symmetric 3x3 matrix.
        """
    @staticmethod
    def Zero(*args, **kwargs):
        """
        
        Zero() -> Symmetric3 :
            Returns a zero 3x3 matrix.
        """
    @staticmethod
    def __add__(*args, **kwargs):
        """
        
        __add__( (Symmetric3)arg1, (Symmetric3)arg2) -> object
        
        __add__( (Symmetric3)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def __copy__(*args, **kwargs):
        """
        
        __copy__( (Symmetric3)self) -> Symmetric3 :
            Returns a copy of *this.
        """
    @staticmethod
    def __deepcopy__(*args, **kwargs):
        """
        
        __deepcopy__( (Symmetric3)self, (dict)memo) -> Symmetric3 :
            Returns a deep copy of *this.
        """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (Symmetric3)arg1, (Symmetric3)arg2) -> object
        """
    @staticmethod
    def __getinitargs__(*args, **kwargs):
        """
        
        __getinitargs__( (Symmetric3)arg1) -> tuple
        """
    @staticmethod
    def __iadd__(*args, **kwargs):
        """
        
        __iadd__( (Symmetric3)arg1, (Symmetric3)arg2) -> object
        """
    @staticmethod
    def __imul__(*args, **kwargs):
        """
        
        __imul__( (Symmetric3)arg1, (float)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor.
        
        __init__( (object)self, (numpy.ndarray)I) -> None :
            Initialize from symmetrical matrix I of size 3x3.
        
        __init__( (object)self, (numpy.ndarray)I) -> None :
            Initialize from vector I of size 6.
        
        __init__( (object)self, (float)a0, (float)a1, (float)a2, (float)a3, (float)a4, (float)a5) -> None :
            Initialize from 6 scalar values.
        
        __init__( (object)self, (Symmetric3)other) -> None :
            Copy constructor.
        
        __init__( (object)arg1, (Symmetric3)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Symmetric3 -> pinocchio.pinocchio_pywrap_default.Symmetric3
        
        __init__( (object)arg1, (Symmetric3)clone) -> object :
            Copy constructor from pinocchio.pinocchio_pywrap_default.Symmetric3 -> pinocchio.pinocchio_pywrap_default.Symmetric3
        """
    @staticmethod
    def __isub__(*args, **kwargs):
        """
        
        __isub__( (Symmetric3)arg1, (object)arg2) -> object
        
        __isub__( (Symmetric3)arg1, (object)arg2) -> object
        
        __isub__( (Symmetric3)arg1, (Symmetric3)arg2) -> object
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (Symmetric3)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (Symmetric3)arg1, (Symmetric3)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (Symmetric3)arg1) -> object
        """
    @staticmethod
    def __str__(*args, **kwargs):
        """
        
        __str__( (Symmetric3)arg1) -> object
        """
    @staticmethod
    def __sub__(*args, **kwargs):
        """
        
        __sub__( (Symmetric3)arg1, (object)arg2) -> object
        
        __sub__( (Symmetric3)arg1, (object)arg2) -> object
        
        __sub__( (Symmetric3)arg1, (Symmetric3)arg2) -> object
        
        __sub__( (Symmetric3)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def cast(*args, **kwargs):
        """
        
        cast( (Symmetric3)arg1) -> Symmetric3 :
            Returns a cast of *this.
        """
    @staticmethod
    def copy(*args, **kwargs):
        """
        
        copy( (Symmetric3)self) -> Symmetric3 :
            Returns a copy of *this.
        """
    @staticmethod
    def decomposeltI(*args, **kwargs):
        """
        
        decomposeltI( (Symmetric3)self) -> object :
            Computes L for a symmetric matrix S.
        """
    @staticmethod
    def fill(*args, **kwargs):
        """
        
        fill( (Symmetric3)self, (float)value) -> None
        """
    @staticmethod
    def inverse(*args, **kwargs):
        """
        
        inverse( (Symmetric3)self, (numpy.ndarray)res) -> None :
            Invert the symmetrical 3x3 matrix.
        """
    @staticmethod
    def isApprox(*args, **kwargs):
        """
        
        isApprox( (Symmetric3)self, (Symmetric3)other [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to other, within the precision given by prec.
        """
    @staticmethod
    def isZero(*args, **kwargs):
        """
        
        isZero( (Symmetric3)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the zero matrix, within the precision given by prec.
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (Symmetric3)self) -> numpy.ndarray :
            Returns a matrix representation of the data.
        """
    @staticmethod
    def rhsMult(*args, **kwargs):
        """
        
        rhsMult( (Symmetric3)SE3, (numpy.ndarray)vin, (numpy.ndarray)vout) -> None
        """
    @staticmethod
    def rotate(*args, **kwargs):
        """
        
        rotate( (Symmetric3)self, (numpy.ndarray)R) -> Symmetric3 :
            Computes R*S*R'
        """
    @staticmethod
    def setDiagonal(*args, **kwargs):
        """
        
        setDiagonal( (Symmetric3)self, (numpy.ndarray)diag) -> None :
            Set the diagonal elements of 3x3 matrix.
        """
    @staticmethod
    def setIdentity(*args, **kwargs):
        """
        
        setIdentity( (Symmetric3)self) -> None :
            Set the components of *this to identity.
        """
    @staticmethod
    def setRandom(*args, **kwargs):
        """
        
        setRandom( (Symmetric3)self) -> None :
            Set all the components of *this randomly.
        """
    @staticmethod
    def setZero(*args, **kwargs):
        """
        
        setZero( (Symmetric3)self) -> None :
            Set all the components of *this to zero.
        """
    @staticmethod
    def svx(*args, **kwargs):
        """
        
        svx( (Symmetric3)v, (numpy.ndarray)S3) -> numpy.ndarray :
            Performs the operation 
        $ M = S_{3} [v]_{	imes} 
        $.
        """
    @staticmethod
    def vtiv(*args, **kwargs):
        """
        
        vtiv( (Symmetric3)self, (numpy.ndarray)v) -> float
        """
    @staticmethod
    def vxs(*args, **kwargs):
        """
        
        vxs( (Symmetric3)v, (numpy.ndarray)S3) -> numpy.ndarray :
            Performs the operation 
        $ M = [v]_{	imes} S_{3} 
        $., Apply the cross product of v on each column of S and return result matrix M.
        """
    @property
    def data(*args, **kwargs):
        """
        6D vector containing the data of the symmetric 3x3 matrix.
        """
    @data.setter
    def data(*args, **kwargs):
        ...
class TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager(Boost.Python.instance):
    """
    Tree-based broad phase manager associated to coal::DynamicAABBTreeArrayCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getBroadPhaseManagers(*args, **kwargs):
        """
        
        getBroadPhaseManagers( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> object :
            Returns the internal broad phase managers
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (TreeBroadPhaseManager_DynamicAABBTreeArrayCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class TreeBroadPhaseManager_DynamicAABBTreeCollisionManager(Boost.Python.instance):
    """
    Tree-based broad phase manager associated to coal::DynamicAABBTreeCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self, (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getBroadPhaseManagers(*args, **kwargs):
        """
        
        getBroadPhaseManagers( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> object :
            Returns the internal broad phase managers
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (TreeBroadPhaseManager_DynamicAABBTreeCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class TreeBroadPhaseManager_IntervalTreeCollisionManager(Boost.Python.instance):
    """
    Tree-based broad phase manager associated to coal::IntervalTreeCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (TreeBroadPhaseManager_IntervalTreeCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self, (TreeBroadPhaseManager_IntervalTreeCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getBroadPhaseManagers(*args, **kwargs):
        """
        
        getBroadPhaseManagers( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self) -> object :
            Returns the internal broad phase managers
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (TreeBroadPhaseManager_IntervalTreeCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class TreeBroadPhaseManager_NaiveCollisionManager(Boost.Python.instance):
    """
    Tree-based broad phase manager associated to coal::NaiveCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (TreeBroadPhaseManager_NaiveCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (TreeBroadPhaseManager_NaiveCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (TreeBroadPhaseManager_NaiveCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (TreeBroadPhaseManager_NaiveCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_NaiveCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_NaiveCollisionManager)self, (TreeBroadPhaseManager_NaiveCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getBroadPhaseManagers(*args, **kwargs):
        """
        
        getBroadPhaseManagers( (TreeBroadPhaseManager_NaiveCollisionManager)self) -> object :
            Returns the internal broad phase managers
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (TreeBroadPhaseManager_NaiveCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (TreeBroadPhaseManager_NaiveCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (TreeBroadPhaseManager_NaiveCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (TreeBroadPhaseManager_NaiveCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (TreeBroadPhaseManager_NaiveCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class TreeBroadPhaseManager_SSaPCollisionManager(Boost.Python.instance):
    """
    Tree-based broad phase manager associated to coal::SSaPCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (TreeBroadPhaseManager_SSaPCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (TreeBroadPhaseManager_SSaPCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (TreeBroadPhaseManager_SSaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (TreeBroadPhaseManager_SSaPCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_SSaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_SSaPCollisionManager)self, (TreeBroadPhaseManager_SSaPCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getBroadPhaseManagers(*args, **kwargs):
        """
        
        getBroadPhaseManagers( (TreeBroadPhaseManager_SSaPCollisionManager)self) -> object :
            Returns the internal broad phase managers
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (TreeBroadPhaseManager_SSaPCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (TreeBroadPhaseManager_SSaPCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (TreeBroadPhaseManager_SSaPCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (TreeBroadPhaseManager_SSaPCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (TreeBroadPhaseManager_SSaPCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class TreeBroadPhaseManager_SaPCollisionManager(Boost.Python.instance):
    """
    Tree-based broad phase manager associated to coal::SaPCollisionManager
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
            Default constructor
        
        __init__( (object)self, (TreeBroadPhaseManager_SaPCollisionManager)other) -> None :
            Copy constructor
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def check(*args, **kwargs):
        """
        
        check( (TreeBroadPhaseManager_SaPCollisionManager)self) -> bool :
            Check whether the base broad phase manager is aligned with the current collision_objects.
        
        check( (TreeBroadPhaseManager_SaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Check whether the callback is inline with *this.
        """
    @staticmethod
    def collide(*args, **kwargs):
        """
        
        collide( (TreeBroadPhaseManager_SaPCollisionManager)self, (CollisionObject)collision_object, (CollisionCallBackBase)callback) -> bool :
            Performs collision test between one object and all the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_SaPCollisionManager)self, (CollisionCallBackBase)callback) -> bool :
            Performs collision test for the objects belonging to the manager.
        
        collide( (TreeBroadPhaseManager_SaPCollisionManager)self, (TreeBroadPhaseManager_SaPCollisionManager)other_manager, (CollisionCallBackBase)callback) -> bool :
            Performs collision test with objects belonging to another manager.
        """
    @staticmethod
    def getBroadPhaseManagers(*args, **kwargs):
        """
        
        getBroadPhaseManagers( (TreeBroadPhaseManager_SaPCollisionManager)self) -> object :
            Returns the internal broad phase managers
        """
    @staticmethod
    def getGeometryData(*args, **kwargs):
        """
        
        getGeometryData( (TreeBroadPhaseManager_SaPCollisionManager)self) -> GeometryData :
            Returns the related geometry data.
        """
    @staticmethod
    def getGeometryModel(*args, **kwargs):
        """
        
        getGeometryModel( (TreeBroadPhaseManager_SaPCollisionManager)self) -> GeometryModel :
            Returns the related geometry model.
        """
    @staticmethod
    def getModel(*args, **kwargs):
        """
        
        getModel( (TreeBroadPhaseManager_SaPCollisionManager)self) -> Model :
            Returns the related model.
        """
    @staticmethod
    def update(*args, **kwargs):
        """
        
        update( (TreeBroadPhaseManager_SaPCollisionManager)self [, (bool)compute_local_aabb=False]) -> None :
            Update the manager from the current geometry positions and update the underlying FCL broad phase manager.
        
        update( (TreeBroadPhaseManager_SaPCollisionManager)self, (GeometryData)geom_data_new) -> None :
            Update the manager with a new geometry data.
        """
class TridiagonalSymmetricMatrix(Boost.Python.instance):
    """
    Tridiagonal symmetric matrix.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (TridiagonalSymmetricMatrix)arg1, (TridiagonalSymmetricMatrix)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (int)size) -> None :
            Default constructor from a given size.
        """
    @staticmethod
    def __mul__(*args, **kwargs):
        """
        
        __mul__( (TridiagonalSymmetricMatrix)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def __ne__(*args, **kwargs):
        """
        
        __ne__( (TridiagonalSymmetricMatrix)arg1, (TridiagonalSymmetricMatrix)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __rmul__(*args, **kwargs):
        """
        
        __rmul__( (TridiagonalSymmetricMatrix)arg1, (numpy.ndarray)arg2) -> object
        """
    @staticmethod
    def cols(*args, **kwargs):
        """
        
        cols( (TridiagonalSymmetricMatrix)self) -> int
        """
    @staticmethod
    def diagonal(*args, **kwargs):
        """
        
        diagonal( (TridiagonalSymmetricMatrix)self) -> object :
            Reference of the diagonal elements of the symmetric tridiagonal matrix.
        """
    @staticmethod
    def isDiagonal(*args, **kwargs):
        """
        
        isDiagonal( (TridiagonalSymmetricMatrix)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the a diagonal matrix, within the precision given by prec.
        """
    @staticmethod
    def isIdentity(*args, **kwargs):
        """
        
        isIdentity( (TridiagonalSymmetricMatrix)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the identity matrix, within the precision given by prec.
        """
    @staticmethod
    def isZero(*args, **kwargs):
        """
        
        isZero( (TridiagonalSymmetricMatrix)self [, (float)prec=1e-12]) -> bool :
            Returns true if *this is approximately equal to the zero matrix, within the precision given by prec.
        """
    @staticmethod
    def matrix(*args, **kwargs):
        """
        
        matrix( (TridiagonalSymmetricMatrix)self) -> numpy.ndarray
        """
    @staticmethod
    def rows(*args, **kwargs):
        """
        
        rows( (TridiagonalSymmetricMatrix)self) -> int
        """
    @staticmethod
    def setDiagonal(*args, **kwargs):
        """
        
        setDiagonal( (TridiagonalSymmetricMatrix)self, (numpy.ndarray)diagonal) -> None :
            Set the current tridiagonal matrix to a diagonal matrix given by the entry vector diagonal.
        """
    @staticmethod
    def setIdentity(*args, **kwargs):
        """
        
        setIdentity( (TridiagonalSymmetricMatrix)self) -> None :
            Set the current tridiagonal matrix to identity.
        """
    @staticmethod
    def setRandom(*args, **kwargs):
        """
        
        setRandom( (TridiagonalSymmetricMatrix)self) -> None :
            Set the current tridiagonal matrix to random.
        """
    @staticmethod
    def setZero(*args, **kwargs):
        """
        
        setZero( (TridiagonalSymmetricMatrix)self) -> None :
            Set the current tridiagonal matrix to zero.
        """
    @staticmethod
    def subDiagonal(*args, **kwargs):
        """
        
        subDiagonal( (TridiagonalSymmetricMatrix)self) -> object :
            Reference of the sub diagonal elements of the symmetric tridiagonal matrix.
        """
class boost_type_index(Boost.Python.instance):
    """
    The class type_index holds implementation-specific information about a type, including the name of the type and means to compare two types for equality or collating order.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (boost_type_index)arg1, (boost_type_index)arg2) -> object
        """
    @staticmethod
    def __ge__(*args, **kwargs):
        """
        
        __ge__( (boost_type_index)arg1, (boost_type_index)arg2) -> object
        """
    @staticmethod
    def __gt__(*args, **kwargs):
        """
        
        __gt__( (boost_type_index)arg1, (boost_type_index)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        Raises an exception
        This class cannot be instantiated from Python
        """
    @staticmethod
    def __le__(*args, **kwargs):
        """
        
        __le__( (boost_type_index)arg1, (boost_type_index)arg2) -> object
        """
    @staticmethod
    def __lt__(*args, **kwargs):
        """
        
        __lt__( (boost_type_index)arg1, (boost_type_index)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def hash_code(*args, **kwargs):
        """
        
        hash_code( (boost_type_index)self) -> int :
            Returns an unspecified value (here denoted by hash code) such that for all std::type_info objects referring to the same type, their hash code is the same.
        """
    @staticmethod
    def name(*args, **kwargs):
        """
        
        name( (boost_type_index)self) -> str :
            Returns an implementation defined null-terminated character string containing the name of the type. No guarantees are given; in particular, the returned string can be identical for several types and change between invocations of the same program.
        """
    @staticmethod
    def pretty_name(*args, **kwargs):
        """
        
        pretty_name( (boost_type_index)self) -> str :
            Human readible name.
        """
class map_indexing_suite_StdMap_String_VectorXd_entry(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 72
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)arg1) -> None
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def __repr__(*args, **kwargs):
        """
        
        __repr__( (map_indexing_suite_StdMap_String_VectorXd_entry)arg1) -> object
        """
    @staticmethod
    def data(*args, **kwargs):
        """
        
        data( (map_indexing_suite_StdMap_String_VectorXd_entry)arg1) -> object
        """
    @staticmethod
    def key(*args, **kwargs):
        """
        
        key( (map_indexing_suite_StdMap_String_VectorXd_entry)arg1) -> str
        """
class std_type_index(Boost.Python.instance):
    """
    The class type_index holds implementation-specific information about a type, including the name of the type and means to compare two types for equality or collating order.
    """
    @staticmethod
    def __eq__(*args, **kwargs):
        """
        
        __eq__( (std_type_index)arg1, (std_type_index)arg2) -> object
        """
    @staticmethod
    def __ge__(*args, **kwargs):
        """
        
        __ge__( (std_type_index)arg1, (std_type_index)arg2) -> object
        """
    @staticmethod
    def __gt__(*args, **kwargs):
        """
        
        __gt__( (std_type_index)arg1, (std_type_index)arg2) -> object
        """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        Raises an exception
        This class cannot be instantiated from Python
        """
    @staticmethod
    def __le__(*args, **kwargs):
        """
        
        __le__( (std_type_index)arg1, (std_type_index)arg2) -> object
        """
    @staticmethod
    def __lt__(*args, **kwargs):
        """
        
        __lt__( (std_type_index)arg1, (std_type_index)arg2) -> object
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def hash_code(*args, **kwargs):
        """
        
        hash_code( (std_type_index)self) -> int :
            Returns an unspecified value (here denoted by hash code) such that for all std::type_info objects referring to the same type, their hash code is the same.
        """
    @staticmethod
    def name(*args, **kwargs):
        """
        
        name( (std_type_index)self) -> str :
            Returns an implementation defined null-terminated character string containing the name of the type. No guarantees are given; in particular, the returned string can be identical for several types and change between invocations of the same program.
        """
    @staticmethod
    def pretty_name(*args, **kwargs):
        """
        
        pretty_name( (std_type_index)self) -> str :
            Human readible name.
        """
def Hlog3(*args, **kwargs):
    """
    
    Hlog3( (numpy.ndarray)R, (numpy.ndarray)v) -> numpy.ndarray :
        Vector v to be multiplied to the hessian
    """
def Jexp3(*args, **kwargs):
    """
    
    Jexp3( (numpy.ndarray)w) -> numpy.ndarray :
        Jacobian of exp(v) which maps from the tangent of SO(3) at R = exp(v) to the tangent of SO(3) at Identity.
    """
def Jexp6(*args, **kwargs):
    """
    
    Jexp6( (Motion)motion) -> numpy.ndarray :
        Jacobian of exp(v) which maps from the tangent of SE(3) at exp(v) to the tangent of SE(3) at Identity.
    
    Jexp6( (numpy.ndarray)v) -> numpy.ndarray :
        Jacobian of exp(v) which maps from the tangent of SE(3) at exp(v) to the tangent of SE(3) at Identity.
    """
def Jlog3(*args, **kwargs):
    """
    
    Jlog3( (numpy.ndarray)R) -> numpy.ndarray :
        Jacobian of log(R) which maps from the tangent of SO(3) at R to the tangent of SO(3) at Identity.
    """
def Jlog6(*args, **kwargs):
    """
    
    Jlog6( (SE3)M) -> numpy.ndarray :
        Jacobian of log(M) which maps from the tangent of SE(3) at M to the tangent of SE(3) at Identity.
    """
def SE3ToXYZQUAT(*args, **kwargs):
    """
    
    SE3ToXYZQUAT( (SE3)arg1) -> numpy.ndarray :
        M
    """
def SE3ToXYZQUATtuple(*args, **kwargs):
    """
    
    SE3ToXYZQUATtuple( (SE3)arg1) -> tuple :
        M
    """
def XYZQUATToSE3(*args, **kwargs):
    """
    
    XYZQUATToSE3( (tuple)tuple) -> SE3 :
        Reverse function of SE3ToXYZQUAT: convert [X,Y,Z,x,y,z,w] to an SE3 element.
    
    XYZQUATToSE3( (list)list) -> SE3 :
        Reverse function of SE3ToXYZQUAT: convert [X,Y,Z,x,y,z,w] to an SE3 element.
    
    XYZQUATToSE3( (numpy.ndarray)array) -> SE3 :
        Reverse function of SE3ToXYZQUAT: convert [X,Y,Z,x,y,z,w] to an SE3 element.
    """
def aba(*args, **kwargs):
    """
    
    aba( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)tau [, (Convention)convention=pinocchio.pinocchio_pywrap_default.Convention.LOCAL]) -> numpy.ndarray :
        Compute ABA, store the result in data.ddq and return it.
        Parameters:
        	 model: Model of the kinematic tree
        	 data: Data related to the kinematic tree
        	 q: joint configuration (size model.nq)
        	 tau: joint velocity (size model.nv)
        	 v: joint torque (size model.nv)	 convention: Convention to use
    
    aba( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)tau, (StdVec_Force)fext [, (Convention)convention=pinocchio.pinocchio_pywrap_default.Convention.LOCAL]) -> numpy.ndarray :
        Compute ABA with external forces, store the result in data.ddq and return it.
        Parameters:
        	 model: Model of the kinematic tree
        	 data: Data related to the kinematic tree
        	 q: joint configuration (size model.nq)
        	 v: joint velocity (size model.nv)
        	 tau: joint torque (size model.nv)
        	 fext: vector of external forces expressed in the local frame of the joint (size model.njoints)	 convention: Convention to use
    """
def appendModel(*args, **kwargs):
    """
    
    appendModel( (Model)modelA, (Model)modelB, (int)frame_in_modelA, (SE3)aMb) -> Model :
        Append a child model into a parent model, after a specific frame given by its index.
        
        Parameters:
        	modelA: the parent model
        	modelB: the child model
        	frameInModelA:  index of the frame of modelA where to append modelB
        	aMb: pose of modelB universe joint (index 0) in frameInModelA
        
    
    appendModel( (Model)modelA, (Model)modelB, (GeometryModel)geomModelA, (GeometryModel)geomModelB, (int)frame_in_modelA, (SE3)aMb) -> tuple :
        Append a child (geometry) model into a parent (geometry) model, after a specific frame given by its index.
        
        Parameters:
        	modelA: the parent model
        	modelB: the child model
        	geomModelA: the parent geometry model
        	geomModelB: the child geometry model
        	frameInModelA:  index of the frame of modelA where to append modelB
        	aMb: pose of modelB universe joint (index 0) in frameInModelA
        
    """
def bodyRegressor(*args, **kwargs):
    """
    
    bodyRegressor( (Motion)velocity, (Motion)acceleration) -> numpy.ndarray :
        Computes the regressor for the dynamic parameters of a single rigid body.
        The result is such that Ia + v x Iv = bodyRegressor(v,a) * I.toDynamicParameters()
        
        Parameters:
        	velocity: spatial velocity of the rigid body
        	acceleration: spatial acceleration of the rigid body
        
    """
def buildGeomFromMJCF(*args, **kwargs):
    """
    
    buildGeomFromMJCF( (Model)model, (object)mjcf_filename, (GeometryType)geom_type) -> GeometryModel :
        Parse the Mjcf file given as input looking for the geometry of the given input model and
        return a GeometryModel containing either the collision geometries (GeometryType.COLLISION) or the visual geometries (GeometryType.VISUAL).
        Parameters:
        	model: model of the robot
        	filename: path to the mjcf file containing the model of the robot
        	geom_type: type of geometry to extract from the mjcf file (either the VISUAL for display or the COLLISION for collision detection).
        
    
    buildGeomFromMJCF( (Model)model, (object)mjcf_filename, (GeometryType)geom_type, (coal.coal_pywrap.MeshLoader)mesh_loader) -> GeometryModel :
        Parse the Mjcf file given as input looking for the geometry of the given input model and
        return a GeometryModel containing either the collision geometries (GeometryType.COLLISION) or the visual geometries (GeometryType.VISUAL).
        Parameters:
        	model: model of the robot
        	filename: path to the mjcf file containing the model of the robot
        	geom_type: type of geometry to extract from the mjcf file (either the VISUAL for display or the COLLISION for collision detection).
        	mesh_loader: an hpp-fcl mesh loader (to load only once the related geometries).
        
    """
def buildGeomFromUrdf(*args, **kwargs):
    """
    
    buildGeomFromUrdf( (Model)model, (object)urdf_filename, (GeometryType)geom_type [, (object)geometry_model=None [, (object)package_dirs=None [, (object)mesh_loader=None]]]) -> GeometryModel :
        Parse the URDF file given as input looking for the geometry of the given input model and
        and store either the collision geometries (GeometryType.COLLISION) or the visual geometries (GeometryType.VISUAL) in a GeometryModel object.
        Parameters:
        	model: model of the robot
        
        urdf_filename: path to the URDF file containing the model of the robot
        	geom_type: type of geometry to extract from the URDF file (either the VISUAL for display or the COLLISION for collision detection).
        	geometry_model: if provided, this geometry model will be used to store the parsed information instead of creating a new one
        	package_dirs: either a single path or a vector of paths pointing to folders containing the model of the robot
        	mesh_loader: an hpp-fcl mesh loader (to load only once the related geometries).
        
        Retuns:
        	a new GeometryModel if `geometry_model` is None else `geometry_model` (that has been updated).
        
    """
def buildGeomFromUrdfString(*args, **kwargs):
    """
    
    buildGeomFromUrdfString( (Model)model, (str)urdf_string, (GeometryType)geom_type [, (object)geometry_model=None [, (object)package_dirs=None [, (object)mesh_loader=None]]]) -> GeometryModel :
        Parse the URDF file given as input looking for the geometry of the given input model and
        and store either the collision geometries (GeometryType.COLLISION) or the visual geometries (GeometryType.VISUAL) in a GeometryModel object.
        Parameters:
        	model: model of the robot
        
        urdf_string: a string containing the URDF model of the robot
        	geom_type: type of geometry to extract from the URDF file (either the VISUAL for display or the COLLISION for collision detection).
        	geometry_model: if provided, this geometry model will be used to store the parsed information instead of creating a new one
        	package_dirs: either a single path or a vector of paths pointing to folders containing the model of the robot
        	mesh_loader: an hpp-fcl mesh loader (to load only once the related geometries).
        
        Retuns:
        	a new GeometryModel if `geometry_model` is None else `geometry_model` (that has been updated).
        
    """
def buildModelFromMJCF(*args, **kwargs):
    """
    
    buildModelFromMJCF( (object)mjcf_filename) -> Model :
        Parse the MJCF file given in input and return a pinocchio Model.
    
    buildModelFromMJCF( (object)mjcf_filename, (JointModel)root_joint) -> Model :
        Parse the MJCF file and return a pinocchio Model with the given root Joint.
    
    buildModelFromMJCF( (object)mjcf_filename, (JointModel)root_joint, (str)root_joint_name) -> tuple :
        Parse the MJCF file and return a pinocchio Model with the given root Joint and its specified name as well as a constraint list if some are present in the MJCF file.
    """
def buildModelFromUrdf(*args, **kwargs):
    """
    
    buildModelFromUrdf( (object)urdf_filename, (JointModel)root_joint) -> Model :
        Parse the URDF file given in input and return a pinocchio Model starting with the given root joint.
    
    buildModelFromUrdf( (object)urdf_filename, (JointModel)root_joint, (str)root_joint_name) -> Model :
        Parse the URDF file given in input and return a pinocchio Model starting with the given root joint with its specified name.
    
    buildModelFromUrdf( (object)urdf_filename) -> Model :
        Parse the URDF file given in input and return a pinocchio Model.
    
    buildModelFromUrdf( (object)urdf_filename, (Model)model) -> Model :
        Append to a given model a URDF structure given by its filename.
    
    buildModelFromUrdf( (object)urdf_filename, (JointModel)root_joint, (Model)model) -> Model :
        Append to a given model a URDF structure given by its filename and the root joint.
        Remark: In the URDF format, a joint of type fixed can be defined. For efficiency reasons,it is treated as operational frame and not as a joint of the model.
    
    buildModelFromUrdf( (object)urdf_filename, (JointModel)root_joint, (str)root_joint_name, (Model)model) -> Model :
        Append to a given model a URDF structure given by its filename and the root joint with its specified name.
        Remark: In the URDF format, a joint of type fixed can be defined. For efficiency reasons,it is treated as operational frame and not as a joint of the model.
    """
def buildModelFromXML(*args, **kwargs):
    """
    
    buildModelFromXML( (str)urdf_xml_stream, (JointModel)root_joint) -> Model :
        Parse the URDF XML stream given in input and return a pinocchio Model starting with the given root joint.
    
    buildModelFromXML( (str)urdf_xml_stream, (JointModel)root_joint, (str)root_joint_name) -> Model :
        Parse the URDF XML stream given in input and return a pinocchio Model starting with the given root joint with its specified name.
    
    buildModelFromXML( (str)urdf_xml_stream, (JointModel)root_joint, (Model)model) -> Model :
        Parse the URDF XML stream given in input and append it to the input model with the given interfacing joint.
    
    buildModelFromXML( (str)urdf_xml_stream, (JointModel)root_joint, (str)root_joint_name, (Model)model) -> Model :
        Parse the URDF XML stream given in input and append it to the input model with the given interfacing joint with its specified name.
    
    buildModelFromXML( (str)urdf_xml_stream) -> Model :
        Parse the URDF XML stream given in input and return a pinocchio Model.
    
    buildModelFromXML( (str)urdf_xml_stream, (Model)model) -> Model :
        Parse the URDF XML stream given in input and append it to the input model.
    """
def buildReducedModel(*args, **kwargs):
    """
    
    buildReducedModel( (Model)model, (StdVec_Index)list_of_joints_to_lock, (numpy.ndarray)reference_configuration) -> Model :
        Build a reduce model from a given input model and a list of joint to lock.
        
        Parameters:
        	model: input kinematic modell to reduce
        	list_of_joints_to_lock: list of joint indexes to lock
        	reference_configuration: reference configuration to compute the placement of the lock joints
        
    
    buildReducedModel( (Model)model, (GeometryModel)geom_model, (StdVec_Index)list_of_joints_to_lock, (numpy.ndarray)reference_configuration) -> tuple :
        Build a reduced model and a reduced geometry model from a given input model,an input geometry model and a list of joints to lock.
        
        Parameters:
        	model: input kinematic model to reduce
        	geom_model: input geometry model to reduce
        	list_of_joints_to_lock: list of joint indexes to lock
        	reference_configuration: reference configuration to compute the placement of the locked joints
        
    
    buildReducedModel( (Model)model, (StdVec_GeometryModel)list_of_geom_models, (StdVec_Index)list_of_joints_to_lock, (numpy.ndarray)reference_configuration) -> tuple :
        Build a reduced model and the related reduced geometry models from a given input model,a list of input geometry models and a list of joints to lock.
        
        Parameters:
        	model: input kinematic model to reduce
        	list_of_geom_models: input geometry models to reduce
        	list_of_joints_to_lock: list of joint indexes to lock
        	reference_configuration: reference configuration to compute the placement of the locked joints
        
    """
def buildSampleGeometryModelHumanoid(*args, **kwargs):
    """
    
    buildSampleGeometryModelHumanoid( (Model)model) -> GeometryModel :
        Generate a (hard-coded) geometry model of a simple humanoid.
    """
def buildSampleGeometryModelManipulator(*args, **kwargs):
    """
    
    buildSampleGeometryModelManipulator( (Model)model) -> GeometryModel :
        Generate a (hard-coded) geometry model of a simple manipulator.
    """
def buildSampleModelHumanoid(*args, **kwargs):
    """
    
    buildSampleModelHumanoid() -> Model :
        Generate a (hard-coded) model of a simple humanoid.
    
    buildSampleModelHumanoid( (bool)using_free_flyer) -> Model :
        Generate a (hard-coded) model of a simple humanoid.
    """
def buildSampleModelHumanoidRandom(*args, **kwargs):
    """
    
    buildSampleModelHumanoidRandom() -> Model :
        Generate a (hard-coded) model of a humanoid robot with 6-DOF limbs and random joint placements.
        Only meant for unit tests.
    
    buildSampleModelHumanoidRandom( (bool)using_free_flyer) -> Model :
        Generate a (hard-coded) model of a humanoid robot with 6-DOF limbs and random joint placements.
        Only meant for unit tests.
    """
def buildSampleModelManipulator(*args, **kwargs):
    """
    
    buildSampleModelManipulator() -> Model :
        Generate a (hard-coded) model of a simple manipulator.
    """
def ccrba(*args, **kwargs):
    """
    
    ccrba( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Computes the centroidal mapping, the centroidal momentum and the Centroidal Composite Rigid Body Inertia, puts the result in Data and returns the centroidal mapping.For the same price, it also computes the total joint jacobians (data.J).
    """
def centerOfMass(*args, **kwargs):
    """
    
    centerOfMass( (Model)model, (Data)data, (numpy.ndarray)q [, (bool)compute_subtree_coms=True]) -> numpy.ndarray :
        Compute the center of mass, putting the result in context::Data and return it.If compute_subtree_coms is True, the algorithm also computes the center of mass of the subtrees.
    
    centerOfMass( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v [, (bool)compute_subtree_coms=True]) -> numpy.ndarray :
        Computes the center of mass position and velocity by storing the result in context::Data. It returns the center of mass position expressed in the WORLD frame.
        If compute_subtree_coms is True, the algorithm also computes the center of mass of the subtrees.
    
    centerOfMass( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a [, (bool)compute_subtree_coms=True]) -> numpy.ndarray :
        Computes the center of mass position, velocity and acceleration by storing the result in context::Data. It returns the center of mass position expressed in the WORLD frame.
        If compute_subtree_coms is True, the algorithm also computes the center of mass of the subtrees.
    
    centerOfMass( (Model)model, (Data)data, (KinematicLevel)kinematic_level [, (bool)compute_subtree_coms=True]) -> numpy.ndarray :
        Computes the center of mass position, velocity or acceleration of a given model according to the current kinematic values contained in data and the requested kinematic_level.
        If kinematic_level = POSITION, computes the CoM position, if kinematic_level = VELOCITY, also computes the CoM velocity and if kinematic_level = ACCELERATION, it also computes the CoM acceleration.
        If compute_subtree_coms is True, the algorithm also computes the center of mass of the subtrees.
    
    centerOfMass( (Model)model, (Data)data [, (bool)compute_subtree_coms=True]) -> numpy.ndarray :
        Computes the center of mass position, velocity and acceleration of a given model according to the current kinematic values contained in data.
        If compute_subtree_coms is True, the algorithm also computes the center of mass of the subtrees.
    """
def checkVersionAtLeast(*args, **kwargs):
    """
    
    checkVersionAtLeast( (int)major, (int)minor, (int)patch) -> bool :
        Checks if the current version of Pinocchio is at least the version provided by the input arguments.
    """
def classicAcceleration(*args, **kwargs):
    """
    
    classicAcceleration( (object)spatial_velocity, (object)spatial_acceleration) -> numpy.ndarray :
        Computes the classic acceleration from a given spatial velocity and spatial acceleration.
    
    classicAcceleration( (object)spatial_velocity, (object)spatial_acceleration, (SE3)placement) -> numpy.ndarray :
        Computes the classic acceleration of a frame B, given the spatial velocity and spatial acceleration of a frame A,
        and the relative placement A^M_B.
    """
def computeABADerivatives(*args, **kwargs):
    """
    
    computeABADerivatives( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)tau) -> tuple :
        Computes the ABA derivatives, store the result in data.ddq_dq, data.ddq_dv and data.Minv (aka ddq_dtau)
        which correspond to the partial derivatives of the joint acceleration vector output with respect to the joint configuration,
        velocity and torque vectors.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	tau: the joint torque vector (size model.nv)
        
        Returns: (ddq_dq, ddq_dv, ddq_da)
    
    computeABADerivatives( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)tau, (StdVec_Force)fext) -> tuple :
        Computes the ABA derivatives with external contact foces,
        store the result in data.ddq_dq, data.ddq_dv and data.Minv (aka ddq_dtau)
        which correspond to the partial derivatives of the acceleration output with respect to the joint configuration,
        velocity and torque vectors.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	tau: the joint torque vector (size model.nv)
        	fext: list of external forces expressed in the local frame of the joints (size model.njoints)
        
        Returns: (ddq_dq, ddq_dv, ddq_da)
    
    computeABADerivatives( (Model)model, (Data)data) -> tuple :
        Computes the ABA derivatives, store the result in data.ddq_dq, data.ddq_dv and data.Minv
        which correspond to the partial derivatives of the joint acceleration vector output with respect to the joint configuration,
        velocity and torque vectors.
        By calling this function, the user assumes that pinocchio.optimized.aba has been called first, allowing to significantly reduce the computation timings by not recalculating intermediate results.
    
    computeABADerivatives( (Model)model, (Data)data, (StdVec_Force)fext) -> tuple :
        Computes the ABA derivatives, store the result in data.ddq_dq, data.ddq_dv and data.Minv
        which correspond to the partial derivatives of the joint acceleration vector output with respect to the joint configuration,
        velocity and torque vectors.
        By calling this function, the user assumes that pinocchio.optimized.aba has been called first, allowing to significantly reduce the computation timings by not recalculating intermediate results.
    """
def computeAllTerms(*args, **kwargs):
    """
    
    computeAllTerms( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> None :
        Compute all the terms M, non linear effects, center of mass quantities, centroidal quantities and Jacobians inin the same loop and store the results in data.
        This algorithm is equivalent to calling:
        	- forwardKinematics
        	- crba
        	- nonLinearEffects
        	- computeJointJacobians
        	- centerOfMass
        	- jacobianCenterOfMass
        	- ccrba
        	- computeKineticEnergy
        	- computePotentialEnergy
        	- computeGeneralizedGravity
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    """
def computeBodyRadius(*args, **kwargs):
    """
    
    computeBodyRadius( (Model)model, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
        Compute the radius of the geometry volumes attached to every joints.
    """
def computeCentroidalDynamicsDerivatives(*args, **kwargs):
    """
    
    computeCentroidalDynamicsDerivatives( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a) -> tuple :
        Computes the analytical derivatives of the centroidal dynamics
        with respect to the joint configuration vector, velocity and acceleration.
    """
def computeCentroidalMap(*args, **kwargs):
    """
    
    computeCentroidalMap( (Model)model, (Data)data, (numpy.ndarray)q) -> numpy.ndarray :
        Computes the centroidal mapping, puts the result in Data.Ag and returns the centroidal mapping.
        For the same price, it also computes the total joint jacobians (data.J).
    """
def computeCentroidalMapTimeVariation(*args, **kwargs):
    """
    
    computeCentroidalMapTimeVariation( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Computes the time derivative of the centroidal momentum matrix Ag, puts the result in Data.Ag and returns the centroidal mapping.
        For the same price, it also computes the centroidal momentum matrix (data.Ag), the total joint jacobians (data.J) and the related joint jacobians time derivative (data.dJ)
    """
def computeCentroidalMomentum(*args, **kwargs):
    """
    
    computeCentroidalMomentum( (Model)model, (Data)data) -> Force :
        Computes the Centroidal momentum, a.k.a. the total momentum of the system expressed around the center of mass.
    
    computeCentroidalMomentum( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> Force :
        Computes the Centroidal momentum, a.k.a. the total momentum of the system expressed around the center of mass.
    """
def computeCentroidalMomentumTimeVariation(*args, **kwargs):
    """
    
    computeCentroidalMomentumTimeVariation( (Model)model, (Data)data) -> Force :
        Computes the Centroidal momentum and its time derivatives, a.k.a. the total momentum of the system and its time derivative expressed around the center of mass.
    
    computeCentroidalMomentumTimeVariation( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a) -> Force :
        Computes the Centroidal momentum and its time derivatives, a.k.a. the total momentum of the system and its time derivative expressed around the center of mass.
    """
def computeCollision(*args, **kwargs):
    """
    
    computeCollision( (GeometryModel)geometry_model, (GeometryData)geometry_data, (int)pair_index [, (coal.coal_pywrap.CollisionRequest)collision_request]) -> bool :
        Check if the collision objects of a collision pair for a given Geometry Model and Data are in collision.
        The collision pair is given by the two index of the collision objects.
    """
def computeCollisions(*args, **kwargs):
    """
    
    computeCollisions( (GeometryModel)geometry_model, (GeometryData)geometry_data [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
    
    computeCollisions( (Model)model, (Data)data, (GeometryModel)geometry_model, (GeometryData)geometry_data, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Update the geometry for a given configuration and determine if all collision pairs are effectively in collision or not.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (object)manager, (CollisionCallBackBase)callback) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (object)manager [, (bool)stop_at_first_collision=False]) -> bool :
        Determine if all collision pairs are effectively in collision or not.
        This function assumes that updateGeometryPlacements and broadphase_manager.update() have been called first.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (numpy.ndarray)q [, (bool)stop_at_first_collision=False]) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    
    computeCollisions( (Model)model, (Data)data, (object)broadphase_manager, (CollisionCallBackBase)callback, (numpy.ndarray)q) -> bool :
        Compute the forward kinematics, update the geometry placements and run the collision detection using the broadphase manager.
    """
def computeComplementarityShift(*args, **kwargs):
    """
    
    computeComplementarityShift( (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)velocities) -> numpy.ndarray :
        Compute the complementarity shift associated to the De Saxé function.
    """
def computeConeProjection(*args, **kwargs):
    """
    
    computeConeProjection( (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)forces) -> numpy.ndarray :
        Project a vector on the cartesian product of cones.
    """
def computeConstraintDynamicsDerivatives(*args, **kwargs):
    """
    
    computeConstraintDynamicsDerivatives( (Model)model, (Data)data, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas [, (ProximalSettings)settings=ProximalSettings(1e-12, 1e-12, 0, 1)]) -> tuple :
        Computes the derivatives of the forward dynamics with kinematic constraints (given in the list of constraint models).
        Assumes that constraintDynamics has been called first. See constraintDynamics for more details.
        This function returns the derivatives of joint acceleration (ddq) and contact forces (lambda_c) of the system with respect to q, v and tau.
        The output is a tuple with ddq_dq, ddq_dv, ddq_da, dlambda_dq, dlambda_dv, dlambda_da.
    """
def computeContactForces(*args, **kwargs):
    """
    
    computeContactForces( (Model)model, (Data)data, (numpy.ndarray)c_ref, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas, (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)R, (numpy.ndarray)constraint_correction, (ProximalSettings)settings [, (numpy.ndarray)lambda_guess=None]) -> numpy.ndarray :
        Compute the inverse dynamics with frictional contacts, store the result in Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	c_ref: the reference velocity of contact points
        	contact_models: list of contact models
        	contact_datas: list of contact datas
        	cones: list of friction cones
        	R: vector representing the diagonal of the compliance matrix
        	constraint_correction: vector representing the constraint correction
        	settings: the settings of the proximal algorithm
        	lambda_guess: initial guess for contact forces
        
    """
def computeCoriolisMatrix(*args, **kwargs):
    """
    
    computeCoriolisMatrix( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Compute the Coriolis Matrix C(q,v) of the Lagrangian dynamics, store the result in data.C and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    """
def computeDampedDelassusMatrixInverse(*args, **kwargs):
    """
    
    computeDampedDelassusMatrixInverse( (Model)arg1, (Data)model, (numpy.ndarray)data, (StdVec_RigidConstraintModel)q, (StdVec_RigidConstraintData)contact_models, (float)contact_datas [, (bool)mu=0]) -> numpy.ndarray :
        Computes the inverse of the Delassus matrix associated to a set of given constraints.
    """
def computeDelassusMatrix(*args, **kwargs):
    """
    
    computeDelassusMatrix( (Model)model, (Data)data, (numpy.ndarray)q, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas [, (float)mu=0]) -> numpy.ndarray :
        Computes the Delassus matrix associated to a set of given constraints.
    """
def computeDistance(*args, **kwargs):
    """
    
    computeDistance( (GeometryModel)geometry_model, (GeometryData)geometry_data, (int)pair_index) -> coal.coal_pywrap.DistanceResult :
        Compute the distance between the two geometry objects of a given collision pair for a GeometryModel and associated GeometryData.
    """
def computeDistances(*args, **kwargs):
    """
    
    computeDistances( (GeometryModel)geometry_model, (GeometryData)geometry_data) -> int :
        Compute the distance between each collision pair for a given GeometryModel and associated GeometryData.
    
    computeDistances( (Model)model, (Data)data, (GeometryModel)geometry_model, (GeometryData)geometry_data, (numpy.ndarray)q) -> int :
        Update the geometry for a given configuration and compute the distance between each collision pair
    """
def computeDualConeProjection(*args, **kwargs):
    """
    
    computeDualConeProjection( (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)velocities) -> numpy.ndarray :
        Project a vector on the cartesian product of dual cones.
    """
def computeForwardKinematicsDerivatives(*args, **kwargs):
    """
    
    computeForwardKinematicsDerivatives( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a) -> None :
        Computes all the terms required to compute the derivatives of the placement, spatial velocity and acceleration
        for any joint of the model.
        The results are stored in data.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        
    """
def computeFrameJacobian(*args, **kwargs):
    """
    
    computeFrameJacobian( (Model)model, (Data)data, (numpy.ndarray)q, (int)frame_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the Jacobian of the frame given by its frame_id in the coordinate system given by reference_frame.
        
    
    computeFrameJacobian( (Model)model, (Data)data, (numpy.ndarray)q, (int)frame_id) -> numpy.ndarray :
        Computes the Jacobian of the frame given by its frame_id.
        The columns of the Jacobian are expressed in the coordinates system of the Frame itself.
        In other words, the velocity of the frame vF expressed in the local coordinate is given by J*v,where v is the joint velocity.
    """
def computeFrameKinematicRegressor(*args, **kwargs):
    """
    
    computeFrameKinematicRegressor( (Model)model, (Data)data, (int)frame_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the kinematic regressor that links the joint placement variations of the whole kinematic tree to the placement variation of the frame given as input.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	frame_id: index of the frame
        	reference_frame: reference frame in which the result is expressed (LOCAL, LOCAL_WORLD_ALIGNED or WORLD)
        
    """
def computeGeneralizedGravity(*args, **kwargs):
    """
    
    computeGeneralizedGravity( (Model)model, (Data)data, (numpy.ndarray)q) -> numpy.ndarray :
        Compute the generalized gravity contribution g(q) of the Lagrangian dynamics, store the result in data.g and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        
    """
def computeGeneralizedGravityDerivatives(*args, **kwargs):
    """
    
    computeGeneralizedGravityDerivatives( (Model)model, (Data)data, (numpy.ndarray)q) -> numpy.ndarray :
        Computes the partial derivative of the generalized gravity contribution
        with respect to the joint configuration.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        Returns: dtau_statique_dq
        
    """
def computeImpulseDynamicsDerivatives(*args, **kwargs):
    """
    
    computeImpulseDynamicsDerivatives( (Model)model, (Data)data, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas [, (float)r_coeff=0 [, (ProximalSettings)prox_settings=ProximalSettings(1e-12, 1e-12, 0, 1)]]) -> None :
        Computes the impulse dynamics derivatives with contact constraints according to a given list of Contact information.
        impulseDynamics should have been called before.
    """
def computeJointJacobian(*args, **kwargs):
    """
    
    computeJointJacobian( (Model)model, (Data)data, (numpy.ndarray)q, (int)joint_id) -> numpy.ndarray :
        Computes the Jacobian of a specific joint frame expressed in the local frame of the joint according to the given input configuration.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	joint_id: index of the joint
        
    """
def computeJointJacobians(*args, **kwargs):
    """
    
    computeJointJacobians( (Model)model, (Data)data, (numpy.ndarray)q) -> numpy.ndarray :
        Computes the full model Jacobian, i.e. the stack of all the motion subspaces expressed in the coordinate world frame.
        The result is accessible through data.J. This function computes also the forward kinematics of the model.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        
    
    computeJointJacobians( (Model)model, (Data)data) -> numpy.ndarray :
        Computes the full model Jacobian, i.e. the stack of all motion subspace expressed in the world frame.
        The result is accessible through data.J. This function assumes that forward kinematics (pinocchio.forwardKinematics) has been called first.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        
    """
def computeJointJacobiansTimeVariation(*args, **kwargs):
    """
    
    computeJointJacobiansTimeVariation( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Computes the full model Jacobian variations with respect to time. It corresponds to dJ/dt which depends both on q and v. It also computes the joint Jacobian of the model (similar to computeJointJacobians).The result is accessible through data.dJ and data.J.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    """
def computeJointKinematicRegressor(*args, **kwargs):
    """
    
    computeJointKinematicRegressor( (Model)model, (Data)data, (int)joint_id, (ReferenceFrame)reference_frame, (SE3)placement) -> numpy.ndarray :
        Computes the kinematic regressor that links the joint placements variations of the whole kinematic tree to the placement variation of the frame rigidly attached to the joint and given by its placement w.r.t. to the joint frame.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	reference_frame: reference frame in which the result is expressed (LOCAL, LOCAL_WORLD_ALIGNED or WORLD)
        	placement: relative placement to the joint frame
        
    
    computeJointKinematicRegressor( (Model)model, (Data)data, (int)joint_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the kinematic regressor that links the joint placement variations of the whole kinematic tree to the placement variation of the joint given as input.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	reference_frame: reference frame in which the result is expressed (LOCAL, LOCAL_WORLD_ALIGNED or WORLD)
        
    """
def computeJointTorqueRegressor(*args, **kwargs):
    """
    
    computeJointTorqueRegressor( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a) -> numpy.ndarray :
        Compute the joint torque regressor that links the joint torque to the dynamic parameters of each link according to the current the robot motion,
        store the result in context::Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        
    """
def computeKKTContactDynamicMatrixInverse(*args, **kwargs):
    """
    
    computeKKTContactDynamicMatrixInverse( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)constraint_jacobian [, (float)damping=0]) -> numpy.ndarray :
        Computes the inverse of the constraint matrix [[M J^T], [J 0]].
    """
def computeKineticEnergy(*args, **kwargs):
    """
    
    computeKineticEnergy( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> float :
        Computes the forward kinematics and the kinematic energy of the system for the given joint configuration and velocity given as input. The result is accessible through data.kinetic_energy.
    
    computeKineticEnergy( (Model)model, (Data)data) -> float :
        Computes the kinematic energy of the system for the given joint placement and velocity stored in data. The result is accessible through data.kinetic_energy.
    """
def computeKineticEnergyRegressor(*args, **kwargs):
    """
    
    computeKineticEnergyRegressor( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Compute the kinetic energy regressor that links the kinetic energyto the dynamic parameters of each link according to the current the robot motion,
        store the result in context::Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    """
def computeMechanicalEnergy(*args, **kwargs):
    """
    
    computeMechanicalEnergy( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> float :
        Computes the forward kinematics and the kinematic energy of the system for the given joint configuration and velocity given as input. The result is accessible through data.mechanical_energy.
        A byproduct of this function is the computation of both data.kinetic_energy and data.potential_energy too.
    
    computeMechanicalEnergy( (Model)model, (Data)data) -> float :
        Computes the mechanical energy of the system for the given joint placement and velocity stored in data. The result is accessible through data.mechanical_energy.
        A byproduct of this function is the computation of both data.kinetic_energy and data.potential_energy too.
    """
def computeMinverse(*args, **kwargs):
    """
    
    computeMinverse( (Model)model, (Data)data, (numpy.ndarray)q) -> numpy.ndarray :
        Computes the inverse of the joint space inertia matrix using an extension of the Articulated Body algorithm.
        The result is stored in data.Minv.
        Parameters:
        	 model: Model of the kinematic tree
        	 data: Data related to the kinematic tree
        	 q: joint configuration (size model.nq)
    
    computeMinverse( (Model)model, (Data)data) -> numpy.ndarray :
        Computes the inverse of the joint space inertia matrix using an extension of the Articulated Body algorithm.
        The result is stored in data.Minv.
        Remarks: pinocchio.aba should have been called first.
        Parameters:
        	 model: Model of the kinematic tree
        	 data: Data related to the kinematic tree
    """
def computePotentialEnergy(*args, **kwargs):
    """
    
    computePotentialEnergy( (Model)model, (Data)data, (numpy.ndarray)q) -> float :
        Computes the potential energy of the system for the given the joint configuration given as input. The result is accessible through data.potential_energy.
    
    computePotentialEnergy( (Model)model, (Data)data) -> float :
        Computes the potential energy of the system for the given joint placement stored in data. The result is accessible through data.potential_energy.
    """
def computePotentialEnergyRegressor(*args, **kwargs):
    """
    
    computePotentialEnergyRegressor( (Model)model, (Data)data, (numpy.ndarray)q) -> numpy.ndarray :
        Compute the potential energy regressor that links the potential energyto the dynamic parameters of each link according to the current the robot motion,
        store the result in context::Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        
    """
def computePrimalFeasibility(*args, **kwargs):
    """
    
    computePrimalFeasibility( (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)forces) -> float :
        Compute the primal feasibility.
    """
def computeRNEADerivatives(*args, **kwargs):
    """
    
    computeRNEADerivatives( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a) -> tuple :
        Computes the RNEA partial derivatives, store the result in data.dtau_dq, data.dtau_dv and data.M (aka dtau_da)
        which correspond to the partial derivatives of the torque output with respect to the joint configuration,
        velocity and acceleration vectors.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        
        Returns: (dtau_dq, dtau_dv, dtau_da)
        
    
    computeRNEADerivatives( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a, (StdVec_Force)fext) -> tuple :
        Computes the RNEA partial derivatives with external contact foces,
        store the result in data.dtau_dq, data.dtau_dv and data.M (aka dtau_da)
        which correspond to the partial derivatives of the torque output with respect to the joint configuration,
        velocity and acceleration vectors.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        	fext: list of external forces expressed in the local frame of the joints (size model.njoints)
        
        Returns: (dtau_dq, dtau_dv, dtau_da)
        
    """
def computeReprojectionError(*args, **kwargs):
    """
    
    computeReprojectionError( (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)forces, (numpy.ndarray)velocities) -> float :
        Compute the reprojection error.
    """
def computeStaticRegressor(*args, **kwargs):
    """
    
    computeStaticRegressor( (Model)model, (Data)data, (numpy.ndarray)q) -> numpy.ndarray :
        Compute the static regressor that links the inertia parameters of the system to its center of mass position,
        store the result in context::Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        
    """
def computeStaticTorque(*args, **kwargs):
    """
    
    computeStaticTorque( (Model)model, (Data)data, (numpy.ndarray)q, (StdVec_Force)fext) -> numpy.ndarray :
        Computes the generalized static torque contribution g(q) - J.T f_ext of the Lagrangian dynamics, store the result in data.tau and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	fext: list of external forces expressed in the local frame of the joints (size model.njoints)
        
    """
def computeStaticTorqueDerivatives(*args, **kwargs):
    """
    
    computeStaticTorqueDerivatives( (Model)model, (Data)data, (numpy.ndarray)q, (StdVec_Force)fext) -> numpy.ndarray :
        Computes the partial derivative of the generalized gravity and external forces contributions (a.k.a static torque vector)
        with respect to the joint configuration.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	fext: list of external forces expressed in the local frame of the joints (size model.njoints)
        Returns: dtau_statique_dq
        
    """
def computeSubtreeMasses(*args, **kwargs):
    """
    
    computeSubtreeMasses( (Model)model, (Data)data) -> None :
        Compute the mass of each kinematic subtree and store it in the vector data.mass.
    """
def computeSupportedForceByFrame(*args, **kwargs):
    """
    
    computeSupportedForceByFrame( (Model)model, (Data)data, (int)frame_id) -> Force :
        Computes the supported force of the frame (given by frame_id) and returns it.
        The supported force corresponds to the sum of all the forces experienced after the given frame.
        You must first call pinocchio::rnea to update placement values in data structure.
    """
def computeSupportedInertiaByFrame(*args, **kwargs):
    """
    
    computeSupportedInertiaByFrame( (Model)model, (Data)data, (int)frame_id, (bool)with_subtree) -> Inertia :
        Computes the supported inertia by the frame (given by frame_id) and returns it.
        The supported inertia corresponds to the sum of the inertias of all the child frames (that belongs to the same joint body) and the child joints, if with_subtree=True.
        You must first call pinocchio::forwardKinematics to update placement values in data structure.
    """
def computeTotalMass(*args, **kwargs):
    """
    
    computeTotalMass( (Model)model) -> float :
        Compute the total mass of the model and return it.
    
    computeTotalMass( (Model)model, (Data)data) -> float :
        Compute the total mass of the model, put it in data.mass[0] and return it.
    """
def constraintDynamics(*args, **kwargs):
    """
    
    constraintDynamics( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)tau, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas [, (ProximalSettings)prox_settings]) -> numpy.ndarray :
        Computes the forward dynamics with contact constraints according to a given list of Contact information.
        When using constraintDynamics for the first time, you should call first initConstraintDynamics to initialize the internal memory used in the algorithm.
        This function returns joint acceleration of the system. The contact forces are stored in the list data.contact_forces.
    """
def contactInverseDynamics(*args, **kwargs):
    """
    
    contactInverseDynamics( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a, (float)dt, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas, (StdVec_CoulombFrictionCone)cones, (numpy.ndarray)R, (numpy.ndarray)constraint_correction, (ProximalSettings)settings [, (numpy.ndarray)lambda_guess=None]) -> numpy.ndarray :
        Compute the inverse dynamics with frictional contacts, store the result in Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        	dt: the time step
        	contact_models: list of contact models
        	contact_datas: list of contact datas
        	cones: list of friction cones
        	R: vector representing the diagonal of the compliance matrix
        	constraint_correction: vector representing the constraint correction
        	settings: the settings of the proximal algorithm
        	lambda_guess: initial guess for contact forces
        
    """
def crba(*args, **kwargs):
    """
    
    crba( (Model)model, (Data)data, (numpy.ndarray)q [, (Convention)convention=pinocchio.pinocchio_pywrap_default.Convention.LOCAL]) -> numpy.ndarray :
        Computes CRBA, store the result in Data and return it.
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	 convention: Convention to use
    """
def dDifference(*args, **kwargs):
    """
    
    dDifference( (Model)model, (numpy.ndarray)q1, (numpy.ndarray)q2) -> tuple :
        Computes the partial derivatives of the difference function with respect to the first and the second argument, and returns the two Jacobians as a tuple.
        
        Parameters:
        	model: model of the kinematic tree
        	q1: the initial joint configuration vector (size model.nq)
        	q2: the terminal joint configuration vector (size model.nq)
        
    
    dDifference( (Model)model, (numpy.ndarray)q1, (numpy.ndarray)q2, (ArgumentPosition)argument_position) -> numpy.ndarray :
        Computes the partial derivatives of the difference function with respect to the first (arg == ARG0) or the second argument (arg == ARG1).
        
        Parameters:
        	model: model of the kinematic tree
        	q1: the initial joint configuration vector (size model.nq)
        	q2: the terminal joint configuration vector (size model.nq)
        	argument_position: either pinocchio.ArgumentPosition.ARG0 or pinocchio.ArgumentPosition.ARG1, depending on the desired Jacobian value.
        
    """
def dIntegrate(*args, **kwargs):
    """
    
    dIntegrate( (Model)model, (numpy.ndarray)q, (numpy.ndarray)v) -> tuple :
        Computes the partial derivatives of the integrate function with respect to the first and the second argument, and returns the two Jacobians as a tuple.
        
        Parameters:
        	model: model of the kinematic tree
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    
    dIntegrate( (Model)model, (numpy.ndarray)q, (numpy.ndarray)v, (ArgumentPosition)argument_position) -> numpy.ndarray :
        Computes the partial derivatives of the integrate function with respect to the first (arg == ARG0) or the second argument (arg == ARG1).
        
        Parameters:
        	model: model of the kinematic tree
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	argument_position: either pinocchio.ArgumentPosition.ARG0 or pinocchio.ArgumentPosition.ARG1, depending on the desired Jacobian value.
        
    """
def dIntegrateTransport(*args, **kwargs):
    """
    
    dIntegrateTransport( (Model)model, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)Jin, (ArgumentPosition)argument_position) -> numpy.ndarray :
        Takes a matrix expressed at q (+) v and uses parallel transport to express it in the tangent space at q.	This operation does the product of the matrix by the Jacobian of the integration operation, but more efficiently.Parameters:
        	model: model of the kinematic tree
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	Jin: the input matrix (row size model.nv)	argument_position: either pinocchio.ArgumentPosition.ARG0 (q) or pinocchio.ArgumentPosition.ARG1 (v), depending on the desired Jacobian value.
        
    """
def dccrba(*args, **kwargs):
    """
    
    dccrba( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Computes the time derivative of the centroidal momentum matrix Ag in terms of q and v.
        For the same price, it also computes the centroidal momentum matrix (data.Ag), the total joint jacobians (data.J) and the related joint jacobians time derivative (data.dJ)
    """
def difference(*args, **kwargs):
    """
    
    difference( (Model)model, (numpy.ndarray)q1, (numpy.ndarray)q2) -> numpy.ndarray :
        Difference between two joint configuration vectors, i.e. the tangent vector that must be integrated during one unit timeto go from q1 to q2.
        
        Parameters:
        	model: model of the kinematic tree
        	q1: the initial joint configuration vector (size model.nq)
        	q2: the terminal joint configuration vector (size model.nq)
        
    """
def distance(*args, **kwargs):
    """
    
    distance( (Model)model, (numpy.ndarray)q1, (numpy.ndarray)q2) -> float :
        Distance between two joint configuration vectors.
        
        Parameters:
        	model: model of the kinematic tree
        	q1: the initial joint configuration vector (size model.nq)
        	q2: the terminal joint configuration vector (size model.nq)
        
    """
def exp3(*args, **kwargs):
    """
    
    exp3( (numpy.ndarray)w) -> numpy.ndarray :
        Exp: so3 -> SO3. Return the integral of the input vector w during time 1. This is also known as the Rodrigues formula.
    """
def exp3_quat(*args, **kwargs):
    """
    
    exp3_quat( (numpy.ndarray)w) -> numpy.ndarray :
        Exp: so3 -> S3. Returns the integral of the input vector w during time 1, represented as a unit Quaternion.
    """
def exp6(*args, **kwargs):
    """
    
    exp6( (Motion)motion) -> SE3 :
        Exp: se3 -> SE3. Return the integral of the input spatial velocity during time 1.
    
    exp6( (numpy.ndarray)v) -> SE3 :
        Exp: se3 -> SE3. Return the integral of the input spatial velocity during time 1.
    """
def exp6_quat(*args, **kwargs):
    """
    
    exp6_quat( (numpy.ndarray)v) -> numpy.ndarray :
        Exp: se3 -> R3 * S3. Return the integral of the input 6D spatial velocity over unit time, using quaternion to represent rotation as in the standard configuration layout for the Lie group SE3.
    """
def findCommonAncestor(*args, **kwargs):
    """
    
    findCommonAncestor( (Model)model, (int)joint1_id, (int)joint2_id) -> tuple :
        Computes the common ancestor between two joints belonging to the same kinematic tree.
        
        Parameters:
        	model: input model
        	joint1_id: index of the first joint
        	joint2_id: index of the second joint
        Returns a tuple containing the index of the common joint ancestor, the position of this ancestor in model.supports[joint1_id] and model.supports[joint2_id].
        
    """
def forwardDynamics(*args, **kwargs):
    """
    
    forwardDynamics( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)tau, (numpy.ndarray)constraint_jacobian, (numpy.ndarray)constraint_drift [, (float)damping=0]) -> numpy.ndarray :
        Solves the constrained dynamics problem with contacts, puts the result in context::Data::ddq and return it. The contact forces are stored in data.lambda_c.
        Note: internally, pinocchio.computeAllTerms is called.
    
    forwardDynamics( (Model)model, (Data)data, (numpy.ndarray)tau, (numpy.ndarray)constraint_jacobian, (numpy.ndarray)constraint_drift [, (float)damping=0]) -> numpy.ndarray :
        Solves the forward dynamics problem with contacts, puts the result in context::Data::ddq and return it. The contact forces are stored in data.lambda_c.
        Note: this function assumes that pinocchio.computeAllTerms has been called first.
    """
def forwardKinematics(*args, **kwargs):
    """
    
    forwardKinematics( (Model)model, (Data)data, (numpy.ndarray)q) -> None :
        Compute the global placements of all the joints of the kinematic tree and store the results in data.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        
    
    forwardKinematics( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> None :
        Compute the global placements and local spatial velocities of all the joints of the kinematic tree and store the results in data.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    
    forwardKinematics( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a) -> None :
        Compute the global placements, local spatial velocities and spatial accelerations of all the joints of the kinematic tree and store the results in data.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        
    """
def frameBodyRegressor(*args, **kwargs):
    """
    
    frameBodyRegressor( (Model)model, (Data)data, (int)frame_id) -> numpy.ndarray :
        Computes the regressor for the dynamic parameters of a rigid body attached to a given frame.
        This algorithm assumes RNEA has been run to compute the acceleration and gravitational effects.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	frame_id: index of the frame
        
    """
def frameJacobianTimeVariation(*args, **kwargs):
    """
    
    frameJacobianTimeVariation( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (int)frame_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the Jacobian Time Variation of the frame given by its frame_id either in the reference frame provided by reference_frame.
        
    """
def framesForwardKinematics(*args, **kwargs):
    """
    
    framesForwardKinematics( (Model)model, (Data)data, (numpy.ndarray)q) -> None :
        Calls first the forwardKinematics(model,data,q) and then update the Frame placement quantities (data.oMf).
    """
def getAcceleration(*args, **kwargs):
    """
    
    getAcceleration( (Model)model, (Data)data, (int)joint_id [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the spatial acceleration of the joint expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v,a) should be called first to compute the joint spatial acceleration stored in data.a .
    """
def getCenterOfMassVelocityDerivatives(*args, **kwargs):
    """
    
    getCenterOfMassVelocityDerivatives( (Model)model, (Data)data) -> numpy.ndarray :
        Computes the partial derivaties of the center of mass velocity with respect to
        the joint configuration.
        You must first call computeAllTerms(model,data,q,v) or centerOfMass(model,data,q,v) before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        
    """
def getCentroidalDynamicsDerivatives(*args, **kwargs):
    """
    
    getCentroidalDynamicsDerivatives( (Model)model, (Data)data) -> tuple :
        Retrive the analytical derivatives of the centroidal dynamics
        from the RNEA derivatives.
        pinocchio.computeRNEADerivatives should have been called first.
    """
def getClassicalAcceleration(*args, **kwargs):
    """
    
    getClassicalAcceleration( (Model)model, (Data)data, (int)joint_id [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the "classical" acceleration of the joint expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v,a) should be called first to compute the joint spatial acceleration stored in data.a .
    """
def getConstraintJacobian(*args, **kwargs):
    """
    
    getConstraintJacobian( (Model)model, (Data)data, (RigidConstraintModel)contact_model, (RigidConstraintData)contact_data) -> numpy.ndarray :
        Computes the kinematic Jacobian associatied to a given constraint model.
    """
def getConstraintsJacobian(*args, **kwargs):
    """
    
    getConstraintsJacobian( (Model)model, (Data)data, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas) -> numpy.ndarray :
        Computes the kinematic Jacobian associatied to a given set of constraint models.
    """
def getCoriolisMatrix(*args, **kwargs):
    """
    
    getCoriolisMatrix( (Model)model, (Data)data) -> numpy.ndarray :
        Retrives the Coriolis Matrix C(q,v) of the Lagrangian dynamics after calling one of the derivative algorithms, store the result in data.C and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        
    """
def getFrameAcceleration(*args, **kwargs):
    """
    
    getFrameAcceleration( (Model)model, (Data)data, (int)frame_id [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the spatial acceleration of the frame expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v,a) should be called first to compute the joint spatial acceleration stored in data.a .
    
    getFrameAcceleration( (Model)model, (Data)data, (int)joint_id, (SE3)placement [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the spatial acceleration of the frame expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v,a) should be called first to compute the joint spatial acceleration stored in data.a .
    """
def getFrameAccelerationDerivatives(*args, **kwargs):
    """
    
    getFrameAccelerationDerivatives( (Model)model, (Data)data, (int)frame_id, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the spatial acceleration of a given frame with respect to
        the joint configuration, velocity and acceleration and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	frame_id: index of the frame
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    
    getFrameAccelerationDerivatives( (Model)model, (Data)data, (int)joint_id, (SE3)placement, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the spatial acceleration of a frame given by its relative placement, with respect to
        the joint configuration, velocity and acceleration and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	placement: placement of the Frame w.r.t. the joint frame.
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getFrameClassicalAcceleration(*args, **kwargs):
    """
    
    getFrameClassicalAcceleration( (Model)model, (Data)data, (int)frame_id [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the "classical" acceleration of the frame expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v,a) should be called first to compute the joint spatial acceleration stored in data.a .
    
    getFrameClassicalAcceleration( (Model)model, (Data)data, (int)joint_id, (SE3)placement [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the "classical" acceleration of the frame expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v,a) should be called first to compute the joint spatial acceleration stored in data.a .
    """
def getFrameJacobian(*args, **kwargs):
    """
    
    getFrameJacobian( (Model)model, (Data)data, (int)frame_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the Jacobian of the frame given by its ID either in the LOCAL, LOCAL_WORLD_ALIGNED or the WORLD coordinates systems.
        In other words, the velocity of the frame vF expressed in the reference frame is given by J*v,where v is the joint velocity vector.
        remarks: computeJointJacobians(model,data,q) must have been called first.
    
    getFrameJacobian( (Model)model, (Data)data, (int)joint_id, (SE3)placement, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the Jacobian of the frame given by its placement with respect to the Joint frame and expressed the solution either in the LOCAL, LOCAL_WORLD_ALIGNED or the WORLD coordinates systems.
        In other words, the velocity of the frame vF expressed in the reference frame is given by J*v,where v is the joint velocity vector.
        
        remarks: computeJointJacobians(model,data,q) must have been called first.
    """
def getFrameJacobianTimeVariation(*args, **kwargs):
    """
    
    getFrameJacobianTimeVariation( (Model)model, (Data)data, (int)frame_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Returns the Jacobian time variation of the frame given by its frame_id either in the reference frame provided by reference_frame.
        You have to call computeJointJacobiansTimeVariation(model,data,q,v) and updateFramePlacements(model,data) first.
    """
def getFrameVelocity(*args, **kwargs):
    """
    
    getFrameVelocity( (Model)model, (Data)data, (int)frame_id [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the spatial velocity of the frame expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v[,a]) should be called first to compute the joint spatial velocity stored in data.v
    
    getFrameVelocity( (Model)model, (Data)data, (int)joint_id, (SE3)placement [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the spatial velocity of the frame expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v[,a]) should be called first to compute the joint spatial velocity stored in data.v
    """
def getFrameVelocityDerivatives(*args, **kwargs):
    """
    
    getFrameVelocityDerivatives( (Model)model, (Data)data, (int)frame_id, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the spatial velocity of a given frame with respect to
        the joint configuration and velocity and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	frame_id: index of the frame
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    
    getFrameVelocityDerivatives( (Model)model, (Data)data, (int)joint_id, (SE3)placement, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the spatial velocity of a frame given by its relative placement, with respect to
        the joint configuration and velocity and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	placement: placement of the Frame w.r.t. the joint frame.
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getJacobianSubtreeCenterOfMass(*args, **kwargs):
    """
    
    getJacobianSubtreeCenterOfMass( (Model)model, (Data)data, (int)subtree_root_joint_id) -> numpy.ndarray :
        Get the Jacobian of the CoM of the given subtree expressed in the world frame, according to the given entries in data. It assumes that jacobianCenterOfMass has been called first.
    """
def getJointAccelerationDerivatives(*args, **kwargs):
    """
    
    getJointAccelerationDerivatives( (Model)model, (Data)data, (int)joint_id, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the spatial acceleration of a given joint with respect to
        the joint configuration, velocity and acceleration and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getJointJacobian(*args, **kwargs):
    """
    
    getJointJacobian( (Model)model, (Data)data, (int)joint_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the jacobian of a given given joint according to the given entries in data.
        If reference_frame is set to LOCAL, it returns the Jacobian expressed in the local coordinate system of the joint.
        If reference_frame is set to LOCAL_WORLD_ALIGNED, it returns the Jacobian expressed in the coordinate system of the frame centered on the joint, but aligned with the WORLD axes.
        If reference_frame is set to WORLD, it returns the Jacobian expressed in the coordinate system of the frame associated to the WORLD.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getJointJacobianTimeVariation(*args, **kwargs):
    """
    
    getJointJacobianTimeVariation( (Model)model, (Data)data, (int)joint_id, (ReferenceFrame)reference_frame) -> numpy.ndarray :
        Computes the Jacobian time variation of a specific joint expressed in the requested frame provided by the value of reference_frame.You have to call computeJointJacobiansTimeVariation first. This function also computes the full model Jacobian contained in data.J.
        If reference_frame is set to LOCAL, it returns the Jacobian expressed in the local coordinate system of the joint.
        If reference_frame is set to LOCAL_WORLD_ALIGNED, it returns the Jacobian expressed in the coordinate system of the frame centered on the joint, but aligned with the WORLD axes.
        If reference_frame is set to WORLD, it returns the Jacobian expressed in the coordinate system of the frame associated to the WORLD.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getJointVelocityDerivatives(*args, **kwargs):
    """
    
    getJointVelocityDerivatives( (Model)model, (Data)data, (int)joint_id, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the spatial velocity of a given joint with respect to
        the joint configuration and velocity and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getKKTContactDynamicMatrixInverse(*args, **kwargs):
    """
    
    getKKTContactDynamicMatrixInverse( (Model)model, (Data)data, (numpy.ndarray)constraint_jacobian) -> numpy.ndarray :
        Computes the inverse of the constraint matrix [[M Jt], [J 0]].
         forwardDynamics or impulseDynamics must have been called first.
        Note: the constraint Jacobian should be the same that was provided to forwardDynamics or impulseDynamics.
    """
def getPointClassicAccelerationDerivatives(*args, **kwargs):
    """
    
    getPointClassicAccelerationDerivatives( (Model)model, (Data)data, (int)joint_id, (SE3)placement, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the classic acceleration of a point given by its placement information w.r.t. the joint frame and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	placement: relative placement of the point w.r.t. the joint frame
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getPointVelocityDerivatives(*args, **kwargs):
    """
    
    getPointVelocityDerivatives( (Model)model, (Data)data, (int)joint_id, (SE3)placement, (ReferenceFrame)reference_frame) -> tuple :
        Computes the partial derivatives of the velocity of a point given by its placement information w.r.t. the joint frame and returns them as a tuple.
        The partial derivatives can be either expressed in the LOCAL frame of the joint, in the LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of reference_frame.
        You must first call computeForwardKinematicsDerivatives before calling this function.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        	placement: relative placement of the point w.r.t. the joint frame
        	reference_frame: reference frame in which the resulting derivatives are expressed
        
    """
def getVelocity(*args, **kwargs):
    """
    
    getVelocity( (Model)model, (Data)data, (int)joint_id [, (ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> Motion :
        Returns the spatial velocity of the joint expressed in the coordinate system given by reference_frame.
        forwardKinematics(model,data,q,v[,a]) should be called first to compute the joint spatial velocity stored in data.v
    """
def impulseDynamics(*args, **kwargs):
    """
    
    impulseDynamics( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v_before, (numpy.ndarray)constraint_jacobian [, (float)restitution_coefficient=0 [, (float)damping=0]]) -> numpy.ndarray :
        Solves the impact dynamics problem with contacts, store the result in context::Data::dq_after and return it. The contact impulses are stored in data.impulse_c.
        Note: internally, pinocchio.crba is called.
    
    impulseDynamics( (Model)model, (Data)data, (numpy.ndarray)v_before, (numpy.ndarray)constraint_jacobian [, (float)restitution_coefficient=0 [, (float)damping=0]]) -> numpy.ndarray :
        Solves the impact dynamics problem with contacts, store the result in context::Data::dq_after and return it. The contact impulses are stored in data.impulse_c.
        Note: this function assumes that pinocchio.crba has been called first.
    
    impulseDynamics( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (StdVec_RigidConstraintModel)contact_models, (StdVec_RigidConstraintData)contact_datas [, (float)r_coeff=0 [, (ProximalSettings)prox_settings=ProximalSettings(1e-12, 1e-12, 0, 1)]]) -> numpy.ndarray :
        Computes the impulse dynamics with contact constraints according to a given list of Contact information.
        When using impulseDynamics for the first time, you should call first initConstraintDynamics to initialize the internal memory used in the algorithm.
        This function returns the after-impulse velocity of the system. The impulses acting on the contacts are stored in the list data.contact_forces.
    """
def initConstraintDynamics(*args, **kwargs):
    """
    
    initConstraintDynamics( (Model)model, (Data)data, (StdVec_RigidConstraintModel)contact_models) -> None :
        This function allows to allocate the memory before hand for contact dynamics algorithms.
        This allows to avoid online memory allocation when running these algorithms.
    """
def integrate(*args, **kwargs):
    """
    
    integrate( (Model)model, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Integrate the joint configuration vector q with a tangent vector v during one unit time.
        This is the canonical integrator of a Configuration Space composed of Lie groups, such as most robots.
        
        Parameters:
        	model: model of the kinematic tree
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    """
def interpolate(*args, **kwargs):
    """
    
    interpolate( (Model)model, (numpy.ndarray)q1, (numpy.ndarray)q2, (float)alpha) -> numpy.ndarray :
        Interpolate between two given joint configuration vectors q1 and q2.
        
        Parameters:
        	model: model of the kinematic tree
        	q1: the initial joint configuration vector (size model.nq)
        	q2: the terminal joint configuration vector (size model.nq)
        	alpha: the interpolation coefficient in [0,1]
        
    """
def isNormalized(*args, **kwargs):
    """
    
    isNormalized( (Model)model, (numpy.ndarray)q [, (float)prec=1e-12]) -> bool :
        Check whether a configuration vector is normalized within the given precision provided by prec.
        
        Parameters:
        	model: model of the kinematic tree
        	q: a joint configuration vector (size model.nq)
        	prec: requested accuracy for the check
        
    """
def isSameConfiguration(*args, **kwargs):
    """
    
    isSameConfiguration( (Model)model, (numpy.ndarray)q1, (numpy.ndarray)q2, (float)prec) -> bool :
        Return true if two configurations are equivalent within the given precision provided by prec.
        
        Parameters:
        	model: model of the kinematic tree
        	q1: a joint configuration vector (size model.nq)
        	q2: a joint configuration vector (size model.nq)
        	prec: requested accuracy for the comparison
        
    """
def jacobianCenterOfMass(*args, **kwargs):
    """
    
    jacobianCenterOfMass( (Model)model, (Data)data, (numpy.ndarray)q [, (bool)compute_subtree_coms=True]) -> numpy.ndarray :
        Computes the Jacobian of the center of mass, puts the result in context::Data and return it.
        If compute_subtree_coms is True, the algorithm also computes the center of mass of the subtrees.
    
    jacobianCenterOfMass( (Model)model, (Data)data [, (bool)compute_subtree_coms=True]) -> numpy.ndarray :
        Computes the Jacobian of the center of mass, puts the result in context::Data and return it.
        If compute_subtree_coms is True, the algorithm also computes the center of mass of the subtrees.
    """
def jacobianSubtreeCenterOfMass(*args, **kwargs):
    """
    
    jacobianSubtreeCenterOfMass( (Model)model, (Data)data, (numpy.ndarray)q, (int)subtree_root_joint_id) -> numpy.ndarray :
        Computes the Jacobian of the CoM of the given subtree (subtree_root_joint_id) expressed in the WORLD frame, according to the given joint configuration.
    
    jacobianSubtreeCenterOfMass( (Model)model, (Data)data, (int)subtree_root_joint_id) -> numpy.ndarray :
        Computes the Jacobian of the CoM of the given subtree (subtree_root_joint_id) expressed in the WORLD frame, according to the given entries in data.
    """
def jointBodyRegressor(*args, **kwargs):
    """
    
    jointBodyRegressor( (Model)model, (Data)data, (int)joint_id) -> numpy.ndarray :
        Compute the regressor for the dynamic parameters of a rigid body attached to a given joint.
        This algorithm assumes RNEA has been run to compute the acceleration and gravitational effects.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	joint_id: index of the joint
        
    """
def loadReferenceConfigurations(*args, **kwargs):
    """
    
    loadReferenceConfigurations( (Model)model, (object)srdf_filename [, (bool)verbose=False]) -> None :
        Retrieve all the reference configurations of a given model from the SRDF file.
        Parameters:
        	model: model of the robot
        	srdf_filename: path to the SRDF file containing the reference configurations
        	verbose: [optional] display to the current terminal some internal information
    """
def loadReferenceConfigurationsFromXML(*args, **kwargs):
    """
    
    loadReferenceConfigurationsFromXML( (Model)model, (str)srdf_xml_stream [, (bool)verbose=False]) -> None :
        Retrieve all the reference configurations of a given model from the SRDF file.
        Parameters:
        	model: model of the robot
        	srdf_xml_stream: XML stream containing the SRDF information with the reference configurations
        	verbose: [optional] display to the current terminal some internal information
    """
def loadRotorParameters(*args, **kwargs):
    """
    
    loadRotorParameters( (Model)model, (object)srdf_filename [, (bool)verbose=False]) -> bool :
        Load the rotor parameters of a given model from a SRDF file.
        Results are stored in model.rotorInertia and model.rotorGearRatio.This function also fills the armature of the model.Parameters:
        	model: model of the robot
        	srdf_filename: path to the SRDF file containing the rotor parameters
        	verbose: [optional] display to the current terminal some internal information
    """
def log3(*args, **kwargs):
    """
    
    log3( (numpy.ndarray)R) -> numpy.ndarray :
        Log: SO3 -> so3 is the pseudo-inverse of Exp: so3 -> SO3. Log maps from SO3 -> { v in so3, ||v|| < 2pi }.
    
    log3( (numpy.ndarray)R, (numpy.ndarray)theta) -> numpy.ndarray :
        Log: SO3 -> so3 is the pseudo-inverse of Exp: so3 -> SO3. Log maps from SO3 -> { v in so3, ||v|| < 2pi }.
        It also returns the angle of rotation theta around the rotation axis.
    
    log3( (numpy.ndarray)R, (float)theta) -> numpy.ndarray :
        Log: SO3 -> so3 is the pseudo-inverse of Exp: so3 -> SO3. Log maps from SO3 -> { v in so3, ||v|| < 2pi }.
        It also returns the angle of rotation theta around the rotation axis.
    
    log3( (coal.coal_pywrap.Quaternion)quat) -> numpy.ndarray :
        Log: S^3 -> so3 is the pseudo-inverse of Exp: so3 -> S^3, the exponential map from so3 to the unit quaternions. It maps from S^3 -> { v in so3, ||v|| < 2pi }.
    
    log3( (numpy.ndarray)quat) -> numpy.ndarray :
        Log: S^3 -> so3 is the pseudo-inverse of Exp: so3 -> S^3, the exponential map from so3 to the unit quaternions. It maps from S^3 -> { v in so3, ||v|| < 2pi }.
    
    log3( (numpy.ndarray)quat, (numpy.ndarray)theta) -> numpy.ndarray :
        Log: S^3 -> so3 is the pseudo-inverse of Exp: so3 -> S^3, the exponential map from so3 to the unit quaternions. It maps from S^3 -> { v in so3, ||v|| < 2pi }.
        It also returns the angle of rotation theta around the rotation axis.
    
    log3( (numpy.ndarray)quat, (float)theta) -> numpy.ndarray :
        Log: S^3 -> so3 is the pseudo-inverse of Exp: so3 -> S^3, the exponential map from so3 to the unit quaternions. It maps from S^3 -> { v in so3, ||v|| < 2pi }.
        It also returns the angle of rotation theta around the rotation axis.
    """
def log6(*args, **kwargs):
    """
    
    log6( (SE3)M) -> Motion :
        Log: SE3 -> se3. Pseudo-inverse of exp from SE3 -> { v,w in se3, ||w|| < 2pi }.
    
    log6( (numpy.ndarray)homegeneous_matrix) -> Motion :
        Log: SE3 -> se3. Pseudo-inverse of Exp: so3 -> SO3. Log maps from SE3 -> { v,w in se3, ||w|| < 2pi }.
    """
def log6_quat(*args, **kwargs):
    """
    
    log6_quat( (numpy.ndarray)q) -> Motion :
        Log: R^3 * S^3 -> se3. Pseudo-inverse of Exp: se3 -> R^3 * S^3,
    """
def neutral(*args, **kwargs):
    """
    
    neutral( (Model)model) -> numpy.ndarray :
        Returns the neutral configuration vector associated to the model.
        
        Parameters:
        	model: model of the kinematic tree
        
    """
def nonLinearEffects(*args, **kwargs):
    """
    
    nonLinearEffects( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v) -> numpy.ndarray :
        Compute the Non Linear Effects (coriolis, centrifugal and gravitational effects), store the result in Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        
    """
def normalize(*args, **kwargs):
    """
    
    normalize( (Model)model, (numpy.ndarray)q) -> numpy.ndarray :
        Returns the configuration normalized.
        For instance, when the configuration vectors contains some quaternion values, it must be required to renormalize these components to keep orthonormal rotation values.
        
        Parameters:
        	model: model of the kinematic tree
        	q: a joint configuration vector to normalize (size model.nq)
        
    """
def printVersion(*args, **kwargs):
    """
    
    printVersion([  (str)delimiter='.']) -> str :
        Returns the current version of Pinocchio as a string.
        The user may specify the delimiter between the different semantic numbers.
    """
def randomConfiguration(*args, **kwargs):
    """
    
    randomConfiguration( (Model)model) -> numpy.ndarray :
        Generate a random configuration in the bounds given by the lower and upper limits contained in model.
        
        Parameters:
        	model: model of the kinematic tree
        
    
    randomConfiguration( (Model)model, (numpy.ndarray)lower_bound, (numpy.ndarray)upper_bound) -> numpy.ndarray :
        Generate a random configuration in the bounds given by the Joint lower and upper limits arguments.
        
        Parameters:
        	model: model of the kinematic tree
        	lower_bound: the lower bound on the joint configuration vectors (size model.nq)
        	upper_bound: the upper bound on the joint configuration vectors (size model.nq)
        
    """
def removeCollisionPairs(*args, **kwargs):
    """
    
    removeCollisionPairs( (Model)model, (GeometryModel)geom_model, (object)srdf_filename [, (bool)verbose=False]) -> None :
        Parse an SRDF file in order to remove some collision pairs for a specific GeometryModel.
        Parameters:
        Parameters:
        	model: model of the robot
        	geom_model: geometry model of the robot
        	srdf_filename: path to the SRDF file containing the collision pairs to remove
        	verbose: [optional] display to the current terminal some internal information
    """
def removeCollisionPairsFromXML(*args, **kwargs):
    """
    
    removeCollisionPairsFromXML( (Model)model, (GeometryModel)geom_model, (str)srdf_xml_stream [, (bool)verbose=False]) -> None :
        Parse an SRDF file in order to remove some collision pairs for a specific GeometryModel.
        Parameters:
        Parameters:
        	model: model of the robot
        	geom_model: geometry model of the robot
        	srdf_xml_stream: XML stream containing the SRDF information with the collision pairs to remove
        	verbose: [optional] display to the current terminal some internal information
    """
def rnea(*args, **kwargs):
    """
    
    rnea( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a) -> numpy.ndarray :
        Compute the RNEA, store the result in Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        
    
    rnea( (Model)model, (Data)data, (numpy.ndarray)q, (numpy.ndarray)v, (numpy.ndarray)a, (StdVec_Force)fext) -> numpy.ndarray :
        Compute the RNEA with external forces, store the result in Data and return it.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        	q: the joint configuration vector (size model.nq)
        	v: the joint velocity vector (size model.nv)
        	a: the joint acceleration vector (size model.nv)
        	fext: list of external forces expressed in the local frame of the joints (size model.njoints)
        
    """
def seed(*args, **kwargs):
    """
    
    seed( (int)seed_value) -> None :
        Initialize the pseudo-random number generator with the argument seed_value.
    """
def sharedMemory(*args, **kwargs):
    """
    
    sharedMemory( (bool)value) -> None :
        Share the memory when converting from Eigen to Numpy.
    
    sharedMemory() -> bool :
        Status of the shared memory when converting from Eigen to Numpy.
        If True, the memory is shared when converting an Eigen::Matrix to a numpy.array.
        Otherwise, a deep copy of the Eigen::Matrix is performed.
    """
def skew(*args, **kwargs):
    """
    
    skew( (numpy.ndarray)u) -> numpy.ndarray :
        Computes the skew representation of a given 3d vector, i.e. the antisymmetric matrix representation of the cross product operator, aka U = [u]x.
        Parameters:
        	u: the input vector of dimension 3
    """
def skewSquare(*args, **kwargs):
    """
    
    skewSquare( (numpy.ndarray)u, (numpy.ndarray)v) -> numpy.ndarray :
        Computes the skew square representation of two given 3d vectors, i.e. the antisymmetric matrix representation of the chained cross product operator, u x (v x w), where w is another 3d vector.
        Parameters:
        	u: the first input vector of dimension 3
        	v: the second input vector of dimension 3
    """
def squaredDistance(*args, **kwargs):
    """
    
    squaredDistance( (Model)model, (numpy.ndarray)q1, (numpy.ndarray)q2) -> numpy.ndarray :
        Squared distance vector between two joint configuration vectors.
        
        Parameters:
        	model: model of the kinematic tree
        	q1: the initial joint configuration vector (size model.nq)
        	q2: the terminal joint configuration vector (size model.nq)
        
    """
def unSkew(*args, **kwargs):
    """
    
    unSkew( (numpy.ndarray)U) -> numpy.ndarray :
        Inverse of skew operator. From a given skew symmetric matrix U (i.e U = -U.T)of dimension 3x3, it extracts the supporting vector, i.e. the entries of U.
        Mathematically speacking, it computes v such that U.dot(x) = cross(u, x).
        Parameters:
        	U: the input skew symmetric matrix of dimension 3x3.
    """
def updateFramePlacement(*args, **kwargs):
    """
    
    updateFramePlacement( (Model)model, (Data)data, (int)frame_id) -> SE3 :
        Computes the placement of the given operational frame (frame_id) according to the current joint placement stored in data, stores the results in data and returns it.
    """
def updateFramePlacements(*args, **kwargs):
    """
    
    updateFramePlacements( (Model)model, (Data)data) -> None :
        Computes the placements of all the operational frames according to the current joint placement stored in dataand puts the results in data.
    """
def updateGeometryPlacements(*args, **kwargs):
    """
    
    updateGeometryPlacements( (Model)model, (Data)data, (GeometryModel)geometry_model, (GeometryData)geometry_data, (numpy.ndarray)q) -> None :
        Update the placement of the collision objects according to the current configuration.
        The algorithm also updates the current placement of the joint in Data.
    
    updateGeometryPlacements( (Model)model, (Data)data, (GeometryModel)geometry_model, (GeometryData)geometry_data) -> None :
        Update the placement of the collision objects according to the current joint placement stored in data.
    """
def updateGlobalPlacements(*args, **kwargs):
    """
    
    updateGlobalPlacements( (Model)model, (Data)data) -> None :
        Updates the global placements of all joint frames of the kinematic tree and store the results in data according to the relative placements of the joints.
        
        Parameters:
        	model: model of the kinematic tree
        	data: data related to the model
        
    """
ACCELERATION: KinematicLevel  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.ACCELERATION
ARG0: ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG0
ARG1: ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG1
ARG2: ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG2
ARG3: ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG3
ARG4: ArgumentPosition  # value = pinocchio.pinocchio_pywrap_default.ArgumentPosition.ARG4
BODY: FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.BODY
COLLISION: GeometryType  # value = pinocchio.pinocchio_pywrap_default.GeometryType.COLLISION
FIXED_JOINT: FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.FIXED_JOINT
JOINT: FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.JOINT
LOCAL: ReferenceFrame  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL
LOCAL_WORLD_ALIGNED: ReferenceFrame  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL_WORLD_ALIGNED
OP_FRAME: FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.OP_FRAME
PINOCCHIO_MAJOR_VERSION: int = 3
PINOCCHIO_MINOR_VERSION: int = 4
PINOCCHIO_PATCH_VERSION: int = 0
POSITION: KinematicLevel  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.POSITION
SENSOR: FrameType  # value = pinocchio.pinocchio_pywrap_default.FrameType.SENSOR
VELOCITY: KinematicLevel  # value = pinocchio.pinocchio_pywrap_default.KinematicLevel.VELOCITY
VISUAL: GeometryType  # value = pinocchio.pinocchio_pywrap_default.GeometryType.VISUAL
WITH_CPPAD: bool = False
WITH_HPP_FCL: bool = True
WITH_OPENMP: bool = False
WITH_SDFORMAT: bool = False
WITH_URDFDOM: bool = True
WORLD: ReferenceFrame  # value = pinocchio.pinocchio_pywrap_default.ReferenceFrame.WORLD
XAxis: numpy.ndarray  # value = array([1., 0., 0.])
YAxis: numpy.ndarray  # value = array([0., 1., 0.])
ZAxis: numpy.ndarray  # value = array([0., 0., 1.])
__raw_version__: str = '3.4.0'
__version__: str = '3.4.0'
StdVec_Double = StdVec_Scalar

from __future__ import annotations
__all__: list[str] = ['computeRpyJacobian', 'computeRpyJacobianInverse', 'computeRpyJacobianTimeDerivative', 'matrixToRpy', 'rotate', 'rpyToMatrix']
def computeRpyJacobian(*args, **kwargs):
    """
    
    computeRpyJacobian( (numpy.ndarray)rpy [, (pinocchio.pinocchio_pywrap_default.ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> numpy.ndarray :
        Compute the Jacobian of the Roll-Pitch-Yaw conversion Given phi = (r, p, y) such that that R = R_z(y)R_y(p)R_x(r) and reference frame F (either LOCAL or WORLD), the Jacobian is such that omega_F = J_F(phi)phidot, where omega_F is the angular velocity expressed in frame F and J_F is the Jacobian computed with reference frame F
        Parameters:
        	rpy Roll-Pitch-Yaw vector	reference_frame  Reference frame in which the angular velocity is expressed. Notice LOCAL_WORLD_ALIGNED is equivalent to WORLD
    """
def computeRpyJacobianInverse(*args, **kwargs):
    """
    
    computeRpyJacobianInverse( (numpy.ndarray)rpy [, (pinocchio.pinocchio_pywrap_default.ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> numpy.ndarray :
        Compute the inverse Jacobian of the Roll-Pitch-Yaw conversion Given phi = (r, p, y) such that that R = R_z(y)R_y(p)R_x(r) and reference frame F (either LOCAL or WORLD), the Jacobian is such that omega_F = J_F(phi)phidot, where omega_F is the angular velocity expressed in frame F and J_F is the Jacobian computed with reference frame F
        Parameters:
        	rpy Roll-Pitch-Yaw vector	reference_frame  Reference frame in which the angular velocity is expressed. Notice LOCAL_WORLD_ALIGNED is equivalent to WORLD
    """
def computeRpyJacobianTimeDerivative(*args, **kwargs):
    """
    
    computeRpyJacobianTimeDerivative( (numpy.ndarray)rpy, (numpy.ndarray)rpydot [, (pinocchio.pinocchio_pywrap_default.ReferenceFrame)reference_frame=pinocchio.pinocchio_pywrap_default.ReferenceFrame.LOCAL]) -> numpy.ndarray :
        Compute the time derivative of the Jacobian of the Roll-Pitch-Yaw conversion Given phi = (r, p, y) such that that R = R_z(y)R_y(p)R_x(r) and reference frame F (either LOCAL or WORLD), the Jacobian is such that omega_F = J_F(phi)phidot, where omega_F is the angular velocity expressed in frame F and J_F is the Jacobian computed with reference frame F
        Parameters:
        	rpy Roll-Pitch-Yaw vector	reference_frame  Reference frame in which the angular velocity is expressed. Notice LOCAL_WORLD_ALIGNED is equivalent to WORLD
    """
def matrixToRpy(*args, **kwargs):
    """
    
    matrixToRpy( (numpy.ndarray)R) -> numpy.ndarray :
        Given a rotation matrix R, the angles (r, p, y) are given so that R = R_z(y)R_y(p)R_x(r), where R_a(theta) denotes the rotation of theta radians axis a. The angles are guaranteed to be in the ranges: r in [-pi,pi], p in[-pi/2,pi/2], y in [-pi,pi]
    """
def rotate(*args, **kwargs):
    """
    
    rotate( (str)axis, (float)angle) -> numpy.ndarray :
        Rotation matrix corresponding to a rotation about x, y or z e.g. R = rot('x', pi / 4): rotate pi/4 rad about x axis
    """
def rpyToMatrix(*args, **kwargs):
    """
    
    rpyToMatrix( (float)roll, (float)pitch, (float)yaw) -> numpy.ndarray :
        Given (r, p, y), the rotation is given as R = R_z(y)R_y(p)R_x(r), where R_a(theta) denotes the rotation of theta radians axis a
    
    rpyToMatrix( (numpy.ndarray)rpy) -> numpy.ndarray :
        Given (r, p, y), the rotation is given as R = R_z(y)R_y(p)R_x(r), where R_a(theta) denotes the rotation of theta radians axis a
    """

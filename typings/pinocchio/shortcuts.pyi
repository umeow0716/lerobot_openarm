from __future__ import annotations
from pinocchio import pinocchio_pywrap_default as pin
__all__: list[str] = ['WITH_HPP_FCL', 'WITH_HPP_FCL_BINDINGS', 'buildModelsFromMJCF', 'buildModelsFromSdf', 'buildModelsFromUrdf', 'createDatas', 'pin']
def _buildModelsFromMJCF(filename, root_joint = None, root_joint_name = None, verbose = False, meshLoader = None, geometry_types = None, contacts = False):
    ...
def _buildModelsFromSdf(filename, package_dirs = None, root_joint = None, root_link_name = '', root_joint_name = None, parent_guidance = None, verbose = False, meshLoader = None, geometry_types = None):
    ...
def _buildModelsFromUrdf(filename, package_dirs = None, root_joint = None, root_joint_name = None, verbose = False, meshLoader = None, geometry_types = None) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    ...
def buildModelsFromMJCF(filename, *args, **kwargs):
    """
    Parse the Mjcf file given in input and return a Pinocchio Model followed by corresponding GeometryModels of types specified by geometry_types, in the same order as listed.
        Arguments:
            - filename - name of the urdf file to load
            - package_dirs - where the meshes of the urdf are located. (default - None)
            - root_joint - Joint at the base of the model (default - None)
            - root_joint_name - Name for the root_joint (default - "root_joint")
            - verbose - print information of parsing (default - False)
            - meshLoader - object used to load meshes (default - hpp::fcl::MeshLoader)
            - geometry_types - Which geometry model to load. Can be pin.GeometryType.COLLISION, pin.GeometryType.VISUAL or both. (default - [pin.GeometryType.COLLISION, pin.GeometryType.VISUAL])
            - contacts - Boolean to know if contraint models are wanted (default - False)
        Return:
            Tuple of the models, in this order : model, collision model, and visual model, or  model, constraint_list, collision model, and visual model, if contacts is True.
    
        Example:
            model, collision_model, visual_model = buildModelsFromMJCF(filename, root_joint, verbose, meshLoader, geometry_types, root_joint_name="root_joint_name")
        
    """
def buildModelsFromSdf(filename, *args, **kwargs) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    """
    Parse the Sdf file given in input and return a Pinocchio Model and a list of Constraint Models, followed by corresponding GeometryModels of types specified by geometry_types, in the same order as listed.
        Arguments:
            - filename - name of the urdf file to load
            - package_dirs - where the meshes of the urdf are located. (default - None)
            - root_joint - Joint at the base of the model (default - None)
            - root_link_name - Name of the body to use as root of the model (default - "")
            - root_joint_name - Name for the root_joint (default - "root_joint")
            - parent_guidance - Joint names which should be preferred for cases where two joints can qualify as parent. The other joint appears in the constraint_model. If empty, joint appearance order in .sdf is taken as default.
            - verbose - print information of parsing (default - False)
            - meshLoader - object used to load meshes (default - hpp::fcl::MeshLoader)
            - geometry_types - Which geometry model to load. Can be pin.GeometryType.COLLISION, pin.GeometryType.VISUAL, both or None. (default - None])
        Return:
            Tuple of the models, in this order : model, collision model, and visual model.
    
        Example:
            model, collision_model, visual_model = buildModelsFromSdf(filename, root_joint, root_link_name, parent_guidance, verbose, meshLoader, geometry_types, root_joint_name="root_joint_name")
        
    """
def buildModelsFromUrdf(filename, *args, **kwargs) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    """
    Parse the URDF file given in input and return a Pinocchio Model followed by corresponding GeometryModels of types specified by geometry_types, in the same order as listed.
        Arguments:
            - filename - name of the urdf file to load
            - package_dirs - where the meshes of the urdf are located. (default - None)
            - root_joint - Joint at the base of the model (default - None)
            - root_joint_name - Name for the root_joint (default - "root_joint")
            - verbose - print information of parsing (default - False)
            - meshLoader - object used to load meshes (default - hpp::fcl::MeshLoader)
            - geometry_types - Which geometry model to load. Can be pin.GeometryType.COLLISION, pin.GeometryType.VISUAL or both. (default - [pin.GeometryType.COLLISION, pin.GeometryType.VISUAL])
        Return:
            Tuple of the models, in this order : model, collision model, and visual model.
    
        Example:
            model, collision_model, visual_model = buildModelsFromUrdf(filename, root_joint, verbose, meshLoader, geometry_types, root_joint_name="root_joint_name")
    
        Remark: In the URDF format, a joint of type fixed can be defined. For efficiency reasons, it is treated as operational frame and not as a joint of the model.
        
    """
def createDatas(*models):
    """
    
        Call createData() on each Model or GeometryModel in input and return the results in
        a tuple. If one of the models is None, the corresponding data object in the result
        is also None.
        
    """
WITH_HPP_FCL: bool = True
WITH_HPP_FCL_BINDINGS: bool = True

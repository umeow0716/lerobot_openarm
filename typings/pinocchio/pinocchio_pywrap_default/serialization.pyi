from __future__ import annotations
import typing
__all__: list[str] = ['StaticBuffer', 'StreamBuffer', 'buffer_copy', 'loadFromBinary', 'saveToBinary']
class StaticBuffer(Boost.Python.instance):
    """
    Static buffer to save/load serialized objects in binary mode with pre-allocated memory.
    """
    __instance_size__: typing.ClassVar[int] = 56
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self, (int)size) -> None :
            Default constructor from a given size capacity.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def reserve(*args, **kwargs):
        """
        
        reserve( (StaticBuffer)arg1, (int)new_size) -> None :
            Increase the capacity of the vector to a value that's greater or equal to new_size.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (StaticBuffer)self) -> int :
            Get the size of the input sequence.
        """
class StreamBuffer(Boost.Python.instance):
    """
    Stream buffer to save/load serialized objects in binary mode.
    """
    __instance_size__: typing.ClassVar[int] = 120
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        __init__( (object)self) -> None :
            Default constructor.
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    @staticmethod
    def max_size(*args, **kwargs):
        """
        
        max_size( (StreamBuffer)arg1) -> int :
            Get the maximum size of the StreamBuffer.
        """
    @staticmethod
    def prepare(*args, **kwargs):
        """
        
        prepare( (StreamBuffer)arg1, (int)arg2) -> StreamBuffer :
            Reserve data.
        """
    @staticmethod
    def size(*args, **kwargs):
        """
        
        size( (StreamBuffer)arg1) -> int :
            Get the size of the input sequence.
        """
    @staticmethod
    def tobytes(*args, **kwargs):
        """
        
        tobytes( (StreamBuffer)arg1) -> object :
            Returns the content of *this as a byte sequence.
        """
    @staticmethod
    def view(*args, **kwargs):
        """
        
        view( (StreamBuffer)arg1) -> object :
            Returns the content of *this as a memory view.
        """
def buffer_copy(*args, **kwargs):
    """
    
    buffer_copy( (StreamBuffer)dest, (StreamBuffer)source) -> None :
        Copy bytes from a source buffer to a target buffer.
    """
def loadFromBinary(*args, **kwargs):
    """
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_SE3)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_SE3)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Force)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Force)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Motion)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Motion)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Inertia)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Inertia)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Symmetric3)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Symmetric3)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Frame)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Frame)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Index)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Index)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_IndexVector)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_IndexVector)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_StdString)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_StdString)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Bool)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Bool)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Scalar)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Scalar)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.Model)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.Model)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.Data)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.Data)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Vector3)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Vector3)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Matrix6x)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Matrix6x)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_int)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_int)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.GeometryObject)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.GeometryObject)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_CollisionPair)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.StdVec_CollisionPair)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.GeometryModel)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.GeometryModel)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.GeometryData)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (pinocchio.pinocchio_pywrap_default.GeometryData)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.TriangleP)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.TriangleP)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Sphere)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Sphere)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Capsule)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Capsule)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Box)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Box)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Cone)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Cone)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Cylinder)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Cylinder)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Plane)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Plane)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Halfspace)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.Halfspace)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.BVHModelOBB)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.BVHModelOBB)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (object)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (object)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.BVHModelOBBRSS)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.BVHModelOBBRSS)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.HeightFieldOBBRSS)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.HeightFieldOBBRSS)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.HeightFieldAABB)object, (StreamBuffer)stream_buffer) -> None :
        Load an object from a binary buffer.
    
    loadFromBinary( (coal.coal_pywrap.HeightFieldAABB)object, (StaticBuffer)static_buffer) -> None :
        Load an object from a static binary buffer.
    """
def saveToBinary(*args, **kwargs):
    """
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_SE3)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_SE3)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Force)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Force)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Motion)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Motion)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Inertia)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Inertia)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Symmetric3)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Symmetric3)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Frame)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Frame)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Index)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Index)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_IndexVector)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_IndexVector)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_StdString)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_StdString)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Bool)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Bool)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Scalar)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Scalar)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.Model)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.Model)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.Data)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.Data)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Vector3)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Vector3)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Matrix6x)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_Matrix6x)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_int)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_int)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.GeometryObject)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.GeometryObject)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_CollisionPair)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.StdVec_CollisionPair)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.GeometryModel)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.GeometryModel)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.GeometryData)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (pinocchio.pinocchio_pywrap_default.GeometryData)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.TriangleP)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.TriangleP)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Sphere)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Sphere)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Capsule)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Capsule)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Box)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Box)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Cone)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Cone)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Cylinder)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Cylinder)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Plane)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Plane)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Halfspace)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.Halfspace)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.BVHModelOBB)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.BVHModelOBB)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (object)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (object)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.BVHModelOBBRSS)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.BVHModelOBBRSS)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.HeightFieldOBBRSS)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.HeightFieldOBBRSS)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    
    saveToBinary( (coal.coal_pywrap.HeightFieldAABB)object, (StreamBuffer)stream_buffer) -> None :
        Save an object to a binary buffer.
    
    saveToBinary( (coal.coal_pywrap.HeightFieldAABB)object, (StaticBuffer)static_buffer) -> None :
        Save an object to a static binary buffer.
    """

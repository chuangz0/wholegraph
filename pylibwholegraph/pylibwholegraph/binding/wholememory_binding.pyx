#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

cimport cpython
from libc cimport stdlib
import functools
import cython
from libc.stdint cimport *
from libcpp.cast cimport *
from cpython cimport Py_buffer
from cpython cimport array
import array
import numpy as np

cdef extern from "wholememory/wholememory.h":
    ctypedef enum wholememory_error_code_t:
        WHOLEMEMORY_SUCCESS                 "WHOLEMEMORY_SUCCESS"               # success
        WHOLEMEMORY_UNKNOW_ERROR            "WHOLEMEMORY_UNKNOW_ERROR"          # unknown error
        WHOLEMEMORY_NOT_IMPLEMENTED         "WHOLEMEMORY_NOT_IMPLEMENTED"       # method is not implemented
        WHOLEMEMORY_LOGIC_ERROR             "WHOLEMEMORY_LOGIC_ERROR"           # logic error
        WHOLEMEMORY_CUDA_ERROR              "WHOLEMEMORY_CUDA_ERROR"            # CUDA error
        WHOLEMEMORY_COMMUNICATION_ERROR     "WHOLEMEMORY_COMMUNICATION_ERROR"   # communication error
        WHOLEMEMORY_INVALID_INPUT           "WHOLEMEMORY_INVALID_INPUT"         # invalid input, e.g. nullptr
        WHOLEMEMORY_INVALID_VALUE           "WHOLEMEMORY_INVALID_VALUE"         # input value is invalid

    ctypedef enum wholememory_memory_type_t:
        WHOLEMEMORY_MT_NONE                 "WHOLEMEMORY_MT_NONE"
        WHOLEMEMORY_MT_CONTINUOUS           "WHOLEMEMORY_MT_CONTINUOUS"
        WHOLEMEMORY_MT_CHUNKED              "WHOLEMEMORY_MT_CHUNKED"
        WHOLEMEMORY_MT_DISTRIBUTED          "WHOLEMEMORY_MT_DISTRIBUTED"

    ctypedef enum wholememory_memory_location_t:
        WHOLEMEMORY_ML_NONE                 "WHOLEMEMORY_ML_NONE"
        WHOLEMEMORY_ML_DEVICE               "WHOLEMEMORY_ML_DEVICE"
        WHOLEMEMORY_ML_HOST                 "WHOLEMEMORY_ML_HOST"

    cdef wholememory_error_code_t wholememory_init(unsigned int flags)

    cdef wholememory_error_code_t wholememory_finalize()

    cdef struct wholememory_unique_id_t:
        char internal[128]

    cdef struct wholememory_comm_:
        pass

    ctypedef wholememory_comm_ * wholememory_comm_t

    cdef wholememory_error_code_t wholememory_create_unique_id(wholememory_unique_id_t* unique_id)

    cdef wholememory_error_code_t wholememory_create_communicator(wholememory_comm_t * comm,
                                                                  wholememory_unique_id_t unique_id,
                                                                  int rank,
                                                                  int size)

    cdef wholememory_error_code_t wholememory_destroy_communicator(wholememory_comm_t comm)

    cdef wholememory_error_code_t wholememory_communicator_get_rank(int* rank, wholememory_comm_t comm)

    cdef wholememory_error_code_t wholememory_communicator_get_size(int* size, wholememory_comm_t comm)

    cdef wholememory_error_code_t wholememory_communicator_barrier(wholememory_comm_t comm)

    cdef struct wholememory_handle_:
        pass

    ctypedef wholememory_handle_ * wholememory_handle_t

    cdef wholememory_error_code_t wholememory_malloc(wholememory_handle_t * wholememory_handle_ptr,
                                                     size_t total_size,
                                                     wholememory_comm_t comm,
                                                     wholememory_memory_type_t memory_type,
                                                     wholememory_memory_location_t memory_location,
                                                     size_t data_granularity)

    cdef wholememory_error_code_t wholememory_free(wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_communicator(wholememory_comm_t * comm,
                                                               wholememory_handle_t wholememory_handle)

    cdef wholememory_memory_type_t wholememory_get_memory_type(wholememory_handle_t wholememory_handle)

    cdef wholememory_memory_location_t wholememory_get_memory_location(wholememory_handle_t wholememory_handle)

    cdef size_t wholememory_get_total_size(wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_local_memory(void** local_ptr,
                                                               size_t * local_size,
                                                               size_t * local_offset,
                                                               wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_rank_memory(void** rank_memory_ptr,
                                                              size_t * rank_memory_size,
                                                              size_t * rank_memory_offset,
                                                              int rank,
                                                              wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_global_pointer(void** global_ptr,
                                                                 wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_determine_partition_plan(size_t * size_per_rank,
                                                                       size_t total_size,
                                                                       size_t data_granularity,
                                                                       int world_size)

    cdef wholememory_error_code_t wholememory_determine_entry_partition_plan(size_t * entry_per_rank,
                                                                             size_t total_entry_count,
                                                                             int world_size)

    cdef wholememory_error_code_t wholememory_get_partition_plan(size_t * size_per_rank,
                                                                 wholememory_handle_t wholememory_handle)

    cdef int fork_get_device_count()


cpdef enum WholeMemoryErrorCode:
    Success = WHOLEMEMORY_SUCCESS
    UnknowError = WHOLEMEMORY_UNKNOW_ERROR
    NotImplemented = WHOLEMEMORY_NOT_IMPLEMENTED
    LogicError = WHOLEMEMORY_LOGIC_ERROR
    CUDAError = WHOLEMEMORY_CUDA_ERROR
    CommunicationError = WHOLEMEMORY_COMMUNICATION_ERROR
    InvalidInput = WHOLEMEMORY_INVALID_INPUT
    InvalidValue = WHOLEMEMORY_INVALID_VALUE


cpdef enum WholeMemoryMemoryType:
    MtNone = WHOLEMEMORY_MT_NONE
    MtContinuous = WHOLEMEMORY_MT_CONTINUOUS
    MtChunked = WHOLEMEMORY_MT_CHUNKED
    MtDistributed = WHOLEMEMORY_MT_DISTRIBUTED


cpdef enum WholeMemoryMemoryLocation:
    MlNone = WHOLEMEMORY_ML_NONE
    MlDevice = WHOLEMEMORY_ML_DEVICE
    MlHost = WHOLEMEMORY_ML_HOST


cdef check_wholememory_error_code(wholememory_error_code_t err):
    cdef WholeMemoryErrorCode err_code = int(err)
    if err_code == Success:
        return
    elif err_code == UnknowError:
        raise Exception('Unknown error')
    elif err_code == NotImplemented:
        raise NotImplementedError('Not implemented')
    elif err_code == LogicError:
        raise RuntimeError('Logic error')
    elif err_code == CUDAError:
        raise RuntimeError('CUDA error')
    elif err_code == CommunicationError:
        raise RuntimeError('Communication error')
    elif err_code == InvalidInput:
        raise ValueError('Invalid input')
    elif err_code == InvalidValue:
        raise ValueError('Invalid value')
    else:
        raise NotImplementedError('Error code %d not recognized' % (int(err), ))


cdef extern from "wholememory/tensor_description.h":
    ctypedef enum wholememory_dtype_t:
        WHOLEMEMORY_DT_UNKNOWN  "WHOLEMEMORY_DT_UNKNOWN"
        WHOLEMEMORY_DT_FLOAT    "WHOLEMEMORY_DT_FLOAT"
        WHOLEMEMORY_DT_HALF     "WHOLEMEMORY_DT_HALF"
        WHOLEMEMORY_DT_DOUBLE   "WHOLEMEMORY_DT_DOUBLE"
        WHOLEMEMORY_DT_BF16     "WHOLEMEMORY_DT_BF16"
        WHOLEMEMORY_DT_INT      "WHOLEMEMORY_DT_INT"
        WHOLEMEMORY_DT_INT64    "WHOLEMEMORY_DT_INT64"
        WHOLEMEMORY_DT_INT16    "WHOLEMEMORY_DT_INT16"
        WHOLEMEMORY_DT_INT8     "WHOLEMEMORY_DT_INT8"
        WHOLEMEMORY_DT_COUNT    "WHOLEMEMORY_DT_COUNT"

    cdef size_t wholememory_dtype_get_element_size(wholememory_dtype_t dtype)


cpdef enum WholeMemoryDataType:
    DtUnknown = WHOLEMEMORY_DT_UNKNOWN
    DtFloat = WHOLEMEMORY_DT_FLOAT
    DtHalf = WHOLEMEMORY_DT_HALF
    DtDouble = WHOLEMEMORY_DT_DOUBLE
    DtBF16 = WHOLEMEMORY_DT_BF16
    DtInt = WHOLEMEMORY_DT_INT
    DtInt64 = WHOLEMEMORY_DT_INT64
    DtInt16 = WHOLEMEMORY_DT_INT16
    DtInt8 = WHOLEMEMORY_DT_INT8
    DtCount = WHOLEMEMORY_DT_COUNT


######################################################################
# dlpack
# https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
# https://github.com/cupy/cupy/blob/master/cupy/_core/dlpack.pyx

cpdef enum DLDeviceType:
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3

ctypedef struct DLDevice:
    DLDeviceType device_type
    int device_id

cdef enum DLDataTypeCode:
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4

ctypedef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes

ctypedef struct DLTensor:
    void * data
    DLDevice device
    int ndim
    DLDataType dtype
    int64_t * shape
    int64_t * strides
    uint64_t byte_offset

ctypedef struct DLManagedTensor:
    DLTensor dl_tensor
    void * manager_ctx
    void (*deleter)(DLManagedTensor *)


cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor* dlm_tensor
    # Do not invoke the deleter on a used capsule
    if cpython.PyCapsule_IsValid(dltensor, 'dltensor'):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        dlm_tensor.deleter(dlm_tensor)


cdef void deleter(DLManagedTensor* tensor) with gil:
    if tensor.manager_ctx is NULL:
        return
    cpython.Py_DECREF(<PyWholeMemoryTensorBuffer>tensor.manager_ctx)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)

# end dlpack
######################################################################

cdef class PyWholeMemoryUniqueID:
    cdef wholememory_unique_id_t wholememory_unique_id
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]
    cdef int64_t shape_int64_t[1]
    cdef int64_t strides_int64_t[1]

    def __cinit__(self):
        self.shape[0] = sizeof(self.wholememory_unique_id.internal)
        self.strides[0] = 1
        self.shape_int64_t[0] = self.shape[0]
        self.strides_int64_t[0] = self.strides[0]

    def __len__(self):
        return self.shape[0]

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = &self.wholememory_unique_id.internal[0]
        buffer.format = 'c'
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.len = self.shape[0]
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        buffer.buf = NULL
        buffer.format = 'c'
        buffer.len = 0
        buffer.ndim = 0
        buffer.obj = None
        buffer.shape = NULL
        buffer.strides = NULL

    def __dlpack__(self, stream=None):
        cdef DLManagedTensor * dlm_tensor = \
            <DLManagedTensor *> stdlib.malloc(sizeof(DLManagedTensor))
        cdef DLTensor * dl_tensor = &dlm_tensor.dl_tensor
        dl_tensor.data = &self.wholememory_unique_id.internal[0]
        dl_tensor.ndim = 1
        dl_tensor.shape = &self.shape_int64_t[0]
        dl_tensor.strides = &self.strides_int64_t[0]
        dl_tensor.byte_offset = 0
        dl_tensor.device.device_type, dl_tensor.device.device_id = self.__dlpack_device__()
        cdef DLDataType * dtype = &dl_tensor.dtype
        dtype.code = <uint8_t> kDLInt
        dtype.lanes = <uint16_t> 1
        dtype.bits = <uint8_t> 8

        dlm_tensor.manager_ctx = <void *> self
        cpython.Py_INCREF(self)
        dlm_tensor.deleter = deleter
        return cpython.PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)

    def __dlpack_device__(self):
        return (kDLCPU, 0)


cdef class PyWholeMemoryComm:
    cdef wholememory_comm_t comm_id

    def __cinit__(self):
        self.comm_id = NULL


def init(unsigned int flags):
    check_wholememory_error_code(wholememory_init(flags))


def finalize():
    check_wholememory_error_code(wholememory_finalize())


def create_unique_id():
    py_uid = PyWholeMemoryUniqueID()
    check_wholememory_error_code(wholememory_create_unique_id(&py_uid.wholememory_unique_id))
    return py_uid


def create_communicator(PyWholeMemoryUniqueID py_uid, int world_rank, int world_size):
    py_comm = PyWholeMemoryComm()
    check_wholememory_error_code(wholememory_create_communicator(&py_comm.comm_id,
                                                                 py_uid.wholememory_unique_id,
                                                                 world_rank,
                                                                 world_size))
    return py_comm


def destroy_communicator(PyWholeMemoryComm py_comm):
    check_wholememory_error_code(wholememory_destroy_communicator(py_comm.comm_id))


def comm_get_rank(PyWholeMemoryComm py_comm):
    cdef int world_rank = -1
    check_wholememory_error_code(wholememory_communicator_get_rank(&world_rank, py_comm.comm_id))
    return world_rank


def comm_get_size(PyWholeMemoryComm py_comm):
    cdef int world_size = -1
    check_wholememory_error_code(wholememory_communicator_get_size(&world_size, py_comm.comm_id))
    return world_size


def comm_barrier(PyWholeMemoryComm py_comm):
    check_wholememory_error_code(wholememory_communicator_barrier(py_comm.comm_id))


cdef class PyWholeMemoryHandle:
    cdef wholememory_handle_t wholememory_handle


def malloc(cython.size_t total_size,
                           PyWholeMemoryComm py_comm,
                           WholeMemoryMemoryType memory_type,
                           WholeMemoryMemoryLocation memory_location,
                           cython.size_t data_granularity):
    handle = PyWholeMemoryHandle()
    check_wholememory_error_code(wholememory_malloc(&handle.wholememory_handle, total_size, py_comm.comm_id,
                                                    int(memory_type), int(memory_location),
                                                    data_granularity))
    return handle


def free(PyWholeMemoryHandle handle):
    check_wholememory_error_code(wholememory_free(handle.wholememory_handle))


def get_communicator_from_handle(PyWholeMemoryHandle handle):
    py_comm = PyWholeMemoryComm()
    check_wholememory_error_code(wholememory_get_communicator(&py_comm.comm_id, handle.wholememory_handle))
    return py_comm


def get_memory_type_from_handle(PyWholeMemoryHandle handle):
    return WholeMemoryMemoryType(wholememory_get_memory_type(handle.wholememory_handle))


def get_memory_location_from_handle(PyWholeMemoryHandle handle):
    return WholeMemoryMemoryLocation(wholememory_get_memory_location(handle.wholememory_handle))


cpdef enum WholeMemoryViewType:
    VtNone = 0
    VtLocal = 1
    VtGlobal = 2
    VtRemote = 3

def get_type_string(WholeMemoryDataType data_type):
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
    if data_type == DtFloat:
        return '<f4'
    elif data_type == DtHalf:
        return '<f2'
    elif data_type == DtDouble:
        return '<f8'
    elif data_type == DtBF16:
        return '<f2'
    elif data_type == DtInt:
        return '<i4'
    elif data_type == DtInt64:
        return '<i8'
    elif data_type == DtInt16:
        return '<i2'
    elif data_type == DtInt8:
        return '|i1'
    else:
        raise ValueError('data type %d not valid' % (int(data_type), ))

cdef class PyWholeMemoryTensorBuffer:
    cdef void* c_ptr
    cdef WholeMemoryDataType data_type
    cdef Py_ssize_t itemsize
    cdef public object typestr
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]
    cdef int64_t shape_int64_t[1]
    cdef int64_t strides_int64_t[1]
    cdef WholeMemoryMemoryLocation device_type
    cdef int device_id

    def __cinit__(self):
        self.c_ptr = NULL
        self.shape[0] = 0
        self.strides[0] = 1
        self.shape_int64_t[0] = 0
        self.strides_int64_t[0] = 1
        self.itemsize = 0
        self.typestr = ''
        self.data_type = DtUnknown
        self.device_type = MlHost
        self.device_id = 0

    def set_view_device(self, WholeMemoryMemoryLocation device_type, int device_id):
        self.device_type = device_type
        self.device_id = device_id

    def get_view(self,
                 PyWholeMemoryHandle handle,
                 WholeMemoryDataType data_type,
                 WholeMemoryViewType view_type,
                 int target_rank):
        """Get view of a WholeMemoryHandle

        Parameters
        ----------
        handle : PyWholeMemoryHandle
            handler to the WholeMemory
        data_type: WholeMemoryDataType
            data type of the WholeMemory
        view_type : WholeMemoryViewType
            view type
        target_rank: int
            if view_type is VtRemote, target_rank is the rank of remote rank's memory, otherwise target_rank is ignored
        """
        self.data_type = data_type
        elt_size = wholememory_dtype_get_element_size(int(data_type))
        self.itemsize = elt_size
        if elt_size <= 0 or elt_size > 8:
            raise ValueError('data_type not supported')
        self.typestr = get_type_string(data_type)
        cdef WholeMemoryMemoryType mem_type
        cdef WholeMemoryMemoryLocation mem_location
        mem_type = int(wholememory_get_memory_type(handle.wholememory_handle))
        mem_location = int(wholememory_get_memory_location(handle.wholememory_handle))
        if self.device_type == MlHost and mem_location == MlDevice:
            raise ValueError('Device WholeMemory cannot get view from host.')
        if mem_type == MtDistributed and (view_type == VtGlobal or view_type == VtRemote):
            raise ValueError('Distributed WholeMemory have no view of Global or Remote')
        cdef size_t map_size
        cdef size_t map_offset
        cdef size_t global_size
        cdef wholememory_comm_t comm
        cdef int world_rank
        cdef int world_size
        global_size = wholememory_get_total_size(handle.wholememory_handle)
        if global_size % elt_size != 0:
            raise ValueError('global_size=%d not multiple of elt_size=%d' % (global_size, elt_size))
        global_elt_count = global_size // elt_size
        if view_type == VtLocal:
            check_wholememory_error_code(
                wholememory_get_local_memory(&self.c_ptr, &map_size, &map_offset, handle.wholememory_handle))
            if map_size % elt_size != 0 or map_offset % elt_size != 0:
                raise ValueError('map_size=%d, map_offset=%d not multiple of elt_size=%d'
                                 % (map_size, map_offset, elt_size))
            local_elt_count = map_size // elt_size
            local_start = map_offset // elt_size
            self.shape[0] = map_size // elt_size
            self.shape_int64_t[0] = map_size // elt_size
            return local_elt_count, local_start
        elif view_type == VtGlobal:
            check_wholememory_error_code(wholememory_get_global_pointer(&self.c_ptr, handle.wholememory_handle))
            self.shape[0] = global_size // elt_size
            self.shape_int64_t[0] = global_size // elt_size
            global_elt_count
            return global_elt_count, 0
        elif view_type == VtRemote:
            check_wholememory_error_code(wholememory_get_communicator(&comm, handle.wholememory_handle))
            check_wholememory_error_code(wholememory_communicator_get_rank(&world_rank, comm))
            check_wholememory_error_code(wholememory_communicator_get_size(&world_size, comm))
            if target_rank < 0 or target_rank >= world_size:
                raise IndexError('target_rank=%d but world_size=%d' % (target_rank, int(world_size)))
            check_wholememory_error_code(wholememory_get_rank_memory(
                &self.c_ptr, &map_size, &map_offset, target_rank, handle.wholememory_handle))
            if map_size % elt_size != 0 or map_offset % elt_size != 0:
                raise ValueError('target_rank=%d map_size=%d, map_offset=%d not multiple of elt_size=%d'
                                 % (target_rank, map_size, map_offset, elt_size))
            target_elt_count = map_size // elt_size
            target_start = map_offset // elt_size
            self.shape[0] = map_size // elt_size
            self.shape_int64_t[0] = map_size // elt_size
            return target_elt_count, target_start
        else:
            raise ValueError('view type should be VtLocal or VtGlobal or VtRemote')

    def __len__(self):
        return self.shape[0]

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = self.c_ptr
        buffer.format = 'c'
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = self.shape[0]
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        buffer.buf = NULL
        buffer.format = 'c'
        buffer.len = 0
        buffer.ndim = 0
        buffer.obj = None
        buffer.shape = NULL
        buffer.strides = NULL

    @property
    def ptr(self):
        return int(<uintptr_t>self.c_ptr)

    @property
    def __cuda_array_interface__(self):
        """See
        https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
        and
        https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        """
        cdef dict intf = {
            "data" : (self.ptr, False),
            "shape": (self.shape[0], ),
            "strides": None,
            "typestr": self.typestr,
            "version": 2
        }
        return intf

    def __dlpack__(self, stream=None):
        cdef DLManagedTensor * dlm_tensor = \
            <DLManagedTensor *> stdlib.malloc(sizeof(DLManagedTensor))
        cdef DLTensor * dl_tensor = &dlm_tensor.dl_tensor
        dl_tensor.data = self.c_ptr
        dl_tensor.ndim = 1
        dl_tensor.shape = &self.shape_int64_t[0]
        dl_tensor.strides = &self.strides_int64_t[0]
        dl_tensor.byte_offset = 0
        dl_tensor.device.device_type, dl_tensor.device.device_id = self.__dlpack_device__()
        cdef DLDataType * dtype = &dl_tensor.dtype
        if self.data_type == DtInt or self.data_type == DtInt64 \
                or self.data_type == DtInt16 or self.data_type == DtInt8:
            dtype.code = <uint8_t> kDLInt
        elif self.data_type == DtFloat or self.data_type == DtDouble \
                or self.data_type == DtHalf:
            dtype.code = <uint8_t> kDLFloat
        elif self.data_type == DtHalf:
            dtype.code = <uint8_t> kDLBfloat
        else:
            raise ValueError('Invalid data_type')
        dtype.lanes = <uint16_t> 1
        dtype.bits = <uint8_t> (self.itemsize * 8)

        dlm_tensor.manager_ctx = <void *> self
        cpython.Py_INCREF(self)
        dlm_tensor.deleter = deleter
        return cpython.PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)

    def __dlpack_device__(self):
        if self.device_type == MlHost:
            return (kDLCPU, 0)
        elif self.device_type == MlDevice:
            return (kDLCUDA, self.device_id)
        else:
            raise ValueError('self.device_type=%d' % (int(self.device_type), ))


def get_global_tensor(PyWholeMemoryHandle handle,
                      object import_dlpack_fn,
                      WholeMemoryDataType data_type,
                      WholeMemoryMemoryLocation view_from_device,
                      int view_from_device_id):
    tb = PyWholeMemoryTensorBuffer()
    tb.set_view_device(view_from_device, view_from_device_id)
    tsize, toffset = tb.get_view(handle, data_type, VtGlobal, 0)
    assert toffset == 0
    return import_dlpack_fn(tb)

def get_local_tensor(PyWholeMemoryHandle handle,
                     object import_dlpack_fn,
                     WholeMemoryDataType data_type,
                     WholeMemoryMemoryLocation view_from_device,
                     int view_from_device_id):
    tb = PyWholeMemoryTensorBuffer()
    tb.set_view_device(view_from_device, view_from_device_id)
    tsize, toffset = tb.get_view(handle, data_type, VtLocal, 0)
    return import_dlpack_fn(tb), toffset

def get_all_chunked_tensor(PyWholeMemoryHandle handle,
                           object import_dlpack_fn,
                           WholeMemoryDataType data_type,
                           WholeMemoryMemoryLocation view_from_device,
                           int view_from_device_id):
    cdef Whole
    cdef int world_rank
    cdef int world_size
    cdef wholememory_comm_t comm
    check_wholememory_error_code(wholememory_get_communicator(&comm, handle.wholememory_handle))
    check_wholememory_error_code(wholememory_communicator_get_rank(&world_rank, comm))
    check_wholememory_error_code(wholememory_communicator_get_size(&world_size, comm))
    chunked_tensors = []
    for r in range(world_size):
        tb = PyWholeMemoryTensorBuffer()
        tb.set_view_device(view_from_device, view_from_device_id)
        tsize, toffset = tb.get_view(handle, data_type, VtRemote, r)
        chunked_tensors.append(import_dlpack_fn(tb))
    return chunked_tensors


def fork_get_gpu_count():
    return fork_get_device_count()

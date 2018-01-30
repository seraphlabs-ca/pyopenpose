%include "common/common.i"

%import(module="common.parametersSWIG") "common/parameters.i"
%import(module="common.serialization.serializationSWIG") "common/serialization.i"

// numpy matrix I/O declaration

%apply (int DIM1, int DIM2, int DIM3, unsigned char* IN_ARRAY3)
        {(int IM_i, int IM_j, int IM_k, unsigned char* IM)}

%apply (int* DIM1, int* DIM2, int* DIM3, double** ARGOUTVIEWM_ARRAY3)
    {(int* POSE_i, int* POSE_j, int* POSE_k, double** POSE)}

// OpenPoseDetector

%shared_ptr(OpenPoseDetector);



%include "pyopenpose/pyopenpose.h"

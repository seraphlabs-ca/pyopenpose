/*
 This is a C++ implementation of components of OpenPose library.
 */

#ifndef _PYOPENPOSE_H_
#define _PYOPENPOSE_H_

#include <atomic>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdio> // sscanf
#include <string>
#include <thread> // std::this_thread
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <common/helper.h>
#include <common/serialization.h>
#include <common/parameters.h>
#include <common/cache.h>

#include "pyopenpose/common.h"

// OpenPose dependencies
#include <openpose/headers.hpp>

///////////////////////////////////////////////////////////////////////////////
// Classes
///////////////////////////////////////////////////////////////////////////////

class OpenPoseDetector;

typedef std::shared_ptr< OpenPoseDetector > OpenPoseDetectorPtr;
static OpenPoseDetectorPtr NullOpenPoseDetectorPtr = OpenPoseDetectorPtr();

///////////////////////////////////////////////////////////////////////////////
// Support classes
///////////////////////////////////////////////////////////////////////////////

struct UserDatum;

// base class for all input methods
class WUserInput;

// This worker will just invert the image
class WUserPostProcessing;

typedef std::vector< UserDatum > UserDatumType;
typedef op::Wrapper< UserDatumType > WrapperType;

///////////////////////////////////////////////////////////////////////////////
// class OpenPoseDetector
///////////////////////////////////////////////////////////////////////////////

class OpenPoseDetector:  virtual public BasePackable, virtual public Parameters {
    SER_CLASS_DECLARATION(OpenPoseDetector, MODULE_NAME);

public:
    OpenPoseDetector();
    ~OpenPoseDetector();

    OpenPoseDetector& start();
    OpenPoseDetector& stop();

    // Python wrapper
    void parse_image(int IM_i, int IM_j, int IM_k, unsigned char* IM);

    std::shared_ptr< UserDatumType >& parse_image(cv::Mat* cv_im = nullptr);

    // returns number of people in last parsed image
    int get_people_num();

    // return poses of last parsed image
    void get_poses(int* POSE_i, int* POSE_j, int* POSE_k, double** POSE);

protected:
    void _init();
    void _init_parameters();
    void _build();

protected:
    // control variables
    bool built;
    bool workerInputOnNewThread;
    bool workerProcessingOnNewThread;

    // params
    std::string model_folder;
    int image_resolution_width, image_resolution_height;
    int num_gpu, num_gpu_start;
    int keypoint_scale;
    std::string input_mode;

    std::string model_pose;
    int net_resolution_width, net_resolution_height, num_scales;
    double scale_gap;

    // ops
    op::PoseModel model_pose_op;
    op::ScaleMode keypoint_scale_op;
    op::Point<int> output_size_op, net_input_size_op, face_net_input_size_op, hand_net_input_size_op;

    // workers
    std::shared_ptr< WUserInput > wUserInput;
    std::shared_ptr< WUserPostProcessing > wUserPostProcessing;
    std::shared_ptr< WrapperType > opWrapper;

    // variables
    std::shared_ptr< UserDatumType > user_datum;
};


#endif // _PYOPENPOSE_H_


/*
 This is a C++ implementation of components of OpenPose library.
 */

#include <common/helper.h> // common/helper.h is included first to apply Eigen assert patch
#include <common/pyhelper.h>

#include <opencv2/opencv.hpp>
#include "pyopenpose.h"


///////////////////////////////////////////////////////////////////////////////
// Support classes
///////////////////////////////////////////////////////////////////////////////

// If the user needs his own variables, he can inherit the op::Datum struct and add them
// UserDatum can be directly used by the OpenPose wrapper because it inherits from op::Datum, just define Wrapper<UserDatum> instead of
// Wrapper<op::Datum>
struct UserDatum : public op::Datum {
    UserDatum() {};
};

// base class for all input methods
class WUserInput : public op::WorkerProducer< std::shared_ptr< UserDatumType > > {
public:
    WUserInput() {};
    void initializationOnThread() {};
    std::shared_ptr<UserDatumType> workProducer() { this->stop(); return nullptr; };
};


// This worker will just invert the image
class WUserPostProcessing : public op::Worker< std::shared_ptr<UserDatumType> >
{
public:
    WUserPostProcessing()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

    // User's post-processing (after OpenPose processing & before OpenPose outputs)
    void work(std::shared_ptr< UserDatumType >& datumsPtr) {};
};

///////////////////////////////////////////////////////////////////////////////
// class OpenPoseDetector
///////////////////////////////////////////////////////////////////////////////

OpenPoseDetector::OpenPoseDetector() {
    _init();
}

OpenPoseDetector::~OpenPoseDetector() {
    this->stop();
}


OpenPoseDetector& OpenPoseDetector::start() {
    if (!built) {
        _build();

        if (opWrapper) {

            // Set to single-thread running (e.g. for debugging purposes)
            // opWrapper.disableMultiThreading();

            PRINT("Starting thread(s)");
            this->opWrapper->start();

            built = true;
        } else {
            PRINT("Failed creating opWrapper!");
        }
    }

    return *this;
}

OpenPoseDetector& OpenPoseDetector::stop() {
    if (built) {
        if (opWrapper) {
            this->opWrapper->stop();
        }
        built = false;
    }

    return *this;
}

// Python wrapper
void OpenPoseDetector::parse_image(int IM_i, int IM_j, int IM_k, unsigned char* IM) {
    if (IM_k != 3) {
        throw DEBUG_STR("Image must have 3 channels (RGB)");
    }
    // Input image is RGB, Convert RGB to BGR
    cv::Mat cv_im_rgb(cv::Size(IM_j, IM_i), CV_8UC3, IM, cv::Mat::AUTO_STEP);
    cv::Mat cv_im;
    cv::cvtColor(cv_im_rgb, cv_im, cv::COLOR_RGB2BGR);

    // parse image
    this->parse_image(&cv_im);
}

// TODO: add push_image/pop_pose for synchronous parsing.

std::shared_ptr< UserDatumType >& OpenPoseDetector::parse_image(cv::Mat* cv_im) {
    if (cv_im && opWrapper) {
        auto datum_to_process = std::make_shared< UserDatumType >();
        datum_to_process->emplace_back();
        auto& datum = datum_to_process->at(0);

        // Fill datum
        datum.cvInputData = *cv_im;

        // clear results
        user_datum = std::shared_ptr< UserDatumType >();
        // If empty frame -> return nullptr
        if (datum.cvInputData.empty()) {
            PRINT("Empty cv_im detected");
        } else {
            auto successfullyEmplaced = opWrapper->waitAndEmplace(datum_to_process);
            // Pop frame
            if (!successfullyEmplaced || !opWrapper->waitAndPop(user_datum)) {
                PRINT("Processed datum could not be emplaced.");
                user_datum = std::shared_ptr< UserDatumType >();
            }
        }
    }

    return user_datum;
}

// returns number of people in last parsed image
int OpenPoseDetector::get_people_num() {
    if (user_datum) {
        return user_datum->at(0).poseKeypoints.getSize(0);
    } else {
        return 0;
    }
}

// return poses of last parsed image
void OpenPoseDetector::get_poses(int* POSE_i, int* POSE_j, int* POSE_k, double** POSE) {

    if (user_datum) {
        // number of people detected
        (*POSE_i) =  user_datum->at(0).poseKeypoints.getSize(0);
        (*POSE_j) = user_datum->at(0).poseKeypoints.getSize(1);
        (*POSE_k) = user_datum->at(0).poseKeypoints.getSize(2);
        int S = (*POSE_i) * (*POSE_j) * (*POSE_k);
        (*POSE) = new double[S];

        // get pointer to raw data
        float* raw_pose_data = user_datum->at(0).poseKeypoints.getPtr();

        // convert results to double
        for (int i = 0; i < S; i++) {
            (*POSE)[i] = (double)(raw_pose_data[i]);
        }

    } else {
        // no results are available
        (*POSE_i) =  (*POSE_j) = (*POSE_k) = 0;
        (*POSE) = new double[(*POSE_i) * (*POSE_j) * (*POSE_k)];
    }
}

void OpenPoseDetector::_init() {
    _init_parameters();

    // initialize variables
    built = false;
    workerInputOnNewThread = false;
    workerProcessingOnNewThread = false;

    // initialize parameters
    model_folder = "../data/assets/models/openpose/";
    // image_resolution_width = 1280;
    image_resolution_width = 640;
    // image_resolution_height = 720;
    image_resolution_height = 480;
    num_gpu = -1;
    num_gpu_start = 0;
    keypoint_scale = 3;
    // input modes available: single-async (a single image blocking call)
    input_mode = "single-async";

    model_pose = "COCO";
    // net_resolution_width = 656;
    net_resolution_width = 368;
    net_resolution_height = 368;
    num_scales = 1;
    scale_gap = 0.3;
}

void OpenPoseDetector::_init_parameters() {
    // OpenPose
    string_params.create("model_folder", model_folder);
    int_params.create("image_resolution_width", image_resolution_width);
    int_params.create("image_resolution_height", image_resolution_height);
    int_params.create("num_gpu", num_gpu);
    int_params.create("num_gpu_start", num_gpu_start);
    int_params.create("keypoint_scale", keypoint_scale);
    string_params.create("input_mode", input_mode);

    // OpenPose Body Pose
    string_params.create("model_pose", model_pose);
    int_params.create("net_resolution_width", net_resolution_width);
    int_params.create("net_resolution_height", net_resolution_height);
    int_params.create("num_scales", num_scales);
    double_params.create("scale_gap", scale_gap);
}

void OpenPoseDetector::_build() {
    // Parse parameters
    if (model_pose == "COCO")
        model_pose_op = op::PoseModel::COCO_18;
    else if (model_pose == "MPI")
        model_pose_op = op::PoseModel::MPI_15;
    else if (model_pose == "MPI_4_layers")
        model_pose_op = op::PoseModel::MPI_15_4;
    else
    {
        throw DEBUG_STR("model_pose does not correspond to any model (COCO, MPI, MPI_4_layers)");
    }

    if (keypoint_scale == 0)
        keypoint_scale_op = op::ScaleMode::InputResolution;
    else if (keypoint_scale == 1)
        keypoint_scale_op = op::ScaleMode::NetOutputResolution;
    else if (keypoint_scale == 2)
        keypoint_scale_op = op::ScaleMode::OutputResolution;
    else if (keypoint_scale == 3)
        keypoint_scale_op = op::ScaleMode::ZeroToOne;
    else if (keypoint_scale == 4)
        keypoint_scale_op = op::ScaleMode::PlusMinusOne;
    else
    {
        throw DEBUG_STR("keypoint_scale does not correspond to any scale mode: (0, 1, 2, 3, 4) for (InputResolution, NetOutputResolution, OutputResolution, ZeroToOne, PlusMinusOne).");
    }

    output_size_op = op::Point<int>(image_resolution_width, image_resolution_height);
    net_input_size_op = op::Point<int>(net_resolution_width, net_resolution_height);

    // setup openpose

    // TODO: add support in synchronous mode for higher performance
    if (input_mode == "single-async") {
        opWrapper = std::shared_ptr< WrapperType >(new WrapperType {op::ThreadManagerMode::Asynchronous});
    } else {
        throw DEBUG_FORMAT_STR("Unknown input_mode=%s, expected: single-async", input_mode);
    }

    // Configure OpenPose
    const op::WrapperStructPose wrapperStructPose {
        net_input_size_op, output_size_op, keypoint_scale_op, num_gpu, num_gpu_start,
                           num_scales, (float)scale_gap, op::RenderMode::None, model_pose_op,
                           false, (float)0.0, (float)0.0,
                           0, model_folder
    };
    // Configure wrapper
    opWrapper->configure(wrapperStructPose, op::WrapperStructFace{}, op::WrapperStructHand{}, op::WrapperStructInput{}, op::WrapperStructOutput{});
}

SER_CLASS_DEFINITION(OpenPoseDetector);
SER_DEFAULT_NEW_FROM_STATE_DEFINITION(OpenPoseDetector);

// loads a state - child classes should read their data
void OpenPoseDetector::_ser_set_state(DataMessage * state) {
    BasePackable::_ser_set_state(state);

    // OpenPose
    SER_GET_STATE_VAR("model_folder", model_folder);
    SER_GET_STATE_VAR("image_resolution_width", image_resolution_width);
    SER_GET_STATE_VAR("image_resolution_height", image_resolution_height);
    SER_GET_STATE_VAR("num_gpu", num_gpu);
    SER_GET_STATE_VAR("num_gpu_start", num_gpu_start);
    SER_GET_STATE_VAR("keypoint_scale", keypoint_scale);

    // OpenPose Body Pose
    SER_GET_STATE_VAR("model_pose", model_pose);
    SER_GET_STATE_VAR("net_resolution_width", net_resolution_width);
    SER_GET_STATE_VAR("net_resolution_height", net_resolution_height);
    SER_GET_STATE_VAR("num_scales", num_scales);
    SER_GET_STATE_VAR("scale_gap", scale_gap);
}

// loads a state - child classes should read their data
void OpenPoseDetector::_ser_get_state(DataMessage * state) {
    BasePackable::_ser_get_state(state);

    // OpenPose
    SER_SET_STATE_VAR("model_folder", model_folder);
    SER_SET_STATE_VAR("image_resolution_width", image_resolution_width);
    SER_SET_STATE_VAR("image_resolution_height", image_resolution_height);
    SER_SET_STATE_VAR("num_gpu", num_gpu);
    SER_SET_STATE_VAR("num_gpu_start", num_gpu_start);
    SER_SET_STATE_VAR("keypoint_scale", keypoint_scale);

    // OpenPose Body Pose
    SER_SET_STATE_VAR("model_pose", model_pose);
    SER_SET_STATE_VAR("net_resolution_width", net_resolution_width);
    SER_SET_STATE_VAR("net_resolution_height", net_resolution_height);
    SER_SET_STATE_VAR("num_scales", num_scales);
    SER_SET_STATE_VAR("scale_gap", scale_gap);
}

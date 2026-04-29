#pragma once

#include "illixr/plugin.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/data_format/scene_reconstruction.hpp"
#include "illixr/data_format/mesh.hpp"
#include "illixr/relative_clock.hpp"
#include "illixr/phonebook.hpp"

#include "ITMLib/Core/ITMBasicEngine.h"
#include "ORUtils/FileUtils.h"

#include <filesystem>

//pyh only extracted partial updated mesh
//uncomment if you want to see what the full mesh looks like
#define ACTIVE_SCENE
#define PARALLEL_COMPRESSION

//pyh need to set env variable COMPRESSION_PARALLELIM & FPS

using namespace ILLIXR;

// aniket: different thresholds evaluated
enum Threshold {
    FPS,
    ALLOCS,
    // UPDATES,
    // AUP,
    VIS_UDELTA,
    VIS_SDELTA,
    POSE_UDELTA
    // POSE_SDELTA
};

class infinitam : public plugin {
public:
    infinitam(const std::string& name_, phonebook *pb_);

    void process_frame(switchboard::ptr<const data_format::scene_recon_type>& datum);

    ~infinitam() override {
        sr_latency_.close();
    }

private:
    //ILLIXR related variables
    const std::shared_ptr<switchboard> switchboard_;
    //for parallel
    switchboard::writer<data_format::mesh_type> mesh_writer_;

    switchboard::writer<data_format::vb_type> vb_list_;

    std::unique_ptr<ORUtils::MemoryBlock<ITMLib::ITMMesh::Triangle>> cpu_triangles_;

    // aniket: decider for threshold signal
    bool thresholdMet();

    // aniket: new allocations since last extraction
    unsigned scene_id_ = 0;
    unsigned alloc_count_ = 0;
    unsigned allocs_threshold_ = 10;
    // unsigned update_count_ = 0;
    // unsigned updates_threshold_ = 10;
    // unsigned aup_count_ = 0;
    // unsigned aup_threshold_ = 1000;
    // aniket: difference in visible bricks since last extraction
    unsigned vis_udelta_count_ = 0;
    unsigned vis_udelta_threshold_ = 10;
    signed long vis_sdelta_count_ = 0;
    unsigned vis_sdelta_threshold_ = 10;
    // aniket: difference in pose since last extraction
    float pose_udelta_count_ = 0;
    float pose_udelta_threshold_ = 10;
    // float pose_sdelta_count_ = 0;
    // unsigned pose_sdelta_threshold_ = 10;

    Threshold threshold_signal_ = Threshold::FPS;

    //InfiniTAM related variables
    ITMLib::ITMRGBDCalib *calib_;
    ITMLib::ITMMainEngine *main_engine_;
    ITMUChar4Image *input_RGB_image_;
    ITMShortImage *input_raw_depth_image_;
    ITMLib::ITMLibSettings *internal_settings_;
    ITMLib::ITMMesh *mesh_;

    std::string scene_number_;
    std::string merge_name_;
    unsigned frame_count_;
    unsigned fps_ = 15;
    unsigned thread_count_;

    std::ofstream sr_latency_;
    const std::string data_path_ = std::filesystem::current_path().string() + "/recorded_data";
};

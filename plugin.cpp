//5/7 This is the version used for final testing
#include "common/plugin.hpp"
#include "common/switchboard.hpp"
#include "common/data_format.hpp"
#include "common/relative_clock.hpp"
#include "common/phonebook.hpp"

#include "ITMLib/Core/ITMBasicEngine.h"
#include "ITMLib/Utils/ITMLibSettings.h"
#include "ITMLib/ITMLibDefines.h"
#include "ORUtils/FileUtils.h"

#include <eigen3/Eigen/Dense>
#include <sys/time.h>
#include <fstream>
#include <opencv/cv.hpp>
#include <stdio.h>
#include <omp.h>
#include <filesystem>
#include <set>
#include <tuple>
#include <iostream>
#include <algorithm>
#include <memory>
#include <spdlog/spdlog.h>
#include <draco/io/ply_reader.h>
#include "draco/io/ply_property_writer.h"
#include "draco/io/ply_decoder.h"
#include "draco/compression/encode.h" 
#include "draco/compression/expert_encode.h"
#include "draco/io/file_utils.h"

//pyh only extracted partial updated mesh
//uncomment if you want to see what the full mesh looks like
#define ACTIVE_SCENE
#define PARALLEL_COMPRESSION

//pyh need to set env variable COMPRESSION_PARALLELIM & FPS 

using namespace ILLIXR;

class infinitam : public plugin {
    public:
        infinitam(std::string name_, phonebook* pb_)
            : plugin{name_, pb_}
        , sb{pb->lookup_impl<switchboard>()}
        , _m_scannet_datum{sb->get_reader<scene_recon_type>("ScanNet_Data")}
        , _m_mesh{sb->get_writer<mesh_type>("requested_scene")}
        , _m_mesh_0{sb->get_writer<mesh_type>("requested_scene_0")}
        , _m_mesh_1{sb->get_writer<mesh_type>("requested_scene_1")}
        , _m_mesh_2{sb->get_writer<mesh_type>("requested_scene_2")}
        , _m_mesh_3{sb->get_writer<mesh_type>("requested_scene_3")}
        , _m_mesh_4{sb->get_writer<mesh_type>("requested_scene_4")}
        , _m_mesh_5{sb->get_writer<mesh_type>("requested_scene_5")}
        , _m_mesh_6{sb->get_writer<mesh_type>("requested_scene_6")}
        , _m_mesh_7{sb->get_writer<mesh_type>("requested_scene_7")}
        //, _m_mesh_8{sb->get_writer<mesh_type>("requested_scene_8")}
        //, _m_mesh_9{sb->get_writer<mesh_type>("requested_scene_9")}
        //, _m_mesh_10{sb->get_writer<mesh_type>("requested_scene_10")}
        //, _m_mesh_11{sb->get_writer<mesh_type>("requested_scene_11")}
        , _m_vb_list{sb->get_writer<vb_type>("unique_VB_list")}
        {
            //pyh For now, I just hardcode internal settings that exists in ScanNet.s
	    //Later we might need to have a more intelligent way to get these variables
            internalSettings = new ITMLib::ITMLibSettings();
            internalSettings->useICP = false;
            internalSettings->useApproximateDepthCheck = false;
            internalSettings->usePreviousVisibilityList = true;
            internalSettings->freqMode = ITMLib::ITMLibSettings::FreqMode::FREQMODE_CONSTANT;
            internalSettings->fusionFreq = 30.0;
            internalSettings->useDecoupledRaycasting = true;
            internalSettings->raycastingFreq=1.0;

            calib = new ITMLib::ITMRGBDCalib();
            const char* illixr_data_c_str = std::getenv("ILLIXR_DATA"); 

	    if (!illixr_data_c_str) {
		    throw std::runtime_error("ILLIXR_DATA not set");
	    }

            std::string illixr_data = std::string{illixr_data_c_str};
            const std::string calib_subpath = "/calibration.txt";
            std::string calib_source{illixr_data + calib_subpath};
            if(!readRGBDCalib(calib_source.c_str(), *calib)){
        spdlog::get("illixr")->error("Read RGBD calibration file failed");
            }
            
            //pyh extract scene name
            std::size_t pos = illixr_data.find_last_of("/");
            scene_number = illixr_data.substr(pos+1);
    //spdlog::get("illixr")->debug("Scene number: {}", scene_number);
            
            mainEngine = new ITMLib::ITMBasicEngine<ITMVoxel, ITMVoxelIndex>(
                    internalSettings,
                    *calib,
                    calib->intrinsics_rgb.imgSize,
                    calib->intrinsics_d.imgSize
            );
            
            //pyh first allocate for incoming depth & RGB image on CPU, then later copy to GPU
            inputRawDepthImage = new ITMShortImage(calib->intrinsics_d.imgSize, true, false);
            inputRGBImage = new ITMUChar4Image(calib->intrinsics_rgb.imgSize, true, false);
            
            if (internalSettings->deviceType == ITMLib::ITMLibSettings::DEVICE_CUDA){
        spdlog::get("illixr")->info("Using the CUDA version of InfiniTAM");
            }
            
            sb->schedule<scene_recon_type>(id,"ScanNet_Data",[&](switchboard::ptr<const scene_recon_type> datum, std::size_t){
                    this->ProcessFrame(datum);
            });

            //pyh initialize mesh used for mesh extraction
            mesh=new ITMLib::ITMMesh(MEMORYDEVICE_CUDA,0);

	    //track how many frame InfiniTAM has processed
            frame_count=0;

	    if (!std::filesystem::exists(data_path)) {
		    if (!std::filesystem::create_directory(data_path)) {
            spdlog::get("illixr")->error("Failed to create data directory.");
		    }
	    }
            sr_latency.open(data_path + "/sr_latency.csv");

	    const char* env_threads = std::getenv("COMPRESSION_PARALLELISM");
	    const char* env_fps = std::getenv("FPS");

	    if (env_threads) {
		    try { THREAD_COUNT = std::stoul(env_threads); }
		    catch (...) { std::cerr << "infinitam: COMPRESSION_PARALLELISM invalid, using default " << THREAD_COUNT << "\n"; }
	    } else {
            spdlog::get("illixr")->error("infinitam: COMPRESSION_PARALLELISM not set; using default {}",
                                         thread_count_);
	    }
	    if (env_fps) {
		    try { FPS = std::stoul(env_fps); }
        spdlog::get("illixr")->error("infinitam: COMPRESSION_PARALLELISM invalid, using default {}",
                                     thread_count_);
    }
	    }	



	    printf("================================InfiniTAM: setup finished==========================\n");
    spdlog::get("illixr")->info("================================InfiniTAM: setup finished==========================");
        }

        void ProcessFrame(switchboard::ptr<const scene_recon_type> datum)
    //spdlog::get("illixr")->debug("================================InfiniTAM: frame %d received==========================", frame_count);
    if (!datum->depth.empty()) {
            if(!datum->depth.empty())
            {
		    //pyh: convert to transformation matrix
		    Eigen::Matrix3f rot = datum->pose.orientation.normalized().toRotationMatrix();
		    ORUtils::Matrix4<float> cur_trans_matrix;
		    cur_trans_matrix = {
			    rot(0, 0), rot(1, 0), rot(2, 0), 0.0f,
			    rot(0, 1), rot(1, 1), rot(2, 1), 0.0f,
			    rot(0, 2), rot(1, 2), rot(2, 2), 0.0f,
			    datum->pose.position.x(), datum->pose.position.y(), datum->pose.position.z(), 1.0f
		    };

		    // Set first pose
		    if(frame_count == 0){
			    mainEngine->SetInitialPose(cur_trans_matrix);
		    }

		    cv::Mat cur_depth = datum->depth.clone();

		    //pyh converting the to the InfiniTAM expected data structure
		    const short *depth_frame = reinterpret_cast<const short*>(cur_depth.datastart);
		    short *cur_depth_head = inputRawDepthImage->GetData(MEMORYDEVICE_CPU);
		    std::memcpy(cur_depth_head, depth_frame, sizeof(short)  *inputRawDepthImage->dataSize);

		    auto frame_start = std::chrono::high_resolution_clock::now();
		
		    //pyh main reconstruction (volumetric fusion) function
		    mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, cur_trans_matrix);

		    auto frame_end = std::chrono::high_resolution_clock::now();
		    auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count();

		    auto sinceEpoch = frame_start.time_since_epoch();
		    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(sinceEpoch).count();
		    sr_latency<<"fuse "<<frame_count<<" " <<(frame_duration / 1000.0) << "\n";	
                
		    if ((frame_count % FPS) == 0 && frame_count > 0){
			    sr_latency<<"start "<<frame_count<<" "<<millis<<"\n";
			    auto start = std::chrono::high_resolution_clock::now();
#if !defined ACTIVE_SCENE
			    mainEngine->GetMesh(mesh, 2);
#else
			    mainEngine->GetMesh(mesh, 1);
#endif
			    //pyh This is for dumping out the mesh directly to file
			    //std::string merge_name = this->scene_number + "_" + std::to_string(frame_count) +".obj";
			    //mesh->WriteOBJ(merge_name.c_str());

			    if (!cpu_triangles_ || cpu_triangles_->dataSize < mesh->noTotalTriangles) {
				    cpu_triangles_.reset(new ORUtils::MemoryBlock<ITMLib::ITMMesh::Triangle>(mesh->noTotalTriangles, MEMORYDEVICE_CPU));
			    }

			    cpu_triangles_->DirectSetFrom(
					    mesh->triangles,
					    ORUtils::MemoryBlock<ITMLib::ITMMesh::Triangle>::CUDA_TO_CPU,
					    mesh->noTotalTriangles);
			    ITMLib::ITMMesh::Triangle *triangleArray = cpu_triangles_->GetData(MEMORYDEVICE_CPU);

			    unsigned face_number = mesh->noTotalTriangles;
			    unsigned scene_id = (frame_count / FPS) - 1;

			    auto end = std::chrono::high_resolution_clock::now();
			    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			    double duration_ms = duration / 1000.0;
			    sr_latency<<"extract "<<scene_id<<" " <<duration_ms<<" "<<face_number << "\n";


			    auto VB_start = std::chrono::high_resolution_clock::now();
			    std::set<std::tuple<int,int,int>> unique_VBs;
			    for(unsigned i=0; i < mesh->updated_voxel_blocks.size(); i++){
				    unique_VBs.emplace(std::make_tuple((mesh->updated_voxel_blocks[i][0]), (mesh->updated_voxel_blocks[i][1]), (mesh->updated_voxel_blocks[i][2])));
			    }
			    //pyh sending UVBL first
			    _m_vb_list.put(_m_vb_list.allocate<vb_type>(vb_type{std::move(unique_VBs),scene_id}));

			    auto VB_end = std::chrono::high_resolution_clock::now();
			    duration = std::chrono::duration_cast<std::chrono::microseconds>(VB_end - VB_start).count();
			    duration_ms = duration / 1000.0;
			    sr_latency<<"vb "<< scene_id << " "<<duration_ms<<" "<<unique_VBs.size()<<"\n";

			    auto roi_start = std::chrono::high_resolution_clock::now();
			    bool set_active = false;
			    std::vector<std::function<void(std::unique_ptr<draco::PlyReader>&&, unsigned, unsigned, unsigned)>> operations = {
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 0 */ 
					    _m_mesh_0.put(_m_mesh_0.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 0, num_partitions,face_number, per_vertices, set_active}));},
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 1 */  
					    _m_mesh_1.put(_m_mesh_1.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 1, num_partitions,face_number, per_vertices, set_active}));},
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 2 */  
					    _m_mesh_2.put(_m_mesh_2.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 2, num_partitions,face_number, per_vertices, set_active}));},
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 3 */  
					    _m_mesh_3.put(_m_mesh_3.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 3, num_partitions,face_number, per_vertices, set_active}));},
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 4 */  
					    _m_mesh_4.put(_m_mesh_4.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 4, num_partitions,face_number, per_vertices, set_active}));},
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 5 */  
					    _m_mesh_5.put(_m_mesh_5.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 5, num_partitions,face_number, per_vertices, set_active}));},
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 6 */  
					    _m_mesh_6.put(_m_mesh_6.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 6, num_partitions,face_number, per_vertices, set_active}));},
				    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 7 */  
					    _m_mesh_7.put(_m_mesh_7.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 7, num_partitions,face_number, per_vertices, set_active}));},
		//		    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 8 */  
		//			    _m_mesh_8.put(_m_mesh_8.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 8, num_partitions,face_number, per_vertices, set_active}));},
		//		    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 9 */  
		//			    _m_mesh_9.put(_m_mesh_9.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 9, num_partitions,face_number, per_vertices, set_active}));},
		//		    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 10 */  
		//			    _m_mesh_10.put(_m_mesh_10.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 10, num_partitions,face_number, per_vertices, set_active}));},
		//		    [&](std::unique_ptr<draco::PlyReader>&& ply_reader, unsigned face_number, unsigned per_vertices, unsigned num_partitions) {  /* do something for case 11 */  
		//			    _m_mesh_11.put(_m_mesh_11.allocate<mesh_type>(mesh_type{std::move(ply_reader), scene_id, 11, num_partitions,face_number, per_vertices, set_active}));},
			    };
			    omp_set_dynamic(0);
			    omp_set_num_threads(THREAD_COUNT);
			    unsigned numThreads = THREAD_COUNT;
			    unsigned trianglesPerThread = (face_number + numThreads - 1)/numThreads;
            spdlog::get("illixr")->info("parallel compression, # of threads: %u, # of triangles/threads: %u ", numThreads,
                   trianglesPerThread);
#pragma omp parallel num_threads(numThreads)
			    {
				    unsigned thread_id = omp_get_thread_num();

				    unsigned startTriangle = thread_id * trianglesPerThread;
				    unsigned endTriangle =  std::min((thread_id + 1) * trianglesPerThread, face_number);
				    unsigned per_faces = endTriangle - startTriangle;
				    unsigned per_vertices =  per_faces * 3;
				    std::unique_ptr<draco::PlyReader> ply_reader(new draco::PlyReader());

				    ply_reader->format_= draco::PlyReader::kAscii;
				    ply_reader->element_index_["vertex"]=0;
				    ply_reader->elements_.emplace_back(draco::PlyElement("vertex", per_vertices));
				    ply_reader->elements_.back().AddProperty(draco::PlyProperty("x",draco::DT_FLOAT32, draco::DT_INVALID)); 
				    ply_reader->elements_.back().AddProperty(draco::PlyProperty("y",draco::DT_FLOAT32, draco::DT_INVALID)); 
				    ply_reader->elements_.back().AddProperty(draco::PlyProperty("z",draco::DT_FLOAT32, draco::DT_INVALID)); 

				    ply_reader->element_index_["face"]=1;
				    ply_reader->elements_.emplace_back(draco::PlyElement("face", per_faces));

				    ply_reader->elements_.back().AddProperty(draco::PlyProperty("vertex_indices",draco::DT_INT32, draco::DT_UINT8));

				    ply_reader->elements_.back().AddProperty(draco::PlyProperty("vb_x",draco::DT_INT32, draco::DT_INVALID)); 
				    ply_reader->elements_.back().AddProperty(draco::PlyProperty("vb_y",draco::DT_INT32, draco::DT_INVALID)); 
				    ply_reader->elements_.back().AddProperty(draco::PlyProperty("vb_z",draco::DT_INT32, draco::DT_INVALID)); 

				    draco::PlyElement &vertex_element = ply_reader->elements_[0];
				    draco::PlyElement &face_element = ply_reader->elements_[1];

				    for(unsigned entry = startTriangle; entry < endTriangle; ++entry){
					    for(int i =0; i < vertex_element.num_properties(); ++i){
						    draco::PlyProperty &prop = vertex_element.property(i);
						    draco::PlyPropertyWriter<float> prop_writer(&prop);
						    switch(i){
							    case 0:
								    prop_writer.PushBackValue(triangleArray[entry].p0.x );	       
								    break;
							    case 1:
								    prop_writer.PushBackValue(triangleArray[entry].p0.y );	       
								    break;
							    case 2:
								    prop_writer.PushBackValue(triangleArray[entry].p0.z );	       
								    break;
							    default: 
                                spdlog::get("illixr")->error("should not happen #1 ");
								    break;
						    }
					    }
					    for(int i =0; i < vertex_element.num_properties(); ++i){
						    draco::PlyProperty &prop = vertex_element.property(i);
						    draco::PlyPropertyWriter<float> prop_writer(&prop);
						    switch(i){
							    case 0:
								    prop_writer.PushBackValue(triangleArray[entry].p1.x );	       
								    break;
							    case 1:
								    prop_writer.PushBackValue(triangleArray[entry].p1.y );	       
								    break;
							    case 2:
								    prop_writer.PushBackValue(triangleArray[entry].p1.z );	       
								    break;
							    default: 
                                spdlog::get("illixr")->error("should not happen #1 ");
								    break;
						    }
					    }
					    for(int i =0; i < vertex_element.num_properties(); ++i){
						    draco::PlyProperty &prop = vertex_element.property(i);
						    draco::PlyPropertyWriter<float> prop_writer(&prop);
						    switch(i){
							    case 0:
								    prop_writer.PushBackValue(triangleArray[entry].p2.x );	       
								    break;
							    case 1:
								    prop_writer.PushBackValue(triangleArray[entry].p2.y );	       
								    break;
							    case 2:
								    prop_writer.PushBackValue(triangleArray[entry].p2.z );	       
								    break;
							    default: 
                                spdlog::get("illixr")->error("should not happen #1 ");
								    break;
						    }
					    }
				    }

				    for(int entry = 0; entry < face_element.num_entries(); ++entry){
					    int actual_entry = startTriangle + entry;
					    for(int i = 0; i < face_element.num_properties(); ++i){
						    draco::PlyProperty &prop = face_element.property(i);
						    draco::PlyPropertyWriter<int32_t> prop_writer(&prop);
						    switch(i){
							    case 0:
								    prop.list_data_.push_back(prop.data_.size() / prop.data_type_num_bytes_);
								    prop.list_data_.push_back(3);
								    int val = entry * 3;
								    int val_1 = entry * 3 + 1;
								    int val_2 = entry * 3 + 2;
								    prop_writer.PushBackValue(val_2);
								    prop_writer.PushBackValue(val_1);
								    prop_writer.PushBackValue(val);
								    break;
							    case 1:
								    prop_writer.PushBackValue(triangleArray[actual_entry].vb_info.x);	       
								    break;
							    case 2:
								    prop_writer.PushBackValue(triangleArray[actual_entry].vb_info.y);	       
								    break;
							    case 3:
								    prop_writer.PushBackValue(triangleArray[actual_entry].vb_info.z);	       
								    break;
						    }
					    }
				    }
				    operations[omp_get_thread_num()](std::move(ply_reader),per_faces,per_vertices,numThreads);
			    }

			    auto roi_end = std::chrono::high_resolution_clock::now();
			    duration = std::chrono::duration_cast<std::chrono::microseconds>(roi_end - roi_start).count();
			    sr_latency<<"gen "<< scene_id << " "<<(duration/1000.0)<<"\n";

			    sr_latency.flush();
			    //pyh reset tracking
			    mainEngine->ResetActiveSceneTracking();
            spdlog::get("illixr")->info("================================InfiniTAM: frame %d finished==========================",
                   frame_count_);

		    }
		    ORcudaSafeCall(cudaThreadSynchronize());
	    }
        if (datum->depth.empty()) { spdlog::get("illixr")->info("depth empty"); }
        if (datum->rgb.empty()) { spdlog::get("illixr")->info("rgb empty"); }
		    if (datum->rgb.empty()){ printf("rgb empty\n"); }
	    }
        spdlog::get("illixr")->info("reached last frame at %d", frame_count_);
            {
                printf("reached last frame at %d\n", frame_count);
		std::cout.flush();
		sr_latency.flush();
            }

            frame_count++;
        }

        virtual ~infinitam() override{
	    sr_latency.close();
        }

    private:
        //ILLIXR related variables
        const std::shared_ptr<switchboard> sb;
        switchboard::reader<scene_recon_type> _m_scannet_datum;
        switchboard::writer<mesh_type> _m_mesh;
        //for parallel
	switchboard::writer<mesh_type> _m_mesh_0;
	switchboard::writer<mesh_type> _m_mesh_1;
	switchboard::writer<mesh_type> _m_mesh_2;
	switchboard::writer<mesh_type> _m_mesh_3;
	switchboard::writer<mesh_type> _m_mesh_4;
	switchboard::writer<mesh_type> _m_mesh_5;
	switchboard::writer<mesh_type> _m_mesh_6;
	switchboard::writer<mesh_type> _m_mesh_7;
	//switchboard::writer<mesh_type> _m_mesh_8;
	//switchboard::writer<mesh_type> _m_mesh_9;
	//switchboard::writer<mesh_type> _m_mesh_10;
	//switchboard::writer<mesh_type> _m_mesh_11;

	switchboard::writer<vb_type> _m_vb_list;

	std::unique_ptr<ORUtils::MemoryBlock<ITMLib::ITMMesh::Triangle>> cpu_triangles_;
	
	//InfiniTAM related variables
        ITMLib::ITMRGBDCalib *calib;
        ITMLib::ITMMainEngine *mainEngine;
        ITMUChar4Image *inputRGBImage;
        ITMShortImage *inputRawDepthImage;
        ITMLib::ITMLibSettings *internalSettings;
        ITMLib::ITMMainEngine::GetImageType reconstructedImageType{ITMLib::ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME};
        ITMLib::ITMMesh *mesh;
        
        std::string scene_number;
        std::string merge_name;
        
        unsigned frame_count;
	unsigned FPS=15;
	unsigned THREAD_COUNT=8;

        std::ofstream sr_latency;
        const std::string data_path = std::filesystem::current_path().string() + "/recorded_data";
};

PLUGIN_MAIN(infinitam)

// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "CLIEngine.h"

#include <fstream>
#include <iostream>
#include <cstring>

#include "../../ORUtils/FileUtils.h"
#include <chrono>
//pyh for ITM Mesh
#include "../../ITMLib/Engines/Meshing/ITMMeshingEngineFactory.h"
//pyh add for draco
#include "draco_illixr/io/ply_reader.h"
#include "draco_illixr/io/file_reader_factory.h"
#include "draco_illixr/io/file_writer_factory.h"
#include "draco_illixr/io/stdio_file_reader.h"
#include "draco_illixr/io/stdio_file_writer.h"
#include "draco_illixr/io/ply_property_writer.h"
#include "draco_illixr/io/ply_decoder.h"
#include "draco_illixr/compression/encode.h"
#include "draco_illixr/compression/expert_encode.h"
#include "draco_illixr/io/file_utils.h"
#include <sstream>

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

CLIEngine* CLIEngine::instance;

void CLIEngine::Initialise(ImageSourceEngine *imageSource, IMUSourceEngine *imuSource, ITMMainEngine *mainEngine, ITMLibSettings *settings)
{
	this->imageSource = imageSource;
	this->imuSource = imuSource;
	this->mainEngine = mainEngine;
	this->internalSettings = settings;

	this->currentFrameNo = 0;

	bool allocateGPU = false;
	if (internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA) allocateGPU = true;

	inputRGBImage = new ITMUChar4Image(imageSource->getRGBImageSize(), true, allocateGPU);
	inputRawDepthImage = new ITMShortImage(imageSource->getDepthImageSize(), true, allocateGPU);
	inputIMUMeasurement = new ITMIMUMeasurement();
	outImage = new ITMUChar4Image(imageSource->getRGBImageSize(), true, allocateGPU);

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif

	sdkCreateTimer(&timer_instant);
	sdkCreateTimer(&timer_average);

	sdkResetTimer(&timer_average);

	freqControl.freqDivisor = mainEngine->GetFreqDivisor();
	freqControl.framesSinceFreqChange = 0;
	freqControl.processed = new std::vector<unsigned>;
	freqControl.frequencies = new std::vector<double>;
	freqControl.newBricks = new std::vector<unsigned>;

	raycastingFreqDivisor = ITMLibSettings::MAX_FREQ / static_cast<unsigned>(internalSettings->raycastingFreq);

    //pyh add mesh count
    mesh=new ITMMesh(MEMORYDEVICE_CUDA,0);
    mesh_count = 0;
    //pyh intialize draco reader
    draco_illixr::FileReaderFactory::RegisterReader(draco_illixr::StdioFileReader::Open);
    draco_illixr::FileWriterFactory::RegisterWriter(draco_illixr::StdioFileWriter::Open);
    printf("initialised.\n");
}


bool CLIEngine::ProcessFrame()
{
	std::cout << "\n============================ Begin Frame =============================\n";

	// Frequency control
	freqControl.frequencies->push_back(ITMLibSettings::MAX_FREQ / static_cast<double>(freqControl.freqDivisor));
	bool shouldSkip = (freqControl.framesSinceFreqChange % freqControl.freqDivisor) != 0;
	if (shouldSkip)
	{
		std::cout << "Skipping frame " << currentFrameNo << "\n";
		imageSource->skipImage();
		mainEngine->SkipFrame();
		freqControl.processed->push_back(0);
		freqControl.newBricks->push_back(0);
	}
	else
	{
		std::cout << "Running frame " << currentFrameNo << "\n";

		if (!imageSource->hasMoreImages()) return false;
		imageSource->getImages(inputRGBImage, inputRawDepthImage);

		if (imuSource != NULL) {
			if (!imuSource->hasMoreMeasurements()) return false;
			else imuSource->getMeasurement(inputIMUMeasurement);
		}

		sdkResetTimer(&timer_instant);
		sdkStartTimer(&timer_instant); sdkStartTimer(&timer_average);

		//actual processing on the mainEngine
		mainEngine->currentTimeStamp = imageSource->currentTimeStamp;
		if (imuSource != NULL) mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, inputIMUMeasurement);
		else mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage);

#ifndef COMPILE_WITHOUT_CUDA
		ORcudaSafeCall(cudaThreadSynchronize());
#endif
		sdkStopTimer(&timer_instant); sdkStopTimer(&timer_average);

		float processedTime_inst = sdkGetTimerValue(&timer_instant);
		float processedTime_avg = sdkGetAverageTimerValue(&timer_average);

		printf("frame %i: time %.2f, avg %.2f\n", currentFrameNo, processedTime_inst, processedTime_avg);

		freqControl.processed->push_back(1);
		freqControl.newBricks->push_back(mainEngine->GetNumNewBricks());
		unsigned newDivisor = mainEngine->GetFreqDivisor();
		if (newDivisor != freqControl.freqDivisor)
		{
			freqControl.freqDivisor = newDivisor;
			freqControl.framesSinceFreqChange = 0;
		}
	}

	// Obtain visualisation if we're running in decoupled mode
	if (internalSettings->useDecoupledRaycasting)
	{
		if ((currentFrameNo % raycastingFreqDivisor) == 0 && currentFrameNo!=0)
		{
            //pyh: this tiggers at raycasting frequency?
			//mainEngine->GetImage(outImage, ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST);

            //pyh add GetMesh Function
            auto start = std::chrono::high_resolution_clock::now();
            mainEngine->GetMesh(mesh, 0);
            //I need to take into acccount mesh transfer time
            ORUtils::MemoryBlock<ITMLib::ITMMesh::Triangle> *cpu_triangles;
            cpu_triangles = new ORUtils::MemoryBlock<ITMLib::ITMMesh::Triangle>(mesh->noMaxTriangles, MEMORYDEVICE_CPU);
            cpu_triangles->SetFrom(mesh->triangles, ORUtils::MemoryBlock<ITMLib::ITMMesh::Triangle>::CUDA_TO_CPU);
            ITMLib::ITMMesh::Triangle *triangleArray = cpu_triangles->GetData(MEMORYDEVICE_CPU);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            double duration_ms = duration / 1000.0;
            printf("Time to get mesh: %.3f milliseconds\n", duration_ms);


            std::string file_name = "ITM-BE_";
            //std::string merge_name = file_name + this->scene_name + "_" + std::to_string(mesh_count) +".obj";

            //mesh->WriteOBJ(merge_name.c_str());
            //auto end_v2 = std::chrono::high_resolution_clock::now();
            //duration = std::chrono::duration_cast<std::chrono::microseconds>(end_v2 - end).count();
            //duration_ms = duration / 1000.0;
            //printf("Time to write mesh out: %.3f milliseconds\n", duration_ms);

            //pyh essentially here is just implemeting a WriteDraco for ITMMesh

            //TODO DEBUG:
            //write this mesh directly into google draco buffer or directly into ply_reader
            //4/23 attempt write to draco_illixr::buffer instead
            unsigned vertices_size = mesh->noTotalTriangles *3;
            std::ostringstream ply_info;
            ply_info << "ply\n";
            ply_info << "format ascii 1.0\n";
            ply_info << "element vertex " << vertices_size << "\n";
            ply_info << "property float x\n";
            ply_info << "property float y\n";
            ply_info << "property float z\n";
            ply_info << "property uint8 red\n";
            ply_info << "property uint8 green\n";
            ply_info << "property uint8 blue\n";
            ply_info << "element face "<< mesh->noTotalTriangles <<"\n";
            ply_info << "property list uint8 int vertex_indices\n";
            ply_info << "end_header\n";
            //printf("vertices size %u \n", vertices_size);
            for(unsigned entry = 0; entry < mesh->noTotalTriangles; ++entry){
                ply_info
                    << triangleArray[entry].p0.x << " "
                    << triangleArray[entry].p0.y << " "
                    << triangleArray[entry].p0.z << " "
                    << static_cast<int>(triangleArray[entry].clr0.r) << " "
                    << static_cast<int>(triangleArray[entry].clr0.g) << " "
                    << static_cast<int>(triangleArray[entry].clr0.b) << "\n";
                ply_info
                    << triangleArray[entry].p1.x << " "
                    << triangleArray[entry].p1.y << " "
                    << triangleArray[entry].p1.z << " "
                    << static_cast<int>(triangleArray[entry].clr1.r) << " "
                    << static_cast<int>(triangleArray[entry].clr1.g) << " "
                    << static_cast<int>(triangleArray[entry].clr1.b) << "\n";
                ply_info
                    << triangleArray[entry].p2.x << " "
                    << triangleArray[entry].p2.y << " "
                    << triangleArray[entry].p2.z << " "
                    << static_cast<int>(triangleArray[entry].clr2.r) << " "
                    << static_cast<int>(triangleArray[entry].clr2.g) << " "
                    << static_cast<int>(triangleArray[entry].clr2.b) << "\n";

            }

            for(unsigned entry = 0; entry < mesh->noTotalTriangles; ++entry){
                unsigned v1 = entry*3 + 2;
                unsigned v2 = entry*3 + 1;
                unsigned v3 = entry*3 + 0;
                ply_info << "3 "<< v1 <<" "<<v2 <<" "<<v3<<"\n";
            }
            std::vector<char> buffer;
            std::string str = ply_info.str();
            buffer.resize(str.size());
            std::copy(str.begin(), str.end(), buffer.begin());

            auto end_v2 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_v2 - end).count();
            duration_ms = duration / 1000.0;
            printf("Time to write mesh to buffer: %.3f milliseconds\n", duration_ms);
            //std::string merge_name = file_name + this->scene_name + "_" + std::to_string(mesh_count) +".ply";
            //std::ofstream outfile(merge_name, std::ios::out);
            //if(!outfile){throw std::runtime_error("failed to openfile");}
            //outfile.write(buffer.data(),buffer.size());
            //outfile.close();

            draco_illixr::PlyDecoder *ply_decoder = new draco_illixr::PlyDecoder();
            ply_decoder->buffer_.Init(&buffer[0], buffer.size());
            printf("buffer size %zu\n",ply_decoder->buffer_.data_size_);

            std::unique_ptr<draco_illixr::Mesh> draco_mesh(new draco_illixr::Mesh());
            ply_decoder->DecodeFromBuffer(&ply_decoder->buffer_, draco_mesh.get());

            std::unique_ptr<draco_illixr::PointCloud> draco_pc;
            draco_illixr::Mesh *temp_mesh = draco_mesh.get();
            printf("mesh face #: %u\n", temp_mesh->num_faces());
            draco_pc = std::move(draco_mesh);

            int pos_quantization_bits = 11;
            int tex_coords_quantization_bits = 10;
            bool tex_coords_deleted = false;
            int normals_quantization_bits = 8;
            bool normals_deleted = false;
            int generic_quantization_bits = 8;
            bool generic_deleted = false;
            int compression_level = 7;
            draco_illixr::Encoder encoder;
            encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::POSITION,pos_quantization_bits);
            encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::TEX_COORD,tex_coords_quantization_bits);
            encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::NORMAL,normals_quantization_bits);
            encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::GENERIC,generic_quantization_bits);
            int speed = 10-compression_level;
            encoder.SetSpeedOptions(speed, speed);

            std::unique_ptr<draco_illixr::ExpertEncoder> expert_encoder;
            expert_encoder.reset(new draco_illixr::ExpertEncoder(*temp_mesh));
            expert_encoder->Reset(encoder.CreateExpertEncoderOptions(*draco_pc));

            draco_illixr::EncoderBuffer draco_buffer;
            const draco_illixr::Status status = expert_encoder->EncodeToBuffer(&draco_buffer);
            if(!status.ok()){
                printf("Failed to encode the mesh\n");
            }
            auto end_v3 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_v3 - end_v2).count();
            duration_ms = duration / 1000.0;
            printf("Time to encode: %.3f milliseconds\n", duration_ms);
            //std::string draco_name = "draco" + this->scene_name + "_" + std::to_string(mesh_count) +".drc";
            //Draco::WriteBufferToFile(draco_buffer.data(), buffer.size(), draco_name);




            //this below version have deduplication mismatch

            //printf("#ofFaces: %u\n",mesh->noTotalTriangles);
            ////pyh creating PLYReader class for parsing ply data
            //draco_illixr::PlyReader *ply_reader = new draco_illixr::PlyReader();
            //ply_reader->format_=draco_illixr::PlyReader::kAscii;
            //
            ////pyh implement ParseHeader & ParseProperty
            //ply_reader->element_index_["vertex"] = static_cast<uint32_t>(ply_reader->elements_.size());
            //ply_reader->elements_.emplace_back(draco_illixr::PlyElement("vertex",mesh->noTotalTriangles*3));
            //
            //// x, y, z float -> DT_FLOAT32
            //ply_reader->elements_.back().AddProperty(draco_illixr::PlyProperty("x", draco_illixr::DT_FLOAT32, draco_illixr::DT_INVALID));
            //ply_reader->elements_.back().AddProperty(draco_illixr::PlyProperty("y", draco_illixr::DT_FLOAT32, draco_illixr::DT_INVALID));
            //ply_reader->elements_.back().AddProperty(draco_illixr::PlyProperty("z", draco_illixr::DT_FLOAT32, draco_illixr::DT_INVALID));
            //
            //// r g b  uchar -> DT_UINT8
            //ply_reader->elements_.back().AddProperty(draco_illixr::PlyProperty("red", draco_illixr::DT_UINT8, draco_illixr::DT_INVALID));
            //ply_reader->elements_.back().AddProperty(draco_illixr::PlyProperty("green", draco_illixr::DT_UINT8, draco_illixr::DT_INVALID));
            //ply_reader->elements_.back().AddProperty(draco_illixr::PlyProperty("blue", draco_illixr::DT_UINT8, draco_illixr::DT_INVALID));
            //
            ////add face
            //ply_reader->element_index_["face"] = static_cast<uint32_t>(ply_reader->elements_.size());
            //ply_reader->elements_.emplace_back(draco_illixr::PlyElement("face",mesh->noTotalTriangles));
            //
            ////add face property, list type uchar, datatype int
            //ply_reader->elements_.back().AddProperty(draco_illixr::PlyProperty("vertex_indices", draco_illixr::DT_INT32, draco_illixr::DT_UINT8));
            //
            ////pyh implmement ParsePropertiesData -> ParseElementDataAscii
            //for(int i=0; i< static_cast<int> (ply_reader->elements_.size()); i++){
            //    //parse element data ascii
            //    //first parse vertex which has 6 properties x,y,z r,g,b
            //    draco_illixr::PlyElement &element = ply_reader->elements_[i];
            //    printf("parse element name %s, # of properties %u, # of entries: %u \n", element.name_.c_str(), element.num_properties(), element.num_entries());
            //    if(strcmp(element.name_.c_str(), "vertex")==0)
            //    {
            //        printf("found vertex\n");
            //        draco_illixr::PlyProperty &prop_x = element.property(0);
            //        draco_illixr::PlyPropertyWriter<double> prop_writer_x(&prop_x);
            //
            //        draco_illixr::PlyProperty &prop_y = element.property(1);
            //        draco_illixr::PlyPropertyWriter<double> prop_writer_y(&prop_y);
            //
            //        draco_illixr::PlyProperty &prop_z = element.property(2);
            //        draco_illixr::PlyPropertyWriter<double> prop_writer_z(&prop_z);
            //
            //        draco_illixr::PlyProperty &prop_r = element.property(3);
            //        draco_illixr::PlyPropertyWriter<double> prop_writer_r(&prop_r);
            //
            //        draco_illixr::PlyProperty &prop_g = element.property(4);
            //        draco_illixr::PlyPropertyWriter<double> prop_writer_g(&prop_g);
            //
            //        draco_illixr::PlyProperty &prop_b = element.property(5);
            //        draco_illixr::PlyPropertyWriter<double> prop_writer_b(&prop_b);
            //
            //        //in the original code, each loop is actually writting three vertices, so modified the loop accordingly
            //        for(unsigned entry = 0; entry < mesh->noTotalTriangles; ++entry){
            //            prop_writer_x.PushBackValue(triangleArray[entry].p0.x);
            //            prop_writer_y.PushBackValue(triangleArray[entry].p0.y);
            //            prop_writer_z.PushBackValue(triangleArray[entry].p0.z);
            //
            //            prop_writer_r.PushBackValue(static_cast<uint8_t>(triangleArray[entry].clr0.r));
            //            prop_writer_g.PushBackValue(static_cast<uint8_t>(triangleArray[entry].clr0.g));
            //            prop_writer_b.PushBackValue(static_cast<uint8_t>(triangleArray[entry].clr0.b));

            //        }
            //    }
            //    else if (strcmp(element.name_.c_str(), "face")==0){
            //        draco_illixr::PlyProperty &prop_face = element.property(0);
            //        draco_illixr::PlyPropertyWriter<double> prop_writer_face(&prop_face);
            //        for(unsigned entry = 0; entry < mesh->noTotalTriangles; ++entry){
            //            //printf("prop.data_.size() %zu, prop.data_type_num_bytes: %d\n", prop_face.data_.size(), prop_face.data_type_num_bytes_);
            //            prop_face.list_data_.push_back(prop_face.data_.size() / prop_face.data_type_num_bytes_);
            //            prop_face.list_data_.push_back(3);
            //            unsigned idx_start = entry*3;
            //            prop_writer_face.PushBackValue(idx_start+3);
            //            prop_writer_face.PushBackValue(idx_start+2);
            //            prop_writer_face.PushBackValue(idx_start+1);
            //        }
            //    }
            //}
            //
            //
            ////ReadMeshFromFile();
            //std::unique_ptr<draco_illixr::Mesh> draco_mesh(new draco_illixr::Mesh());
            //draco_illixr::PlyDecoder *ply_decoder = new draco_illixr::PlyDecoder();
            //ply_decoder->out_mesh_ = draco_mesh.get();
            //
            ////ply decoder DecodeFromFile
            //ply_decoder->out_point_cloud_ = static_cast<draco_illixr::PointCloud *>(draco_mesh.get());

            ////pyh decode face data (DecodeInternal()
            //printf("DecodeInternal\n");
            //ply_decoder->DecodeFaceData(ply_reader->GetElementByName("face"));
            //ply_decoder->DecodeVertexData(ply_reader->GetElementByName("vertex"));
            //printf("point cloud data before deduplication %u\n", ply_decoder->out_point_cloud_->num_points());
            //ply_decoder->out_point_cloud_->DeduplicateAttributeValues();
            //ply_decoder->out_point_cloud_->DeduplicatePointIds();
            //printf("point cloud data after deduplication %u\n", ply_decoder->out_point_cloud_->num_points());
            //
            ////pyh this move the ownership to draco_pc
            //draco_illixr::PointCloud* draco_pc = static_cast<draco_illixr::PointCloud *>(draco_mesh.get());
            //
            //int pos_quantization_bits = 11;
            //int tex_coords_quantization_bits = 10;
            //bool tex_coords_deleted = false;
            //int normals_quantization_bits = 8;
            //bool normals_deleted = false;
            //int generic_quantization_bits = 8;
            //bool generic_deleted = false;
            //int compression_level = 7;
            //draco_illixr::Encoder encoder;
            //encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::POSITION,pos_quantization_bits);
            //encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::TEX_COORD,tex_coords_quantization_bits);
            //encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::NORMAL,normals_quantization_bits);
            //encoder.SetAttributeQuantization(draco_illixr::GeometryAttribute::GENERIC,generic_quantization_bits);
            //int speed = 10-compression_level;
            //encoder.SetSpeedOptions(speed, speed);

            //std::unique_ptr<draco_illixr::ExpertEncoder> expert_encoder;
            //expert_encoder.reset(new draco_illixr::ExpertEncoder(*draco_mesh));
            //expert_encoder->Reset(encoder.CreateExpertEncoderOptions(*draco_pc));
            //
            ////EncodeMeshToFile
            //printf("Encode Mesh to File\n");
            //draco_illixr::EncoderBuffer buffer;
            //expert_encoder->EncodeToBuffer(&buffer);
            //printf("reached here\n");
            //std::string draco_name = "draco" + this->scene_name + "_" + std::to_string(mesh_count) +".drc";
            //printf("WriteBufferToFile\n");
            //draco_illixr::WriteBufferToFile(buffer.data(), buffer.size(), draco_name);
            mesh_count++;
            //
            //////pyh delete the object we created here
            //delete ply_reader;
            delete ply_decoder;
            //draco_mesh=nullptr;
            draco_pc = nullptr;
            //expert_encoder=nullptr;
            delete cpu_triangles;


            //load the obj
            //draco_illixr::Options load_options;
            //load_options.SetBool("use_metadata", false);
            //load_options.SetBool("preserve_polygons", false);
            //auto maybe_mesh = draco_illixr::ReadMeshFromFile(merge_name, load_options);
            //if(!maybe_mesh.ok())
            //{
            //    printf("Failed loading the input mesh: %s.\n",
            //            maybe_mesh.status().error_msg());
            //    return false;
            //}
            //printf("finished loading mesh\n");

		}
	}

	currentFrameNo++;
	freqControl.framesSinceFreqChange++;
	std::cout << "============================= End Frame ==============================\n";
	return true;
}

void CLIEngine::Run()
{
	while (true) {
		if (!ProcessFrame()) break;
	}
}

void CLIEngine::Shutdown()
{
	sdkDeleteTimer(&timer_instant);
	sdkDeleteTimer(&timer_average);

	delete inputRGBImage;
	delete inputRawDepthImage;
	delete inputIMUMeasurement;
	delete outImage;

	delete instance;

    std::string file_name="execution_data_";
	//std::ofstream frameFile("execution_data.csv");

    //pyh add scene dependent output file
    std::string merge_name=file_name + this->scene_name+".csv";
	std::ofstream frameFile(merge_name.c_str());

    frameFile << "#Frame,New Bricks,Frequency\n";
	for (unsigned idx = 0; idx < freqControl.processed->size(); idx++)
		frameFile << idx << "," << freqControl.newBricks->at(idx) << "," << freqControl.frequencies->at(idx) << "\n";
	frameFile.close();

	delete freqControl.processed;
	delete freqControl.frequencies;
	delete freqControl.newBricks;
}

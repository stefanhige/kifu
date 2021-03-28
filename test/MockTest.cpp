//#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>

#include "DataTypes.h"
#include "SurfaceMeasurer.h"
#include "SurfacePredictor.h"
#include "SurfaceReconstructor.h"
#include "PoseEstimator.h"
#include "KinectFusion.h"


class MockSurfaceMeasurer : public ISurfaceMeasurer
{
public:
    MockSurfaceMeasurer(){}
    MOCK_METHOD(void, registerInput, (float*), (override));
    MOCK_METHOD(void, saveDepthMap, (std::string), (override));
    MOCK_METHOD(void, process, (), (override));
    MOCK_METHOD(PointCloud, getPointCloud, (), (override));
};

class MockSurfacePredictor : public ISurfacePredictor
{
public:
    MockSurfacePredictor(){}
    MOCK_METHOD(PointCloud, predict, (const ImageSize &, const Matrix4f), (const, override));
    MOCK_METHOD(void, predictColor, (uint8_t*, const ImageSize&, const Matrix4f), (const, override));

};

class MockSurfaceReconstructor : public ISurfaceReconstructor
{
public:
    MOCK_METHOD(void, reconstruct, (const float*, const uint8_t*, const ImageSize&, const Matrix4f&), (override));
};

class MockPoseEstimator : public IPoseEstimator
{
public:
    MockPoseEstimator(){}
    MOCK_METHOD(Matrix4f, estimatePose, (Matrix4f), (override));

};

KiFuModel constructKiFu(std::shared_ptr<VirtualSensor> sensor)
{
    // 512 will be ~500MB ram
    // 1024 -> 4GB
    auto tsdf = std::make_shared<Tsdf>(256, 1);

    auto surfaceMeasurer = std::make_unique<MockSurfaceMeasurer>();
    auto surfaceReconstructor = std::make_unique<MockSurfaceReconstructor>();
    auto poseEstimator = std::make_unique<MockPoseEstimator>();
    auto surfacePredictor = std::make_unique<MockSurfacePredictor>();
    return KiFuModel(sensor, std::move(surfaceMeasurer), std::move(surfaceReconstructor), std::move(poseEstimator), std::move(surfacePredictor), tsdf);
}


std::shared_ptr<VirtualSensor> constructVirtualSensor()
{
    // load video

    std::string filenameIn = std::string("../kifu/data/rgbd_dataset_freiburg1_xyz/");

    //Verify that folder exists
    std::filesystem::path executableFolderPath =  std::filesystem::canonical("/proc/self/exe").parent_path();    //Folder of executable from system call
    std::filesystem::path dataFolderLocation = executableFolderPath.parent_path() / filenameIn;

    if (!std::filesystem::exists(dataFolderLocation))
    {
        std::cout << "No input files at folder " << dataFolderLocation << std::endl;
    }

    std::cout << "Initialize virtual sensor..." << std::endl;

    auto sensor = std::make_shared<VirtualSensor>();
    if (!sensor->init(dataFolderLocation))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
    }
    return sensor;
}


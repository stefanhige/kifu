//#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>

#include "DataTypes.h"
#include "VirtualSensor.h"
#include "SurfaceMeasurer.h"
#include "SurfacePredictor.h"
#include "SurfaceReconstructor.h"
#include "PoseEstimator.h"
#include "KinectFusion.h"

class MockVirtualSensor : public IVirtualSensor
{
public:
    MockVirtualSensor(){}
   MOCK_METHOD(bool, processNextFrame, (), (override));
   MOCK_METHOD(Eigen::Matrix4f, getTrajectory, (), (const, override));
   MOCK_METHOD(float*, getDepth, (), (override));
   MOCK_METHOD(BYTE*, getColorRGBX, (), (override));
   MOCK_METHOD(ImageSize, getDepthImageSize, (), (const, override));

};


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


TEST(KifuTest, ConstructorTest)
{
    auto sensor = std::make_shared<MockVirtualSensor>();

    // 512 will be ~500MB ram
    // 1024 -> 4GB
    auto tsdf = std::make_shared<Tsdf>(16, 1);

    auto surfaceMeasurer = std::make_unique<MockSurfaceMeasurer>();
    auto surfaceReconstructor = std::make_unique<MockSurfaceReconstructor>();
    auto poseEstimator = std::make_unique<MockPoseEstimator>();
    auto surfacePredictor = std::make_unique<MockSurfacePredictor>();

    EXPECT_CALL(*surfaceMeasurer, registerInput).Times(2);
    EXPECT_CALL(*surfaceMeasurer, process).Times(2);
    EXPECT_CALL(*surfaceMeasurer, getPointCloud).Times(2);

    EXPECT_CALL(*surfaceReconstructor, reconstruct).Times(1);



    KiFuModel model(sensor, std::move(surfaceMeasurer), std::move(surfaceReconstructor), std::move(poseEstimator), std::move(surfacePredictor), tsdf);
}


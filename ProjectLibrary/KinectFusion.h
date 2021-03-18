#pragma once

#include <string>
#include <assert.h>
#include <memory>
#include <mutex>
#include <thread>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "NearestNeighbor.h"
#include "DataTypes.h"
#include "SurfaceReconstructor.h"
#include "SurfaceMeasurer.h"
#include "PoseEstimator.h"
#include "SurfacePredictor.h"

// debug
#include "SimpleMesh.h"

//template<class InputType>
class KiFuModel
{
public:
    KiFuModel(const std::shared_ptr<VirtualSensor> & inputHandle,
              std::unique_ptr<ISurfaceMeasurer> && surfaceMeasurer,
              std::unique_ptr<ISurfaceReconstructor> && surfaceReconstructor,
              std::unique_ptr<IPoseEstimator> && poseEstimator,
              std::unique_ptr<ISurfacePredictor> && surfacePredictor,
              std::shared_ptr<Tsdf> tsdf);

    bool processNextFrame();

    // debug method
    void saveTsdf(std::string filename, float tsdfThreshold = 0.01, float weightThreshold = 0) const;

    // debug method
    void saveScreenshot(std::string filename, const Matrix4f pose=Matrix4f::Identity()) const;

private:
    void prepareNextFrame(bool &result);

    std::shared_ptr<VirtualSensor> m_InputHandle;

    std::unique_ptr<ISurfaceMeasurer> m_SurfaceMeasurer;
    std::unique_ptr<ISurfaceReconstructor> m_SurfaceReconstructor;
    std::unique_ptr<IPoseEstimator> m_PoseEstimator;
    std::unique_ptr<ISurfacePredictor> m_SurfacePredictor;

    std::shared_ptr<Tsdf> m_tsdf;

    PointCloud m_nextFrame;
    std::mutex m_nextFrameMutex;

    Matrix4f m_CamToWorld;
    std::vector<Matrix4f> m_currentPose;
    std::vector<Matrix4f> m_currentPoseGroundTruth;
    const Matrix4f m_refPoseGroundTruth;

};

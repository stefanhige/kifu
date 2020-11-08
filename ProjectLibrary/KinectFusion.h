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
    KiFuModel(VirtualSensor & InputHandle);

    bool processNextFrame();

    // debug method
    void saveTsdf(std::string filename, float tsdfThreshold = 0.01, float weightThreshold = 0) const;

    // debug method
    void saveScreenshot(std::string filename, const Matrix4f pose=Matrix4f::Identity()) const;

private:
    void prepareNextFrame(bool &result);

    std::unique_ptr<SurfaceMeasurer> m_SurfaceMeasurer;
    std::unique_ptr<PoseEstimator> m_PoseEstimator;
    std::unique_ptr<SurfaceReconstructor> m_SurfaceReconstructor;
    std::unique_ptr<SurfacePredictor> m_SurfacePredictor;

    PointCloud m_nextFrame;
    std::mutex m_nextFrameMutex;


    Matrix4f m_CamToWorld;
    Matrix4f m_currentPose;

    VirtualSensor* m_InputHandle;
    std::shared_ptr<Tsdf> m_tsdf;
};

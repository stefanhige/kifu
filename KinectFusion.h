#pragma once

#include <string>
#include <assert.h>
#include <memory>

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

    void saveTsdf(std::string filename);
    // debug method

    void saveScreenshot(std::string filename, const Matrix4f pose=Matrix4f::Identity());

    std::unique_ptr<SurfaceMeasurer> m_SurfaceMeasurer;
    std::unique_ptr<PoseEstimator> m_PoseEstimator;
    std::unique_ptr<SurfaceReconstructor> m_SurfaceReconstructor;
    std::unique_ptr<SurfacePredictor> m_SurfacePredictor;

    Matrix4f m_CamToWorld;
    Matrix4f m_currentPose;
    VirtualSensor* m_InputHandle;
    std::string param;
    std::shared_ptr<Tsdf> m_tsdf;
};

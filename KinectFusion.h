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

typedef VirtualSensor InputType;
//template<class InputType>
class KiFuModel
{
public:
    KiFuModel(InputType& InputHandle)
        : m_InputHandle(&InputHandle)
    {
        m_InputHandle->processNextFrame();
        m_SurfaceMeasurer = std::make_unique<SurfaceMeasurer>(m_InputHandle->getDepthIntrinsics(),
                                                m_InputHandle->getDepthImageHeight(),
                                                m_InputHandle->getDepthImageWidth());
        m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());
        //m_SurfaceMeasurer->smoothInput();
        //m_SurfaceMeasurer->displayDepthMap();
        m_SurfaceMeasurer->process();

        PointCloud Frame0 = m_SurfaceMeasurer->getPointCloud();

        std::vector<bool> pointsAndNormalsValid;
        pointsAndNormalsValid.reserve( Frame0.pointsValid.size() );

        std::transform(Frame0.pointsValid.begin(), Frame0.pointsValid.end(), Frame0.normalsValid.begin(),
                       std::back_inserter(pointsAndNormalsValid), std::logical_and<>());


        m_PoseEstimator = std::make_unique<NearestNeighborPoseEstimator>();
        m_PoseEstimator->setTarget(PoseEstimator::pruneVector(Frame0.points, pointsAndNormalsValid),
                                   PoseEstimator::pruneVector(Frame0.normals, pointsAndNormalsValid));

        // set to 64
        // 512 will be ~500MB ram
        // 1024 -> 4GB
        m_tsdf = std::make_shared<Tsdf>(128, 1);
        m_tsdf->calcVoxelSize(Frame0);

        m_SurfaceReconstructor = std::make_unique<SurfaceReconstructor>(m_tsdf, m_InputHandle->getDepthIntrinsics());
        m_SurfaceReconstructor->reconstruct(m_InputHandle->getDepth(),
                                            m_InputHandle->getDepthImageHeight(),
                                            m_InputHandle->getDepthImageWidth(),
                                            Matrix4f::Identity());

        //m_tsdf->writeToFile("tsdf-test.ply", 0.01, 0);

        m_SurfacePredictor = std::make_unique<SurfacePredictor>(m_tsdf, m_InputHandle->getDepthIntrinsics());

        PointCloud Frame0_predicted = m_SurfacePredictor->predict(m_InputHandle->getDepthImageHeight(),
                                        m_InputHandle->getDepthImageWidth());

        Matrix4f pose = Matrix4f::Identity();
        /* DEBUG
        SimpleMesh Frame0_predicted_mesh(Frame0_predicted,
                                         m_InputHandle->getDepthImageHeight(),
                                         m_InputHandle->getDepthImageWidth(),
                                         1);


        SimpleMesh Frame0_mesh(Frame0,
                               m_InputHandle->getDepthImageHeight(),
                               m_InputHandle->getDepthImageWidth());

        //SimpleMesh Frame0_mesh(*m_InputHandle, pose);

        Frame0_mesh.writeMesh("Frame0_mesh.off");
        Frame0_predicted_mesh.writeMesh("Frame0_predicted_mesh.off");
        */

    }


    bool processNextFrame()
    {
        if(!m_InputHandle->processNextFrame())
        {
            return true;
        }
        m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());
        m_SurfaceMeasurer->process();

        PointCloud nextFrame = m_SurfaceMeasurer->getPointCloud();

        std::vector<bool> pointsAndNormalsValid;
        pointsAndNormalsValid.reserve( nextFrame.pointsValid.size() );

        std::transform(nextFrame.pointsValid.begin(), nextFrame.pointsValid.end(), nextFrame.normalsValid.begin(),
            std::back_inserter(pointsAndNormalsValid), std::logical_and<>());

        m_PoseEstimator->setSource(PoseEstimator::pruneVector(nextFrame.points, pointsAndNormalsValid),
                                   PoseEstimator::pruneVector(nextFrame.normals, pointsAndNormalsValid), 8);


        // matrix inverse????
        Matrix4f currentCamToWorld = m_PoseEstimator->estimatePose();
        Matrix4f currentPose = currentCamToWorld.inverse();


        std::cout << currentPose << std::endl;

        return false;
    }


private:
    std::unique_ptr<SurfaceMeasurer> m_SurfaceMeasurer;
    std::unique_ptr<PoseEstimator> m_PoseEstimator;
    std::unique_ptr<SurfaceReconstructor> m_SurfaceReconstructor;
    std::unique_ptr<SurfacePredictor> m_SurfacePredictor;

    InputType* m_InputHandle;
    std::string param;
    std::shared_ptr<Tsdf> m_tsdf;

};

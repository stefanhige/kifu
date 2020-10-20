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

        PointCloud Frame0_pruned = Frame0;
        Frame0_pruned.prune();

        m_PoseEstimator = std::make_unique<NearestNeighborPoseEstimator>();
        m_PoseEstimator->setTarget(Frame0_pruned.points, Frame0.normals);

        // 512 will be ~500MB ram
        // 1024 -> 4GB
        m_tsdf = std::make_shared<Tsdf>(256, 1);
        m_tsdf->calcVoxelSize(Frame0);

        m_SurfaceReconstructor = std::make_unique<SurfaceReconstructor>(m_tsdf, m_InputHandle->getDepthIntrinsics());
        m_SurfaceReconstructor->reconstruct(m_InputHandle->getDepth(),
                                            m_InputHandle->getDepthImageHeight(),
                                            m_InputHandle->getDepthImageWidth(),
                                            Matrix4f::Identity());

        m_tsdf->writeToFile("tsdf-test.ply", 0.01, 0);

        m_SurfacePredictor = std::make_unique<SurfacePredictor>(m_tsdf, m_InputHandle->getDepthIntrinsics());

        PointCloud Frame0_predicted = m_SurfacePredictor->predict(m_InputHandle->getDepthImageHeight(),
                                        m_InputHandle->getDepthImageWidth());

        PointCloud Frame0_predicted_pruned = Frame0_predicted;
        Frame0_predicted_pruned.prune();

        Matrix4f pose = Matrix4f::Identity();

        SimpleMesh Frame0_predicted_mesh(Frame0_predicted,
                                         m_InputHandle->getDepthImageHeight(),
                                         m_InputHandle->getDepthImageWidth(),
                                         true);


        SimpleMesh Frame0_mesh(Frame0,
                               m_InputHandle->getDepthImageHeight(),
                               m_InputHandle->getDepthImageWidth(),
                               true);

        //SimpleMesh Frame0_mesh(*m_InputHandle, pose);

        Frame0_mesh.writeMesh("Frame0_mesh.off");
        Frame0_predicted_mesh.writeMesh("Frame0_predicted_mesh.off");


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

        PointCloud nextFramePruned = nextFrame;
        nextFramePruned.prune();

        // set targetfrom SurfacePredictor

        m_PoseEstimator->setSource(nextFramePruned.points, nextFramePruned.normals, 8);

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

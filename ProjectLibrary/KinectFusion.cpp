#include "KinectFusion.h"
#include "StopWatch.h"


KiFuModel::KiFuModel(const std::shared_ptr<IVirtualSensor> & InputHandle,
                     std::unique_ptr<ISurfaceMeasurer> && surfaceMeasurer,
                     std::unique_ptr<ISurfaceReconstructor> && surfaceReconstructor,
                     std::unique_ptr<IPoseEstimator> && poseEstimator,
                     std::unique_ptr<ISurfacePredictor> && surfacePredictor,
                     std::shared_ptr<Tsdf> tsdf)
    : m_InputHandle(InputHandle),
      m_SurfaceMeasurer(std::forward<std::unique_ptr<ISurfaceMeasurer>>(surfaceMeasurer)),
      m_SurfaceReconstructor(std::forward<std::unique_ptr<ISurfaceReconstructor>>(surfaceReconstructor)),
      m_PoseEstimator(std::forward<std::unique_ptr<IPoseEstimator>>(poseEstimator)),
      m_SurfacePredictor(std::forward<std::unique_ptr<ISurfacePredictor>>(surfacePredictor)),
      m_tsdf(tsdf),
      m_refPoseGroundTruth((m_InputHandle->processNextFrame(), m_InputHandle->getTrajectory()))
{

    m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());
    //m_SurfaceMeasurer->smoothInput();

    m_SurfaceMeasurer->process();

    PointCloud Frame0 = m_SurfaceMeasurer->getPointCloud();

    PointCloud Frame0_pruned = Frame0;
    Frame0_pruned.prune();


    m_tsdf->calcVoxelSize(Frame0);

    //m_SurfaceReconstructor = std::make_unique<SurfaceReconstructor>(m_tsdf, m_InputHandle->getDepthIntrinsics());

    m_SurfaceReconstructor->reconstruct(m_InputHandle->getDepth(),
                                        m_InputHandle->getColorRGBX(),
                                        m_InputHandle->getDepthImageSize(),
                                        Matrix4f::Identity());

    //m_tsdf->writeToFile("tsdf_frame0.ply", 0.01, 0);

    m_currentPose.push_back(Matrix4f::Identity());
    m_CamToWorld = Matrix4f::Identity();
    m_currentPoseGroundTruth.push_back(m_InputHandle->getTrajectory() * m_refPoseGroundTruth.inverse());

    // TODO: check return value
    prepareNextFrame();

    //SimpleMesh camMesh = SimpleMesh::camera(m_currentPose.back());
    //camMesh.writeMesh("CamMesh.off");

//        PointCloud Frame0_predicted_pruned = Frame0_predicted;
//        Frame0_predicted_pruned.prune();

//        Matrix4f pose = Matrix4f::Identity();

//        SimpleMesh Frame0_predicted_mesh(Frame0_predicted,
//                                         m_InputHandle->getDepthImageHeight(),
//                                         m_InputHandle->getDepthImageWidth(),
//                                         true);


//        SimpleMesh Frame0_mesh(Frame0,
//                               m_InputHandle->getDepthImageHeight(),
//                               m_InputHandle->getDepthImageWidth(),
//                               true);

//        //SimpleMesh Frame0_mesh(*m_InputHandle, pose);

//        Frame0_mesh.writeMesh("Frame0_mesh.off");
//        Frame0_predicted_mesh.writeMesh("Frame0_predicted_mesh.off");


}

bool KiFuModel::prepareNextFrame()
{
    //StopWatch watch;
    if(m_InputHandle->processNextFrame())
    {
        return true;
    }


    m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());
    m_SurfaceMeasurer->process();

    const std::lock_guard<std::mutex> lock(m_nextFrameMutex);
    m_nextFrame = m_SurfaceMeasurer->getPointCloud();
    m_nextFrame.prune();
    return false;
}

bool KiFuModel::processNextFrame()
{
    // get V_k-1 N_k-1 from global model
    PointCloud prevFrame;
    {
        //StopWatch watch("SurfacePredictor");
        prevFrame = m_SurfacePredictor->predict(m_InputHandle->getDepthImageSize(),
                                                m_currentPose.back());
        prevFrame.prune();
    }


    //StopWatch watch("PoseEstimator");

    m_PoseEstimator->setTarget(prevFrame.points, prevFrame.normals);
    {
        const std::lock_guard<std::mutex> lock(m_nextFrameMutex);
        m_PoseEstimator->setSource(m_nextFrame.points, m_nextFrame.normals, 8);
    }
    // read out current ground truth before launching thread
    m_currentPoseGroundTruth.push_back(m_InputHandle->getTrajectory() * m_refPoseGroundTruth.inverse());

    std::future<bool> isLastFrameFut = std::async(std::launch::async, &KiFuModel::prepareNextFrame, this);

    m_CamToWorld = m_PoseEstimator->estimatePose(m_CamToWorld);
    m_currentPose.push_back(m_CamToWorld.inverse());
    ASSERT_NDBG(m_currentPose.size() == m_currentPoseGroundTruth.size())

    const bool isLastFrame = isLastFrameFut.get();

    // integrate the new frame in the tsdf
    m_SurfaceReconstructor->reconstruct(m_InputHandle->getDepth(),
                                        m_InputHandle->getColorRGBX(),
                                        m_InputHandle->getDepthImageSize(),
                                        m_currentPose.back());

    // actually skips the last frame, but that's ok
    if(isLastFrame)
    {
        return false;
    }
    return true;
}

void KiFuModel::saveTsdf(std::string filename, float tsdfThreshold, float weightThreshold) const
{
    m_tsdf->writeToFile(filename, tsdfThreshold, weightThreshold);
}

void KiFuModel::saveScreenshot(std::string filename, const Matrix4f pose) const
{
    ImageSize imageSize = m_InputHandle->getDepthImageSize();
    FreeImageB image(imageSize.w, imageSize.h, 3);

    m_SurfacePredictor->predictColor(image.data,
                                     m_InputHandle->getDepthImageSize(),
                                     pose);

    image.SaveImageToFile(filename);

}


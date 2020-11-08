#include "Eigen.h"
#include "DataTypes.h"

// takes the raw depth data and backprojects it into 3D camera space
class SurfaceMeasurer
{
public:
    SurfaceMeasurer(Eigen::Matrix3f DepthIntrinsics, uint DepthImageHeight, uint DepthImageWidth);

    // set pointer of m_rawDepthMap. SurfaceMeasurer does not take care of memory management for depthMap
    void registerInput(float* depthMap);
    // not implemented
    void smoothInput()
    {
        /*
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        cv::Mat rawDepthMapCopy = rawDepthMap.clone();
        cv::bilateralFilter(rawDepthMapCopy, rawDepthMap, 5, 10, 10);
        */

    }
    // main method: process current depth map
    void process();

    PointCloud getPointCloud();

private:
    // backproject into camera space
    void computeVertexAndNormalMap();
    // paramters needed for backprojection
    Matrix3f m_DepthIntrinsics;
    size_t m_DepthImageHeight;
    size_t m_DepthImageWidth;
    float* m_rawDepthMap;

    PointCloud m_pointCloud;
};

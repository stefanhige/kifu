#include "Eigen.h"
#include "DataTypes.h"
#include "StopWatch.h"
#include "FreeImageHelper.h"
#include "BilateralFilter.h"
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// takes the raw depth data and backprojects it into 3D camera space
class SurfaceMeasurer
{
public:
    SurfaceMeasurer(Eigen::Matrix3f DepthIntrinsics, uint DepthImageHeight, uint DepthImageWidth);

    // set pointer of m_rawDepthMap. SurfaceMeasurer does not take care of memory management for depthMap
    void registerInput(float* depthMap);

    void saveDepthMap(std::string filename);

    // main method: process current depth map
    void process();

    PointCloud getPointCloud();

private:
    void smoothInputManual();
    void smoothInput();

    // backproject into camera space
    void computeVertexAndNormalMap();
    // paramters needed for backprojection
    Matrix3f m_DepthIntrinsics;
    size_t m_DepthImageHeight;
    size_t m_DepthImageWidth;
    float* m_rawDepthMap;

    PointCloud m_pointCloud;
};

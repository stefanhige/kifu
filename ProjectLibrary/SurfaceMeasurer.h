#pragma once
#include <iterator>

#include "Eigen.h"
#include "DataTypes.h"
#include "StopWatch.h"
#include "FreeImageHelper.h"
#include "BilateralFilter.h"

//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>

// takes the raw depth data and backprojects it into 3D camera space
class ISurfaceMeasurer
{
public:
    virtual ~ISurfaceMeasurer() = default;

    virtual void registerInput(float*) = 0;
    virtual void saveDepthMap(std::string) = 0;
    virtual void process() = 0;
    virtual PointCloud getPointCloud() = 0;
};

class SurfaceMeasurer : public ISurfaceMeasurer
{
public:
    SurfaceMeasurer(Eigen::Matrix3f DepthIntrinsics, const ImageSize& depthImageSize);

    // set pointer of m_rawDepthMap. SurfaceMeasurer does not take care of memory management for depthMap
    void registerInput(float* depthMap) override;

    void saveDepthMap(std::string filename) override;

    // main method: process current depth map
    void process() override;

    PointCloud getPointCloud() override;

private:
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

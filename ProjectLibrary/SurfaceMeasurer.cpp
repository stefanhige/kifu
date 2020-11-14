#include "SurfaceMeasurer.h"

SurfaceMeasurer::SurfaceMeasurer(Eigen::Matrix3f DepthIntrinsics, uint DepthImageHeight, uint DepthImageWidth)
    : m_DepthIntrinsics(DepthIntrinsics),
      m_DepthImageHeight(DepthImageHeight),
      m_DepthImageWidth(DepthImageWidth),
      m_pointCloud(DepthImageHeight*DepthImageWidth)
{}

void SurfaceMeasurer::registerInput(float* depthMap)
{
    m_rawDepthMap = depthMap;
}

void SurfaceMeasurer::smoothInput()
{
    {
        StopWatch watch("smoothInput_OpenCV");
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        cv::Mat rawDepthMapCopy = rawDepthMap.clone();
        cv::bilateralFilter(rawDepthMapCopy, rawDepthMap, 5, 10, 10);

        // needed, because opencv places nans instead of MINF
        std::transform(m_rawDepthMap, m_rawDepthMap + m_DepthImageHeight*m_DepthImageWidth, m_rawDepthMap,
                       [](float in) -> float {return std::isnan(in) ? MINF : in;});
    }
}
void SurfaceMeasurer::smoothInputManual()
{

    saveDepthMap("before_m.png");
    {
        StopWatch watch("manual filtering");
        auto filter = BilateralFilter<5,5>(m_DepthImageWidth, m_DepthImageHeight);
        filter.apply(m_rawDepthMap);
    }
    saveDepthMap("after_m.png");

    exit(0);

}

void SurfaceMeasurer::saveDepthMap(std::string filename)
{
    FreeImage image(m_DepthImageWidth, m_DepthImageHeight, 1);
    std::copy(m_rawDepthMap, m_rawDepthMap + m_DepthImageHeight*m_DepthImageWidth, image.data);
    image.normalize();
    image.SaveImageToFile(filename);
}

void SurfaceMeasurer::process()
{
    smoothInputManual();
    computeVertexAndNormalMap();
}

PointCloud SurfaceMeasurer::getPointCloud()
{
    return m_pointCloud;
}

void SurfaceMeasurer::computeVertexAndNormalMap()
{
    //m_pointCloud.points.reserve(m_DepthImageHeight*m_DepthImageWidth);
    //m_pointCloud.pointsValid.reserve(m_DepthImageHeight*m_DepthImageWidth);

    //m_pointCloud.normals = std::vector<Vector3f>(m_DepthImageHeight*m_DepthImageWidth);
    //m_pointCloud.normalsValid = std::vector<bool>(m_DepthImageHeight*m_DepthImageWidth);

    //#pragma omp parallel for

    float fovX = m_DepthIntrinsics(0, 0);
    float fovY = m_DepthIntrinsics(1, 1);
    float cX = m_DepthIntrinsics(0, 2);
    float cY = m_DepthIntrinsics(1, 2);

    #pragma omp parallel for collapse(2)
    for(uint y = 0; y < m_DepthImageHeight; ++y)
    {
        for(uint x = 0; x < m_DepthImageWidth; ++x)
        {
            uint idx = y*m_DepthImageWidth + x;
            const float depth = m_rawDepthMap[idx];
            if (depth == MINF || depth == NAN)
            {
                m_pointCloud.points[idx] = Vector3f(MINF, MINF, MINF);
                m_pointCloud.pointsValid[idx] = false;
            }
            else
            {
                // backproject to camera space
                m_pointCloud.points[idx] = Vector3f((x - cX) / fovX * depth, (y - cY) / fovY * depth, depth);
                m_pointCloud.pointsValid[idx] = true;
            }

        }
    }

    const float maxDistHalve = 0.05f;
    #pragma omp parallel for collapse(2)
    for(uint y = 1; y < m_DepthImageHeight-1; ++y)
    {
        for(uint x = 1; x < m_DepthImageWidth-1; ++x)
        {
            uint idx = y*m_DepthImageWidth + x;
            const float du = 0.5f * (m_rawDepthMap[idx + 1] - m_rawDepthMap[idx - 1]);
            const float dv = 0.5f * (m_rawDepthMap[idx + m_DepthImageWidth] - m_rawDepthMap[idx - m_DepthImageWidth]);
            if (!std::isfinite(du) || !std::isfinite(dv) || std::abs(du) > maxDistHalve || std::abs(dv) > maxDistHalve)
            {
                m_pointCloud.normals[idx] = Vector3f(MINF, MINF, MINF);
                m_pointCloud.normalsValid[idx] = false;
            }
            else
            {
                m_pointCloud.normals[idx] = Vector3f(du, dv, -1);
                m_pointCloud.normals[idx].normalize();
                m_pointCloud.normalsValid[idx] = true;
            }
        }
    }
    // edge regions
    for (uint x = 0; x < m_DepthImageWidth; ++x) {
        m_pointCloud.normals[x] = Vector3f(MINF, MINF, MINF);
        m_pointCloud.normals[x + (m_DepthImageHeight - 1) * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
    }
    for (uint y = 0; y < m_DepthImageHeight; ++y) {
        m_pointCloud.normals[y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
        m_pointCloud.normals[(m_DepthImageWidth - 1) + y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
    }
}


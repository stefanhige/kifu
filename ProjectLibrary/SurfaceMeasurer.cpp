#include "SurfaceMeasurer.h"

SurfaceMeasurer::SurfaceMeasurer(Eigen::Matrix3f DepthIntrinsics, uint DepthImageHeight, uint DepthImageWidth)
    : m_DepthIntrinsics(DepthIntrinsics),
      m_DepthImageHeight(DepthImageHeight),
      m_DepthImageWidth(DepthImageWidth),
      m_vertexMap(DepthImageHeight*DepthImageWidth),
      m_vertexValidityMap(DepthImageHeight*DepthImageWidth),
      m_normalMap(DepthImageHeight*DepthImageWidth),
      m_normalValidityMap(DepthImageHeight*DepthImageWidth)
{}

void SurfaceMeasurer::registerInput(float* depthMap)
{
    m_rawDepthMap = depthMap;
}

void SurfaceMeasurer::process()
{
    computeVertexAndNormalMap();
}

PointCloud SurfaceMeasurer::getPointCloud()
{
    return PointCloud{m_vertexMap, m_vertexValidityMap, m_normalMap, m_normalValidityMap};
}

void SurfaceMeasurer::computeVertexAndNormalMap()
{
    m_vertexMap.reserve(m_DepthImageHeight*m_DepthImageWidth);
    m_vertexValidityMap.reserve(m_DepthImageHeight*m_DepthImageWidth);

    m_normalMap = std::vector<Vector3f>(m_DepthImageHeight*m_DepthImageWidth);
    m_normalValidityMap = std::vector<bool>(m_DepthImageHeight*m_DepthImageWidth);

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
                m_vertexMap[idx] = Vector3f(MINF, MINF, MINF);
                m_vertexValidityMap[idx] = false;
            }
            else
            {
                // backproject to camera space
                m_vertexMap[idx] = Vector3f((x - cX) / fovX * depth, (y - cY) / fovY * depth, depth);
                m_vertexValidityMap[idx] = true;
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
                m_normalMap[idx] = Vector3f(MINF, MINF, MINF);
                m_normalValidityMap[idx] = false;
            }
            else
            {
                m_normalMap[idx] = Vector3f(du, dv, -1);
                m_normalMap[idx].normalize();
                m_normalValidityMap[idx] = true;
            }
        }
    }
    // edge regions
    for (uint x = 0; x < m_DepthImageWidth; ++x) {
        m_normalMap[x] = Vector3f(MINF, MINF, MINF);
        m_normalMap[x + (m_DepthImageHeight - 1) * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
    }
    for (uint y = 0; y < m_DepthImageHeight; ++y) {
        m_normalMap[y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
        m_normalMap[(m_DepthImageWidth - 1) + y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
    }
}


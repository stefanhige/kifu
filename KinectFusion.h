#pragma once

#include <string>
#include "Eigen.h"
#include "VirtualSensor.h"

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// compute surface and normal maps
class SurfaceMeasurer
{
public:
    SurfaceMeasurer(Eigen::Matrix3f DepthIntrinsics, unsigned int DepthImageHeight, unsigned int DepthImageWidth)
        : m_DepthIntrinsics(DepthIntrinsics),
          m_DepthImageHeight(DepthImageHeight),
          m_DepthImageWidth(DepthImageWidth)
    {}

    void registerInput(float* depthMap)
    {
        m_rawDepthMap = depthMap;
    }

    void smoothInput()
    {
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        cv::Mat rawDepthMapCopy = rawDepthMap.clone();
        cv::bilateralFilter(rawDepthMapCopy, rawDepthMap, 5, 10, 10);

    }
    void process()
    {
        computeVertexAndNormalMap();
    }

    void printDepthMap()
    {
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        std::cout << rawDepthMap;
    }

    void displayDepthMap()
    {
        // TODO
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        cv::Mat normDepthMap;
        //cv::patchNaNs(rawDepthMap);
        std::cout << cv::checkRange(rawDepthMap);
        cv::normalize(rawDepthMap, normDepthMap, 0, 1, cv::NORM_MINMAX);
        std::cout << cv::checkRange(normDepthMap);
        //std::cout << normDepthMap;
        std::string windowName = "rawDepthMap";
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

        cv::Mat test = cv::Mat::zeros(100,100,CV_32F);
        cv::imshow(windowName, rawDepthMap);
        cv::waitKey(0);
    }


private:
    void computeVertexAndNormalMap()
    {
        m_vertexMap.reserve(m_DepthImageHeight*m_DepthImageWidth);
        m_normalMap.reserve(m_DepthImageHeight*m_DepthImageWidth);

        m_vertexValidityMap.reserve(m_DepthImageHeight*m_DepthImageWidth);
        m_normalValidityMap.reserve(m_DepthImageHeight*m_DepthImageWidth);

        //#pragma omp parallel for

        float fovX = m_DepthIntrinsics(0, 0);
        float fovY = m_DepthIntrinsics(1, 1);
        float cX = m_DepthIntrinsics(0, 2);
        float cY = m_DepthIntrinsics(1, 2);
        for(int y = 0; y < m_DepthImageHeight; ++y)
        {
            for(int x = 0; x < m_DepthImageWidth; ++x)
            {
                unsigned int idx = y*m_DepthImageWidth + x;
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
        for(int y = 1; y < m_DepthImageHeight-1; ++y)
        {
            for(int x = 1; x < m_DepthImageWidth-1; ++x)
            {
                unsigned int idx = y*m_DepthImageWidth + x;
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
        for (int x = 0; x < m_DepthImageWidth; ++x) {
            m_normalMap[x] = Vector3f(MINF, MINF, MINF);
            m_normalMap[x + (m_DepthImageHeight - 1) * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
        }
        for (int y = 0; y < m_DepthImageHeight; ++y) {
            m_normalMap[y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
            m_normalMap[(m_DepthImageWidth - 1) + y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
        }

//        for(int i= 0; i<10; ++i)
//        {
//            std::cout << "v " << m_vertexMap[i] << std::endl;
//            std::cout << "n " << m_normalMap[i] << std::endl;

//        }
    }


    Matrix3f m_DepthIntrinsics;
    unsigned int m_DepthImageHeight;
    unsigned int m_DepthImageWidth;
    float* m_rawDepthMap;
    std::vector<Vector3f> m_vertexMap;
    std::vector<bool> m_vertexValidityMap;
    std::vector<Vector3f> m_normalMap;
    std::vector<bool> m_normalValidityMap;

};

class PoseEstimator
{

};

class SurfaceReconstructor
{

};

class SurfacePredictor
{

};

typedef VirtualSensor InputType;
//template<class InputType>
class KiFuModel
{
public:
    KiFuModel(InputType& InputHandle)
        : m_InputHandle(&InputHandle)
    {
        m_InputHandle->processNextFrame();
        m_SurfaceMeasurer = new SurfaceMeasurer(m_InputHandle->getDepthIntrinsics(),
                                                m_InputHandle->getDepthImageHeight(),
                                                m_InputHandle->getDepthImageWidth());
        m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());
        //m_SurfaceMeasurer->smoothInput();
        //m_SurfaceMeasurer->displayDepthMap();
        m_SurfaceMeasurer->process();
    }


    bool processNextFrame()
    {
        if(!m_InputHandle->processNextFrame())
        {
            return true;
        }
        m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());


        return false;
    }


private:
    SurfaceMeasurer* m_SurfaceMeasurer;
    PoseEstimator m_PoseEstimator;
    SurfaceReconstructor m_SurfaceReconstructor;
    SurfacePredictor m_SurfacePredictor;

    InputType* m_InputHandle;
    std::string param;
    float* tsdf;

};


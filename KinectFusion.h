#pragma once

#include <string>
#include "Eigen.h"
#include "VirtualSensor.h"

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
        std::cout << "not implemented." << std::endl;
        // smooth with gaussian kernel
//        float coeff[] = {0.0545, 0.2442, 0.4026, 0.2442, 0.0545};
//        int size = 5;
//        float* smoothDepthMap1 = new float[m_DepthImageHeight*(m_DepthImageWidth-4)];
//        float* smoothDepthMap1ptr = smoothDepthMap1;

//        // not yet implemented
//        for(int row=0;row<m_DepthImageHeight;++row)
//        {
//            for(int idx=row*m_DepthImageWidth+2; idx < (row+1)*m_DepthImageWidth-2; ++idx)
//            {
//                float sum = 0;
//                for(int pos=0; pos<5; ++pos)
//                {
//                    sum += coeff[pos] * m_rawDepthMap[idx+pos-2];
//                }
//                *smoothDepthMap1ptr = sum;
//                smoothDepthMap1ptr++;
//            }
//        }
//        //for(int col=0;col<m_DepthImageWidth-4;++col)

//        for(int col=0;col<2;++col)
//        {
//            // for loop not correct
//            for(int idx=col; idx<col+(m_DepthImageHeight-2)*(m_DepthImageWidth-4);idx += m_DepthImageWidth-4)
//            {
//                std::cout << idx << " ";
//                float sum = 0;
//                for(int pos=0; pos<5; ++pos)
//                {
//                    sum += coeff[pos] * smoothDepthMap1[idx+(pos-2)*(m_DepthImageWidth-4)];
//                }
//            }

//            std::cout << " end" << std::endl;
//        }
    }


private:
    void computeVertexMap()
    {

    }
    void computeNormalMap()
    {

    }

    Matrix3f m_DepthIntrinsics;
    unsigned int m_DepthImageHeight;
    unsigned int m_DepthImageWidth;
    float* m_rawDepthMap;

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
        m_SurfaceMeasurer->smoothInput();
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


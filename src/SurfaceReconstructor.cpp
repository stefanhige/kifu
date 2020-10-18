#include "SurfaceReconstructor.h"

SurfaceReconstructor::SurfaceReconstructor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics)
    : m_tsdf(tsdf),
      m_cameraIntrinsics(cameraIntrinsics)

{
}

void SurfaceReconstructor::reconstruct(const float* rawDepthMap, const unsigned int depthImageHeight, const unsigned int depthImageWidth, const Matrix4f cameraToWorld)
{

    // for each point in the tsdf:
    // loop over idx
    double begin = tic();

    #pragma omp parallel for
    for(uint idx=0; idx < (m_tsdf->getSize()*m_tsdf->getSize()*m_tsdf->getSize()); ++idx)
    {

        Vector4f globalPoint = m_tsdf->getPoint(idx);
        Vector4f cameraPoint = cameraToWorld*globalPoint;
        Vector3f cameraPoint_ = m_cameraIntrinsics*cameraPoint.block<3,1>(0,0);

        int x_pixel = floor(cameraPoint_.x()/cameraPoint_.z());
        int y_pixel = floor(cameraPoint_.y()/cameraPoint_.z());

        if (!(x_pixel < 0 || x_pixel >= static_cast<int>(depthImageWidth) || y_pixel < 0 || y_pixel >= static_cast<int>(depthImageHeight)))
        {
            // look up depth value of raw depth map
            float depth = rawDepthMap[x_pixel + depthImageWidth*y_pixel];
            // filter out -inf or nan
            if(std::isgreaterequal(depth, 0))
            {
                float lambda = (m_cameraIntrinsics.inverse()*Vector3f(x_pixel, y_pixel, 1)).norm();
                Vector3f translation = (cameraToWorld.inverse()).block<3,1>(0,3);
                float eta = (translation - (m_tsdf->getPoint(idx)).block<3,1>(0,0)).norm() / lambda - depth;
                float mu = 1;

                if (eta > -mu)
                {
                    //                                     v sign(eta)
                    // float sdf = std::min<float>(1, eta/mu)*((eta > 0) - (eta < 0));
                    float sdf = std::min<float>(1, eta/mu);
                    // update tsdf and weight (weight increase is 1)
                    (*m_tsdf)(idx) = (m_tsdf->weight(idx)*(*m_tsdf)(idx) + sdf) / (m_tsdf->weight(idx) + 1);

                    m_tsdf->weight(idx) = (m_tsdf->weight(idx) < m_tsdf->max_weight()) ? m_tsdf->weight(idx) + 1 : m_tsdf->max_weight();

                    //if (0 && eta<0)
                    //    std::cout << "x: " << x_pixel << " y: " << y_pixel << " depth: " << depth << " lambda: " << lambda
                    //          << " eta: " << eta << " sdf: " << sdf << " weight: " << static_cast<int>(m_tsdf->weight(idx)) << std::endl;

                }
            }
        }
    }
    toc(begin);
}

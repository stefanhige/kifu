#include "SurfaceReconstructor.h"

SurfaceReconstructor::SurfaceReconstructor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics)
    : m_tsdf(tsdf),
      m_cameraIntrinsics(cameraIntrinsics)

{
}

void SurfaceReconstructor::reconstruct(const float* rawDepthMap,
                                       const uint8_t* rawColorMap,
                                       const uint imageHeight,
                                       const uint imageWidth,
                                       const Matrix4f cameraToWorld)
{

    // for each point in the tsdf:
    // loop over idx

    #pragma omp parallel for
    for(size_t idx=0; idx < (m_tsdf->getSize()*m_tsdf->getSize()*m_tsdf->getSize()); ++idx)
    {

        Vector4f globalPoint = m_tsdf->getPoint(idx);
        Vector4f cameraPoint = cameraToWorld*globalPoint;
        Vector3f cameraPoint_ = m_cameraIntrinsics*cameraPoint.block<3,1>(0,0);

        int x_pixel = floor(cameraPoint_.x()/cameraPoint_.z());
        int y_pixel = floor(cameraPoint_.y()/cameraPoint_.z());

        if (!(x_pixel < 0 || x_pixel >= static_cast<int>(imageWidth) || y_pixel < 0 || y_pixel >= static_cast<int>(imageHeight)))
        {
            // look up depth value of raw depth map
            float depth = rawDepthMap[x_pixel + imageWidth*y_pixel];
            // filter out -inf or nan
            if(std::isgreaterequal(depth, 0))
            {
                float lambda = (m_cameraIntrinsics.inverse()*Vector3f(x_pixel, y_pixel, 1)).norm();
                Vector3f translation = (cameraToWorld.inverse()).block<3,1>(0,3);
                float eta = (translation - (m_tsdf->getPoint(idx)).block<3,1>(0,0)).norm() / lambda - depth;
                float mu = 1;

                if (eta > -mu)
                {
                    //                                                v -sign(eta)
                    float sdf = std::min<float>(1, std::abs(eta)/mu)*((eta < 0) - (eta > 0));

                    //float sdf = std::min<float>(1, eta/mu);
                    // update tsdf and weight (weight increase is 1)
                    (*m_tsdf)(idx) = (m_tsdf->weight(idx)*(*m_tsdf)(idx) + sdf) / (m_tsdf->weight(idx) + 1);

                    m_tsdf->weight(idx) = (m_tsdf->weight(idx) < m_tsdf->max_weight()) ? m_tsdf->weight(idx) + 1 : m_tsdf->max_weight();

                    // update colors
                    // TODO: update constraint
                    if(std::abs(sdf) < m_tsdf->getVoxelSize())
                    {
                        // ingore alpha channel: rawColorMap is RGBX, we only use RGB
                        uint16_t color[3] = {rawColorMap[(x_pixel + imageWidth*y_pixel)*4],
                                                  rawColorMap[(x_pixel + imageWidth*y_pixel)*4+1],
                                                  rawColorMap[(x_pixel + imageWidth*y_pixel)*4+2]};
                        m_tsdf->colorR(idx) = static_cast<uint16_t>((static_cast<uint16_t>(m_tsdf->weight(idx))*m_tsdf->colorR(idx) + color[0]) / (m_tsdf->weight(idx) + 1));
                        m_tsdf->colorG(idx) = static_cast<uint16_t>((static_cast<uint16_t>(m_tsdf->weight(idx))*m_tsdf->colorG(idx) + color[1]) / (m_tsdf->weight(idx) + 1));
                        m_tsdf->colorB(idx) = static_cast<uint16_t>((static_cast<uint16_t>(m_tsdf->weight(idx))*m_tsdf->colorB(idx) + color[2]) / (m_tsdf->weight(idx) + 1));
                    }
                }
            }
        }
    }
}

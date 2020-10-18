#include <memory>

#include "Eigen.h"
#include "DataTypes.h"

class SurfaceReconstructor
{
public:
    SurfaceReconstructor()
    {
    }
    SurfaceReconstructor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics);

    void reconstruct(const float* rawDepthMap, const unsigned int depthImageHeight, const unsigned int depthImageWidth, const Matrix4f cameraToWorld);

private:
    std::shared_ptr<Tsdf> m_tsdf;
    Matrix3f m_cameraIntrinsics;
};

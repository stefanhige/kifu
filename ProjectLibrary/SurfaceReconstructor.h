#include <memory>

#include "Eigen.h"
#include "DataTypes.h"
// integrates a depth frame into the global model
class SurfaceReconstructor
{
public:
    SurfaceReconstructor(){}
    SurfaceReconstructor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics);

    // reconstruct surfaces from rawDepthMap with pose cameraToWorld and integrate it into the global model
    void reconstruct(const float* rawDepthMap, const uint8_t* rawColorMap, const unsigned int imageHeight, const unsigned int imageWidth, const Matrix4f cameraToWorld);

private:
    std::shared_ptr<Tsdf> m_tsdf;
    Matrix3f m_cameraIntrinsics;
};

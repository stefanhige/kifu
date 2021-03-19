#include <memory>

#include "Eigen.h"
#include "DataTypes.h"
class ISurfaceReconstructor
{
public:
    virtual void reconstruct(const float*, const uint8_t*, const ImageSize&, const Matrix4f&) = 0;
};


// integrates a depth frame into the global model
class SurfaceReconstructor : public ISurfaceReconstructor
{
public:
    SurfaceReconstructor(){}
    SurfaceReconstructor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics);

    // reconstruct surfaces from rawDepthMap with pose cameraToWorld and integrate it into the global model
    void reconstruct(const float* rawDepthMap, const uint8_t* rawColorMap, const ImageSize& imageSize, const Matrix4f& cameraToWorld) override;

private:
    std::shared_ptr<Tsdf> m_tsdf;
    Matrix3f m_cameraIntrinsics;
};

#include <memory>

#include "Eigen.h"
#include "DataTypes.h"

class SurfacePredictor
{
public:
    SurfacePredictor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics);

    PointCloud predict(const uint depthImageHeight, const uint depthImageWidth, const Matrix4f pose = Matrix4f::Identity());

private:
   float trilinear_interpolate(const Vector3f& point) const;
   bool trilinear_interpolate(const Vector3f& point, float& value) const;
   float compute_min_t(Vector3f origin, Vector3f direction) const;
   float compute_max_t(Vector3f origin, Vector3f direction) const;
   bool compute_normal(const Vector3f& point, Vector3f& normal) const;

   std::shared_ptr<Tsdf> m_tsdf;
   Matrix3f m_cameraIntrinsics;

};

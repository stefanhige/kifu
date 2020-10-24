#include <memory>

#include "Eigen.h"
#include "DataTypes.h"
// predict an image to a certain pose from the global model
// this is equivalent to taking a shapshot of the global model with a 'virutal' camera from a certain pose.
class SurfacePredictor
{
public:
    SurfacePredictor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics);

    // predict a PointCloud to a certain pose (depth information only)
    PointCloud predict(const uint depthImageHeight, const uint depthImageWidth, const Matrix4f pose = Matrix4f::Identity());
    // predict a color image from a certain pose
    // color image gets stored in the memory pointed to by colorMap
    void predictColor(uint8_t* colorMap, const uint depthImageHeight, const uint depthImageWidth, const Matrix4f pose = Matrix4f::Identity());

private:
   // interpolate m_tsdf to continous locations
   float trilinear_interpolate(const Vector3f& point) const;
   bool trilinear_interpolate(const Vector3f& point, float& value) const;
   // estimate parameter 't' for raycasting
   float compute_min_t(Vector3f origin, Vector3f direction) const;
   float compute_max_t(Vector3f origin, Vector3f direction) const;
   // compute the normal on the surface at point using m_tsdf
   bool compute_normal(const Vector3f& point, Vector3f& normal) const;

   std::shared_ptr<Tsdf> m_tsdf;
   Matrix3f m_cameraIntrinsics;

};

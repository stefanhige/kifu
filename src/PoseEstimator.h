#include <memory>

#include "Eigen.h"
#include "DataTypes.h"
#include "NearestNeighbor.h"

class PoseEstimator
{
public:
    PoseEstimator(){}

    void setTarget(PointCloud& input);
    void setSource(PointCloud& input);
    void setTarget(std::vector<Vector3f> points, std::vector<Vector3f> normals);
    void setSource(std::vector<Vector3f> points, std::vector<Vector3f> normals);
    void setSource(std::vector<Vector3f> points, std::vector<Vector3f> normals, unsigned int downsample);
    void printPoints();
    virtual Matrix4f estimatePose(Matrix4f = Matrix4f::Identity()) = 0;
    static std::vector<Vector3f> pruneVector(std::vector<Vector3f>& input, std::vector<bool>& validity);
    static std::vector<Vector3f> transformPoint(const std::vector<Vector3f>& input, const Matrix4f& pose);
    static std::vector<Vector3f> transformNormal(const std::vector<Vector3f>& input, const Matrix4f& pose);

protected:
    PointCloud m_target;
    PointCloud m_source;
    int m_nIter = 10;
};

class NearestNeighborPoseEstimator : public PoseEstimator
{
public:
    NearestNeighborPoseEstimator();

    virtual Matrix4f estimatePose(Matrix4f initialPose = Matrix4f::Identity()) override;

private:
    Matrix4f solvePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals);

    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;
};

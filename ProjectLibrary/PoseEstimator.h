#include <memory>

#include "Eigen.h"
#include "DataTypes.h"
#include "NearestNeighbor.h"

// estimate a 4x4 transformation matrix 'pose',
// which alignes PointCloud Source with PointCloud Target.
class IPoseEstimator
{
public:
    IPoseEstimator(){}

    void setTarget(PointCloud& input);
    void setSource(PointCloud& input);
    void setTarget(const std::vector<Vector3f>& points, const std::vector<Vector3f>& normals);
    void setSource(const std::vector<Vector3f>& points, const std::vector<Vector3f>& normals);
    // for a downsample factor of n: only take every n-th point.
    void setSource(const std::vector<Vector3f>& points, const std::vector<Vector3f>& normals, unsigned int downsample);
    // debug method
    void printPoints();
    // estimate pose. optional argument: initial pose
    virtual Matrix4f estimatePose(Matrix4f = Matrix4f::Identity()) = 0;
    // helper methods
    static std::vector<Vector3f> transformPoint(const std::vector<Vector3f>& input, const Matrix4f& pose);
    static std::vector<Vector3f> transformNormal(const std::vector<Vector3f>& input, const Matrix4f& pose);

protected:
    PointCloud m_target;
    PointCloud m_source;
    int m_nIter = 10;
};

// see also: Kok-Lim Low "Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration"
class NearestNeighborPoseEstimator : public IPoseEstimator
{
public:
    NearestNeighborPoseEstimator();

    virtual Matrix4f estimatePose(Matrix4f initialPose = Matrix4f::Identity()) override;

private:
    Matrix4f solvePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals);
    void pruneCorrespondences(const std::vector<Vector3f> &sourceNormals, const std::vector<Vector3f> &targetNormals, std::vector<Match> &matches);
};

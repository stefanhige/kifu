#include "PoseEstimator.h"
#include "StopWatch.h"


void IPoseEstimator::setTarget(PointCloud& input)
{
    m_target = input;
}

void IPoseEstimator::setSource(PointCloud& input)
{
    m_source = input;
}

void IPoseEstimator::setTarget(const std::vector<Vector3f>& points, const std::vector<Vector3f>& normals)
{
    m_target.points = points;
    m_target.normals = normals;

    m_target.normalsValid = std::vector<bool>(normals.size(), true);
    m_target.pointsValid = std::vector<bool>(points.size(), true);

}

void IPoseEstimator::setSource(const std::vector<Vector3f>& points, const std::vector<Vector3f>& normals)
{
    setSource(points, normals, 1);
}

void IPoseEstimator::setSource(const std::vector<Vector3f>& points, const std::vector<Vector3f>& normals, unsigned int downsample)
{
    if (downsample == 1)
    {
        m_source.points = points;
        m_source.normals = normals;

        m_source.normalsValid = std::vector<bool>(normals.size(), true);
        m_source.pointsValid = std::vector<bool>(points.size(), true);
    }
    else
    {
        size_t nPoints = std::min(points.size(), normals.size()) / downsample;
        m_source.points = std::vector<Vector3f>(nPoints);
        m_source.normals = std::vector<Vector3f>(nPoints);
        for (size_t i = 0; i < nPoints; ++i)
        {
            m_source.points[i] = points[i*downsample];
            m_source.normals[i] = normals[i*downsample];
        }

        m_source.normalsValid = std::vector<bool>(nPoints, true);
        m_source.pointsValid = std::vector<bool>(nPoints, true);
    }
}

void IPoseEstimator::printPoints()
{
    std::cout << "first 10 points " << std::endl;
    for(int i =0; i<std::min<int>(10, m_target.points.size());++i)
    {
        std::cout << m_target.points[i].transpose() << std::endl;
    }
}

std::vector<Vector3f> IPoseEstimator::transformPoint(const std::vector<Vector3f>& input, const Matrix4f& pose)
{
    std::vector<Vector3f> output;
    output.reserve(input.size());

    const auto rotation = pose.block(0, 0, 3, 3);
    const auto translation = pose.block(0, 3, 3, 1);

    for (const auto& point : input)
    {
        output.push_back(rotation * point + translation);
    }

    return output;
}

std::vector<Vector3f> IPoseEstimator::transformNormal(const std::vector<Vector3f>& input, const Matrix4f& pose)
{
    std::vector<Vector3f> output;
    output.reserve(input.size());

    const auto rotation = pose.block(0, 0, 3, 3);

    for (const auto& normal : input)
    {
        output.push_back(rotation.inverse().transpose() * normal);
    }

    return output;
}

NearestNeighborPoseEstimator::NearestNeighborPoseEstimator()
{}

Matrix4f NearestNeighborPoseEstimator::estimatePose(Matrix4f initialPose)
{
    std::unique_ptr<NearestNeighborSearch> nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
    // Build the index of the FLANN tree (for fast nearest neighbor lookup).
    nearestNeighborSearch->buildIndex(m_target.points);

    // The initial estimate can be given as an argument.
    Matrix4f estimatedPose = initialPose;

    for (int i = 0; i < m_nIter; ++i)
    {
        auto transformedPoints = transformPoint(m_source.points, estimatedPose);
        auto transformedNormals = transformNormal(m_source.normals, estimatedPose);
        auto matches = nearestNeighborSearch->queryMatches(transformedPoints);

        pruneCorrespondences(transformedNormals, m_target.normals, matches);

        std::vector<Vector3f> sourcePoints;
        std::vector<Vector3f> targetPoints;

        // Add all matches to the sourcePoints and targetPoints vectors,
        // so that sourcePoints[i] matches targetPoints[i].
        for (size_t j = 0; j < transformedPoints.size(); j++)
        {
            const auto& match = matches[j];
            if (match.idx >= 0)
            {
                sourcePoints.push_back(transformedPoints[j]);
                targetPoints.push_back(m_target.points[match.idx]);
            }
        }

        // need at least 3 points
        ASSERT_NDBG(sourcePoints.size() >= 3);
        estimatedPose = solvePointToPlane(sourcePoints, targetPoints, m_target.normals) * estimatedPose;
    }
    return estimatedPose;

}

Matrix4f NearestNeighborPoseEstimator::solvePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals)
{
    const size_t nPoints = sourcePoints.size();

    // Build the system
    MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
    VectorXf b = VectorXf::Zero(4 * nPoints);

    for (size_t i = 0; i < nPoints; i++)
    {
        const auto& s = sourcePoints[i];
        const auto& d = targetPoints[i];
        const auto& n = targetNormals[i];

        // Add the point-to-plane constraints to the system
        b[i] = n.dot(d) - n.dot(s);
        A.block<1,3>(i,0) << n[2]*s[1] - n[1]*s[2],
                             n[0]*s[2] - n[2]*s[0],
                             n[1]*s[0] - n[0]*s[1];
        A.block<1,3>(i,3) = n;
        // Add the point-to-point constraints to the system
        // for x-coords
        b[nPoints+i] = d[0] - s[0];
        A.block<1,6>(nPoints+i,0) << 0, s[2], -s[1], 1, 0, 0;

        // for y-coords
        b[2*nPoints+i] = d[1] - s[1];
        A.block<1,6>(2*nPoints+i,0) << -s[2], 0, s[0], 0, 1, 0;

        // for z-coords
        b[3*nPoints+i] = d[2] - s[2];
        A.block<1,6>(3*nPoints+i,0) << s[1], -s[0], 0, 0, 0, 1;
    }

    // Solve the system
    VectorXf x(6);
    JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);

    x = svd.solve(b);

    // TODO: constraint watching d/(d n_iter) MSE and adjusting n_iter if necessary
    // residuals
    ArrayXf res = (b - A * x).array();
    // std::cout << "avg MSE per eqn: " << (res.abs().square().sum()) / (4*nPoints) << std::endl;


    float alpha = x(0), beta = x(1), gamma = x(2);

    // Build the pose matrix
    Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
                        AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
                        AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

    Vector3f translation = x.tail(3);

    // Build the pose matrix using the rotation and translation matrices
    Matrix4f estimatedPose = Matrix4f::Identity();
    estimatedPose.block<3,3>(0,0) = rotation;
    estimatedPose.block<3,1>(0,3) = translation;

    return estimatedPose;
}

void NearestNeighborPoseEstimator::pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches)
{
    const size_t nPoints = sourceNormals.size();

    for (size_t i = 0; i < nPoints; ++i)
    {
        Match& match = matches[i];
        if (match.idx >= 0)
        {
            const auto& sourceNormal = sourceNormals[i];
            const auto& targetNormal = targetNormals[match.idx];
            float angle = std::acos(sourceNormal.dot(targetNormal));
            // larger than 60 deg
            if(angle > 1.0472f)
            {
                match.idx = -1;
            }
        }
    }
}


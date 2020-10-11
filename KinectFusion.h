#pragma once

#include <string>
#include <assert.h>
#include <memory>

#include "Eigen.h"
#include "VirtualSensor.h"
#include "NearestNeighbor.h"
#include "DataTypes.h"

// debug
#include "SimpleMesh.h"

#define tic() (omp_get_wtime())
#define toc(a) (printf("%s, %i: dur %f s\n", __FILE__, __LINE__, omp_get_wtime() - a))

// compute surface and normal maps
class SurfaceMeasurer
{
public:
    SurfaceMeasurer(Eigen::Matrix3f DepthIntrinsics, uint DepthImageHeight, uint DepthImageWidth)
        : m_DepthIntrinsics(DepthIntrinsics),
          m_DepthImageHeight(DepthImageHeight),
          m_DepthImageWidth(DepthImageWidth),
          m_vertexMap(DepthImageHeight*DepthImageWidth),
          m_vertexValidityMap(DepthImageHeight*DepthImageWidth),
          m_normalMap(DepthImageHeight*DepthImageWidth),
          m_normalValidityMap(DepthImageHeight*DepthImageWidth)
    {}

    void registerInput(float* depthMap)
    {
        m_rawDepthMap = depthMap;
    }

    void smoothInput()
    {
        /*
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        cv::Mat rawDepthMapCopy = rawDepthMap.clone();
        cv::bilateralFilter(rawDepthMapCopy, rawDepthMap, 5, 10, 10);
        */

    }
    void process()
    {
        computeVertexAndNormalMap();
        //std::cout << "size " << m_vertexMap.size() << std::endl;
    }

    void printDepthMap()
    {
        /*
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        std::cout << rawDepthMap;
        */
    }

    void displayDepthMap()
    {
        /*
        // TODO
        cv::Mat rawDepthMap = cv::Mat{static_cast<int>(m_DepthImageHeight), static_cast<int>(m_DepthImageWidth), CV_32F, m_rawDepthMap};
        cv::Mat normDepthMap;
        //cv::patchNaNs(rawDepthMap);
        std::cout << cv::checkRange(rawDepthMap);
        cv::normalize(rawDepthMap, normDepthMap, 0, 1, cv::NORM_MINMAX);
        std::cout << cv::checkRange(normDepthMap);
        //std::cout << normDepthMap;
        std::string windowName = "rawDepthMap";
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

        cv::Mat test = cv::Mat::zeros(100,100,CV_32F);
        cv::imshow(windowName, rawDepthMap);
        cv::waitKey(0);
        */
    }

    PointCloud getPointCloud()
    {
        return PointCloud{m_vertexMap, m_vertexValidityMap, m_normalMap, m_normalValidityMap};
    }


private:
    void computeVertexAndNormalMap()
    {
        m_vertexMap.reserve(m_DepthImageHeight*m_DepthImageWidth);
        m_vertexValidityMap.reserve(m_DepthImageHeight*m_DepthImageWidth);

        m_normalMap = std::vector<Vector3f>(m_DepthImageHeight*m_DepthImageWidth);
        m_normalValidityMap = std::vector<bool>(m_DepthImageHeight*m_DepthImageWidth);

        //#pragma omp parallel for

        float fovX = m_DepthIntrinsics(0, 0);
        float fovY = m_DepthIntrinsics(1, 1);
        float cX = m_DepthIntrinsics(0, 2);
        float cY = m_DepthIntrinsics(1, 2);

        auto begin = tic();
        #pragma omp parallel for collapse(2)
        for(uint y = 0; y < m_DepthImageHeight; ++y)
        {
            for(uint x = 0; x < m_DepthImageWidth; ++x)
            {
                uint idx = y*m_DepthImageWidth + x;
                const float depth = m_rawDepthMap[idx];
                if (depth == MINF || depth == NAN)
                {
                    m_vertexMap[idx] = Vector3f(MINF, MINF, MINF);
                    m_vertexValidityMap[idx] = false;
                }
                else
                {
                    // backproject to camera space
                    m_vertexMap[idx] = Vector3f((x - cX) / fovX * depth, (y - cY) / fovY * depth, depth);
                    m_vertexValidityMap[idx] = true;
                }

            }
        }
        toc(begin);

        const float maxDistHalve = 0.05f;
        begin = tic();
        #pragma omp parallel for collapse(2)
        for(uint y = 1; y < m_DepthImageHeight-1; ++y)
        {
            for(uint x = 1; x < m_DepthImageWidth-1; ++x)
            {
                uint idx = y*m_DepthImageWidth + x;
                const float du = 0.5f * (m_rawDepthMap[idx + 1] - m_rawDepthMap[idx - 1]);
                const float dv = 0.5f * (m_rawDepthMap[idx + m_DepthImageWidth] - m_rawDepthMap[idx - m_DepthImageWidth]);
                if (!std::isfinite(du) || !std::isfinite(dv) || std::abs(du) > maxDistHalve || std::abs(dv) > maxDistHalve)
                {
                    m_normalMap[idx] = Vector3f(MINF, MINF, MINF);
                    m_normalValidityMap[idx] = false;
                }
                else
                {
                    m_normalMap[idx] = Vector3f(du, dv, -1);
                    m_normalMap[idx].normalize();
                    m_normalValidityMap[idx] = true;
                }
            }
        }
        toc(begin);
        // edge regions
        for (uint x = 0; x < m_DepthImageWidth; ++x) {
            m_normalMap[x] = Vector3f(MINF, MINF, MINF);
            m_normalMap[x + (m_DepthImageHeight - 1) * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
        }
        for (uint y = 0; y < m_DepthImageHeight; ++y) {
            m_normalMap[y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
            m_normalMap[(m_DepthImageWidth - 1) + y * m_DepthImageWidth] = Vector3f(MINF, MINF, MINF);
        }
    }


    Matrix3f m_DepthIntrinsics;
    uint m_DepthImageHeight;
    uint m_DepthImageWidth;
    float* m_rawDepthMap;
    std::vector<Vector3f> m_vertexMap;
    std::vector<bool> m_vertexValidityMap;
    std::vector<Vector3f> m_normalMap;
    std::vector<bool> m_normalValidityMap;

};

class PoseEstimator
{
public:
    PoseEstimator()
    {}

    void setTarget(PointCloud& input)
    {
        m_target = input;
    }
    void setSource(PointCloud& input)
    {
        m_source = input;
    }
    void setTarget(std::vector<Vector3f> points, std::vector<Vector3f> normals)
    {
        m_target.points = points;
        m_target.normals = normals;

        m_target.normalsValid = std::vector<bool>(normals.size(), true);
        m_target.pointsValid = std::vector<bool>(points.size(), true);

    }

    void setSource(std::vector<Vector3f> points, std::vector<Vector3f> normals)
    {
        setSource(points, normals, 1);
    }

    void setSource(std::vector<Vector3f> points, std::vector<Vector3f> normals, unsigned int downsample)
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
            int nPoints = std::min(points.size(), normals.size()) / downsample;
            m_source.points = std::vector<Vector3f>(nPoints);
            m_source.normals = std::vector<Vector3f>(nPoints);
            for (int i = 0; i < nPoints; ++i)
            {
                m_source.points[i] = points[i*downsample];
                m_source.normals[i] = normals[i*downsample];
            }

            m_source.normalsValid = std::vector<bool>(nPoints, true);
            m_source.pointsValid = std::vector<bool>(nPoints, true);
        }
    }


    void printPoints()
    {
        std::cout << "first 10 points " << std::endl;
        for(int i =0; i<std::min<int>(10, m_target.points.size());++i)
        {
            std::cout << m_target.points[i].transpose() << std::endl;
        }
    }
    virtual Matrix4f estimatePose(Matrix4f = Matrix4f::Identity()) = 0;

    static std::vector<Vector3f> pruneVector(std::vector<Vector3f>& input, std::vector<bool>& validity)
    {
        assert((input.size() == validity.size()));

        std::vector<Vector3f> output;
        for (uint i = 0; i < input.size(); ++i)
        {
            if(validity[i])
            {
                output.push_back(input[i]);
            }
        }
        return output;
    }
    static std::vector<Vector3f> transformPoint(const std::vector<Vector3f>& input, const Matrix4f& pose)
    {
        std::vector<Vector3f> output;
        output.reserve(input.size());

        const auto rotation = pose.block(0, 0, 3, 3);
        const auto translation = pose.block(0, 3, 3, 1);

        for (const auto& point : input) {
            output.push_back(rotation * point + translation);
        }

        return output;
    }
    static std::vector<Vector3f> transformNormal(const std::vector<Vector3f>& input, const Matrix4f& pose)
    {
        std::vector<Vector3f> output;
        output.reserve(input.size());

        const auto rotation = pose.block(0, 0, 3, 3);

        for (const auto& normal : input) {
            output.push_back(rotation.inverse().transpose() * normal);
        }

        return output;
    }
protected:
    PointCloud m_target;
    PointCloud m_source;
    int m_nIter = 10;
};

class NearestNeighborPoseEstimator : public PoseEstimator
{
public:
    NearestNeighborPoseEstimator()
        : m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>()}
    {}

    virtual Matrix4f estimatePose(Matrix4f initialPose = Matrix4f::Identity()) override
    {
        // Build the index of the FLANN tree (for fast nearest neighbor lookup).
        m_nearestNeighborSearch->buildIndex(m_target.points);

        // The initial estimate can be given as an argument.
        Matrix4f estimatedPose = initialPose;

        for (int i = 0; i < m_nIter; ++i) {
            auto transformedPoints = transformPoint(m_source.points, estimatedPose);
            auto transformedNormals = transformNormal(m_source.normals, estimatedPose);
            auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);

            // TODO
            // pruneCorrespondences(transformedNormals, m_target.getNormals(), matches);

            std::vector<Vector3f> sourcePoints;
            std::vector<Vector3f> targetPoints;

            // Add all matches to the sourcePoints and targetPoints vectors,
            // so thath sourcePoints[i] matches targetPoints[i].
            for (uint j = 0; j < transformedPoints.size(); j++)
            {
                const auto& match = matches[j];
                if (match.idx >= 0) {
                    sourcePoints.push_back(transformedPoints[j]);
                    targetPoints.push_back(m_target.points[match.idx]);
                }
            }

            estimatedPose = solvePointToPlane(sourcePoints, targetPoints, m_target.normals) * estimatedPose;
        }

        return estimatedPose;

    }

private:
    Matrix4f solvePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals)
    {
        const unsigned nPoints = sourcePoints.size();

        // Build the system
        MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
        VectorXf b = VectorXf::Zero(4 * nPoints);

        for (unsigned i = 0; i < nPoints; i++) {
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
    std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;
};

class SurfaceReconstructor
{
public:

    SurfaceReconstructor()
    {
    }

    SurfaceReconstructor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics)
        : m_tsdf(tsdf),
          m_cameraIntrinsics(cameraIntrinsics)

    {
    }

    void reconstruct(const float* rawDepthMap, const unsigned int depthImageHeight, const unsigned int depthImageWidth, const Matrix4f cameraToWorld)
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


private:
    std::shared_ptr<Tsdf> m_tsdf;
    Matrix3f m_cameraIntrinsics;
};

class SurfacePredictor
{
public:
    SurfacePredictor(std::shared_ptr<Tsdf> tsdf, Matrix3f cameraIntrinsics)
        : m_tsdf(tsdf),
          m_cameraIntrinsics(cameraIntrinsics)
    {
    }

   PointCloud predict(const uint depthImageHeight, const uint depthImageWidth, const Matrix4f pose = Matrix4f::Identity())
   {
       float fovX = m_cameraIntrinsics(0, 0);
       float fovY = m_cameraIntrinsics(1, 1);
       float cX = m_cameraIntrinsics(0, 2);
       float cY = m_cameraIntrinsics(1, 2);

       Matrix3f rotMatrix = pose.block<3,3>(0,0);
       Vector3f tranVector = pose.block<3,1>(0,3);

       PointCloud pointCloud;
       pointCloud.pointsValid.reserve(depthImageHeight*depthImageWidth);
       pointCloud.points.reserve(depthImageHeight*depthImageWidth);
       pointCloud.normalsValid.reserve(depthImageHeight*depthImageWidth);
       pointCloud.normals.reserve(depthImageHeight*depthImageWidth);

       auto begin = tic();
       #pragma omp parallel for
       // collapse(2) seems to make it slower
       for(uint y_pixel=0; y_pixel<depthImageHeight; ++y_pixel)
       {
           for(uint x_pixel=0; x_pixel < depthImageWidth; ++x_pixel)
           {
               uint idx = y_pixel*depthImageWidth + x_pixel;

               float depth = 1;
               Vector3f rayDirCamera = Vector3f((x_pixel - cX) / fovX * depth, (y_pixel - cY) / fovY * depth, depth);
               Vector3f rayDirWorld = (rotMatrix*rayDirCamera).normalized();

               // position of the camera
               Vector3f rayOriginWorld = tranVector;

               float min_t = compute_min_t(rayOriginWorld, rayDirWorld);
               float max_t = compute_max_t(rayOriginWorld, rayDirWorld);

               float mu = 1;
               float t_step_size = 0.05; // function of truncation distance

               // loop
               float prev_sdf;
               float sdf;

               bool is_first_sdf = true;
               bool found_sign_change = false;

               for(float t=min_t; t<max_t; t+= t_step_size)
               {
                   Vector3f currPoint = rayOriginWorld + t * rayDirWorld;

                   //Vector3f relPoint = currPoint - m_tsdf->getOrigin();

                   if(is_first_sdf)
                   {
                       sdf = trilinear_interpolate(currPoint);
                       is_first_sdf = false;
                       continue;
                   }

                   prev_sdf = sdf;
                   sdf = trilinear_interpolate(currPoint);


                   if ((prev_sdf > 0 && sdf < 0)  || (prev_sdf == 0 && sdf < 0) || (prev_sdf > 0 && sdf == 0))
                   {
                       // found a surface
                       float t_star = t - t_step_size - (t_step_size * prev_sdf) / (sdf - prev_sdf);
                       Vector3f surfaceVertex = rayOriginWorld + t_star * rayDirWorld;

                       pointCloud.points[idx] = surfaceVertex;
                       pointCloud.pointsValid[idx] = true;
                       //pointCloud.points.push_back(surfaceVertex);
                       //pointCloud.pointsValid.push_back(true);

                       found_sign_change = true;
                       break;

                   }
                   else if ((prev_sdf < 0 && sdf > 0 ) || (prev_sdf == 0 && sdf > 0) || (prev_sdf < 0 && sdf == 0))
                   {
                       // back of surface
                       pointCloud.points[idx] = Vector3f(MINF, MINF, MINF);
                       pointCloud.pointsValid[idx] = false;

                       //pointCloud.points.push_back(Vector3f(MINF, MINF, MINF));
                       //pointCloud.pointsValid.push_back(false);
                       found_sign_change = true;
                       break;
                   }
                   else
                   {
                       // continue iteration
                   }
               }
               if(!found_sign_change)
               {
                   pointCloud.points[idx] = Vector3f(MINF, MINF, MINF);
                   pointCloud.pointsValid[idx] = false;
               }
           }
       }
       toc(begin);

       return pointCloud;
   }

private:

   float trilinear_interpolate(const Vector3f point) const
   {
       Vector3f relPoint = point - m_tsdf->getOrigin();

       float x = relPoint.x() / m_tsdf->getVoxelSize();
       float y = relPoint.y() / m_tsdf->getVoxelSize();
       float z = relPoint.z() / m_tsdf->getVoxelSize();

       // for numeric stability: set negative values within 0.5 index to small positive number
       x = (-0.5 < x && x < 0) ? std::numeric_limits<float>::epsilon() : x;
       y = (-0.5 < y && y < 0) ? std::numeric_limits<float>::epsilon() : y;
       z = (-0.5 < z && z < 0) ? std::numeric_limits<float>::epsilon() : z;

       // valid interpolation only possible with:
       // x >= 0, y>=0, z>=0 with equality
       // x < max_x, y < max_y ... no equality!
       assert(!((x < 0) || (y < 0) || (z < 0)));
       assert(!((x >= m_tsdf->getSize() - 1) || (y >= m_tsdf->getSize() - 1) || (z >= m_tsdf->getSize() - 1)));

       // notation follows
       // S. Parker: "Interactive Ray Tracing for Isosurface Rendering" 1999

       int x_0 = std::floor(x);
       int y_0 = std::floor(y);
       int z_0 = std::floor(z);

       float u_ = x - x_0;
       float v_ = y - y_0;
       float w_ = z - z_0;

       float u[] = {1 - u_, u_};
       float v[] = {1 - v_, v_};
       float w[] = {1 - w_, w_};

       float p = 0;

       for(int i=0; i<2; ++i)
       {
           for(int j=0; j<2; ++j)
           {
               for(int k=0; k<2; ++k)
               {
                   p += u[i] * v[j] * w[k] * (*m_tsdf)(x_0+i, y_0+j, z_0+k);
               }
           }
       }

       return p;


   }

   float compute_min_t(Vector3f origin, Vector3f direction) const
   {
       // get point at highest index: size^3 - 1
       Vector3f vol_max = m_tsdf->getPoint(pow(m_tsdf->getSize(), 3) - 1).head(3);

       // get point at lowest index: 0
       Vector3f vol_min = m_tsdf->getPoint(0).head(3);

       float min_t_x = ((direction.x() > 0 ? vol_min.x() : vol_max.x()) - origin.x()) / direction.x();
       float min_t_y = ((direction.y() > 0 ? vol_min.y() : vol_max.y()) - origin.y()) / direction.y();
       float min_t_z = ((direction.z() > 0 ? vol_min.z() : vol_max.z()) - origin.z()) / direction.z();

       return std::max<float>(std::max<float>(min_t_x, min_t_y), min_t_z);
   }

   float compute_max_t(Vector3f origin, Vector3f direction) const
   {
       // get point at highest index: size^3 - 1
       Vector3f vol_max = m_tsdf->getPoint(pow(m_tsdf->getSize(), 3) - 1).head(3);

       // get point at lowest index: 0
       Vector3f vol_min = m_tsdf->getPoint(0).head(3);

       float min_t_x = ((direction.x() > 0 ? vol_max.x() : vol_min.x()) - origin.x()) / direction.x();
       float min_t_y = ((direction.y() > 0 ? vol_max.y() : vol_min.y()) - origin.y()) / direction.y();
       float min_t_z = ((direction.z() > 0 ? vol_max.z() : vol_min.z()) - origin.z()) / direction.z();

       return std::min<float>(std::min<float>(min_t_x, min_t_y), min_t_z);
   }

   std::shared_ptr<Tsdf> m_tsdf;
   Matrix3f m_cameraIntrinsics;

};

typedef VirtualSensor InputType;
//template<class InputType>
class KiFuModel
{
public:
    KiFuModel(InputType& InputHandle)
        : m_InputHandle(&InputHandle)
    {
        m_InputHandle->processNextFrame();
        m_SurfaceMeasurer = std::make_unique<SurfaceMeasurer>(m_InputHandle->getDepthIntrinsics(),
                                                m_InputHandle->getDepthImageHeight(),
                                                m_InputHandle->getDepthImageWidth());
        m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());
        //m_SurfaceMeasurer->smoothInput();
        //m_SurfaceMeasurer->displayDepthMap();
        m_SurfaceMeasurer->process();

        PointCloud Frame0 = m_SurfaceMeasurer->getPointCloud();

        std::vector<bool> pointsAndNormalsValid;
        pointsAndNormalsValid.reserve( Frame0.pointsValid.size() );

        std::transform(Frame0.pointsValid.begin(), Frame0.pointsValid.end(), Frame0.normalsValid.begin(),
                       std::back_inserter(pointsAndNormalsValid), std::logical_and<>());


        m_PoseEstimator = std::make_unique<NearestNeighborPoseEstimator>();
        m_PoseEstimator->setTarget(PoseEstimator::pruneVector(Frame0.points, pointsAndNormalsValid),
                                   PoseEstimator::pruneVector(Frame0.normals, pointsAndNormalsValid));

        // set to 64
        // 512 will be ~500MB ram
        // 1024 -> 4GB
        m_tsdf = std::make_shared<Tsdf>(128, 1);
        m_tsdf->calcVoxelSize(Frame0);

        m_SurfaceReconstructor = std::make_unique<SurfaceReconstructor>(m_tsdf, m_InputHandle->getDepthIntrinsics());
        m_SurfaceReconstructor->reconstruct(m_InputHandle->getDepth(),
                                            m_InputHandle->getDepthImageHeight(),
                                            m_InputHandle->getDepthImageWidth(),
                                            Matrix4f::Identity());

        //m_tsdf->writeToFile("tsdf-test.ply", 0.01, 0);

        m_SurfacePredictor = std::make_unique<SurfacePredictor>(m_tsdf, m_InputHandle->getDepthIntrinsics());

        PointCloud Frame0_predicted = m_SurfacePredictor->predict(m_InputHandle->getDepthImageHeight(),
                                        m_InputHandle->getDepthImageWidth());

        Matrix4f pose = Matrix4f::Identity();
        //SimpleMesh Frame0_mesh(*m_InputHandle, pose);

        /*
        SimpleMesh Frame0_predicted_mesh(Frame0_predicted,
                                         m_InputHandle->getDepthImageHeight(),
                                         m_InputHandle->getDepthImageWidth(),
                                         1);


        SimpleMesh Frame0_mesh(Frame0,
                               m_InputHandle->getDepthImageHeight(),
                               m_InputHandle->getDepthImageWidth());

        Frame0_mesh.writeMesh("Frame0_mesh.off");
        Frame0_predicted_mesh.writeMesh("Frame0_predicted_mesh.off");
        */

    }


    bool processNextFrame()
    {
        if(!m_InputHandle->processNextFrame())
        {
            return true;
        }
        m_SurfaceMeasurer->registerInput(m_InputHandle->getDepth());
        m_SurfaceMeasurer->process();

        PointCloud nextFrame = m_SurfaceMeasurer->getPointCloud();

        std::vector<bool> pointsAndNormalsValid;
        pointsAndNormalsValid.reserve( nextFrame.pointsValid.size() );

        std::transform(nextFrame.pointsValid.begin(), nextFrame.pointsValid.end(), nextFrame.normalsValid.begin(),
            std::back_inserter(pointsAndNormalsValid), std::logical_and<>());

        m_PoseEstimator->setSource(PoseEstimator::pruneVector(nextFrame.points, pointsAndNormalsValid),
                                   PoseEstimator::pruneVector(nextFrame.normals, pointsAndNormalsValid), 8);


        // matrix inverse????
        Matrix4f currentCamToWorld = m_PoseEstimator->estimatePose();
        Matrix4f currentPose = currentCamToWorld.inverse();


        std::cout << currentPose << std::endl;

        return false;
    }


private:
    std::unique_ptr<SurfaceMeasurer> m_SurfaceMeasurer;
    std::unique_ptr<PoseEstimator> m_PoseEstimator;
    std::unique_ptr<SurfaceReconstructor> m_SurfaceReconstructor;
    std::unique_ptr<SurfacePredictor> m_SurfacePredictor;

    InputType* m_InputHandle;
    std::string param;
    std::shared_ptr<Tsdf> m_tsdf;

};


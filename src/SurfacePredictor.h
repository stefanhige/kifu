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
       pointCloud.pointsValid.resize(depthImageHeight*depthImageWidth);
       pointCloud.points.resize(depthImageHeight*depthImageWidth);
       pointCloud.normalsValid.resize(depthImageHeight*depthImageWidth);
       pointCloud.normals.resize(depthImageHeight*depthImageWidth);

       auto begin = tic();
       //#pragma omp parallel for
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

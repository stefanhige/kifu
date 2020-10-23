#pragma once

#include <string>
#include <iostream>
#include <assert.h>
#include <cstdint>
#include "Eigen.h"

//#define TIMING_ENABLED

#ifdef TIMING_ENABLED
#define tic() (omp_get_wtime())
#define toc(a) (printf("%s, %i: dur %f s\n", __FILE__, __LINE__, omp_get_wtime() - a))
#else
#define tic() ((double)0)
#define toc(a) ((void)a)
#endif

#define assert_ndbg(expr) {if(!(expr)){ \
    std::cerr << __FILE__ << ":" << __LINE__ << " " << __PRETTY_FUNCTION__ << ":" << \
    " Assertion '" << \
    #expr \
    << "' failed." << std::endl; exit(1);}}

struct PointCloud
{
    std::vector<Vector3f> points;
    std::vector<bool> pointsValid;
    std::vector<Vector3f> normals;
    std::vector<bool> normalsValid;

    void prune()
    {
        std::vector<bool> pointsAndNormalsValid;
        std::transform(pointsValid.begin(), pointsValid.end(), normalsValid.begin(),
                       std::back_inserter(pointsAndNormalsValid), std::logical_and<>());

        std::vector<Vector3f> points_;
        std::vector<Vector3f> normals_;
        for (uint i = 0; i < pointsAndNormalsValid.size(); ++i)
        {
            if(pointsAndNormalsValid[i])
            {
                points_.push_back(points[i]);
                normals_.push_back(normals[i]);
            }
        }
        points = points_;
        normals  = normals_;
        pointsValid = std::vector<bool>(points.size(), true);
        normalsValid = std::vector<bool>(normals.size(), true);
        assert_ndbg((points.size() == normals.size()) && (pointsValid.size() == normalsValid.size()) && (points.size() == pointsValid.size()));
    }
};

// truncated signed distance function
class Tsdf
{
public:
    Tsdf(unsigned int size, float voxelSize)
        : m_voxelSize(voxelSize)
    {
        assert_ndbg(!(size % 2));
        m_tsdf = new float[size*size*size];

        // initialize with zeros
        m_weight = new uint_least8_t[size*size*size]();

        // initialize with zeros
        m_color = new uint_least8_t[size*size*size*3]();

        m_size = size;
    }

    ~Tsdf()
    {
        delete [] m_tsdf;
        delete [] m_weight;
        delete [] m_color;
    }

    // set m_voxelSize according to the points
    // TODO: missing edge cases for pointCloud
    void calcVoxelSize(const PointCloud& pointCloud)
    {
        float x_min = 0, x_max = 0, y_min = 0, y_max = 0, z_min = 0, z_max = 0;
        bool flag = true;
        for(uint i = 0; i<pointCloud.points.size(); ++i)
        {
            if(pointCloud.pointsValid[i] && pointCloud.normalsValid[i])
            {
                // the first point
                if(flag)
                {
                    x_min = x_max = pointCloud.points[i].x();
                    y_min = y_max = pointCloud.points[i].y();
                    z_min = z_max = pointCloud.points[i].z();
                    flag = false;
                    continue;
                }

                x_max = (pointCloud.points[i].x() > x_max) ? pointCloud.points[i].x() : x_max;
                x_min = (pointCloud.points[i].x() < x_min) ? pointCloud.points[i].x() : x_min;

                y_max = (pointCloud.points[i].y() > y_max) ? pointCloud.points[i].y() : y_max;
                y_min = (pointCloud.points[i].y() < y_min) ? pointCloud.points[i].y() : y_min;

                z_max = (pointCloud.points[i].z() > z_max) ? pointCloud.points[i].z() : z_max;
                z_min = (pointCloud.points[i].z() < z_min) ? pointCloud.points[i].z() : z_min;
            }
        }
        //std::cout << "x " << x_min << " " << x_max << std::endl;
        //std::cout << "y " << y_min << " " << y_max << std::endl;
        //std::cout << "z " << z_min << " " << z_max << std::endl;

        m_origin = Vector3f(x_min, y_min, z_min);
        float max_span = std::max({x_max - x_min, y_max - y_min, z_max - z_min});
        //std::cout << max_span << std::endl;
        m_voxelSize = max_span / (m_size - 1);
        //std::cout << m_voxelSize << std::endl;
    }

    float& operator()(int x, int y, int z)
    {
        assert_ndbg(x < m_size && x >= 0);
        assert_ndbg(y < m_size && y >= 0);
        assert_ndbg(z < m_size && z >= 0);
        return m_tsdf[x + y*m_size + z*m_size*m_size];
    }

    float operator()(int x, int y, int z) const
    {
        assert_ndbg(x < m_size && x >= 0);
        assert_ndbg(y < m_size && y >= 0);
        assert_ndbg(z < m_size && z >= 0);
        return m_tsdf[x + y*m_size + z*m_size*m_size];
    }

    float operator()(Vector3f pos)
    {
       Vector3f rel_pos = pos - m_origin;
       int x = rel_pos.x() / m_voxelSize;
       int y = rel_pos.y() / m_voxelSize;
       int z = rel_pos.z() / m_voxelSize;
       return this->operator()(x, y, z);

    }

    float& operator()(const int idx)
    {
        return m_tsdf[idx];
    }

    uint_least8_t& weight(const int idx)
    {
        return m_weight[idx];
    }

    uint_least8_t weight(const int idx) const
    {
        return m_weight[idx];
    }

    uint_least8_t& colorR(const int idx)
    {
        return m_color[idx*3];
    }

    uint_least8_t colorR(const int idx) const
    {
        return m_color[idx*3];
    }

    uint_least8_t& colorG(const int idx)
    {
        return m_color[idx*3+1];
    }

    uint_least8_t colorG(const int idx) const
    {
        return m_color[idx*3+1];
    }

    uint_least8_t& colorB(const int idx)
    {
        return m_color[idx*3+2];
    }

    uint_least8_t colorB(const int idx) const
    {
        return m_color[idx*3+2];
    }

    uint_least8_t max_weight() const
    {
        // usually 255
        return UINT_LEAST8_MAX;
    }

    Vector4f getPoint(const int idx) const
    {
       auto indices = unravel_index(idx);
       int x = std::get<0>(indices);
       int y = std::get<1>(indices);
       int z = std::get<2>(indices);

       return Vector4f(x*m_voxelSize + m_origin.x(),
                       y*m_voxelSize + m_origin.y(),
                       z*m_voxelSize + m_origin.z(),
                       1);
    }

    Vector3f getOrigin() const
    {
        return m_origin;
    }

    float getVoxelSize() const
    {
        return m_voxelSize;
    }


    int ravel_index(const int x, const int y, const int z) const
    {
        assert_ndbg(x < m_size && x >= 0);
        assert_ndbg(y < m_size && y >= 0);
        assert_ndbg(z < m_size && z >= 0);
        return x + y*m_size + z*m_size*m_size;
    }

    int ravel_index(const std::tuple<int, int, int> xyz) const
    {
        return ravel_index(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz));
    }


     std::tuple<int, int, int> unravel_index(const int idx) const
    {
        assert_ndbg(static_cast<uint>(idx) < m_size*m_size*m_size && idx >= 0);
        const int x = idx % m_size;
        const int z = idx / (m_size*m_size);
        const int y = (idx / m_size) % m_size;

        return std::tuple<int, int, int>(x, y, z);
    }

    unsigned int getSize() const
    {
        return m_size;
    }


    void writeToFile(const std::string &file_name, float tsdf_threshold = 0.1, float weight_threshold = 0)
    {
      // number of points in point cloud
      int num_pts = 0;
      for (int i = 0; i < m_size * m_size * m_size; ++i)
      {
          if (std::abs(m_tsdf[i]) < tsdf_threshold && m_weight[i] > weight_threshold)
          {
              num_pts++;
          }
      }

      // .ply file header
      FILE *fp = fopen(file_name.c_str(), "w");
      fprintf(fp, "ply\n");
      fprintf(fp, "format binary_little_endian 1.0\n");
      fprintf(fp, "element vertex %d\n", num_pts);
      fprintf(fp, "property float x\n");
      fprintf(fp, "property float y\n");
      fprintf(fp, "property float z\n");
      fprintf(fp, "end_header\n");

      // point cloud for ply file
      for (int i = 0; i < m_size * m_size * m_size; ++i)
      {
        if (std::abs(m_tsdf[i]) < tsdf_threshold && m_weight[i] > weight_threshold)
        {
          std::tuple<int, int, int> xyz = unravel_index(i);
          float pt_base_x = m_origin.x() + std::get<0>(xyz) * m_voxelSize;
          float pt_base_y = m_origin.y() + std::get<1>(xyz) * m_voxelSize;
          float pt_base_z = m_origin.z() + std::get<2>(xyz) * m_voxelSize;
          fwrite(&pt_base_x, sizeof(float), 1, fp);
          fwrite(&pt_base_y, sizeof(float), 1, fp);
          fwrite(&pt_base_z, sizeof(float), 1, fp);
        }
      }
      fclose(fp);
    }

    bool isValid(const Vector3f& point) const
    {
        Vector3f relPoint = point - getOrigin();

        float x = relPoint.x() / getVoxelSize();
        float y = relPoint.y() / getVoxelSize();
        float z = relPoint.z() / getVoxelSize();

        // for numeric stability: set negative values within 0.5 index to small positive number
        x = (-0.5 < x && x < 0) ? std::numeric_limits<float>::epsilon() : x;
        y = (-0.5 < y && y < 0) ? std::numeric_limits<float>::epsilon() : y;
        z = (-0.5 < z && z < 0) ? std::numeric_limits<float>::epsilon() : z;

        // valid interpolation only possible with:
        // x >= 0, y>=0, z>=0 with equality
        // x < max_x, y < max_y ... no equality!
        if((x < 0) || (y < 0) || (z < 0))
        {
            return false;
        }
        if((x >= getSize() - 1) || (y >= getSize() - 1) || (z >= getSize() - 1))
        {
            return false;
        }
        return true;
    }



private:
    float* m_tsdf;
    uint_least8_t* m_weight;
    uint_least8_t* m_color;
    uint m_size;
    Vector3f m_origin = Vector3f(0, 0, 0);
    float m_voxelSize;
};

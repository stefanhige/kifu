#pragma once

#include <string>
#include <iostream>
#include <assert.h>
#include <cstdint>
#include "Eigen.h"

struct PointCloud
{
    std::vector<Vector3f> points;
    std::vector<bool> pointsValid;
    std::vector<Vector3f> normals;
    std::vector<bool> normalsValid;
};

// truncated signed distance function
class Tsdf
{
public:
    Tsdf(unsigned int size, float voxelSize)
        : m_voxelSize(voxelSize)
    {
        assert(!(size % 2));
        m_tsdf = new float[size*size*size];

        // initialize with zeros
        m_weight = new uint_least8_t[size*size*size]();

        m_size = size;
    }

    ~Tsdf()
    {
        delete [] m_tsdf;
        delete [] m_weight;
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
        assert(x < m_size && x >= 0);
        assert(y < m_size && y >= 0);
        assert(z < m_size && z >= 0);
        return m_tsdf[x + y*m_size + z*m_size*m_size];
    }

    float operator()(int x, int y, int z) const
    {
        assert(x < m_size && x >= 0);
        assert(y < m_size && y >= 0);
        assert(z < m_size && z >= 0);
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
        assert(x < m_size && x >= 0);
        assert(y < m_size && y >= 0);
        assert(z < m_size && z >= 0);
        return x + y*m_size + z*m_size*m_size;
    }

    int ravel_index(const std::tuple<int, int, int> xyz) const
    {
        return ravel_index(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz));
    }


     std::tuple<int, int, int> unravel_index(const int idx) const
    {
        assert(static_cast<uint>(idx) < m_size*m_size*m_size && idx >= 0);
        const int x = idx % m_size;
        const int z = idx / (m_size*m_size);
        const int y = (idx / m_size) % m_size;

        return std::tuple<int, int, int>(x, y, z);
    }

    unsigned int getSize()
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



private:
    float* m_tsdf;
    uint_least8_t* m_weight;
    uint m_size;
    Vector3f m_origin = Vector3f(0, 0, 0);
    float m_voxelSize;
};

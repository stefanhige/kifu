#pragma once

#include <string>
#include <assert.h>
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
        m_size = size;
    }

    ~Tsdf()
    {
        delete [] m_tsdf;
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

    float& operator()(int idx)
    {
        return m_tsdf[idx];
    }

    Vector4f getPoint(int idx)
    {
       return Vector4f(0, 0, 0, 1);
    }


    int ravel_index(int x, int y, int z) const
    {
        assert(x < m_size && x >= 0);
        assert(y < m_size && y >= 0);
        assert(z < m_size && z >= 0);
        return x + y*m_size + z*m_size*m_size;
    }

    int ravel_index(std::tuple<int, int, int> xyz) const
    {
        return ravel_index(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz));
    }


     std::tuple<int, int, int> unravel_index(const int idx) const
    {
        assert(idx < m_size*m_size*m_size && idx >= 0);

        const int x = idx % m_size;
        const int z = idx / (m_size*m_size);
        const int y = (idx / m_size) % m_size;

        return std::tuple<int, int, int>(x, y, z);
    }


private:
    float* m_tsdf;
    unsigned int m_size;
    Vector3f m_origin = Vector3f(0, 0, 0);
    float m_voxelSize;
};

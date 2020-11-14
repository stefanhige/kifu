#pragma once
#include <array>

template <int size, int sigma>
struct GaussianKernel
{
    constexpr GaussianKernel()
        : kernel()
    {
        static_assert(size>0);
        static_assert(size % 2 == 1);
        static_assert(size < 13);
        int it_r = size/2;

        float r = 0;
        float s = 2.0 * sigma * sigma;

        float sum = 0.0;

        for (int x = -it_r; x <= it_r; x++)
        {
            for (int y = -it_r; y <= it_r; y++)
            {
                r = std::sqrt(x * x + y * y);
                kernel.at(x + it_r + size*(y + it_r)) = (std::exp(-(r * r) / s)) / (M_PI * s);
                sum += kernel.at(x + it_r + size*(y + it_r));
            }
        }
        // std::transform not supported in constexpr
        //std::transform(kernel.begin(), kernel.end(), kernel.begin(),
        //               [&sum](float in) -> float {return in/sum;});

        for(auto& el : kernel)
        {
            el /= sum;
        }


    }
    std::array<float, size*size> kernel;
};

template <int size, int sigma>
class BilateralFilter
{
public:
    BilateralFilter(size_t depthImageWidth, size_t depthImageHeight)
        : kernel(GaussianKernel<size,sigma>()),
          w(depthImageWidth),
          h(depthImageHeight),
          it_s(size/2),
          tempImage(new float[w*h])
    {}

    ~BilateralFilter()
    {
        delete[] tempImage;
    }

    void apply(float* image)
    {
        std::copy(image, image+w*h, tempImage);

#pragma omp parallel for collapse(2)
        for(size_t x=it_s; x<= w-it_s; ++x)
        {
            for(size_t y=it_s; y<= h-it_s; ++y)
            {
                image[x + w*y] = evalKernel(tempImage, x, y);
            }
        }
    }

private:
    float evalKernel(const float* in, size_t x, size_t y) const
    {
        float res = 0;

        for (int i = -it_s; i <= it_s; i++)
        {
            for (int j = -it_s; j <= it_s; j++)
            {
                //res += kernel.at() * in[x+i + (y+j)*w];
                res += kernel.kernel.at(i+it_s + (j+it_s)*size) * in[x+i + (y+j)*w];
            }
        }
        return res;
    }
    const GaussianKernel<size,sigma> kernel;
    const size_t w;
    const size_t h;
    const int it_s;
    float* tempImage;
};

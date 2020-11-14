#pragma once
#include <array>
#include <math.h>

/**
 * Calculates a square gaussian kernel
 * @tparam size The size of size^2 gaussian kernel.
 * @tparam sigma Sigma of gaussian. Type is int because of being a template paramter.
 */
template<int size, int sigma, typename T = float>
struct GaussianKernel
{
    constexpr GaussianKernel()
        : kernel()
    {
        static_assert(size > 0, "Size has to be bigger than 0");
        static_assert(size % 2 == 1, "Size needs to be a multiple of two");
        static_assert(size < 13, "Size smaller than 14 not allowed"); //because 13 is also not allowed as it is not a multiple of two
        static_assert(sigma > 0, "Sima needs to be bigger than 0");

        int it_r = size/2;

        float r = 0;
        float s = 2.0 * sigma * sigma;

        float sum = 0.0;

        for (int x = -it_r; x <= it_r; ++x)
        {
            for (int y = -it_r; y <= it_r; ++y)
            {
                r = std::sqrt(x * x + y * y);
                //kernel.at(x + it_r + size*(y + it_r)) = (std::exp(-(r * r) / s)) / (M_PI * s);
                kernel.at(x + it_r + size*(y + it_r)) = (std::exp(-(r * r) / s));
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
    std::array<T, size*size> kernel;
};

/**
 * Performs bilateral filtering
 * @tparam size The size of size^2 gaussian kernel.
 * @tparam sigma Sigma of gaussian. Type is int because of being a template paramter.
 */
template <int size, int sigma>
class BilateralFilter
{
public:
    /**
     * @brief BilateralFilter
     * @param imageWidth Width of Image to be processed.
     * @param imageHeight Height of Image to be processed.
     */
    BilateralFilter(size_t imageWidth, size_t imageHeight)
        : w(imageWidth),
          h(imageHeight),
          it_s(size/2),
          tempImage(new float[w*h])
    {}

    ~BilateralFilter()
    {
        delete[] tempImage;
    }

    /**
     * @brief apply Apply filter object to image.
     * @param image pointer to image
     */
    void apply(float* image)
    {
        std::copy(image, image+w*h, tempImage);

        //#pragma omp parallel for
        for(size_t x = 0; x < w; ++x)
        {
            for(size_t y = 0; y < h; ++y)
            {
                image[x + w*y] = evalKernel(tempImage, x, y);
            }
        }
    }

private:
    float evalKernel(const float* in, size_t x, size_t y) const
    {
        float res = 0;

        for (int i = -it_s; i <= it_s; ++i)
        {
            for (int j = -it_s; j <= it_s; ++j)
            {
                // skip values at the corner (simulates zero-padding)
                if(int(x)+i < 0 || int(x)+i >= w || int(y)+j < 0 || int(y)+j >= h)
                {
                    //std::cout << "skipping index[" << static_cast<int>(x+i) <<
                    //             " " << static_cast<int>(y+j) << "]" << std::endl;
                    continue;
                }
                res += kernel.kernel.at(i+it_s + (j+it_s)*size) * in[x+i + (y+j)*w];
            }
        }
        return res;
    }
    // const GaussianKernel<size,sigma> kernel = GaussianKernel<size,sigma>();
    static constexpr GaussianKernel<size,sigma> kernel = GaussianKernel<size,sigma>();
    const size_t w;
    const size_t h;
    const int it_s;
    float* tempImage;
};

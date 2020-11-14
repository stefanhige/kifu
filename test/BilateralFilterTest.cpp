#include <gtest/gtest.h>
#include <numeric>
#include "BilateralFilter.h"

class BilateralFilterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_img = new float[3*3]();
    }

    void TearDown() override
    {
        delete[] m_img;
    }

    float* m_img;
};

TEST(GaussianKernelTest, TestTrivial)
{
    const auto kernel_1 = GaussianKernel<1,1>();
    const auto kernel_5 = GaussianKernel<5,1>();
    const auto kernel_7 = GaussianKernel<7,1>();

    // kernels have the correct size
    EXPECT_EQ(kernel_1.kernel.size(), 1);
    EXPECT_EQ(kernel_5.kernel.size(), 5*5);
    EXPECT_EQ(kernel_7.kernel.size(), 7*7);


    // kernels are normalized
    auto sum_1 = std::accumulate(kernel_1.kernel.begin(), kernel_1.kernel.end(), 0.0f);
    EXPECT_FLOAT_EQ(sum_1, 1);
    auto sum_5 = std::accumulate(kernel_5.kernel.begin(), kernel_5.kernel.end(), 0.0f);
    EXPECT_FLOAT_EQ(sum_5, 1);
    auto sum_7 = std::accumulate(kernel_7.kernel.begin(), kernel_7.kernel.end(), 0.0f);
    EXPECT_FLOAT_EQ(sum_7, 1);
}

float gauss_weight(int dx, int dy, float sigma)
{
    float weight = std::exp(-(dx*dx + dy*dy)/(2*sigma*sigma));
    return weight;
}

float normalization_constant(int kernel_size, float sigma)
{
    int sz = kernel_size/2;
    int ctr = 0;
    float sum = 0;
    for(int dx=-sz; dx<=sz; dx++)
    {
        for(int dy=-sz; dy<=sz; dy++)
        {
           ctr++;
           sum += gauss_weight(dx, dy, sigma);
        }
    }
    EXPECT_EQ(ctr, kernel_size*kernel_size);
    return sum;

}

TEST(GaussianKernelTest, TestTrivialValues)
{
    constexpr int sigma = 1;
    const auto kernel_1 = GaussianKernel<1,sigma>();
    const float center_1 = gauss_weight(0,0,sigma) / normalization_constant(1,sigma);

    // test the testing funcitons
    EXPECT_FLOAT_EQ(center_1, 1);

    // test GaussianKernel
    EXPECT_FLOAT_EQ(center_1, kernel_1.kernel.at(0));

}

TEST(GaussianKernelTest, TestValuesSigma1)
{
    constexpr int sigma = 1;
    constexpr int kernel_size = 3;
    const auto kernel_1 = GaussianKernel<kernel_size,sigma>();
    const float center_1 = gauss_weight(0,0,sigma) / normalization_constant(kernel_size, sigma);
    const float side_1 = gauss_weight(1,0,sigma) / normalization_constant(kernel_size, sigma);
    const float edge_1 = gauss_weight(1,1,sigma) / normalization_constant(kernel_size, sigma);

    // test GaussianKernel
    EXPECT_FLOAT_EQ(center_1, kernel_1.kernel.at(4));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(0));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(2));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(6));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(8));

    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(1));
    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(3));
    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(5));
    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(7));

}

TEST(GaussianKernelTest, TestValuesSigma10)
{
    constexpr int sigma = 10;
    constexpr int kernel_size = 3;
    const auto kernel_1 = GaussianKernel<kernel_size,sigma>();
    const float center_1 = gauss_weight(0,0,sigma) / normalization_constant(kernel_size, sigma);
    const float side_1 = gauss_weight(1,0,sigma) / normalization_constant(kernel_size, sigma);
    const float edge_1 = gauss_weight(1,1,sigma) / normalization_constant(kernel_size, sigma);

    // test GaussianKernel
    EXPECT_FLOAT_EQ(center_1, kernel_1.kernel.at(4));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(0));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(2));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(6));
    EXPECT_FLOAT_EQ(edge_1, kernel_1.kernel.at(8));

    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(1));
    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(3));
    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(5));
    EXPECT_FLOAT_EQ(side_1, kernel_1.kernel.at(7));

}
/*
TEST_F(BilateralFilterTest, TestTrivial)
{
    auto filter = BilateralFilter<1,1>(3,3);

    // set center to 1
    m_img[4] = 1;

    filter.apply(m_img);

    // expect no change
    EXPECT_FLOAT_EQ(m_img[0], 0);
    EXPECT_FLOAT_EQ(m_img[1], 0);
    EXPECT_FLOAT_EQ(m_img[4], 1);
}
*/

TEST_F(BilateralFilterTest, TestFilterSize3)
{
    constexpr int sigma = 1;
    constexpr int kernel_size = 3;
    auto filter = BilateralFilter<kernel_size,sigma>(3,3);

    // set center to 1
    m_img[4] = 1;

    filter.apply(m_img);

    // expect change of central value
    EXPECT_FLOAT_EQ(m_img[0], gauss_weight(1,1,sigma) / normalization_constant(kernel_size, sigma));
    EXPECT_FLOAT_EQ(m_img[1], gauss_weight(1,0,sigma) / normalization_constant(kernel_size, sigma));
    EXPECT_FLOAT_EQ(m_img[4], gauss_weight(0,0,sigma) / normalization_constant(kernel_size, sigma));
}


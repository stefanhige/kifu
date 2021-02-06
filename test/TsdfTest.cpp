#include <gtest/gtest.h>
#include "DataTypes.h"

TEST(TsdfTest, TestConstructor)
{
    Tsdf tsdf(10,1);

    Tsdf tsdf2;

    EXPECT_FALSE(tsdf.isEmpty());
    EXPECT_TRUE(tsdf2.isEmpty());

    tsdf2 = std::move(tsdf);

    EXPECT_TRUE(tsdf.isEmpty());
    EXPECT_FALSE(tsdf2.isEmpty());

    Tsdf tsdf3(std::move(tsdf2));

    EXPECT_TRUE(tsdf2.isEmpty());
    EXPECT_FALSE(tsdf3.isEmpty());
}

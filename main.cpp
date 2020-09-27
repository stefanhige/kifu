#include "KinectFusion.h"
#include "VirtualSensor.h"


int main()
{

    // load video
    std::string filenameIn = std::string("../3dscanning_lecture/data/rgbd_dataset_freiburg1_xyz/");
    std::cout << "Initialize virtual sensor..." << std::endl;

    VirtualSensor sensor;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    //KiFuModel<VirtualSensor> model(sensor);
    if (1)
    {
        Tsdf tsdf(6, 5);

        for (int i = 0; i<6*6*6; ++i)
        {
            tsdf(i) = 0;
        }

        // test
        std::vector<Vector3f> points;
        points.push_back(Vector3f(-1,1,1));
        points.push_back(Vector3f(2,2,2));

        std::vector<bool> pointsValid;
        pointsValid.push_back(true);
        pointsValid.push_back(true);

        PointCloud pointCloud;
        pointCloud.points = points;
        pointCloud.normals = points;
        pointCloud.pointsValid = pointsValid;
        pointCloud.normalsValid = pointsValid;

        tsdf.calcVoxelSize(pointCloud);

        Vector4f point = tsdf.getPoint(tsdf.ravel_index(0,3,5));
        std::cout << "point " << point << std::endl << std::endl;


        std::cout << tsdf(0,0,0) << std::endl;
        std::cout << tsdf(6*6) << std::endl << std::endl;

        int in = 6*6*6-1;
        std::cout << std::get<0>(tsdf.unravel_index(in)) << std::endl;
        std::cout << std::get<1>(tsdf.unravel_index(in)) << std::endl;
        std::cout << std::get<2>(tsdf.unravel_index(in)) << std::endl;
        std::cout << tsdf.ravel_index(tsdf.unravel_index(in)) << std::endl;


    }

    if (0)
    {
        KiFuModel model(sensor);
        //model.addInputHandle(sensor);

        //model.processNextFrame();
        //model.processNextFrame();
    }



    return 0;
}

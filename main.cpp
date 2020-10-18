#include "KinectFusion.h"
#include "VirtualSensor.h"


int main()
{

    // load video
    std::string filenameIn = std::string("../kifu/data/rgbd_dataset_freiburg1_xyz/");
    std::cout << "Initialize virtual sensor..." << std::endl;

    VirtualSensor sensor;
    if (!sensor.init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    KiFuModel model(sensor);
    model.processNextFrame();
    return 0;
}

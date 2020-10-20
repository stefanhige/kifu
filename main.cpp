#include "KinectFusion.h"
#include "VirtualSensor.h"


int main()
{

    // load video
    std::string filenameIn = std::string("../kifu/data/rgbd_dataset_freiburg1_plant/");
    std::cout << "Initialize virtual sensor..." << std::endl;

    VirtualSensor sensor;
    if (!sensor.init(filenameIn))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    KiFuModel model(sensor);
    for(int i=0; i<100; i++)
    {
        model.processNextFrame();
    }

    model.saveTsdf();
    return 0;
}

#include "KinectFusion.h"
#include "VirtualSensor.h"
#include <filesystem>

#include "StopWatch.h"

int main()
{
    // load video

    std::string filenameIn = std::string("kifu/data/rgbd_dataset_freiburg1_xyz/");

    //Verify that folder exists
    std::filesystem::path executableFolderPath =  std::filesystem::canonical("/proc/self/exe").parent_path();    //Folder of executable from system call
    std::filesystem::path dataFolderLocation = executableFolderPath.parent_path() / filenameIn;


    if (!std::filesystem::exists(dataFolderLocation))
    {
        std::cout << "No input files at folder " << dataFolderLocation << std::endl;
        return 0;
    }

    std::cout << "Initialize virtual sensor..." << std::endl;

    VirtualSensor sensor;
    if (!sensor.init(dataFolderLocation))
    {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    KiFuModel model(sensor);
    for(int i=0; i<100; i++)
    {
        StopWatch watch;
        model.processNextFrame();
    }

    model.saveTsdf("tsdf_frame3.ply");
    model.saveScreenshot("screenshot.png");
    return 0;
}

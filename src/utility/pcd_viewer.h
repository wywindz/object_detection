#ifndef PCD_VIEWER_H
#define PCD_VIEWER_H

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace detection {
  class pcd_viewer
  {
  public:
    pcd_viewer();

    /**
     * @brief show pointcloud from pcdfile
     * @param pcdfileName
     */
    inline static void showPCD(std::string pcdfileName)
    {
      typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudT;
      PointCloudT sourceCloud(new pcl::PointCloud<pcl::PointXYZ>);
      if(pcl::io::loadPCDFile(pcdfileName,*sourceCloud)==-1)
      {
        std::cout<<">>>ERROR: "<<"Load pcd file failed"<<std::endl;
        return;
      }
      else
        std::cout<<">>>INFO: "<<"Load pcd file finished"<<std::endl;

      pcl::visualization::PCLVisualizer viewer(pcdfileName);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(
            sourceCloud,150,150,150);
      viewer.addPointCloud(sourceCloud,handler,pcdfileName);
      viewer.spin();
    }

    /**
     * @brief show the input pointcloud
     * @param input pointcloud
     */
    template<typename pointT> inline static void showPCD(
        boost::shared_ptr<pcl::PointCloud<pointT> > cloudPtr)
    {
      pcl::visualization::PCLVisualizer viewer("pcd viewer");
      //pcl::visualization::PointCloudColorHandlerCustom<pointT> handler(
       //    cloudPtr,150,150,150);
      viewer.addPointCloud(cloudPtr,"sourceCloud");
      viewer.addCoordinateSystem(1.0,"coordframe",0);
      viewer.spin();
    }
  };
}


#endif // PCD_VIEWER_H

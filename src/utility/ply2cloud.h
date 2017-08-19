#ifndef PLY2CLOUD_H
#define PLY2CLOUD_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace detection
{
  class ply2cloud
  {
  public:
    typedef pcl::PointXYZRGB pointT;
    typedef pcl::PointCloud<pcl::PointXYZRGB> pointcloudT;
    ply2cloud();
    static int convert(std::string plyfileName, pointcloudT::Ptr & output);

  private:
    static double computeCloudResolution (pointcloudT::Ptr & cloud);
    static void convert2meterUnit(pointcloudT::Ptr &cloud);
  };
}
#endif // PLY2CLOUD_H

#include "ply2cloud.h"
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <exception>

namespace detection
{
  ply2cloud::ply2cloud()
  {

  }

  /**
   * @brief convert from ply to pointXYZRGB
   * @param plyfileName
   * @param outputCloud
   * @return
   */
  int ply2cloud::convert(std::string plyfileName, pointcloudT::Ptr & outputCloud)
  {
    pointcloudT::Ptr cloud(new pointcloudT);
    pcl::PolygonMesh mesh;
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

    try
    {
      pcl::io::loadPolygonFilePLY(plyfileName, mesh);
      pcl::io::mesh2vtk(mesh, polydata);
      pcl::io::vtkPolyDataToPointCloud(polydata, *cloud);
    }
    catch(std::exception & e)
    {
      return -1;
    }

    convert2meterUnit(cloud);

    double resolution=computeCloudResolution(cloud);
    if(resolution<0.01)
      {
        outputCloud=cloud;
        return 1;
      }

    //upsampling
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> filter;
    filter.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree;
    filter.setSearchMethod(kdtree);
    filter.setSearchRadius(0.01);
    //DISTINCT_CLOUD, or RANDOM_UNIFORM_DENSITY
    filter.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB>::RANDOM_UNIFORM_DENSITY);
    filter.setUpsamplingRadius(0.01);
    filter.setUpsamplingStepSize(0.005);
    filter.process(*outputCloud);

    return 1;
  }

  /**
   * @brief ply2cloud::convert2meterUnit
   * @param cloud
   */
  void ply2cloud::convert2meterUnit(pointcloudT::Ptr &cloud)
  {
    for(int i=0;i<cloud->points.size();++i)
      {
        cloud->points[i].x/=1000;
        cloud->points[i].y/=1000;
        cloud->points[i].z/=1000;
      }
  }

  /**
   * @brief ply2cloud::computeCloudResolution
   * @param cloud
   * @return
   */
  double ply2cloud::computeCloudResolution (pointcloudT::Ptr &cloud)
  {
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices (2);
    std::vector<float> sqr_distances (2);
    pcl::search::KdTree<pcl::PointXYZRGB> tree;
    tree.setInputCloud (cloud);

    for (size_t i = 0; i < cloud->size (); ++i)
    {
      if (! pcl_isfinite ((*cloud)[i].x))
      {
        continue;
      }
      //Considering the second neighbor since the first is the point itself.
      nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
      if (nres == 2)
      {
        res += sqrt (sqr_distances[1]);
        ++n_points;
      }
    }
    if (n_points != 0)
    {
      res /= n_points;
    }
    return res;
  }

}

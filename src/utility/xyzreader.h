#ifndef XYZREADER_H
#define XYZREADER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <exception>

namespace detection {

  class xyzReader
  {
  public:
    xyzReader();
    template<typename pointT> static int
    read(std::string fileName,pcl::PointCloud<pointT>& cloud)
    {
      std::ifstream stream;
      stream.open(fileName.c_str(),std::ios::in);
      if(!stream.is_open())
      {
        std::cerr<<">>> Error: cannot open model file "
          <<fileName<<std::endl;
        return -1;
      }

      int stp=0;
      cloud.points.clear();
      while(!stream.eof())
      {
        std::string line="";
        std::getline(stream,line);
        boost::trim(line);

        std::vector<std::string> pointData;
        boost::split(pointData,line,boost::is_any_of(" "),boost::token_compress_on);
        pointT point;
        //std::cout<<"x= "<<pointData[0]<<"  y= "<<pointData[1]<<"  z= "<<pointData[2]<<std::endl;
        try
        {
          point.x=boost::lexical_cast<float>(pointData[0]);
          point.y=boost::lexical_cast<float>(pointData[1]);
          point.z=boost::lexical_cast<float>(pointData[2]);

          cloud.points.push_back(point);
        }
        catch(std::exception e)
        {
          continue;
        }
      }

      cloud.height=1;
      cloud.width=cloud.points.size();
      return 0;
    }
  };
}


#endif // XYZREADER_H

#include <iostream>
#include <vector>
#include "utility/pcd_viewer.h"
#include "utility/ply2cloud.h"
#include "utility/xyzreader.h"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/sampling_surface_normal.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/correspondence.h>
#include <pcl/features/board.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/features//fpfh_omp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/vfh.h>

bool greater_size(const pcl::Correspondences& c1,const pcl::Correspondences& c2)
{
  return c1.size()>c2.size();
}

int main(int argc, char** argv)
{
    std::cout<<">>> INFO: runing object detect demo..."<<std::endl;

    /*if(argc!=2)
      {
        std::cout<<"You should input at least 1 params"<<std::endl;
        return -1;
      }
    std::string pcdFileName=argv[1];
    */

    //using RAM datasets from the paper:
    //Tutorial-Point Cloud Library Three-Dimensional Object Recognition and 6 DOF Pose Estimation

    const std::string pcdFileName="//home//wangy//dev//3dvision_ws//projects//RAM_dataset"
                            "//pcd_files//no_occlusions//frame_20111220T111153.549117.pcd";
    const std::string modelFileName="//home//wangy//dev//3dvision_ws//projects//RAM_dataset"
                            "//cad_models//Brita_water_filter_pitcher_6cup_x2.ply";
    const std::string pcdModelFileName="//home//wangy//dev//3dvision_ws//projects//RAM_dataset"
                            "//pcd_models//Brita_water_cup.pcd";
    const std::string xyzModelFileName="//home//wangy//dev//3dvision_ws//projects//RAM_dataset"
                            "//ply_new//Brita_water_filter_pitcher_6cup_x2.xyz";

    Eigen::Matrix4d ground_truth_transform = Eigen::Matrix4d::Identity ();
    ground_truth_transform(0,0)=-0.000661341e3;//-0.000661341e3;
    ground_truth_transform(0,1)=0.000750091e3;//0.000750091e3;
    ground_truth_transform(0,2)=1.94377e-03;
    ground_truth_transform(0,3)=0.21325;//0.21325;
    ground_truth_transform(1,0)=0.000422076e3;
    ground_truth_transform(1,1)=0.000374273e3;
    ground_truth_transform(1,2)=-0.000825705e3;
    ground_truth_transform(1,3)=-0.114822;
    ground_truth_transform(2,0)=-0.000620085e3;
    ground_truth_transform(2,1)=-0.000545246e3;
    ground_truth_transform(2,2)=-0.000564112e3;
    ground_truth_transform(2,3)=1.11771;

    typedef pcl::PointXYZRGB pointT;
    typedef pcl::PointCloud<pointT> pointCloudT;

    //load pcd file
    pointCloudT::Ptr sourceCloudPtr(new pointCloudT);
    if(pcl::io::loadPCDFile(pcdFileName,*sourceCloudPtr)==-1)
      {
        std::cout<<">>> ERROR: "<<"load pcd file failed"<<std::endl
                <<"pcd file: "<<pcdFileName<<std::endl;
        return -1;
      }
    else
      std::cout<<">>> INFO: load pcd file: "<<pcdFileName<<std::endl;

    int pointsize=sourceCloudPtr->points.size();
    std::cout<<">>> INFO: "<<"source cloud pointsize "<<pointsize<<std::endl;
    //visualize source cloud
    //detection::pcd_viewer::showPCD<pointT>(sourceCloudPtr);

    //Filter by passthrough
    pointCloudT::Ptr filteredCloudPtr(new pointCloudT);
    pcl::PassThrough<pointT> pass;
    double xmin=-1.0,xmax=1.0,ymin=-1.0,ymax=1.0,zmin=0,zmax=2.0;
    pass.setInputCloud(sourceCloudPtr);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(zmin,zmax);
    pass.filter(*filteredCloudPtr);

    pass.setInputCloud(filteredCloudPtr);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(xmin,xmax);
    pass.filter(*filteredCloudPtr);

    pass.setInputCloud(filteredCloudPtr);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(ymin,ymax);
    pass.filter(*filteredCloudPtr);
    std::cout<<">>> INFO: after pass through filter, cloud size is "
            <<filteredCloudPtr->points.size()<<std::endl;
    //detection::pcd_viewer::showPCD<pointT>(filteredCloudPtr);

    //downsample
    pointCloudT::Ptr processedCloudPtr(new pointCloudT);
    /*
    pcl::VoxelGrid<pointT> voxelgrid;
    voxelgrid.setInputCloud(filteredCloudPtr);
    voxelgrid.setLeafSize(0.002,0.002,0.002);
    voxelgrid.filter(*processedCloudPtr);
    std::cout<<">>> INFO: after voxelgrid filter, cloud size is "
            <<processedCloudPtr->points.size()<<std::endl;
    //detection::pcd_viewer::showPCD<pointT>(processedCloudPtr);
    */
    processedCloudPtr=filteredCloudPtr;

    //plane segment
    pcl::SACSegmentation<pointT> plane_seg;
    pcl::ModelCoefficients::Ptr coeffPtr(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    plane_seg.setOptimizeCoefficients(true);
    plane_seg.setModelType(pcl::SACMODEL_PLANE);
    plane_seg.setMethodType(pcl::SAC_RANSAC);
    plane_seg.setInputCloud(processedCloudPtr);
    plane_seg.setDistanceThreshold(0.01);
    plane_seg.segment(*inliers,*coeffPtr);
    if(inliers->indices.size()==0)
      {
        std::cout<<">>> INFO: no plane extracted"<<std::endl;
      }
    else
      std::cout<<">>> INFO: plane extracted, point size: "<<inliers->indices.size()<<std::endl;

    //extract plane and scene-without-plane
    pointCloudT::Ptr planeCloudPtr(new pointCloudT);
    pointCloudT::Ptr sceneCloudPtr(new pointCloudT);
    pcl::ExtractIndices<pointT> extractor;
    extractor.setInputCloud(processedCloudPtr);
    extractor.setIndices(inliers);
    extractor.setNegative(false);
    extractor.filter(*planeCloudPtr);
    extractor.setNegative(true);
    extractor.filter(*sceneCloudPtr);
    std::cout<<">>> INFO: scene extracted, point size: "<<sceneCloudPtr->points.size()<<std::endl;
    //detection::pcd_viewer::showPCD<pointT>(sceneCloudPtr);

    //outliers removal
    pcl::StatisticalOutlierRemoval<pointT> sor;
    sor.setInputCloud (sceneCloudPtr);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*sceneCloudPtr);
    std::cout<<">>> INFO: after outliers removal, scene point size: "<<sceneCloudPtr->points.size()<<std::endl;
    //detection::pcd_viewer::showPCD<pointT>(sceneCloudPtr);

    //euclidean cluster
    pcl::search::KdTree<pointT>::Ptr tree(new pcl::search::KdTree<pointT>);
    tree->setInputCloud(sceneCloudPtr);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pointT> clusterExtrac;
    clusterExtrac.setInputCloud(sceneCloudPtr);
    clusterExtrac.setSearchMethod(tree);
    clusterExtrac.setClusterTolerance(0.01);
    clusterExtrac.setMinClusterSize(100);
    clusterExtrac.setMaxClusterSize(25000);
    clusterExtrac.extract(cluster_indices);
    if(cluster_indices.size()==0)
      {
        std::cout<<">>> ERROR: no cluster extracted"<<std::endl;
        return -1;
      }
    else
      std::cout<<">>> INFO: extracted "<<cluster_indices.size()<<" clusters"<<std::endl;

    //extract the clusters
    std::vector<pointCloudT::Ptr> clusterClouds;
    pcl::ExtractIndices<pointT> extc;
    extc.setInputCloud(sceneCloudPtr);
    extc.setNegative(false);

    std::vector<pcl::PointIndices>::iterator iter;
    int idx=0;
    for(iter=cluster_indices.begin();iter!=cluster_indices.end();++iter)
      {
        pcl::PointIndices _indices=*iter;
        pcl::PointIndices::Ptr cluster=boost::make_shared<pcl::PointIndices>(_indices);
        std::cout<<"    cluster #"<<++idx<<" size: "<<cluster->indices.size()<<std::endl;
        pointCloudT::Ptr tmpCloud(new pointCloudT);
        extc.setIndices(cluster);
        extc.filter(*tmpCloud);
        clusterClouds.push_back(tmpCloud);
      }

    //load ply model, convert to pointcloud
    pointCloudT::Ptr modelPtr(new pointCloudT);
    //detection::ply2cloud::convert(modelFileName,modelPtr);

    /*
    if(pcl::io::loadPCDFile(pcdModelFileName,*modelPtr)==-1)
      {
        std::cerr<<">>> ERROR: "<<"failed to load model pcd file"<<std::endl;
      }
      */

    detection::xyzReader::read<pointT>(xyzModelFileName,*modelPtr);

    for(size_t i=0;i<modelPtr->points.size();++i)
      {
        modelPtr->points[i].x/=1000;
        modelPtr->points[i].y/=1000;
        modelPtr->points[i].z/=1000;
      }

    std::string savedPCDName="Brita_water_filter_pitcher_6cup_model.pcd";
    pcl::io::savePCDFileASCII(savedPCDName,*modelPtr);

    std::cout<<">>> INFO: model point size= "<<modelPtr->points.size()<<std::endl;
    pointCloudT::Ptr tmpModelCloud(new pointCloudT);
    pcl::transformPointCloud(*modelPtr,*tmpModelCloud,ground_truth_transform);
    modelPtr=tmpModelCloud;


    pointCloudT::Ptr scenePtr(clusterClouds.at(0));
    //Save cup pcd
    //std::string savedPCDName="Brita_water_filter_pitcher_6cup_scene.pcd";
    //pcl::io::savePCDFileASCII(savedPCDName,*scenePtr);

    //compute normals
    pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<pointT, pcl::Normal> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (modelPtr);
    norm_est.compute (*model_normals);

    norm_est.setInputCloud (scenePtr);
    norm_est.compute (*scene_normals);

    //downsample to get keypoints
    pointCloudT::Ptr model_keypoints(new pointCloudT);
    pointCloudT::Ptr scene_keypoints(new pointCloudT);

    double sample_leaf=0.01;
    pcl::VoxelGrid<pointT> sampleGrid;
    sampleGrid.setLeafSize(sample_leaf,sample_leaf,sample_leaf);
    sampleGrid.setInputCloud(modelPtr);
    sampleGrid.filter(*model_keypoints);
    std::cout << ">>> INFO: Model total points: " << modelPtr->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

    sample_leaf=0.01;
    sampleGrid.setLeafSize(sample_leaf,sample_leaf,sample_leaf);
    sampleGrid.setInputCloud(scenePtr);
    sampleGrid.filter(*scene_keypoints);
    std::cout << ">>> INFO: Scene total points: " << scenePtr->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

    //compute descriptors
    typedef pcl::SHOT352 descriptorT;
    pcl::PointCloud<descriptorT>::Ptr model_descriptors (new pcl::PointCloud<descriptorT> ());
    pcl::PointCloud<descriptorT>::Ptr scene_descriptors (new pcl::PointCloud<descriptorT> ());
    pcl::SHOTEstimationOMP<pointT, pcl::Normal, descriptorT> descr_est;
    descr_est.setRadiusSearch (0.05);

    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (modelPtr);
    descr_est.compute (*model_descriptors);
    std::cout<<">>> INFO: model descriptors size "<<model_descriptors->points.size()<<std::endl;

    descr_est.setInputCloud (scene_keypoints);
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (scenePtr);
    descr_est.compute (*scene_descriptors);
    std::cout<<">>> INFO: scene descriptors size "<<scene_descriptors->points.size()<<std::endl;

    //find model-scene correspondences
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::KdTreeFLANN<descriptorT> match_search;
    match_search.setInputCloud (model_descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
      std::vector<int> neigh_indices (1);
      std::vector<float> neigh_sqr_dists (1);
      if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
      {
        continue;
      }
      int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
      //std::cout<<"neign_sqr_dists:  "<<neigh_sqr_dists[0]<<std::endl;
      if(found_neighs == 1 && neigh_sqr_dists[0] < 0.3f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
      {
        pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
        model_scene_corrs->push_back (corr);
        //std::cout<<"  model_point_id: "<<neigh_indices[0]<<"  scene_point_id: "<<i<<std::endl;
      }
    }
    std::cout << ">>> INFO: Correspondences found: " << model_scene_corrs->size () << std::endl;

    //estimating pose
    typedef pcl::PointNormal PointNT;
    typedef pcl::PointCloud<PointNT> PointCloudNT;
    typedef pcl::FPFHSignature33 FeatureT;
    typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
    typedef pcl::PointCloud<FeatureT> FeatureCloudT;

    FeatureCloudT::Ptr model_features (new FeatureCloudT);
    FeatureCloudT::Ptr scene_features (new FeatureCloudT);

    PointCloudNT::Ptr fpfh_input_model(new PointCloudNT);
    PointCloudNT::Ptr fpfh_input_scene(new PointCloudNT);

    pcl::console::print_highlight ("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<pointT,PointNT> nest;
    nest.setRadiusSearch (0.01);
    nest.setInputCloud (model_keypoints);
    nest.compute (*fpfh_input_model);
    nest.setInputCloud (scene_keypoints);
    nest.compute (*fpfh_input_scene);

    // Estimate features
    FeatureEstimationT fest;
    fest.setRadiusSearch (0.025);
    //pointCloudT::ConstPtr model_keypoints_const=model_keypoints;
    fest.setInputCloud (fpfh_input_model);
    fest.setInputNormals (fpfh_input_model);
    fest.compute (*model_features);
    fest.setInputCloud (fpfh_input_scene);
    fest.setInputNormals (fpfh_input_scene);
    fest.compute (*scene_features);

    //SampleConsensusPrerejective

    double leaf=0.005;
    PointCloudNT::Ptr object_aligned(new PointCloudNT);

    pcl::SampleConsensusInitialAlignment<PointNT,PointNT,FeatureT> sac_ia;
    sac_ia.setInputSource(fpfh_input_model);
    sac_ia.setSourceFeatures(model_features);
    sac_ia.setInputTarget(fpfh_input_scene);
    sac_ia.setTargetFeatures(scene_features);
    sac_ia.setMaximumIterations(50);
    sac_ia.setNumberOfSamples(5);
    sac_ia.setCorrespondenceRandomness(5);
    sac_ia.setMaxCorrespondenceDistance(2.5f*0.01);
    sac_ia.align(*object_aligned);

    Eigen::Matrix4f transformation = sac_ia.getFinalTransformation ();
    pointCloudT::Ptr icp_init_guess(new pointCloudT);
    pcl::transformPointCloud(*model_keypoints,*icp_init_guess,transformation);


    //estimate the pose by icp, which has better accuracy
    pcl::IterativeClosestPoint<pointT, pointT> icp;
    icp.setInputSource(icp_init_guess);
    icp.setInputTarget(scene_keypoints);
    //icp.setMaxCorrespondenceDistance(0.5);
    pointCloudT Final;
    icp.align(Final);

    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;
    pointCloudT::Ptr final_cloud_(new pointCloudT);
    pcl::transformPointCloud(*icp_init_guess,*final_cloud_,icp.getFinalTransformation());

    //pcl::transformPointCloud(*model_keypoints,*final_cloud_,ground_truth_transform);

    pcl::visualization::PCLVisualizer alignViewer("Align results");
    pcl::visualization::PointCloudColorHandlerCustom<pointT> model_color_handler (final_cloud_, 255, 100, 100);
    alignViewer.addPointCloud (final_cloud_, model_color_handler, "model_cloud");
    pcl::visualization::PointCloudColorHandlerCustom<pointT> scene_color_handler (scene_keypoints, 128, 128, 128);
    alignViewer.addPointCloud (scene_keypoints, scene_color_handler, "scene_cloud");
    alignViewer.spin();


    //Actual Clustering

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    bool useHough=false;
    if(useHough)
      {
        //Compute (Keypoints) Reference Frames only for Hough
        typedef pcl::ReferenceFrame RFType;
        typedef pcl::Normal NormalType;
        pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
        pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

        pcl::BOARDLocalReferenceFrameEstimation<pointT, NormalType, RFType> rf_est;
        rf_est.setFindHoles (true);
        rf_est.setRadiusSearch (0.015);

        rf_est.setInputCloud (model_keypoints);
        rf_est.setInputNormals (model_normals);
        rf_est.setSearchSurface (modelPtr);
        rf_est.compute (*model_rf);

        rf_est.setInputCloud (scene_keypoints);
        rf_est.setInputNormals (scene_normals);
        rf_est.setSearchSurface (scenePtr);
        rf_est.compute (*scene_rf);

        //  Clustering
        pcl::Hough3DGrouping<pointT, pointT, RFType, RFType> clusterer;
        clusterer.setHoughBinSize (0.02);
        clusterer.setHoughThreshold (5);
        clusterer.setUseInterpolation (true);
        clusterer.setUseDistanceWeight (false);

        clusterer.setInputCloud (model_keypoints);
        clusterer.setInputRf (model_rf);
        clusterer.setSceneCloud (scene_keypoints);
        clusterer.setSceneRf (scene_rf);
        clusterer.setModelSceneCorrespondences (model_scene_corrs);

        clusterer.recognize (rototranslations, clustered_corrs);
      }
    else
      {
        pcl::GeometricConsistencyGrouping<pointT, pointT> gc_clusterer;
        gc_clusterer.setGCSize (0.03);
        gc_clusterer.setGCThreshold (5);

        gc_clusterer.setInputCloud (model_keypoints);
        gc_clusterer.setSceneCloud (scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

        //gc_clusterer.cluster (clustered_corrs);
        gc_clusterer.recognize (rototranslations, clustered_corrs);
      }

    std::sort(clustered_corrs.begin(),clustered_corrs.end(),greater_size);

    //Output results
    std::cout << ">>> INFO: Model instances found: " << rototranslations.size () << std::endl; 
    //for (size_t i = 0; i < rototranslations.size (); ++i)
    {
      size_t i=0;
      std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
      std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

      // Print the rotation matrix and translation vector
      Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
      Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

      printf ("\n");
      printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
      printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
      printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
      printf ("\n");
      printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
    }


    //results
    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    //viewer.addPointCloud (scenePtr, "scene_cloud");

    pcl::PointCloud<pointT>::Ptr off_scene_model (new pcl::PointCloud<pointT> ());
    pcl::PointCloud<pointT>::Ptr off_scene_model_keypoints (new pcl::PointCloud<pointT> ());

    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud (*modelPtr, *off_scene_model, Eigen::Vector3f (-0.8,0,0), Eigen::Quaternionf (1, 0, 0, 0));
    pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-0.8,0,0), Eigen::Quaternionf (1, 0, 0, 0));

    //pcl::visualization::PointCloudColorHandlerCustom<pointT> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
    //viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");

    pcl::visualization::PointCloudColorHandlerCustom<pointT> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
    viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<pointT> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "off_scene_model_keypoints");

    /*
    //draw lines of model_scene_correspondences
    for(size_t i=0;i<model_scene_corrs->size();++i)
    {
      pointT& model_p=off_scene_model_keypoints->at(model_scene_corrs->at(i).index_query);
      pointT& scene_p=scene_keypoints->at(model_scene_corrs->at(i).index_match);
      std::stringstream ss_line;
      ss_line << "line" << i;
      viewer.addLine<pointT, pointT> (model_p, scene_p, 0, 255, 0, ss_line.str ());
    }
    */

    int i=0;
    //for (size_t i = 0; i < rototranslations.size (); ++i)
    {
      pcl::PointCloud<pointT>::Ptr rotated_model (new pcl::PointCloud<pointT> ());
      pcl::transformPointCloud (*modelPtr, *rotated_model, rototranslations[i]);

      std::stringstream ss_cloud;
      ss_cloud << "instance" << i;

      pcl::visualization::PointCloudColorHandlerCustom<pointT> rotated_model_color_handler (rotated_model, 255, 0, 0);
      pcl::visualization::PointCloudColorHandlerCustom<pointT> final_model_color_handler (rotated_model, 0, 255, 50);

      viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
      //viewer.addPointCloud (boost::make_shared<pcl::PointCloud<pointT> >(Final), final_model_color_handler, "final_model");

      for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
      {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << i << "_" << j;
        pointT& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
        pointT& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        viewer.addLine<pointT, pointT> (model_point, scene_point, 0, 255, 0, ss_line.str ());
      }

    }

    while (!viewer.wasStopped ())
    {
      viewer.spinOnce ();
    }

    return 1;
}



#include <stdio.h>
#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/nonfree/features2d.hpp"
#include "openni2_wrapper.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"


using namespace cv;


void cameraCalibration(Mat image, std::vector<Point2f> obj_corners, std::vector<std::vector<Point2f>> scenes_corners, Mat &cameraMatrix, Mat &distCoeffs ) {

  std::vector<Point3f> obj_corners_3;
  std::vector<std::vector<Point3f>> obj_corners_3_vector;
  std::vector<std::vector<Point2f>> scenes_corners_3;
  scenes_corners_3.resize(scenes_corners.size());
  obj_corners_3_vector.resize(scenes_corners.size());
  obj_corners_3.resize(obj_corners.size());

  for (int i = 0; i < obj_corners.size(); i++)
    obj_corners_3[i] = Point3f(obj_corners[i].x, obj_corners[i].y, 0);

  for (int i = 0; i < scenes_corners.size(); i++)
    obj_corners_3_vector[i] = obj_corners_3;

  int tot = 0;
  for (int i = 0; i < scenes_corners.size(); i++)
    for(int j = 0; j < scenes_corners[i].size(); j++){
      scenes_corners_3[i].push_back(Point2f(scenes_corners[i][j].x, scenes_corners[i][j].y));
      tot++;
  }

  for(int i = 0 ; i < obj_corners_3_vector.size(); i++){
    std::cout << "obj at " << i;
    for (int j = 0 ; j < obj_corners_3.size(); j++)
        std::cout << "  " << obj_corners_3_vector[i][j];
    std::cout << std::endl;
  }

  for(int i = 0 ; i < scenes_corners_3.size(); i++){
    std::cout << "scene at " << i;
    for (int j = 0 ; j < scenes_corners_3[i].size(); j++)
      std::cout << "  " << scenes_corners_3[i][j];
    std::cout << std::endl;
  }

  cameraMatrix = Mat::eye(3, 3, CV_64F);
  distCoeffs = Mat::zeros(8, 1, CV_64F);
  std::vector<Mat> rvecs;
  std::vector<Mat> tvecs;

  std::cout << "obj size: " << obj_corners_3_vector.size() << " scene size: " << scenes_corners_3.size() <<" obj points: " << obj_corners.size()* obj_corners_3_vector.size()  << " scene points: " << tot << std::endl;
  calibrateCamera(obj_corners_3_vector, scenes_corners_3, image.size(), cameraMatrix, distCoeffs, rvecs, tvecs);
}







int main( int argc, char** argv )
{
  if( argc != 2 )
  { return -1; }

  Mat image = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat frame;

  if( !image.data ){ 
    std::cout<< " --(!) Error reading images " << std::endl; 
    return -1; 
  }
  

  openni::Status rc;
  rc = openni::OpenNI::initialize(); // Initialize OpenNI 
  if(rc != openni::STATUS_OK){ 
    std::cout << "OpenNI initialization failed" << std::endl; 
    openni::OpenNI::shutdown(); 
  } 
  else 
    std::cout << "OpenNI initialization successful" << std::endl; 

  openni::Device device;
  rc = device.open(openni::ANY_DEVICE); 
  if(rc != openni::STATUS_OK){ 
    std::cout << "Device initialization failed" << std::endl; 
    device.close(); 
  }

  VideoWrapper video_stream;
  video_stream.open(device, NI_SENSOR_COLOR);
  
  //Detect the keypoints using SURF Detector
  int minHessian = 1200;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_image, keypoints_frame;

  detector.detect( image, keypoints_image );

  //Calculate descriptors (feature vectors)

  Mat descriptors_image, descriptors_frame;

  SurfDescriptorExtractor extractor;
  extractor.compute( image, keypoints_image, descriptors_image );

  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;


  double max_dist = 0; double min_dist = 1000;
  std::vector< DMatch > good_matches;
  Mat img_matches;
  namedWindow("Good Matches", WINDOW_NORMAL);
  //video_stream.setUndistortParameters(argv[2]);
  double dist;

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  Mat H;
  std::vector<Point2f> obj_corners(4);
  std::vector<Point2f> scene_corners(4);

  
  //-- Get the corners from the image_1 ( the object to be "detected" )
  obj_corners[0] = cvPoint(0,0); 
  obj_corners[1] = cvPoint( image.cols, 0 );
  obj_corners[2] = cvPoint( image.cols, image.rows ); 
  obj_corners[3] = cvPoint( 0, image.rows );

  std::vector<std::vector<Point2f>> scene_corners_vector;
  bool calibrated = false;
  Mat cameraMatrix;
  Mat distCoeffs;
  Mat H_1;

  while(true){

    //video_stream.readFrameUndistorted(frame);
    video_stream.readFrame(frame);
    cvtColor(frame, frame, CV_BGR2GRAY);
  
    detector.detect( frame, keypoints_frame );
    extractor.compute( frame, keypoints_frame, descriptors_frame );

    //Matching descriptor vectors using FLANN matcher
  
    matcher.match( descriptors_image, descriptors_frame, matches );
    
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_image.rows; i++ ){
      dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
  
    for( int i = 0; i < descriptors_image.rows; i++ ){
      if( matches[i].distance <= max(2*min_dist, 0.1))
        good_matches.push_back( matches[i]); 
    }
    //-- Draw only "good" matches
    drawMatches( image, keypoints_image, frame, keypoints_frame, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if (good_matches.size() >= 10){
      for( int i = 0; i < good_matches.size(); i++ ){
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_image[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_frame[ good_matches[i].trainIdx ].pt );
      }

      H = findHomography( obj, scene, CV_RANSAC );

      if(H.rows == 3 && H.cols == 3){
        perspectiveTransform( obj_corners, scene_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f( image.cols, 0), scene_corners[1] + Point2f( image.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f( image.cols, 0), scene_corners[2] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f( image.cols, 0), scene_corners[3] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f( image.cols, 0), scene_corners[0] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
      }
      scene_corners_vector.push_back(scene_corners);
    }

    if(scene_corners_vector.size() == 25 && calibrated == false){
      std::cout << "Collected 25 images!" << std::endl;
      cameraCalibration(image, obj_corners, scene_corners_vector, cameraMatrix, distCoeffs );
      std::cout << "ciao!!!" << std::endl;
      calibrated = true;
    }
    
    //-- Show detected matches
    if(calibrated){
      Mat temp = img_matches.clone();
      //undistort(temp, img_matches, cameraMatrix, distCoeffs);
      imshow( "Good Matches", temp );
    } else
      imshow( "Good Matches", img_matches );

    matches.clear();
    good_matches.clear();
    obj.clear();
    scene.clear();
    scene_corners.resize(4);
    char key = waitKey(30);
    if(key == 27) break;

  }
  return 0;
}



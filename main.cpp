#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>         		// std::thread
#include <mutex>          		// std::mutex
#include <condition_variable>	// std::condition_variable

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>	// cvSize ... why ever this doenst come from core directly?!

const char* keys =
"{help h usage ? | | Usage examples: \n\
\t\t./simpleYoloObjectDetection --image=asdf.jpg \n\
\t\t./simpleYoloObjectDetection --video=asdf.mp4 \n\
\t\t./simpleYoloObjectDetection --device=0 \n\
\t\t./simpleYoloObjectDetection --device=0 --title=ObjDetectCam0 --yolo=/home/root/yolofiles \n\
\t\t./simpleYoloObjectDetection --device=0 cw=1920 ch=1080 nw=416 nh=416 \n\
}"
"{image i        |<none>| input image }"
"{video v        |<none>| input video }"
"{device d       |<none>| input device id }"
"{title t        |<none>| window title (default ObjDetect) }"
"{yolo y         |<none>| path prefix to yolo files (default \"\") }"
"{capturew cw    |<none>| camera device capture width (default 1920) }"
"{captureh ch    |<none>| camera device capture height (default 1080) }"
"{networkw nw    |<none>| blob width for the dnn (default 416) }"
"{networkh nh    |<none>| blob height for the dnn (default 416) }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
int capWidth = 1920;
int capHeight = 1080;
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

Net net;

// stuff for threaded detection
std::mutex detectionRunningMutex;
std::condition_variable detectionRunCondition;
bool detectionRunning = false;
bool detectionRdy = false;
Mat detectionFrame;
unsigned long detectionFrameCnt = 0;
vector<Mat> detectedOuts;
bool stopThread = false;

// Thread method waiting for a condvar to pop and do the processing
void detectionThread()
{
	while(stopThread == false)
	{
		// Wait until main() sends data
		{
			std::unique_lock<std::mutex> lk(detectionRunningMutex);
			detectionRunCondition.wait(lk, [] {return detectionRunning; });
			std::cout << "start with detection of detectionFrameCnt=" << detectionFrameCnt << std::endl;
		}

		if(!stopThread)
		{
			// Create a 4D blob from a frame.
			Mat blob;
			blobFromImage(detectionFrame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

			//Sets the input to the network
			net.setInput(blob);

			// Runs the forward pass to get output of the output layers
			net.forward(detectedOuts, getOutputsNames(net));

			std::cout << "done with  detection of detectionFrameCnt=" << detectionFrameCnt << std::endl;
			{
				std::unique_lock<std::mutex> lk(detectionRunningMutex);
				detectionRunning = false;
				detectionRdy = true;
			}
		}
	}
}

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);
	parser.about("Use this binary runs object detection using YOLO3 in OpenCV.");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	// Open a video file or an image file or a camera stream.
	string str, outputFile, windowTitle, pathYoloPrefix;
	VideoCapture cap;
	VideoWriter video;
	//Mat frame, blob;
	Mat frame;

	bool isLiveCameraMode = false;

	try
	{
		// defaults
		outputFile = "yolo_out_cpp.avi";
		windowTitle = "ObjDetect";
		pathYoloPrefix = "";

		if(parser.has("title"))
		{
			windowTitle = parser.get<String>("title");
			cout << "windowTitle=" << windowTitle << endl;
		}

		if(parser.has("yolo"))
		{
			pathYoloPrefix = parser.get<String>("yolo");
			cout << "yolo path prefix=" << pathYoloPrefix << endl;
		}

		if(parser.has("capturew") && parser.has("captureh"))
		{
			capWidth = parser.get<int>("capturew");
			capHeight = parser.get<int>("captureh");
			cout << "using capture size=[" << capWidth << "x" << capHeight << "]" << endl;
		}

		if(parser.has("networkw") && parser.has("networkh"))
		{
			inpWidth = parser.get<int>("networkw");
			inpHeight = parser.get<int>("networkh");
			cout << "using network size=[" << inpWidth << "x" << inpHeight << "]" << endl;
		}

		if (parser.has("image"))
		{
			// Open the image file
			str = parser.get<String>("image");
			ifstream ifile(str);
			if (!ifile) throw("failed to open file stream");
			if(cap.open(str) == false)
			{
				throw("failed to open capture");
			}
			str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
			outputFile = str;
			cout << "[IMAGE FILE MODE] input=" << str << " outputFile=" << outputFile << endl;
		}
		else if (parser.has("video"))
		{
			// Open the video file
			str = parser.get<String>("video");
			ifstream ifile(str);
			if (!ifile) throw("failed to open file stream");
			if(cap.open(str) == false)
			{
				throw("failed to open capture");
			}
			str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
			outputFile = str;
			cout << "[VIDEO FILE MODE] input=" << str << " outputFile=" << outputFile << endl;
		}
		else if (parser.has("device"))
		{
			// Open video device by ID
			int devId = parser.get<int>("device");
			if(cap.open(devId) == false)
			{
				throw("failed to open capture device");
			}
			cap.set(CAP_PROP_FRAME_WIDTH, capWidth);
			cap.set(CAP_PROP_FRAME_HEIGHT, capHeight);
			cout << "[CAM MODE] devId=" << devId << " capWidth=" << capWidth << " capHeight=" << capHeight << endl;

			isLiveCameraMode = true;
		}
		else
		{
			// Open Video device first
			if(cap.open(0) == false)
			{
				throw("failed to open capture device");
			}
			cap.set(CAP_PROP_FRAME_WIDTH, capWidth);
			cap.set(CAP_PROP_FRAME_HEIGHT, capHeight);
			cout << "[CAM MODE] video device capWidth=" << capWidth << " capHeight=" << capHeight << endl;

			isLiveCameraMode = true;
		}
	}
	catch(const std::exception& e)
	{
		cout << "Could not open the input image/video stream: " << e.what() << endl;
		return 0;
	}

	// Load names of classes
	string classesFile = pathYoloPrefix + "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);
	
	// Give the configuration and weight files for the model
	String modelConfiguration = pathYoloPrefix + "yolov3.cfg";
	String modelWeights = pathYoloPrefix + "yolov3.weights";
	//String modelConfiguration = pathYoloPrefix + "yolov3-spp.cfg";
	//String modelWeights = pathYoloPrefix + "yolov3-spp.weights";

	// Load the network
	/*Net*/ net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	//net.setPreferableTarget(DNN_TARGET_CPU);
	net.setPreferableTarget(DNN_TARGET_OPENCL);

	// Get the video writer initialized to save the output video
	if (parser.has("video")) {
		video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	}
    
	// Create a window
	static const string kWinName = windowTitle;
	namedWindow(kWinName, WINDOW_NORMAL);
	setWindowProperty(kWinName, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

	unsigned long frameCnt = 0;
	std::thread decodingThead = std::thread(detectionThread);;
	vector<Mat> outs;

	// Process frames.
	while (waitKey(1) < 0)
	{
		// get frame from the video
		cap >> frame;


		if(isLiveCameraMode == true)
		{
			bool newDetectionStarted = false;
			{
				std::lock_guard<std::mutex> lk(detectionRunningMutex);
				if (detectionRunning == false)
				{
					// check if there is was a frame process before
					if (detectionRdy == true)
					{
						outs = detectedOuts;
						detectionRdy = false;
					}

					// start a thread for the next one
					detectionFrame = frame;
					detectionFrameCnt = frameCnt;
					detectionRunning = true;
					newDetectionStarted = true;
				}
			}
			
			if(newDetectionStarted)
				detectionRunCondition.notify_one();
		}
		else
		{
			// Stop the program if reached end of video
			if (frame.empty())
			{
					cout << "Done processing !!!" << endl;
					cout << "Output file is stored as " << outputFile << endl;
					waitKey(3000);
					break;
			}
			// Create a 4D blob from a frame.
			Mat blob;
			blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
			
			//Sets the input to the network
			net.setInput(blob);
			
			// Runs the forward pass to get output of the output layers
			//vector<Mat> outs;
			net.forward(outs, getOutputsNames(net));
		}

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);
		
		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		/*vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));*/
		
		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (parser.has("image")) imwrite(outputFile, detectedFrame);
      else video.write(detectedFrame);
        
		std::cout << "draw frameCnt=" << frameCnt << std::endl;
		frameCnt++;
      imshow(kWinName, frame);
   }

	// signal termination and wait till the thread exits clean
	stopThread = true;
	detectionRunCondition.notify_one();
	decodingThead.join();
    
   cap.release();
   if (!parser.has("image")) video.release();

   return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	
	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;
					
					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
			}
		}
	}
	
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
					box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
	
	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();
		
		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();
		
		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
		names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

package org.opencv.samples.tutorial2;

import java.util.Date;
import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;
import java.util.Vector;

import android.content.Context;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.Sensor;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.Point;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.features2d.*;
import org.opencv.calib3d.*;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class Tutorial2Activity extends Activity implements CvCameraViewListener2 {
	private static final String    TAG = "OCVSample::Activity";
	private static final String    Debug = "Debug";

    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_CANNY    = 2;
    private static final int       VIEW_MODE_FEATURES = 5;
    private static final int       VIEW_MODE_THRESH = 3;
    private static final int       VIEW_MODE_SAVE = 4;
    
    private Date                   sampleDate = new Date();
    private long                   sampleTimer_old = sampleDate.getTime() / 1000;
    private long                   sampleTimer = -1;
    
    float[] Acceleration = new float [3];
	float[] Magnetic = new float [3];
    private float[]                Rotation_init = new float [9];
    private Mat                    Rotation_init_mat;
    
    private static SensorManager   sensorService;
    private Sensor                 sensorAccelerometer;
    private Sensor                 sensorMagneticField;
    private int                    MinAccelerometerDelay;
    private int                    MinMagneticFieldDelay;
            
    private int                    mViewMode;
    private Mat                    mRgba;
    private ArrayList<Mat>         samples = new ArrayList<Mat>();
    private Mat                    mIntermediateMat;
    private Mat                    mGray;

    private MenuItem               mItemPreviewRGBA;
    private MenuItem               mItemPreviewGray;
    private MenuItem               mItemPreviewCanny;
    private MenuItem               mItemPreviewFeatures;
    private MenuItem 			   mItemPreviewThresh;
    private MenuItem 			   mItemPreviewSave;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("mixed_sample");

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public Tutorial2Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.tutorial2_surface_view);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial2_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        
        sensorService = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        sensorAccelerometer = sensorService.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        MinAccelerometerDelay = sensorAccelerometer.getMinDelay ();
        if (sensorAccelerometer != null) {
        	sensorService.registerListener(sampleListener, sensorAccelerometer, MinAccelerometerDelay);
            Log.i("Tag", "Registered for Accelerometer Sensor");
        } 
        else {
            Log.e("Tag", "Acceleromter Sensor not found");
            finish();
        }
        sensorMagneticField = sensorService.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        MinMagneticFieldDelay = sensorMagneticField.getMinDelay();
        if (sensorMagneticField != null) {
        	sensorService.registerListener(sampleListener, sensorMagneticField, MinMagneticFieldDelay);
            Log.i("Tag", "Registered for Magnetic Field Sensor");
        } 
        else {
            Log.e("Tag", "Magnetic Field Sensor not found");
            finish();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewFeatures = menu.add("Find features");
        mItemPreviewThresh = menu.add("Thresh"); 
        mItemPreviewSave = menu.add("Save");
        return true;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        sensorService.unregisterListener(sampleListener);
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
        sensorService.registerListener(sampleListener, sensorAccelerometer, MinAccelerometerDelay);
        sensorService.registerListener(sampleListener, sensorMagneticField, MinMagneticFieldDelay);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        sensorService.unregisterListener(sampleListener);        
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        final int viewMode = mViewMode;
        switch (viewMode) {
        case VIEW_MODE_GRAY:
            // input frame has gray scale format
            Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            break;
        case VIEW_MODE_RGBA:
            // input frame has RBGA format
            mRgba = inputFrame.rgba();
            break;
        case VIEW_MODE_CANNY:
            // input frame has gray scale format
            mRgba = inputFrame.rgba();
            Imgproc.Canny(inputFrame.gray(), mIntermediateMat, 80, 100);
            Imgproc.cvtColor(mIntermediateMat, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            break;
        case VIEW_MODE_FEATURES:
            // input frame has RGBA format
            mRgba = inputFrame.rgba();
            mGray = inputFrame.gray();
            FindFeatures(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());
            break;
        case VIEW_MODE_THRESH:
        	mRgba = inputFrame.rgba(); 
        	int maxValue = 255; 
        	int blockSize = 61; 
        	int meanOffset = 15; 
        	Imgproc.adaptiveThreshold( 
        	inputFrame.gray(), 
        	 mIntermediateMat, 
        	 maxValue, 
        	 Imgproc.ADAPTIVE_THRESH_MEAN_C, 
        	 Imgproc.THRESH_BINARY_INV, 
        	 blockSize, 
        	 meanOffset 
        	); 
        	Imgproc.cvtColor( 
        	mIntermediateMat, 
        	mRgba, 
        	Imgproc.COLOR_GRAY2RGBA, 
        	4 
        	); 
        	break;
        case VIEW_MODE_SAVE:
        	sampleDate = new Date();
        	sampleTimer = sampleDate.getTime() / 5000;
        	if ((sampleTimer > sampleTimer_old || samples.isEmpty()) && samples.size() < 2) {
        		mGray = inputFrame.gray();        		
        		if (samples.isEmpty()) {
        			samples.add(mGray.clone());
        			Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
        		}
        		else {        			
        			DescriptorExtractor OrbExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        			FeatureDetector OrbDetector = FeatureDetector.create(FeatureDetector.ORB);
        			DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            		MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
            	    MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
                    Mat[] descriptors = new Mat[2];
                    descriptors[0] = new Mat();
                    descriptors[1] = new Mat();
                    
                    Date temp1 = new Date();
                    long temp1_t = temp1.getTime();
                    OrbDetector.detect(samples.get(samples.size()-1), keypoints1);
                    OrbExtractor.compute(samples.get(samples.size()-1), keypoints1, descriptors[0]); 
                    KeyPoint[] ArrayOfKeyPoints1 = keypoints1.toArray();
                    OrbDetector.detect(mGray, keypoints2);
                    OrbExtractor.compute(mGray, keypoints2, descriptors[1]);
                    KeyPoint[] ArrayOfKeyPoints2 = keypoints2.toArray();
                   
                    ArrayList<MatOfDMatch> AllMatches = new ArrayList<MatOfDMatch>();
                    AllMatches.add(new MatOfDMatch());
                    AllMatches.add(new MatOfDMatch());
                    matcher.knnMatch(descriptors[0], descriptors[1], AllMatches, 2);
                    
                    int length = AllMatches.size();
                    
                    ArrayList<Point> ArrayListOfPoints1 = new ArrayList<Point>();
                    ArrayList<Point> ArrayListOfPoints2 = new ArrayList<Point>();
                    for (int i = 0; i < length; i++) {
                    	DMatch[] ArrayOfDMatch = AllMatches.get(i).toArray();
                    	int index1 = ArrayOfDMatch[0].queryIdx;
                    	float distance1 = ArrayOfDMatch[0].distance;
                    	int index2 = ArrayOfDMatch[0].trainIdx;
                    	float distance2 = ArrayOfDMatch[1].distance;
                    	if (distance1 < 0.6 * distance2) {
                    		if (Math.abs(ArrayOfKeyPoints1[index1].pt.y - ArrayOfKeyPoints2[index2].pt.y) < 20) {
                    			ArrayListOfPoints1.add(ArrayOfKeyPoints1[index1].pt);
                    			ArrayListOfPoints2.add(ArrayOfKeyPoints2[index2].pt);
                    		}
                    	}
                    }
                    
                    Point [] ArrayOfPoints1 = new Point [ArrayListOfPoints1.size()];
                    Point [] ArrayOfPoints2 = new Point [ArrayListOfPoints2.size()];
                    ArrayListOfPoints1.toArray(ArrayOfPoints1);
                    ArrayListOfPoints2.toArray(ArrayOfPoints2);
                    for (int i = 0; i < ArrayOfPoints1.length; i++) {
                    	Imgproc.cvtColor(samples.get(0), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                    	
                    	Core.circle(mRgba, ArrayOfPoints1[i], 10, new Scalar(255, 255, 255, 255));
                    	Imgproc.cvtColor(mRgba, samples.get(0), Imgproc.COLOR_RGBA2GRAY, 1);                    	
                    }             
                    MatOfPoint2f MatOfpoints1 = new MatOfPoint2f(ArrayOfPoints1);
                    MatOfPoint2f MatOfpoints2 = new MatOfPoint2f(ArrayOfPoints2);
                    double[] HomographyArray = new double [9];
                    Mat HomographyMatrix = Calib3d.findHomography(MatOfpoints2, MatOfpoints1, Calib3d.RANSAC, 1);
                    Mat mGray_transformed = new Mat(mGray.size(), CvType.CV_8UC1);
                    Imgproc.warpPerspective(mGray, mGray_transformed, HomographyMatrix, mGray.size());
                    HomographyMatrix.get(0,  0, HomographyArray);                    
                    Date temp2 = new Date();
                    long temp2_t = temp2.getTime();
                    //Log.i(Debug, "" + temp1_t + " " + temp2_t + " ");
                    //Log.i(Debug, "" + HomographyArray[0] + " " + HomographyArray[1] + " " + HomographyArray[2]);
                    //Log.i(Debug, "" + HomographyArray[3] + " " + HomographyArray[4] + " " + HomographyArray[5]);
                    //Log.i(Debug, "" + HomographyArray[6] + " " + HomographyArray[7] + " " + HomographyArray[8]);
                    samples.add(mGray_transformed.clone());
                    for (int i = 0; i < ArrayOfPoints2.length; i++) {
                    	Imgproc.cvtColor(samples.get(1), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                    	Mat HomoPoint = new Mat(3, 1, CvType.CV_64FC1);
                    	Mat TransHomoPoint = new Mat(3, 1, CvType.CV_64FC1);
                    	HomoPoint.put(0, 0, (double)ArrayOfPoints2[i].x);
                    	HomoPoint.put(1, 0, (double)ArrayOfPoints2[i].y);
                    	HomoPoint.put(2, 0, 1.0);
                    	Core.gemm(HomographyMatrix, HomoPoint, 1.0, Mat.zeros(3, 1, CvType.CV_64FC1), 0.0, TransHomoPoint);  
                    	ArrayOfPoints2[i].x = TransHomoPoint.get(0, 0)[0] / TransHomoPoint.get(2, 0)[0];
                    	ArrayOfPoints2[i].y = TransHomoPoint.get(1, 0)[0] / TransHomoPoint.get(2, 0)[0];                    	
                    	Core.circle(mRgba, ArrayOfPoints2[i], 10, new Scalar(255, 255, 255, 255));
                    	Imgproc.cvtColor(mRgba, samples.get(1), Imgproc.COLOR_RGBA2GRAY, 4);                    	
                    }               
        	    }
                sampleTimer_old = sampleTimer;
                break;
        	}
        	else {
        		if (samples.size() == 1) {
        			mGray = samples.get(0);
        		}
        		else {
        			Date temp3 = new Date();
        			long temp3_t = temp3.getTime();
        			if ((double)temp3_t / 1000 - temp3_t / 1000 < 0.5) {
        				mGray = samples.get(0);
        			}
        			else {
        				mGray = samples.get(1);
        			}
        		}
        		Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
        		break;
        	}
        }

        return mRgba;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemPreviewRGBA) {
            mViewMode = VIEW_MODE_RGBA;
        } else if (item == mItemPreviewGray) {
            mViewMode = VIEW_MODE_GRAY;
        } else if (item == mItemPreviewCanny) {
            mViewMode = VIEW_MODE_CANNY;
        } else if (item == mItemPreviewFeatures) {
            mViewMode = VIEW_MODE_FEATURES;
        } else if (item == mItemPreviewThresh) {
        	mViewMode = VIEW_MODE_THRESH; 
        } else if (item == mItemPreviewSave) {
        	mViewMode = VIEW_MODE_SAVE; 
        }

        return true;
    }
    
    private SensorEventListener sampleListener = new SensorEventListener () { 
      
        @Override
        public final void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
        }
      
        @Override
        public final void onSensorChanged(SensorEvent event) {
        	if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
        		Acceleration[0] = event.values[0];
        		Acceleration[1] = event.values[1];
        		Acceleration[2] = event.values[2];
        	}
        	else {
        		if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
        			Magnetic[0] = event.values[0];
        			Magnetic[1] = event.values[1];
        			Magnetic[2] = event.values[2];
        		}
            }
        }
    };
    

    public native void FindFeatures(long matAddrGr, long matAddrRgba);
}

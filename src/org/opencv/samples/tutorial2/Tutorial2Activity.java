package org.opencv.samples.tutorial2;

import java.io.File;
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
import android.os.AsyncTask;
import android.os.Environment;

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
import org.opencv.highgui.Highgui;
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

	private static final int       CAMERA = 0;
    private static final int       GALLERY = 1;
    private static final int       STITCH = 2;
    
    private boolean				   needCapture = false;
    private boolean                needClear = false;
    private boolean                notStart = true;
    private boolean                stitchDone = false;
    
    BackgroundStitching            task;                
    private int                    mViewMode;
    private Mat                    mRgba;
    private Mat                    result;
    private byte[]                 black;
    private ArrayList<Mat>         processing_samples = new ArrayList<Mat>();
    private ArrayList<Long>        processing_addrs = new ArrayList<Long>();
    Mat errorCode;

    
    private MenuItem               mItemCamera;
    private MenuItem               mItemGallery;
    private MenuItem               mItemStitch;
    private MenuItem 			   mItemClear;
    private MenuItem               mItemCapture;

    private CameraBridgeViewBase   mOpenCvCameraView;
    
    private class BackgroundStitching extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... A) {
			byte[] errorCounter = new byte [1];
			errorCounter[0] = 5;
			errorCode.put(0, 0, errorCounter);
    		FindFeatures(processing_addrs, errorCode.getNativeObjAddr());	   
        	return "Done!";
        }
        
        @Override
        protected void onPostExecute(String done) {
        	notStart = true;
        	stitchDone = true;
        }
    }
    

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
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemCamera = menu.add("Camera");
        mItemGallery = menu.add("Gallery");
        mItemStitch  = menu.add("Stitch");
        mItemClear = menu.add("Clear");
        mItemCapture = menu.add("Capture");
        return true;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        black = new byte [height * width * 4];
        for (int i = 0; i < height * width * 4; i++) {
        	black[i] = 0;
        }
        errorCode = new Mat(1, 1, CvType.CV_8UC1);
        byte[] errorCounter = new byte [1];
        errorCounter[0] = 0;
        errorCode.put(0, 0, errorCounter);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        final int viewMode = mViewMode;
        switch (viewMode) {
        case CAMERA:
        	mRgba = inputFrame.rgba();        	
        	if (needClear == true) {
        		processing_samples.clear();
        		processing_addrs.clear();
        		result = null;
        		needClear = false;
        		stitchDone = false;
        		byte [] errorCounter = new byte[1];
        		errorCounter[0] = 0;
    			errorCode.put(0, 0, errorCounter);
        	} else {
        		if (needCapture == true) {
	        		if (stitchDone == false) {
		        		/*if (processing_samples.size() < 7) {
		        			File root = Environment.getExternalStorageDirectory();        		
		                	File file = new File(root, "DCIM/Camera/" + (processing_samples.size() + 1) + ".jpg");
		                    processing_samples.add(Highgui.imread(file.getAbsolutePath()));
		                    processing_addrs.add(Long.valueOf(processing_samples.get(processing_samples.size() - 1).getNativeObjAddr()));
		        		}*/
	        			Mat mRgb = new Mat(mRgba.size(), CvType.CV_8UC3);
	        			Imgproc.cvtColor(mRgba, mRgb, Imgproc.COLOR_RGBA2RGB, 3);
	        			processing_samples.add(mRgb);
	        			processing_addrs.add(Long.valueOf(processing_samples.get(processing_samples.size() - 1).getNativeObjAddr()));
	        		}
	        		needCapture = false;
        		}
        		Core.putText(mRgba, "Image captured: " + processing_samples.size(),  new Point(100, 100), Core.FONT_HERSHEY_SIMPLEX, 3.0, new Scalar(0, 255, 0, 255), 5);
        	}
        	break;
        case GALLERY:
        	mRgba = inputFrame.rgba();
        	if (needClear == true) {
        		processing_samples.clear();
        		processing_addrs.clear();
        		result = null;
    			needClear = false;
    			stitchDone = false;
    			byte [] errorCounter = new byte[1];
    			errorCounter[0] = 0;
    			errorCode.put(0, 0, errorCounter);
    			mViewMode = CAMERA;
        	}
        	if (processing_samples.size() > 0) {
        		long sampleTime = new Date().getTime() / 1000;
        		Mat sampleImage = processing_samples.get((int)sampleTime % processing_samples.size()); 
        		Imgproc.resize(sampleImage, mRgba, mRgba.size());
        		Core.putText(mRgba, "Image captured: " + (int)sampleTime % processing_samples.size(),  new Point(100, 100), Core.FONT_HERSHEY_SIMPLEX, 3.0, new Scalar(0, 255, 0, 255), 5);
        	}
        	else {
        		mRgba.put(0, 0, black);
        	    Core.putText(mRgba, "No captured image! ",  new Point(mRgba.cols() / 2 - 800, mRgba.rows() / 2), Core.FONT_HERSHEY_SIMPLEX, 5.0, new Scalar(255, 128, 0, 255), 3);
        	}
        	break;
        case STITCH:
        	mRgba = inputFrame.rgba();
        	byte[] errorCounter = new byte [1];
        	errorCode.get(0, 0, errorCounter);
        	if (needClear == true) {
        		if ((notStart == false || stitchDone == true) &&  errorCounter[0] > 0) {
        			task.cancel(true);
        		}
        		processing_samples.clear();
        		processing_samples.clear();
        		result = null;
        		notStart = true;
    			needClear = false;
    			stitchDone = false;
    			errorCounter[0] = 0;
    			errorCode.put(0, 0, errorCounter);
    			mViewMode = CAMERA;
    			break;
        	}
        	else {
	        	if (notStart == true && stitchDone == false) {
	        		if (processing_samples.size() >= 2) {
	        			notStart = false;
	        			task = new BackgroundStitching();
	        			task.execute(new String[] { "start" });
	        		}
	        		else {
	        			stitchDone = true;
	        			notStart = true;
	        			errorCounter[0] = 0;
	                	errorCode.put(0, 0, errorCounter);
	        		}
	        	}
        	}	   
        	errorCode.get(0, 0, errorCounter);
        	if (stitchDone == true) {
        		Log.i(Debug, "" + errorCounter[0]);
        		if (errorCounter[0] == 100) {
        			if (result == null) {
        				result = new Mat(mRgba.size(), CvType.CV_8UC4);
        				File root = Environment.getExternalStorageDirectory();        		
        				File file = new File(root, "DCIM/Camera/result.jpg");
        				Imgproc.resize(Highgui.imread(file.getAbsolutePath()), result, result.size());
        			}
        		}
        		else {
        			if (result == null) {
        				result = new Mat(mRgba.size(), CvType.CV_8UC4);
        			}
        			result.put(0, 0, black);
        			Core.putText(result, "Can't stitch! ",  new Point(result.cols() / 2 - 400, result.rows() / 2), Core.FONT_HERSHEY_SIMPLEX, 5.0, new Scalar(255, 0, 0, 255), 3);
        		}
        		mRgba = result;        		
        	}
        	else {
        		mRgba.put(0, 0, black);
        		if (errorCounter[0] < 10) {
        			Core.putText(mRgba, "" + (int)errorCounter[0] + "%",  new Point(mRgba.cols() / 2 - 200, mRgba.rows() / 2), Core.FONT_HERSHEY_SCRIPT_COMPLEX, 10.0, new Scalar(255, 255, 0, 255), 5);
        		}
        		else {
        			Core.putText(mRgba, "" + (int)errorCounter[0] + "%",  new Point(mRgba.cols() / 2 - 400, mRgba.rows() / 2), Core.FONT_HERSHEY_SCRIPT_COMPLEX, 10.0, new Scalar(255, 255, 0, 255), 5);
        		}
        	}        	
        	break;
        }
        return mRgba;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemCamera) {
        	if (notStart == true) {
        		mViewMode = CAMERA;
        	}
        } else if (item == mItemGallery) {
        	if (notStart == true) {
        		mViewMode = GALLERY;
        	}
        } else if (item == mItemStitch) {
            mViewMode = STITCH;
        } else if (item == mItemClear) {
        	needClear = true;
        } else if (item == mItemCapture) {
        	needCapture = true;
        }
        return true;
    }
    public native void FindFeatures(ArrayList<Long> processingAddr, long errorCodeAddr);
}

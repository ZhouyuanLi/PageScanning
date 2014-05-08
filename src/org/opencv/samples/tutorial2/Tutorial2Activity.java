package org.opencv.samples.tutorial2;

import java.util.Date;
import java.util.ArrayList;

import android.content.Context;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.Sensor;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
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
    
    private static SensorManager   sensorService;
    private Sensor                 sensorAccelerometer;
    private float[]                Acceleration = new float [3];
    private Sensor                 sensorMagneticField;
    private float[]                Magnetic = new float [3];
            
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
        Acceleration[0] = 0;
        Acceleration[1] = 0;
        Acceleration[2] = 0;
        Magnetic[0] = 0;
        Magnetic[1] = 0;
        Magnetic[2] = 0;
        
        sensorService = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        sensorAccelerometer = sensorService.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        if (sensorAccelerometer != null) {
        	sensorService.registerListener(sampleListener, sensorAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            Log.i("Tag", "Registered for Accelerometer Sensor");
        } 
        else {
            Log.e("Tag", "Acceleromter Sensor not found");
            finish();
        }
        sensorMagneticField = sensorService.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        if (sensorMagneticField != null) {
        	sensorService.registerListener(sampleListener, sensorMagneticField, SensorManager.SENSOR_DELAY_NORMAL);
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
        sensorService.registerListener(sampleListener, sensorAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        sensorService.registerListener(sampleListener, sensorMagneticField, SensorManager.SENSOR_DELAY_NORMAL);
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
        	sampleTimer = sampleDate.getTime() / 1000;
        	if (sampleTimer > sampleTimer_old || samples.isEmpty()) {
        		mRgba = inputFrame.rgba();        		
        		samples.add(mRgba.clone());
        		sampleTimer_old = sampleTimer;
        		Log.i(Debug, "acc:" + Acceleration[0] + " " + Acceleration[1] + " " + Acceleration[2]); 
        		Log.i(Debug, "mag:" + Magnetic[0] + " " + Magnetic[1] + " " + Magnetic[2]); 
        		break;
        	}
        	else {
        		mRgba = samples.get(samples.size() - 1);
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

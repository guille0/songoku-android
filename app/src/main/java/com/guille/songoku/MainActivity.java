package com.guille.songoku;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.content.pm.PackageManager;

import android.media.Image;
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import android.graphics.YuvImage;
import android.graphics.ImageFormat;
import java.io.ByteArrayOutputStream;
import android.graphics.Rect;

import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import androidx.constraintlayout.widget.ConstraintLayout;
import android.widget.Toast;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.CvType;

// For NN
import org.tensorflow.lite.Interpreter;
import java.nio.channels.FileChannel;
import android.app.Activity;
import java.nio.MappedByteBuffer;
import java.io.IOException;
import android.content.res.AssetFileDescriptor;
import java.io.FileInputStream;

import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private Boolean drawBorders = false;
    private Boolean manualBorders = false;
    private int currentResolution = 0;
    private double borderExt = 62.5;
    private double borderInt = 50;

    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    ImageView ivBitmap;

    Menu optionsMenu;

    private Classifier mClassifier;

    Size screen;
    Rational aspectRatio;

    ImageAnalysis imageAnalysis;


    // Load OpenCV
    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ivBitmap = findViewById(R.id.ivBitmap);

        // Initialize the neural network
        try {
            mClassifier = new Classifier(this);
        } catch (IOException e) {
            Log.e("tag", "init(): Failed to create Classifier", e);
        }

        Sudoku sudoku = Sudoku.getInstance();
        sudoku.setClassifier(mClassifier);

        aspectRatio = new Rational(ivBitmap.getWidth(), ivBitmap.getHeight());
        screen = new Size(ivBitmap.getWidth(), ivBitmap.getHeight());

        // Check for permissions/request them
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }

        // SeekbarExt changes the variable borderExt
        final SeekBar seekBarExt = findViewById(R.id.seekBarExt);
        seekBarExt.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                borderExt = (double) progress;
            }
        });

        // SeekbarInt changes the variable borderInt
        final SeekBar seekBarInt = findViewById(R.id.seekBarInt);
        seekBarInt.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                borderInt = (double) progress;
            }
        });


    }

    private void startCamera() {

        CameraX.unbindAll();
        imageAnalysis = setImageAnalysis();

        // Bind to lifecycle
        CameraX.bindToLifecycle(this, imageAnalysis);
    }



    private ImageAnalysis setImageAnalysis() {
        // Setup image analysis pipeline that computes average pixel luminance
        HandlerThread analyzerThread = new HandlerThread("OpenCVAnalysis");
        analyzerThread.start();
        ImageAnalysisConfig imageAnalysisConfig;

        if (currentResolution == 0) {
            // Low resolution
            imageAnalysisConfig = new ImageAnalysisConfig.Builder()
                    .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                    .setCallbackHandler(new Handler(analyzerThread.getLooper()))
                    .setImageQueueDepth(1).build();

        } else {
            // High resolution
            imageAnalysisConfig = new ImageAnalysisConfig.Builder()
                    .setTargetAspectRatio(aspectRatio)
                    .setTargetResolution(screen)
                    .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                    .setCallbackHandler(new Handler(analyzerThread.getLooper()))
                    .setImageQueueDepth(1).build();
        }


        ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);


        imageAnalysis.setAnalyzer(
                new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(ImageProxy imageProxy, int rotationDegrees) {
                        //Analyzing live camera feed begins.

                        // Transform Image to Bitmap
                        final Image cameraImage = imageProxy.getImage();
                        final Bitmap bitmap = imageToBitmap(cameraImage, rotationDegrees);

                        if(bitmap==null)
                            return;

                        Mat mat = new Mat();
                        Utils.bitmapToMat(bitmap, mat);
//                        mat.convertTo(mat, CvType.CV_8UC4);

                        // Sends image to ImageProcessing class and returns final image
                        mat = ImageProcessing.preprocess(mat, drawBorders, manualBorders, borderExt, borderInt);

                        Utils.matToBitmap(mat, bitmap);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ivBitmap.setImageBitmap(bitmap);
                            }
                        });

                    }
                });

        return imageAnalysis;

    }


    public Bitmap imageToBitmap(Image image, int rotationDegrees) {

        assert (image.getFormat() == ImageFormat.NV21);

        // NV21 is a plane of 8 bit Y values followed by interleaved  Cb Cr
        ByteBuffer ib = ByteBuffer.allocate(image.getHeight() * image.getWidth() * 2);

        ByteBuffer y = image.getPlanes()[0].getBuffer();
        ByteBuffer cr = image.getPlanes()[1].getBuffer();
        ByteBuffer cb = image.getPlanes()[2].getBuffer();
        ib.put(y);
        ib.put(cb);
        ib.put(cr);

        YuvImage yuvImage = new YuvImage(ib.array(),
                ImageFormat.NV21, image.getWidth(), image.getHeight(), null);


        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0,
                image.getWidth(), image.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        Bitmap bm = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        Bitmap bitmap = bm;


        // Rotation by 90º
        Matrix matrix = new Matrix();
        matrix.postRotate(90);

        bitmap = Bitmap.createBitmap(bm, 0, 0,
                bm.getWidth(), bm.getHeight(), matrix, true);
        return bitmap;
    }


//    PERMISSIONS


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {

        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }


//    BUTTONS AND OPTIONS


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        optionsMenu = menu;
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        switch (item.getItemId()) {

            case R.id.resolution:
                if (item.isChecked() == false) {
                    currentResolution = 1;
                    item.setChecked(true);
                    startCamera();
                    return true;
                } else {
                    currentResolution = 0;
                    item.setChecked(false);
                    startCamera();
                    return true;
                }

            case R.id.drawBorders:
                if (item.isChecked() == false) {
                    drawBorders = true;
                    item.setChecked(true);
                    return true;
                } else {
                    drawBorders = false;
                    item.setChecked(false);
                    return true;
                }

            case R.id.manualBorders:
                ConstraintLayout borderOptions = (ConstraintLayout)findViewById(R.id.borderOptions);
                if (item.isChecked() == false) {
                    manualBorders = true;
                    item.setChecked(true);

                    // Also activate drawBorders so we can see them
                    MenuItem drawBs = optionsMenu.findItem(R.id.drawBorders);
                    drawBorders = true;
                    drawBs.setChecked(true);

                    // Seekbars made visible
                    borderOptions.setVisibility(View.VISIBLE);
                    return true;
                } else {
                    manualBorders = false;
                    item.setChecked(false);
                    borderOptions.setVisibility(View.INVISIBLE);
                    return true;
                }
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onClick(View view) {
    }
}

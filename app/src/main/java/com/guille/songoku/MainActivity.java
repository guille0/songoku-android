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

// For NN
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private boolean drawBorders = false;
    private boolean manualBorders = false;
    private int currentResolution = 0;
    private double borderExt = 62.5;
    private double borderInt = 50;

    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};

    private Menu optionsMenu;
    private ImageView imageView;

    private Classifier classifier;

    private Size screen;
    private Rational aspectRatio;

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
        imageView = findViewById(R.id.ivBitmap);

        // Initialize the neural network
        try {
            classifier = new Classifier(this);
        } catch (IOException e) {
            Log.e("tag", "init(): Failed to create Classifier", e);
        }

        Sudoku sudoku = Sudoku.getInstance();
        sudoku.setClassifier(classifier);

        // Check for permissions/request them
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(MainActivity.this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }

        aspectRatio = new Rational(imageView.getWidth(), imageView.getHeight());
        screen = new Size(imageView.getWidth(), imageView.getHeight());

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
        ImageAnalysis imageAnalysis = setImageAnalysis();
        CameraX.bindToLifecycle(this, imageAnalysis);
    }

    private ImageAnalysis setImageAnalysis() {
        // Get latest frame as an Image, process it and project it onto ImageView
        HandlerThread analyzerThread = new HandlerThread("OpenCVAnalysis");
        analyzerThread.start();
        ImageAnalysisConfig imageAnalysisConfig;

        // Low or high resolution
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
                (imageProxy, rotationDegrees) -> {
                    //Analyzing live camera feed begins.

                    // Transform Image to Bitmap
                    final Image cameraImage = imageProxy.getImage();
                    final Bitmap bitmap = imageToBitmap(cameraImage, rotationDegrees);

                    if (bitmap == null) return;

                    Mat mat = new Mat();
                    Utils.bitmapToMat(bitmap, mat);

                    // Sends image to ImageProcessing class and returns final image
                    mat = ImageProcessing.preprocess(mat, drawBorders, manualBorders, borderExt, borderInt);

                    // Write final image
                    Utils.matToBitmap(mat, bitmap);
                    runOnUiThread(() -> imageView.setImageBitmap(bitmap));

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

        // Rotation by 90º
        Matrix matrix = new Matrix();
        matrix.postRotate(90);

        return Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), matrix, true);
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
                if (!item.isChecked()) {
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
                if (!item.isChecked()) {
                    drawBorders = true;
                    item.setChecked(true);
                    return true;
                } else {
                    drawBorders = false;
                    item.setChecked(false);
                    return true;
                }

            case R.id.manualBorders:
                ConstraintLayout borderOptions = findViewById(R.id.borderOptions);
                if (!item.isChecked()) {
                    manualBorders = true;
                    item.setChecked(true);

                    // Also activates drawBorders by default
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
}

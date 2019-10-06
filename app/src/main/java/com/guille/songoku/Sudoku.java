package com.guille.songoku;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import org.opencv.core.CvException;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

import java.util.List;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.lang.System;
import java.util.HashMap;
import java.util.Map;

// For testing
import android.os.Environment;
import java.io.File;
import org.opencv.imgcodecs.Imgcodecs;

import org.opencv.core.CvType;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import org.opencv.android.Utils;
import android.util.Log;

// SUdoku solver
import de.sfuhrm.sudoku.GameMatrix;
import de.sfuhrm.sudoku.Riddle;
import de.sfuhrm.sudoku.GameMatrixFactory;
import de.sfuhrm.sudoku.Solver;


public class Sudoku {
    private Scalar colorBorder = new Scalar(0, 40, 0, 128);
    private Scalar colorNumber = new Scalar(0, 20, 128, 180);

    // Arrays that contain the sudoku numbers
    private byte[][] sudokuArray = new byte[9][9];
    private byte[][] sudokuSolvedArray = new byte[9][9];

    // Hashmap for already solved sudokus
    private Map<byte[][], byte[][]> alreadySolved = new HashMap<byte[][], byte[][]>();

    private Mat imgEdges = new Mat();

    private Classifier mClassifier;

    // Singleton stuff
    private static final Sudoku ourInstance = new Sudoku();

    public static Sudoku getInstance() {
        return ourInstance;
    }

    private Sudoku() {
    }

    // Main methods
    public Boolean buildFromImageManual(Mat imgSudoku, Boolean drawBorders, double borderInt, double borderExt) {
        // Black and white
        Imgproc.cvtColor(imgSudoku, imgEdges, Imgproc.COLOR_RGB2GRAY);

        // Adaptive threshold
        Imgproc.adaptiveThreshold(imgEdges, imgEdges, 255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 7, 9);

        // Grab images from every square (manual borders)
        // Get the corners for each number
        final double w = imgSudoku.width();
        final double h = imgSudoku.height();
        final double x = w/9;
        final double y = h/9;
        double top, left, bottom, right;
        for (int i=0; i<9; i++) {
            for (int j=0; j<9; j++) {
                if (i%3 == 0) {
                    left = x*i+borderExt;
                } else {
                    left = x*i+borderInt;
                }

                if ((i+1)%3 == 0) {
                    right = x*(i+1)-borderExt;
                } else {
                    right = x*(i+1)-borderInt;
                }

                if (j%3 == 0) {
                    top = y*j+borderExt;
                } else {
                    top = y*j+borderInt;
                }

                if ((j+1)%3 == 0) {
                    bottom = y*(j+1)-borderExt;
                } else {
                    bottom = y*(j+1)-borderInt;
                }

                // We now have the coordinates top left bottom right
                // Extract the images and save them

                try {
                    Rect rectCrop = new Rect(
                            (int) Math.ceil(left),
                            (int) Math.ceil(top),
                            (int) Math.ceil(right - left + 1),
                            (int) Math.ceil(bottom - top + 1));
                    Mat imgSquare = imgEdges.submat(rectCrop);

                    guessNumber(imgSquare, i, j);
                } catch (CvException e) {
                    Log.i("exception", "caught exception, not analyzing sudoku");
                    return false;
                }
            }
        }

        if (drawBorders == true) {
            drawBordersManual(imgSudoku, borderExt, borderInt);
        }

        return true;

    }

    public Boolean buildFromImageAuto(Mat imgSudoku, Boolean drawBorders) {
        Mat imgEdges = new Mat();
        // Black and white
        Imgproc.cvtColor(imgSudoku, imgEdges, Imgproc.COLOR_RGB2GRAY);

        // For numbers
        Imgproc.adaptiveThreshold(imgEdges, imgEdges, 255,
                Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 7, 9);

        // Go diagonally down right the sudoku
        // Find what we guess might be the middle of each number
        // Loop to up, left, down and right
        // Those are the borders for the whole line

        Point center = new Point();
        int[] top = new int[9];
        int[] bottom = new int[9];
        int[] left = new int[9];
        int[] right = new int[9];
        Arrays.fill(top, 0);
        Arrays.fill(bottom, imgEdges.height()-1);
        Arrays.fill(left, 0);
        Arrays.fill(right, imgEdges.width()-1);

        double w = imgEdges.width()/9;
        double h = imgEdges.height()/9;

        double threshold = imgEdges.width() * 0.6;
        int border = 4;

        for (int i=0; i<9; i++) {
            center = new Point( w*i+(w/2), h*i+(h/2));
            for (int y = (int)center.y; y>0; y--) {
                if (Core.countNonZero(imgEdges.row(y)) < threshold) {
                    top[i] = (int) y+border;
                    break;
                }
            }
            for (int y = (int)center.y; y<imgEdges.height(); y++) {
                if (Core.countNonZero(imgEdges.row(y)) < threshold) {
                    bottom[i] = y-border;
                    break;
                }
            }
            for (int x = (int)center.x; x>0; x--) {
                if (Core.countNonZero(imgEdges.col(x)) < threshold) {
                    left[i] = x+border;
                    break;
                }
            }
            for (int x = (int)center.x; x<imgEdges.width(); x++) {
                if (Core.countNonZero(imgEdges.col(x)) < threshold) {
                    right[i] = x-border;
                    break;
                }
            }
            // Drawing lines
            if (drawBorders == true) {
                Imgproc.line(imgSudoku, new Point(0, top[i]), new Point(imgSudoku.width() - 1, top[i]), colorBorder, 1);
                Imgproc.line(imgSudoku, new Point(0, bottom[i]), new Point(imgSudoku.width() - 1, bottom[i]), colorBorder, 1);
                Imgproc.line(imgSudoku, new Point(left[i], 0), new Point(left[i], imgSudoku.height() - 1), colorBorder, 1);
                Imgproc.line(imgSudoku, new Point(right[i], 0), new Point(right[i], imgSudoku.height() - 1), colorBorder, 1);
            }
        }


        for (int i=0; i<9; i++) {
            for (int j=0; j<9; j++) {

                // Try and catch error here
                try {
                    Rect rectCrop = new Rect(
                            (int) Math.ceil(left[i]),
                            (int) Math.ceil(top[j]),
                            (int) Math.ceil(right[i] - left[i] + 1),
                            (int) Math.ceil(bottom[j] - top[j] + 1));
                    Mat imgSquare = imgEdges.submat(rectCrop);

                    guessNumber(imgSquare, i, j);
                } catch (CvException e) {
                    Log.i("exception", "caught exception, not analyzing sudoku");
                    return false;
                }
            }
        }

        return true;

    }

    public void solve() {
        // Save solutions in a dictionary
        if (alreadySolved.containsValue(sudokuArray)) {
            Log.i("sudoku.solve()", "loading solved");
            sudokuSolvedArray = alreadySolved.get(sudokuArray);

        } else {
            // Maximum time before giving up on a sudoku (ms)
            long maxTime = 200;

            // Solving the sudoku
            Riddle riddle = new GameMatrixFactory().newRiddle();
            riddle.setAll(sudokuArray);

            Solver solver = new Solver(riddle);
            solver.setLimit(1);
            List<GameMatrix> solutions = solver.solve(maxTime);

            if (!solutions.isEmpty()) {
//                Log.i("sudoku.solve()", "Saving sudoku solution");
                sudokuSolvedArray = solutions.get(0).getArray();
                alreadySolved.put(sudokuArray, sudokuSolvedArray);
            }
        }
    }

    public void guessNumber(Mat imgSquare, int i, int j) {
        // Check how many non-blank pixels there are and assume if there is a number
        if (Core.countNonZero(imgSquare) < imgSquare.total()*0.95) {
            // Resize to 28x28
            Mat resizeImage = new Mat(28, 28, CvType.CV_32F);
            int interpolation = Imgproc.INTER_CUBIC;
            Imgproc.resize(imgSquare, resizeImage, resizeImage.size(), 0, 0, interpolation);

            // Saving it to a file (for testing) (delete)
//            if (i == 5) {
//                if (j == 1 || j==2 || j==4 || j==5 || j==7 || j==8){
//                    count++;
//                    File path = new File(Environment.getExternalStorageDirectory() + "/Images/sixes/" + Integer.toString(i) + "/");
//                    path.mkdirs();
//                    File file = new File(path, "square" + Integer.toString(count) + ".png");
//                    Imgcodecs.imwrite(file.toString(), resizeImage);
//                }
//            }

            // Mat to Bitmap
            Bitmap bmp = null;
            bmp = Bitmap.createBitmap(resizeImage.cols(), resizeImage.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(resizeImage, bmp);

            // Guess the number
            Result result = mClassifier.classify(bmp);

            sudokuArray[i][j] = (byte) result.getNumber();
        } else {
            sudokuArray[i][j] = 0;
        }

    }

    // Setters
    public void setClassifier(Classifier classer) {
        mClassifier = classer;
    }

    public void setOptionsColorNumber(Scalar color) {
        colorNumber = color;
    }

    public void setOptionsColorBorder(Scalar color) {
        colorBorder = color;
    }

    // Drawing
    public void drawNumbers(Mat imgSudoku) {
        final double w = imgSudoku.width();
        final double h = imgSudoku.height();
        final double x = w/9;
        final double y = h/9;

        final int fontface = 1;
        final double fontsize = x*0.05;
        final int thickness = 2;
        int [] baseline = new int[1];

        final Size digitSize = Imgproc.getTextSize("0",
                fontface,
                fontsize,
                thickness,
                baseline);


        for (int i=0; i<9; i++) {
            for (int j=0; j<9; j++) {
                // draw the detected numbers
//                if (sudokuArray[i][j] != 0) {
//                    Imgproc.putText(imgSudoku, Byte.toString(sudokuArray[i][j]),
//                            new Point(x * i + x / 2, y * j + y / 2), 1, 1.0, blue);
//                }

                // Draw the solution
                if (sudokuArray[i][j] == 0) {

                    Imgproc.putText(imgSudoku, Byte.toString(sudokuSolvedArray[i][j]),
                            new Point(x * i + (x/2) - digitSize.width/2, y * j + (y/2) + (digitSize.height + baseline[0])/2),
                            fontface,
                            fontsize,
                            colorNumber,
                            thickness);
                }
            }
        }
    }

    public void drawBordersManual(Mat imgSudoku, double borderExt, double borderInt) {
        final double w = imgSudoku.width();
        final double h = imgSudoku.height();
        final double x = w/9;
        final double y = h/9;
        double x1, x2, y1, y2;
        double max;
        for (int i=0; i<9; i++) {
            for (int b=0; b<Math.max(borderInt, borderExt); b++) {
                if (i%3 == 0) {
                    max = borderExt;
                } else {
                    max = borderInt;
                }

                if (b < max) {
                    // left
                    x1 = x * i + b;
                    x2 = x * i + b;
                    y1 = 0;
                    y2 = h;
                    Imgproc.line(imgSudoku, new Point(x1, y1), new Point(x2, y2), colorBorder, 1);
                    // top
                    x1 = 0;
                    x2 = w;
                    y1 = y * i + b;
                    y2 = y * i + b;
                    Imgproc.line(imgSudoku, new Point(x1, y1), new Point(x2, y2), colorBorder, 1);
                }

                if ((i+1)%3 == 0) {
                    max = borderExt;
                } else {
                    max = borderInt;
                }

                if (b<max) {
                    // right
                    x1 = x * (i + 1) - b;
                    x2 = x * (i + 1) - b;
                    y1 = 0;
                    y2 = h;
                    Imgproc.line(imgSudoku, new Point(x1, y1), new Point(x2, y2), colorBorder, 1);
                    // bottom
                    x1 = 0;
                    x2 = w;
                    y1 = y * (i + 1) - b;
                    y2 = y * (i + 1) - b;
                    Imgproc.line(imgSudoku, new Point(x1, y1), new Point(x2, y2), colorBorder, 1);
                }
            }
        }
    }


}

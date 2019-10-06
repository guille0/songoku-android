package com.guille.songoku;

import android.os.Environment;
import java.io.File;

import java.util.List;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.CvType;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

import org.opencv.imgcodecs.Imgcodecs;

import android.util.Log;


public class ImageProcessing {
    public static Mat preprocess(Mat imgOriginal, Boolean drawBorders, Boolean manualBorders,
                                 double borderExtSize, double borderIntSize) {
        double borderInt = 4;   // default value
        double borderExt = 4;   // default value

        // TODO: Inside findSudoku, make sure it has at least 9 inner contours
        // To be more sure it's a real sudoku
        MatOfPoint contourSudoku = findSudoku(imgOriginal);

        if (!contourSudoku.empty()) {
            // If we have a contour, extract it into a new, warped image
            TransformedImage transImg = extractSudoku(imgOriginal, contourSudoku);

            if (transImg.isSudoku == true) {
                Sudoku sudoku = Sudoku.getInstance();

                Boolean success = false;

                if (manualBorders == true) {
                    // NOTE: Load these 2 from options
                    // we get them from 0 to 100
                    // just divide by 50
//                    borderIntSize = 1;       // From 0.0 to 2.0
//                    borderExtSize = 1.25;    // From 0.0 to 2.0
                    borderIntSize = borderIntSize/50;
                    borderExtSize = borderExtSize/50;

                    borderIntSize = (2.0 - borderIntSize)*100;
                    borderExtSize = (2.0 - borderExtSize)*100;

                    borderInt = Math.ceil(transImg.imgSudoku.height() / borderIntSize);
                    borderExt = Math.ceil(transImg.imgSudoku.height() / borderExtSize);
                    success = sudoku.buildFromImageManual(transImg.imgSudoku, drawBorders,
                            borderInt, borderExt);
                } else {
                    success = sudoku.buildFromImageAuto(transImg.imgSudoku, drawBorders);
                }

                // If we could not read the image properly
                if (success == false) {
                    return imgOriginal;
                }

                sudoku.solve();
                sudoku.drawNumbers(transImg.imgSudoku);

                // And paste it back into the imgOriginal
                Mat invPerspective = transImg.perspectiveMatrix.inv();

                // Create a blank image of the same size as the original
                Mat imgFloatingSudoku = new Mat(imgOriginal.width(), imgOriginal.height(),
                        imgOriginal.type(), new Scalar(0,0,0,0));

                // Paste the edited sudoku into the image
                Imgproc.warpPerspective(transImg.imgSudoku, imgFloatingSudoku, invPerspective,
                        new Size (imgOriginal.width(), imgOriginal.height()));

                // Replace the blank parts with the original image
                Mat mask = new Mat();
                Core.inRange(imgFloatingSudoku, new Scalar(0,0,0,0), new Scalar(10,10,10,0), mask);
                imgOriginal.copyTo(imgFloatingSudoku, mask);

                return imgFloatingSudoku;
            }



        }
//        Log.i("hey", "we found no sudoku this frame");

        return imgOriginal;

    }

    public static MatOfPoint findSudoku(Mat imgOriginal) {
        // Black and white
        Mat imgEdges = new Mat();
        Imgproc.cvtColor(imgOriginal, imgEdges, Imgproc.COLOR_RGB2GRAY);

        // Blur
        Imgproc.GaussianBlur(imgEdges, imgEdges, new Size(7, 7), 0);

        // Open
        Mat kernel = Imgproc.getStructuringElement(0, new Size(3, 3));
        Imgproc.morphologyEx(imgEdges, imgEdges, Imgproc.MORPH_OPEN, kernel);

        // Adaptive threshold
        Imgproc.adaptiveThreshold(imgEdges, imgEdges, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 19, 2);

        // Get contours:
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(imgEdges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Sort contours by cv2.contourArea to get the biggest one
        Collections.sort(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                if (Imgproc.contourArea(o1) > Imgproc.contourArea(o2)) {
                    return -1;
                }
                if (Imgproc.contourArea(o2) > Imgproc.contourArea(o1)) {
                    return 1;
                } else {
                    return 0;
                }
            }
        } );

        // TODO: loop through them all, biggest to smallest, until we find one with 9 or more inner contours
        // The biggest contour with 9+ subcontours gets returned
        // If we don't find one, return new MatOfPoint();
//
//        List<MatOfPoint> contours = new ArrayList<>();
//        Mat hierarchy = new Mat();
//        Imgproc.findContours(imgEdges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
//
//
//        if (contours.size() > 0) {
//            List<MatOfPoint> contoursSquares = new ArrayList<>();
//
//            double currentContour = 0;
//            while (currentContour != -1) {
//                contoursSquares.add(contours.get((int) currentContour));
//                currentContour = hierarchy.get(0, (int) currentContour)[0];
//            }
//            if (!contoursSquares.isEmpty()) {
//                Imgproc.drawContours(imgSudoku, contoursSquares, -1, new Scalar(255, 0, 0, 0), 1);
//            }

        // If we don't find any contours, return empty Mat


        // If we find a contour of a decent size, return it
        if (contours.size() > 0) {
//            int minSize = 100;
//            if (contours.get(0).total() > minSize) {
                return contours.get(0);
//            }
        }
        return new MatOfPoint();
    }

    public static TransformedImage extractSudoku(Mat imgOriginal, MatOfPoint contourSudoku) {
        // Approximate a rectangle from the contour
        MatOfPoint2f contourSudoku2f = new MatOfPoint2f(contourSudoku.toArray());
        MatOfPoint2f contourApprox2f = new MatOfPoint2f();
        double epsilon = 0.01*Imgproc.arcLength(contourSudoku2f, true);
        Imgproc.approxPolyDP(contourSudoku2f, contourApprox2f, epsilon, true);

        // If it has 4 corners it might just be a sudoku
        if (contourApprox2f.total() == 4) {
            // Extract it
            TransformedImage transImg = cropFromPoints(imgOriginal, contourApprox2f);
            return transImg;
        }

        return new TransformedImage(false);
    }

    public static TransformedImage cropFromPoints(Mat imgOriginal, MatOfPoint2f corners) {
        // Get all the 4 corners (there's only 4, always)
        List<Point> points = corners.toList();
        points = orderCorners(points);

        // Get width and height
        Size size = new Size();
        size = getDestinationShape(points);

        double dimension = Math.max(size.width, size.height);

        List<Point> dstListPoints = new ArrayList<Point>();
        dstListPoints.add(new Point(0,0));
        dstListPoints.add(new Point(0,dimension));
        dstListPoints.add(new Point(dimension,0));
        dstListPoints.add(new Point(dimension,dimension));

        // Get a MatOfPoint with 4 corners in the image
        // Make a square MatOfPoint with 4 corners (topleft, bottomleft, topright, bottomright)
        Mat srcCorners = new Mat(4,1,CvType.CV_32FC2);
        srcCorners.put(0,0,
                points.get(0).x, points.get(0).y,
                points.get(1).x, points.get(1).y,
                points.get(2).x, points.get(2).y,
                points.get(3).x, points.get(3).y);
        Mat dstCorners = new Mat(4,1,CvType.CV_32FC2);
        dstCorners.put(0,0,
                dstListPoints.get(0).x, dstListPoints.get(0).y,
                dstListPoints.get(1).x, dstListPoints.get(1).y,
                dstListPoints.get(2).x, dstListPoints.get(2).y,
                dstListPoints.get(3).x, dstListPoints.get(3).y);

        // Perspective transform the original image into the new one
        Mat perspective = Imgproc.getPerspectiveTransform(srcCorners, dstCorners);

        Mat transformedImage = imgOriginal.clone();
        Imgproc.warpPerspective(imgOriginal, transformedImage, perspective, new Size(dimension,dimension));

        // We now have our transformedImage
        return new TransformedImage(true, perspective, transformedImage, corners);
    }

    // HELPER FUNCTIONS

    public static List<Point> orderCorners(List<Point> points) {
        // returns in order topleft, bottomleft, topright, bottomright
        // First order them by X
        Collections.sort(points, new Comparator<Point>() {
            @Override
            public int compare(Point p1, Point p2) {
                if (p1.x > p2.x) {
                    return 1;
                }
                if (p1.x < p2.x) {
                    return -1;
                } else {
                    return 0;
                }
            }
        } );

        // Now get the first 2, order them by Y, first is topleft
        List<Point> left = new ArrayList<>();
        Collections.addAll(left, points.get(0), points.get(1));
        left = sortByY(left);

        List<Point> right = new ArrayList<>();
        Collections.addAll(right, points.get(2), points.get(3));
        right = sortByY(right);


        List<Point> corners = new ArrayList<>();
        corners.addAll(left);
        corners.addAll(right);

        return corners;

    }


    public static List<Point> sortByY(List<Point> list) {
        Collections.sort(list, new Comparator<Point>() {
            @Override
            public int compare(Point p1, Point p2) {
                if (p1.y > p2.y) {
                    return 1;
                }
                if (p1.y < p2.y) {
                    return -1;
                } else {
                    return 0;
                }
            }
        } );

        return list;
    }


    public static Size getDestinationShape(List<Point> corners) {
        // Takes an ordered list of points and returns a Size(width, height)
        double width = Math.max(euclideanDistance(corners.get(0), corners.get(2)),
            euclideanDistance(corners.get(0), corners.get(2)));
        double height = Math.max(euclideanDistance(corners.get(0), corners.get(1)),
                euclideanDistance(corners.get(2), corners.get(3)));

        return new Size(width, height);
    }


    public static double euclideanDistance(Point a, Point b){
        double distance = 0.0;
        try {
            if(a != null && b != null) {
                double xDiff = a.x - b.x;
                double yDiff = a.y - b.y;
                distance = Math.sqrt(Math.pow(xDiff,2) + Math.pow(yDiff, 2));
            }
        } catch(Exception e) {
            System.err.println("Something went wrong in euclideanDistance function");
        }
        return distance;
    }

}

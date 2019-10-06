package com.guille.songoku;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;

public class TransformedImage {
    Boolean isSudoku = false;
    Mat perspectiveMatrix;
    Mat imgSudoku;
    MatOfPoint2f contourApprox2f;

    public TransformedImage(Boolean isS, Mat perspM, Mat imgS, MatOfPoint2f contourApprox) {
        isSudoku = isS;
        perspectiveMatrix = perspM;
        imgSudoku = imgS;
        contourApprox2f = contourApprox;
    }
    public TransformedImage(Boolean isS) {
        isSudoku = isS;
    }
}
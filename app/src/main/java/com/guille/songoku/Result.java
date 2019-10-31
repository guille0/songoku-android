package com.guille.songoku;

public class Result {
    // The result of a guess with our classifier

    private final int number;
    private final float probability;
    private final long timeCost;

    public Result(float[] probs, long time) {
        number = argmax(probs);
        probability = probs[number];
        timeCost = time;
    }

    public int getNumber() {
        return number;
    }

    public float getProbability() {
        return probability;
    }

    public long getTimeCost() {
        return timeCost;
    }

    private static int argmax(float[] probs) {
        int maxIdx = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

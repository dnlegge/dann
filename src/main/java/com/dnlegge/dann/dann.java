package com.dnlegge.dann;

public interface dann {
    void setupNet(int[] numNodesInEachLayer);

    void randomiseWeights();

    double[][][] forward(double[][] inputArray);

    void backward(double reward, double[][] previousNodeValues);

    double sigmoid(double input);

    void setLearning(boolean newLearning);

    void setBatch(boolean newBatch);

    void setSeed(long newSeed);

    void setLearningRate(double newLearningRate);

    void setMomentum(double newMomentum);

    void setDiscountRate(double newDiscountRate);

    void setReward(double newReward);

    void writeWeightsOut(String suffix);
}

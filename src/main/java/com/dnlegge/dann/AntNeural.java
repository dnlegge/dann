package com.dnlegge.dann;

import com.dnlegge.dann.impl.dannImpl;

public class
        AntNeural {
    /*
    David's AntNeural Class:
        Public Methods:
            SetupNet - Consult Net

    */
    private double rewardArray[][] = new double[1][1];
    private dannImpl nnet;
    private int[] tally = new int[3];

    // Constructor calls
    public AntNeural() {
        // this sets the default NN-topolgy - overridden by weights.in
        int layerArray[] = {7, 3, 1};
        nnet = new dannImpl(layerArray);
        nnet.setLearning(true);
        nnet.setLearningRate(0.1);
        nnet.setMomentum(0.0);
        nnet.setDiscountRate(0.3);

    }

    public double[][][] ConsultNet(double[][] inputArray, double previousNodeValues[][]) {
        double outputArray[][][];//= new double[size][1] ;

        // The Reward Function
        //double reward =  inputArray[0][0] * inputArray[0][1] * ( inputArray[0][1] ) *( 1-inputArray[0][2] )* inputArray[0][3] ;
        //double reward = inputArray[0][1] ;// * inputArray[0][0] ;//* inputArray[0][1] * inputArray[0][1] ;//* inputArray[0][3]* ( 1 - inputArray[0][2] ) ;
        // reward = ( 3 / ( 1 + Math.exp( - reward ) ) ) - 1.5 ;

        System.out.println("rewardbase = " + inputArray[0][0]);

        double reward = 0;
        if (inputArray[0][0] > 0.85) reward = 0.4;
        if (inputArray[0][0] > 0.90) reward = 0.6;

        System.out.println("rewardpattern = " + reward);

        //System.out.println( "Calling Forward Method " ) ;
        outputArray = nnet.forward(inputArray);

        //System.out.println( "Calling Backward Method " ) ;
        nnet.backward(reward, previousNodeValues);

        return (outputArray);
    }

    public String toString() {
        return nnet.toString();

    }

    public void writeWeightsOut(String suffix) {
        nnet.writeWeightsOut(suffix);

    }
}

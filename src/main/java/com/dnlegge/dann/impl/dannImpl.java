package com.dnlegge.dann.impl;

import com.dnlegge.dann.dann;

import java.io.*;
import java.util.Random;

public class dannImpl implements dann {
    /**
     * David's Artificial Neural Network 
     */

// Class Variables:
    private int numLayers;
    private int numNodes[];
    private int maxNumNodes;
    private int numPatterns = 3;
    private int patternNum;
    private boolean learning = true;
    private boolean batch = true;
    private boolean firstTime = true;
    private double learningRate = 0.1;
    private double momentum = 0.1;
    private double discountRate = 0.1;
    private double nodeValues[][];
    private double weights[][][];
    private double deltaWeight[][];
    private double oldDeltaWeight[][];
    private double outputArray[][][];
    private double reward;
    private double GlobalError = 0;
    private double MaxQ = 0;

    private String fileName = "weights";

    private Random generator;
    private long SEED = 299;


// Constructor:

    public dannImpl(int[] numNodesInEachLayer) {
        setupNet(numNodesInEachLayer);
    }

    public void setupNet(int[] numNodesInEachLayer) {
        this.numLayers = numNodesInEachLayer.length;
        numNodes = numNodesInEachLayer;
        if (!readInWeightsFromFile()) {
            randomiseWeights();
        }

        maxNumNodes = 0;
        System.out.println("numLayers = " + numLayers);
        for (int i = 0; i < numLayers; i++) {
            System.out.println("numNodes[" + i + "] = " + numNodes[i]);
            if (numNodes[i] > maxNumNodes) maxNumNodes = numNodes[i];
        }

    }

    private boolean readInWeightsFromFile() {
        // if weights.in exists use it, otherwise randomise
        try {
            FileReader inputFile = new FileReader(fileName + ".in");
            StreamTokenizer tokenizer = new StreamTokenizer(inputFile);

            tokenizer.nextToken();
            numLayers = (int) tokenizer.nval + 1;
            weights = new double[numLayers - 1][][];
            deltaWeight = new double[numLayers][];

            System.err.println("Number of Layers = " + numLayers);
            for (int i = 0; i < (numLayers - 1); i++) {
                tokenizer.nextToken();
                numNodes[i] = (int) tokenizer.nval;
                weights[i] = new double[numNodes[i]][];
                System.err.println("J Length = " + tokenizer.nval);
                tokenizer.nextToken();
                numNodes[i + 1] = (int) tokenizer.nval;
                deltaWeight[i + 1] = new double[numNodes[i + 1]];
                for (int j = 0; j < numNodes[i]; j++) {
                    weights[i][j] = new double[numNodes[i + 1]];
                    for (int k = 0; k < numNodes[i + 1]; k++) {
                        tokenizer.nextToken();
                        weights[i][j][k] = tokenizer.nval;
                        System.err.println("weights[" + i + "][" + j + "][" + k + "] = " + weights[i][j][k]);
                    }
                    System.err.println();
                }
                System.err.println();
            }


            inputFile.close();
            //AntManager.RANDOM_WEIGHTS = false ;

        } catch (Exception e) {
            System.err.println("No weights.in file found - generating randomly");
            return false;
        }
        return true;
    }

    // Generates Weights in Region of -0.1 to 0.1

    public void randomiseWeights() {
        generator = new Random(SEED);
        weights = new double[numLayers - 1][][];
        deltaWeight = new double[numLayers][];
        for (int i = 0; i < (numLayers - 1); i++) {
            weights[i] = new double[numNodes[i]][];
            deltaWeight[i + 1] = new double[numNodes[i + 1]];
            for (int j = 0; j < numNodes[i]; j++) {
                weights[i][j] = new double[numNodes[i + 1]];
                for (int k = 0; k < numNodes[i + 1]; k++) {
                    weights[i][j][k] = (generator.nextDouble() * 0.2) - 0.1;
                }
            }
        }
        System.out.println(toString());
        writeWeightsOut("New");
    }

    public double[][][] forward(double[][] inputArray) {
        //System.out.println("Called Forward Method") ;
        int action = -1;
        // Check for right Num of Inputs
        if (inputArray[0].length != numNodes[0]) {
            System.out.println("Number of Inputs does not equal Number of Input Nodes!");
            System.out.println("Inputs = " + inputArray.length + ", Input Nodes = " + numNodes[0] + numLayers + numNodes[1] + numNodes[2]);
            System.exit(0);
        }
        int numPatterns = inputArray.length;
        //System.out.println("Number of patterns = " + numPatterns ) ;

        outputArray = new double[numPatterns][numLayers][];
        nodeValues = new double[numLayers][];

        for (int eachPattern = 0; eachPattern < numPatterns; eachPattern++) {
            // Set inputs as first layer and take sigmoid function
            nodeValues[0] = new double[numNodes[0]];
            //System.out.println("Calculating Layer 0 " ) ;
            for (int eachNodeinLayerZero = 0; eachNodeinLayerZero < numNodes[0]; eachNodeinLayerZero++) {
                //Dont do sigmoid	nodeValues[0][eachNodeinLayerZero] = sigmoid( inputArray[eachPattern][eachNodeinLayerZero] ) ;
                nodeValues[0][eachNodeinLayerZero] = inputArray[eachPattern][eachNodeinLayerZero];
                //		System.out.println("Output = " + nodeValues[0][eachNodeinLayerZero] ) ;
            }
            if (nodeValues[0][4] == 1) action = 0;
            if (nodeValues[0][5] == 1) action = 1;
            if (nodeValues[0][6] == 1) action = 2;

            // Calculate Nodevalues:
            //	First set Nodevalues to zero
            //	Then go through multiplying each node in layer A by appropriate weight
            //	and sum to give nodeValue on layer B
            //	Finally, put through Sigmoid Function

            for (int i = 0; i < (numLayers - 1); i++) {
                //	System.out.println("Calculating Layer " +(i+1) ) ;
                nodeValues[i + 1] = new double[numNodes[i + 1]];
                for (int j = 0; j < numNodes[i]; j++) {
                    for (int k = 0; k < numNodes[i + 1]; k++) {
                        nodeValues[i + 1][k] += nodeValues[i][j] * weights[i][j][k];
                    }
                }
                for (int k = 0; k < numNodes[i + 1]; k++) {
                    nodeValues[i + 1][k] = sigmoid(nodeValues[i + 1][k]);
                }
            }
            // Return Final Layer as output ( has been Sigmoided )
            System.out.println("q" + action + " = " + nodeValues[numLayers - 1][0]);

            if (nodeValues[numLayers - 1][0] > MaxQ) {
                MaxQ = nodeValues[numLayers - 1][0];
                patternNum = eachPattern;
            }

            //outputArray[eachPattern] = nodeValues ;
            for (int i = 0; i < numLayers; i++) {
                outputArray[eachPattern][i] = (double[]) nodeValues[i].clone();
//				for ( int j = 0 ; j < nodeValues[i].length ; j++ ) {
//					outputArray[eachPattern][i][j] = nodeValues[i][j] ;
                //outputArray[eachPattern][i][j] = (double) nodeValues[i][j].clone() ;
//				 }
            }


        }

//	//	for ( int eachPattern = 0 ; eachPattern < numPatterns ; eachPattern++ ) {
        //		System.out.println("Qs" + outputArray[eachPattern][ numLayers - 1 ][0] ) ;
        //	 }

        System.out.println("Max Q = " + MaxQ);

        return outputArray;
    }

    // This takes the GlobalError term and implements Backpropagation weight changes

    public void backward(double reward, double[][] previousNodeValues) {

        double previousQvalue = previousNodeValues[numLayers - 1][0];
        //System.out.println("previousQvalue = " + previousQvalue ) ;

        if (previousQvalue != 0.0) {

            //	Could do this, but values already sigmoided!
            //	previousNodeValues = ( forward( new double[][] { previousNodeValues[0] } ) )[0] ;

            double GlobalError = (reward + (discountRate * MaxQ) - previousQvalue);
            GlobalError = GlobalError * GlobalError / 2;

            System.out.println("Applying Training - GlobalError = " + GlobalError);

            // First Store Previous Delta Weights for use in momentum term
            //oldDeltaWeight = deltaWeight ;

            double[][] oldDeltaWeight = new double[numLayers][];
            for (int i = 0; i < numLayers; i++) {
                if (deltaWeight[i] != null) oldDeltaWeight[i] = (double[]) deltaWeight[i].clone();
            }

            // Calculate Error Vector for Output Layer
            // dError = (d-o)(1-o^2)/2
            for (int eachNeuron = 0; eachNeuron < numNodes[numLayers - 1]; eachNeuron++) {
                //System.out.println(numLayers + "<-l n-> " + eachNeuron  ) ;

                deltaWeight[numLayers - 1][eachNeuron]
                        = 0.5 *
                        (reward + (discountRate * MaxQ) - previousQvalue)
                        * (1 - previousQvalue * previousQvalue);
            }

            // Calculate Error Vector for Hidden Layers
            // dW = (1-h^2)/2* E[dW(pk)W(kj)] - In this case only single output neuron
            for (int eachlayer = (numLayers - 2); eachlayer > 0; eachlayer--) {
                //System.out.println("Calculating hidden layer weight changes " + eachlayer ) ;
                for (int eachNeuron = 0; eachNeuron < numNodes[eachlayer]; eachNeuron++) {
                    for (int eachOutput = 0; eachOutput < numNodes[eachlayer + 1]; eachOutput++) {
                        deltaWeight[eachlayer][eachNeuron]
                                = 0.5 * (1 - (previousNodeValues[eachlayer][eachNeuron] * previousNodeValues[eachlayer + 1][eachOutput]))
                                * (deltaWeight[eachlayer + 1][eachOutput] * weights[eachlayer][eachNeuron][eachOutput]);
                    }
                }
            }

            //System.out.println("Calculated Changes, now do Weights") ;

            // First do output layer as this has different eqn
            for (int eachnode = 0; eachnode < numNodes[numLayers - 2]; eachnode++) {
                weights[numLayers - 2][eachnode][0]
                        += ((learningRate * (deltaWeight[numLayers - 1][0] * previousNodeValues[numLayers - 2][eachnode]))
                        + momentum * oldDeltaWeight[numLayers - 1][0]);
            }

            //  Do all Layers
            for (int eachlayer = (numLayers - 3); eachlayer >= 0; eachlayer--) {
                //	System.out.println("applying hidden layer weight changes " + eachlayer ) ;
                for (int eachNeuron = 0; eachNeuron < numNodes[eachlayer]; eachNeuron++) {
                    for (int eachWeight = 0; eachWeight < numNodes[eachlayer + 1]; eachWeight++) {
                        weights[eachlayer][eachNeuron][eachWeight]
                                += ((learningRate * (deltaWeight[eachlayer + 1][eachWeight] * previousNodeValues[eachlayer][eachNeuron]))
                                + momentum * oldDeltaWeight[eachlayer + 1][eachWeight]);
                    }
                }
            }
        }
        MaxQ = 0;

    }

    // Smoothly limits output between 0 and 1

    public double sigmoid(double input) {
        double output = 1 / (1 + Math.exp(-input));
        return output;
    }

    public void setLearning(boolean newLearning) {
        learning = newLearning;
    }

    public void setBatch(boolean newBatch) {
        batch = newBatch;
    }

    public void setSeed(long newSeed) {
        SEED = newSeed;
    }

    public void setLearningRate(double newLearningRate) {
        learningRate = newLearningRate;
    }

    public void setMomentum(double newMomentum) {
        momentum = newMomentum;
    }

    public void setDiscountRate(double newDiscountRate) {
        discountRate = newDiscountRate;
    }

    public void setReward(double newReward) {
        reward = newReward;
    }

    public String toString() {
        StringBuffer returnStringBuffer = new StringBuffer("");
        StringBuffer fileStringBuffer = new StringBuffer("");
        for (int i = 0; i < (numLayers - 1); i++) {
            for (int j = 0; j < numNodes[i]; j++) {
                for (int k = 0; k < numNodes[i + 1]; k++) {
                    //if ( weights[i][j][k] != null )
                    returnStringBuffer.append("\nW(" + i + "," + j + "," + k + ") = " + weights[i][j][k]);
                    fileStringBuffer.append("\n" + weights[i][j][k]);
                }
            }
            returnStringBuffer = returnStringBuffer.append("\n");
        }

/*		try {
			FileWriter outputFile = new FileWriter ( fileName + ".out" ) ;
			outputFile.write(fileStringBuffer.toString( )) ;
			outputFile.close() ; 
		} catch ( FileNotFoundException e ) { }
		catch ( IOException e2 ) { }
*/
        return (returnStringBuffer.toString());
    }

    public void writeWeightsOut(String suffix) {

        try {
            FileWriter outputFile = new FileWriter(fileName + suffix + ".out");

            // write out the file
            outputFile.write(weights.length + "\n\n");
//			for ( int x = 0; x < weights.length; x++ ) {
//				outputFile.write( weights[x].length + "\n");
//			 }

            for (int x = 0; x < weights.length; x++) {
                outputFile.write(weights[x].length + " " + weights[x][0].length + "\n");

                for (int y = 0; y < weights[x].length; y++) {
//				 outputFile.write( weights[x][y].length + " ");
                    for (int z = 0; z < weights[x][y].length; z++) {
                        outputFile.write(Double.toString(weights[x][y][z]) + " ");
                    }
                    outputFile.write("\n");
                }
                outputFile.write("\n");
            }
            outputFile.close();
        } catch (IOException e2) {
            System.err.println("Problem with weights.out ");
            System.exit(0);
        }

        return;//( toString( ) ) ;
    }

    /*public void writeNewWeights( ) {

         try {
             FileWriter outputFile = new FileWriter ( fileName + ".new" ) ;

         // write out the file
             outputFile.write( weights.length + "\n\n");
 //			for ( int x = 0; x < weights.length; x++ ) {
 //				outputFile.write( weights[x].length + "\n");
 //			 }

             for ( int x = 0; x < weights.length; x++ )
              {
                  outputFile.write( weights[x].length + " " + weights[x][0].length + "\n");

                  for ( int y = 0; y < weights[x].length; y++ )
                  {
 //				 outputFile.write( weights[x][y].length + " ");
                  for ( int z = 0; z < weights[x][y].length; z++ )
                  {
                      outputFile.write( Double.toString( weights[x][y][z] ) + " ");
                  }
                  outputFile.write("\n");
                  }
                  outputFile.write("\n");
          }
          outputFile.close();
          } catch (IOException e2 ) { System.err.println("Problem with weights.out " ) ;
              System.exit(0) ; }

         return ;//( toString( ) ) ;
      }*/

} // END OF CLASS



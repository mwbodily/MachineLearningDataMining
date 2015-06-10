package automatedneuralnetwork;


import java.util.ArrayList;
import java.util.List;
//import java.util.*;
import java.util.Random;
import weka.core.Instance;
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Mackenzie
 */
public class Node {
    List<Double> inputList;
    List<Double> weightList;
    int numberOfInputs;
    double threshold;
    double biasInput;  // THIS STILL NEEDS TO BE IMPLEMENTED and GIVEN A WEIGHT!!!
    double biasWeight;
    double learningConstant;
    int nodeIndex;
    double output;
    double target;
    double error;
    
    /*************************************************************
    * Each node, as it is created, needs to know the number of
    * inputs that it will have and to assign weights. This
    * constructor takes care of all of that using the 
    * features parameter.
    * 
    * @param features - this is the number of features that the 
    *                   instance has. Because each is an input
    *                   and each input must have a weight, we
    *                   need this number so that we can generate
    *                   the same number of weights as there are
    *                   features.
    * @param classNum - the numeric value of the class to which
    *                   the node belongs. If the instance is of
    *                   class classNum, the node should fire. 
    *                   otherwise, the node should not fire.
    */
    Node(int numInputs, double pLearningConstant, int pIndex, double targ)
    {
        target = targ;
        inputList = new ArrayList();
        weightList = new ArrayList();
        generateWeights(numInputs);
        threshold = .5;
        biasInput = -1;
        learningConstant = pLearningConstant;
        nodeIndex = pIndex;
        biasWeight = generateBiasWeight();
    }
    
    public void calculateOLError()
    {
   //     System.out.println("\t(OL) The output for this node is: " + output);
   //     System.out.println("\t(OL) The target for this node is: " + target);
        error = (output - target) * output * (1 - output);
   //     System.out.println("\t(OL) The resulting error for this target is: " + error);
   //     System.out.println();
    }
    
    public void calculateHLError(Layer nextLayer)
    {
        error = output*(1 - output);
    //    System.out.println("The current error is: " + error);
   //     System.out.println("The output for this node is: " + output);
        double sum = 0.0;
        
        for(int i = 0; i < nextLayer.numberOfNodes; i++)
        {
            double tempError = nextLayer.Nodes.get(i).error;
            double tempWeight = nextLayer.Nodes.get(i).weightList.get(nodeIndex);
  //          System.out.println("\tError of next layer: " + tempError);
  //          System.out.println("\tWeight of node to that layer: " + tempWeight);
  //          System.out.println("\tAdding " + (tempError * tempWeight) + " to " + sum);
  //          System.out.println("\t\tFor the HL node " + nodeIndex + " I will be multiplying: " +
   //                 tempError + " and " + tempWeight + " to get: " + (tempError*tempWeight));
            sum += (tempError * tempWeight);
        }
        
   //     System.out.println("The error is " + error + "*" + sum);
        error *= sum;
   //     System.out.println("\tThe resulting error is: " + error);
   //     System.out.println();
    }
    
    public double generateBiasWeight()
    {
        Random rand = new Random();
        double max = .3;
        double min = -.3;
        
        return .1;
        //return (min + (max - min) * rand.nextDouble());
    }

    public void generateWeights(int features)
    {
 //       System.out.println("this is a test...");
        //weightList.add(.2);
        //weightList.add(-.1);
        //weightList.add(.05);
        //weightList.add(-.01);
        numberOfInputs = features;
        //numberOfInputs = features;
        Random rand = new Random();
        double max = .3;
        double min = -.3;
        
        //generate the weights for each of the inputs.
        for(int i = 0; i < features; i++)
        {
            double theNum = min + (max - min) * rand.nextDouble();
            weightList.add(theNum);           
        }
    }
    
    public void adjustTheWeights()
    {
        double tempWeight;
        double newWeight;
        
        for(int i = 0; i < numberOfInputs; i++)
        {
//            System.out.println("\tAdjusting the weights for node " + nodeIndex);
            tempWeight = weightList.get(i);
//            System.out.println("\t\tCurrent weight: " + tempWeight);
//        System.out.println("\t\tLearning Constant" + learningConstant);
//           System.out.println("\t\tError" + error);
            //System.out.println("\t\tNode output" + output);
            //System.out.println("\t\tValue over that weight: " + inputList.get(i));
            
            //this output needs to be the input from the PREVIOUS node.

        //    System.out.println("HERE:" + inputList.get(i));
            newWeight = tempWeight - (learningConstant * error * inputList.get(i));
       //     System.out.println("\t\t\tThe result is: " + newWeight);
            weightList.set(i, newWeight);
        //    System.out.println("The new weight is: " + newWeight + " TEST: " + weightList.get(i));
            

        }
      //  System.out.println();
        biasWeight = (biasWeight - (learningConstant * error * (-1)));
        outputDebugData();
    }
    
    //This function sees if the node would fire
    //then using the class attribute of the 
    //instance determines IF the node should
    //have fired. It then adjusts the Node's
    //weights accordingly.
    public double trainOnInstance(Instance inst)
    {
        for(int i = 0; i < inst.numAttributes(); i++)
        {
            inputList.add(inst.value(i));
        }
        
        output = fires(inst);

        return output;
    }
    
    //Does just what thainOnInstance does, but uses a list.
    //TODO: fix this...
    public double trainOnList(List input, int items)
    {
        inputList = input;
        output = fires(input);

        return output;
    }
    
    public void outputDebugData()
    {
    //    System.out.println("\tI am node " + nodeIndex + " and my current target value is: " 
     //           + target);
     //   System.out.println("\tMy weights are: ");
        for(int i = 0; i < numberOfInputs; i++)
        {
     //       System.out.println("\t\tWeight " + i + ": " + weightList.get(i));
        }
     //   System.out.println("\t\tMy bias weight is: " + biasWeight);
        
    }

   /***************************************************************
    * Tests to see if a node fires. It does so using the formula:
    * 
    *               Sigma(weight * input)
    *
    * If this result is above the threshold (generally set to 0), the
    * node will fire (return a 1). Otherwise it does not fire (returns 0).
    * 
    * @param inst - the instance upon which to see if the node fires.
    * 
    *************************************************************/    
    public double fires(Instance inst)
    {       
        double sum = 0.0;
    //    System.out.println("I am in fires inst. The number of inputs is " + numberOfInputs);
        for(int i = 0; i < numberOfInputs; i++)
        {
    //        System.out.println("\tWe are adding (" + inst.value(i) + " * " + weightList.get(i) + ") to " + sum);
            sum += (inst.value(i) * weightList.get(i));
        }
        
        //lastly, we need to add the bias node.
    //    System.out.println("\tWe are adding (" + biasInput + " * " + biasWeight + ") to " + sum);
        sum += (biasInput * biasWeight);
        sum *= -1;
   //     System.out.println("h = " + sum);
        double result = 1 / (1 + Math.exp(sum));
    //    System.out.println("Final: " + result);
        output = result;
        return result;
    }
    
    public double fires(List input)
    {       
        double sum = 0;
    //    System.out.println("I am in fires list. The number of inputs is " + numberOfInputs);
    //    System.out.println("The input I'm using is: " + input);
        for(int i = 0; i < numberOfInputs; i++)
        {
    //        System.out.println("\t(OL) We are adding (" + input.get(i) + " * " + weightList.get(i) + ") to " + sum);
            sum += (double) input.get(i) * (double) weightList.get(i);
        }
        
    //    System.out.println("\t(OL) We are adding (" + biasInput + " * " + biasWeight + ") to " + sum);
        sum += (biasInput * biasWeight);
        sum *= -1;
    //    System.out.println("h = " + sum);
       
        double result = 1 / (1 + Math.exp(sum));
    //    System.out.println("Final: " + result);
        output = result;
        return result;
    }
}

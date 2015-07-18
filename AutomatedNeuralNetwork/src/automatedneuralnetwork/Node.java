package automatedneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.core.Instance;

/**
 *
 * @author Mackenzie
 */
public class Node {
    List<Double> inputList;
    List<Double> weightList;
    int numberOfInputs;
    double threshold;
    double biasInput;  
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
    *************************************************************/
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
    
    /*****************************************************
    * Finds the error for an output layer node
    *****************************************************/
    public void calculateOLError()
    {
        error = (output - target) * output * (1 - output);
    }
    
    /******************************************************
    * Finds the error for an hidden layer node.
    ******************************************************/
    public void calculateHLError(Layer nextLayer)
    {
        error = output*(1 - output);
        double sum = 0.0;
        
        for(int i = 0; i < nextLayer.numberOfNodes; i++)
        {
            double tempError = nextLayer.Nodes.get(i).error;
            double tempWeight = nextLayer.Nodes.get(i).weightList.get(nodeIndex);
            sum += (tempError * tempWeight);
        }
        error *= sum;
    }
    
    /***************************************************************
    * Generates a weight for the bias node.
    ***************************************************************/
    public double generateBiasWeight()
    {
        Random rand = new Random();
        double max = .3;
        double min = -.3;
        
        return .1;
        //return (min + (max - min) * rand.nextDouble());
    }
    
    /***************************************************************
    * Randomly generates node weights.
    ***************************************************************/
    public void generateWeights(int features)
    {
        numberOfInputs = features;
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
    
    /***************************************************************
    * Sends the signal for each node to adjust its weights
    ***************************************************************/
    public void adjustTheWeights()
    {
        double tempWeight;
        double newWeight;
        
        for(int i = 0; i < numberOfInputs; i++)
        {
            tempWeight = weightList.get(i);  
            newWeight = tempWeight - (learningConstant * error * inputList.get(i));
            weightList.set(i, newWeight);
        }
        biasWeight = (biasWeight - (learningConstant * error * (-1)));
    }
    
    /***************************************************************
    * This function sees if the node would fire
    * then using the class attribute of the 
    * instance determines IF the node should
    * have fired. It then adjusts the Node's
    * weights accordingly.
    ***************************************************************/
    public double trainOnInstance(Instance inst)
    {
        for(int i = 0; i < inst.numAttributes(); i++)
        {
            inputList.add(inst.value(i));
        }
        
        output = fires(inst);
        return output;
    }
    
    /***************************************************************
    * Trains using a list of inputs
    ***************************************************************/
    public double trainOnList(List input, int items)
    {
        inputList = input;
        output = fires(input);

        return output;
    }
    
    public void outputDebugData()
    {
        System.out.println("\tI am node " + nodeIndex + " and my current target value is: " 
                + target);
        System.out.println("\tMy weights are: ");
        for(int i = 0; i < numberOfInputs; i++)
        {
            System.out.println("\t\tWeight " + i + ": " + weightList.get(i));
        }
        System.out.println("\t\tMy bias weight is: " + biasWeight);    
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
        for(int i = 0; i < numberOfInputs; i++)
        {
            sum += (inst.value(i) * weightList.get(i));
        }
        
        //lastly, we need to add the bias node.
        sum += (biasInput * biasWeight);
        sum *= -1;
        output = 1 / (1 + Math.exp(sum));
        return output;
    }
    
   /***************************************************************
    * Tests to see if a node fires. It does so using the formula:
    * 
    *               Sigma(weight * input)
    *
    * If this result is above the threshold (generally set to 0), the
    * node will fire (return a 1). Otherwise it does not fire (returns 0).
    * 
    * @param input - the data upon which to see if the node fires.
    * 
    *************************************************************/  
    public double fires(List input)
    {       
        double sum = 0;
        for(int i = 0; i < numberOfInputs; i++)
        {
            sum += (double) input.get(i) * (double) weightList.get(i);
        }
        
        sum += (biasInput * biasWeight);
        sum *= -1;
        output = 1 / (1 + Math.exp(sum));
        return output;
    }
}

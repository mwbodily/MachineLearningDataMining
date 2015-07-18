package automatedneuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Mackenzie
 */
public class NeuralClassifier extends Classifier{
    Layer theLayer;
    List<Layer> theLayers; //this is for the next iteration of the ANN.
    Instances dataSet;
    List<Double> layerInput;
    int numberOfAttributes;
    int numberOfLayers;
    /*******************************************************************
     * Default constructor, currently empty. 
     ********************************************************************/
    public NeuralClassifier(int numLayers, int numAttributes, ArrayList<Integer> numNodes, int numClasses)
    {
        theLayers = new ArrayList();
        layerInput = new ArrayList();
        
        numberOfAttributes = numAttributes;
        numberOfLayers = numLayers;
        
        //Create the network
        for(int i = 0; i < (numLayers - 1); i++)
        {
            //construct a layer, the target starts out as -1 since it's not set.
            Layer temp = new Layer((int) numNodes.get(i), numAttributes, -1);
            theLayers.add(temp);
        }
        //add the output layer. This should have as many nodes as there are classes.
        theLayers.add(new Layer((int) numNodes.get((numLayers - 1)), theLayers.get(numberOfLayers-2).numberOfNodes, -1));

    }
    
   /******************************************************
    * Outputs the contents of the layers. Used for 
    * Debugging
    *****************************************************/
    public void outputLayersForDebug()
    {
        System.out.println("\t---------DEBUG TESTING------------");
        for(int i = 0; i < numberOfLayers; i++)
        {
            System.out.println("\tLAYER " + i);
            theLayers.get(i).outputDebugData();
            System.out.println();
        }
        System.out.println("\t---------END TESTING------------");
    }
    
    /*****************************************************
    * Trains the classifier by sending each instance through.
    *****************************************************/
    public void trainNetwork()
    {
        //for each instance, convert it to input for the list then run it 
        //through the layers       

        for(int i = 0; i < dataSet.numInstances(); i++)
        {            
            theLayers.get((numberOfLayers - 1)).setTarget(dataSet.instance(i).classValue());
                   
            for(int k = 0; k < numberOfLayers; k++)
            {
                if(k != 0)
                {
                    theLayers.get(k).trainWithList((theLayers.get(k-1).getOutputList()), theLayers.get(k-1).numberOfNodes);
                }
                else
                {
                    theLayers.get(k).trainWithInstance(dataSet.instance(i));
                }
                
            }

            adjustNetwork();
        }
    }
    
    /*****************************************************
    * Adjusts the weights of the network.
    *****************************************************/
    public void adjustNetwork()
    {
        //find the error of each node starting from the output layer
        
        theLayers.get((numberOfLayers - 1)).calculateOLErrors();
        
        for(int i = (numberOfLayers - 2); i >= 0; i--)
        {
            theLayers.get(i).calculateHLErrors(theLayers.get(i+1));
        }
        
        for(int i = (numberOfLayers - 1); i >= 0; i--)
        {
            theLayers.get(i).adjustWeights();
        }

    }
    
    public double classifyAnInstance(Instance inst)
    {
        for(int i = 0; i < (numberOfLayers); i++)
        {
            if(i != 0)
            {
                theLayers.get(i).propogateList(theLayers.get(i-1).getOutputList());
            }
            else
            {
                theLayers.get(i).propogateInstance(inst);
            }
        }
        return theLayers.get((numberOfLayers-1)).classify();
    }
    /********************************************************************
     * Builds the classifier using the instances.
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        dataSet = i;
        for(int it = 0; it < 5; it++)
        {
            trainNetwork();
            dataSet.randomize(new Random(1));
        }

    }
   
    /********************************************************************
     * Classifies the instance passed to it.
     ********************************************************************/
    @Override
    public double classifyInstance(Instance inst) throws Exception
    {
        return classifyAnInstance(inst);
    }
}

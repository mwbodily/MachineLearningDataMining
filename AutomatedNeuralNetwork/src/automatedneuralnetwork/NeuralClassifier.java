/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
            System.out.println("I am creating layer number " + i);
            //construct a layer, the target starts out as -1 since it's not set.
            Layer temp = new Layer((int) numNodes.get(i), numAttributes, false, -1);
            theLayers.add(temp);
        }
        //add the output layer. This should have as many nodes as there are classes.
        theLayers.add(new Layer((int) numNodes.get((numLayers - 1)), theLayers.get(numberOfLayers-2).numberOfNodes, true, -1));
        System.out.println("We now have layers. As you can see.");
        //Output the layers, for debugging.
        //outputLayersForDebug();

    }
    
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
    
    //trains the classifier by sending each instance through.
    public void trainNetwork()
    {
        //for each instance, convert it to input for the list then run it 
        //through the layers       

        for(int i = 0; i < dataSet.numInstances(); i++)
        {            
            System.out.println("Instance: " + dataSet.instance(i));
            System.out.println("Checking..." + dataSet.instance(i).classValue());
            //set up the outer layer to have the correct output.
            theLayers.get((numberOfLayers - 1)).setTarget(dataSet.instance(i).classValue());
            //outputLayersForDebug();
                   
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
            outputLayersForDebug();
            adjustNetwork();
            System.out.println("END OF INSTANCE");

        }
    }
    
    public void adjustNetwork()
    {
        System.out.println("-----------Beginning back propogation-----------------");
        //find the error of each node starting from the output layer
        
        System.out.println("Propogation for outer layer");
        theLayers.get((numberOfLayers - 1)).calculateOLErrors();
        
        for(int i = (numberOfLayers - 2); i >= 0; i--)
        {
            System.out.println("Propogation for hidden layer" + i);
            theLayers.get(i).calculateHLErrors(theLayers.get(i+1));
        }
        

        for(int i = (numberOfLayers - 1); i <= 0; i--)
        {
            System.out.println("Weight adjustments for layer " + i);
            theLayers.get(i).adjustWeights();
        }

    }
    
    public double classifyAnInstance(Instance inst)
    {
        System.out.println("I am classifying an instance: " + inst);
        for(int i = 0; i < (numberOfLayers); i++)
        {
            if(i != 0)
            {
                System.out.println("Output Layers");
                theLayers.get(i).propogateList(theLayers.get(i-1).getOutputList());
                outputLayersForDebug();
            }
            else
            {
                System.out.println("Hidden Layers");
                theLayers.get(i).propogateInstance(inst);
            }
        }
        return theLayers.get((numberOfLayers - 1)).classify();
    }
    /********************************************************************
     * Builds the classifier using the instances.
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        dataSet = i;
        for(int it = 0; it < 3; it++)
        {
            trainNetwork();
            dataSet.randomize(new Random(1));
            //outputLayersForDebug();
            System.out.println("------------------------------------------------------------------------------------here " + it);
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

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import java.util.ArrayList;
import java.util.List;
import weka.core.Instance;

/**
 *
 * @author Mackenzie
 */
public class Layer {
        List<Node> Nodes;
        int numberOfNodes;
        int numberOfFeatures;
        double learningConstant;
        Boolean isOutput;
        double target;
        List<Double> nodeOutputs;
        
        public Layer(int numNodes, int numFeatures, Boolean out, double t)
        {
            System.out.println("I am being created with " + numNodes + "nodes.");
            Nodes = new ArrayList();
            nodeOutputs = new ArrayList();
            numberOfNodes = numNodes;
            numberOfFeatures = numFeatures;
            learningConstant = .1;
            target = t;
            isOutput = out;
            instantiateNodes();
            instantiateNodeOutputs();
        }
        
        public void instantiateNodeOutputs()
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                nodeOutputs.add(0.0);
            }
        }
        
        public void calculateOLErrors()
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).calculateOLError();
            }
        }
        
        public void calculateHLErrors(Layer nextLayer)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).calculateHLError(nextLayer);
            }
        }
        
        public void adjustWeights()
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).adjustTheWeights();
            }
        }
        /******************************************************
         * Instantiates the list of nodes by creating 
         * numberOfNodes nodes.
         *****************************************************/
        public void instantiateNodes()
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.add(new Node(numberOfFeatures, learningConstant, i, target));
            }  
        }
  
        //outputs data about the layer and it's nodes for debugging.
        public void outputDebugData()
        {
            System.out.println("\t\tCreated a layer with " + numberOfNodes + 
                    " nodes and a target of " + target);
            System.out.println("\t\tFor validation, these are the nodes I contain.");
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).outputDebugData();
            }
            
            if(nodeOutputs.isEmpty())
            {
                System.out.println("\t\tAs of yet, I have no outputs.");
            }
            else
            {
                System.out.println("\t\tThe following is my output layer: ");
                for(int i = 0; i < numberOfNodes; i++)
                {
                    System.out.println("\t\t\t" + nodeOutputs.get(i));
                }
            }
        }
        
        //calls the training protocol of the Node class
        //if it's off the Node will be able to adjust
        //it's weigths. This function requires an instance.
        public void trainWithInstance(Instance inst)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                double result = Nodes.get(i).trainOnInstance(inst);
                nodeOutputs.set(i, result);
            }
        }
        
        public void trainWithList(List inputList, int items)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                double result = Nodes.get(i).trainOnList(inputList, items);
                nodeOutputs.set(i, result);
            }
        }
        
        //gets the output of all of the nodes and places them into 
        //a list to serve as inputs to the next layer.
        public List getOutputList()
        {            
            return nodeOutputs;
        }
        
        //sets the target of the layer and each node to the class value of
        //the instance.
        public void setTarget(double targ)
        {
            target = targ;
            
            for(int i = 0; i < numberOfNodes; i++)
            {
                if(Nodes.get(i).nodeIndex == targ)
                {
                    Nodes.get(i).target = 1.0;
                }
                else
                {
                    Nodes.get(i).target = 0.0;
                }
            }
        }
        
        public void propogateList(List inputList)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                double result = Nodes.get(i).fires(inputList);
                nodeOutputs.set(i, result);
            }
        }
        
        public void propogateInstance(Instance inst)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                double result = Nodes.get(i).fires(inst);
                nodeOutputs.set(i, result);
            }
        }
        
        public double classify()
        {
            double max = Double.NEGATIVE_INFINITY;
            int indexOfMax = -1;
            
            for(int i = 0; i < numberOfNodes; i++)
            {
                if(Nodes.get(i).output > max)
                {
                    max = Nodes.get(i).output;
                    indexOfMax = i;
                }
            }
            System.out.println(indexOfMax + " asdf");
            return indexOfMax;
        }
}

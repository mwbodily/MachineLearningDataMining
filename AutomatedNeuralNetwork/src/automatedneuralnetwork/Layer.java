package automatedneuralnetwork;

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
        double target;
        
       /******************************************************
       * Basic constructor. Takes in the number of nodes
       * features and the target.
       *****************************************************/
        public Layer(int numNodes, int numFeatures, double t)
        {
            Nodes = new ArrayList();
            numberOfNodes = numNodes;
            numberOfFeatures = numFeatures;
            learningConstant = .2;
            target = t;
            instantiateNodes();
        }
        
       /******************************************************
        * Calculates the error for the output layer 
        *****************************************************/
        public void calculateOLErrors()
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).calculateOLError();
            }
        }
        
       /******************************************************
        * Calculates the error for hidden nodes.
        *****************************************************/
        public void calculateHLErrors(Layer nextLayer)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).calculateHLError(nextLayer);
            }
        }
        
       /******************************************************
        * Adjusts the weights for the nodes
        *****************************************************/
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
  
        /******************************************************
        * outputs data about the layer and it's nodes for debugging.
        ******************************************************/
        public void outputDebugData()
        {
            System.out.println("\t\tCreated a layer with " + numberOfNodes + 
                    " nodes and a target of " + target);
            System.out.println("\t\tFor validation, these are the nodes I contain.");
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).outputDebugData();
            }
            
            List nodeOutputs = getOutputList();
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
        
        /******************************************************
        * calls the training protocol of the Node class
        * if it's off the Node will be able to adjust
        * it's weigths. This function requires an instance.
        ******************************************************/
        public void trainWithInstance(Instance inst)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).trainOnInstance(inst);
            }
        }
        
        /******************************************************
        * calls the training protocol of the Node class
        * if it's off the Node will be able to adjust
        * it's weigths. This function requires a list.
        ******************************************************/
        public void trainWithList(List inputList, int items)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).trainOnList(inputList, items);
            }
        }
        
        /******************************************************
        * gets the output of all of the nodes and places them into 
        * a list to serve as inputs to the next layer.'
        ******************************************************/
        public List getOutputList()
        {            
            List<Double> outputList = new ArrayList();
            for(int i = 0; i < numberOfNodes; i++)
            {
                outputList.add(Nodes.get(i).output);
            
            }
            
            return outputList;
        }
        
        /******************************************************
        * sets the target of the layer and each node to the class value of
        * the instance.
        ******************************************************/
        public void setTarget(double targ)
        {
            target = targ;
            
            for(int i = 0; i < numberOfNodes; i++)
            {
                if(Nodes.get(i).nodeIndex == (int) targ)
                {
                    Nodes.get(i).target = 1.0;
                }
                else
                {
                    Nodes.get(i).target = 0.0;
                }
            }
        }
        
        /******************************************************
        * Goes through the list and makes each node fire if it
        * should be firing
        ******************************************************/
        public void propogateList(List inputList)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).fires(inputList);
            }
        }
        
       /******************************************************
        * Goes through the list and makes each node fire if it
        * should be firing
        ******************************************************/
        public void propogateInstance(Instance inst)
        {
            for(int i = 0; i < numberOfNodes; i++)
            {
                Nodes.get(i).fires(inst);
            }
        }
        
       /******************************************************
        * Classifies an instance
        ******************************************************/
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
            return indexOfMax;
        }
}

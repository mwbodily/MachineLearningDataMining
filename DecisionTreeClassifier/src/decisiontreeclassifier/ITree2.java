/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import decisiontreeclassifier.InstanceTree.Node;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Mackenzie Bodily
 */
public class ITree2 extends Classifier{
    private Instances dataSet;
    InstanceTree iTree;
    //List attributeSelection;
    int binNum;

    /********************************************************************
     * Default constructor. 
     ********************************************************************/
    public ITree2(int bn)
    {
        //attributeSelection = new ArrayList();
        binNum = bn;
    }
    
    public double findEntropy(Node theNode)
    {
        double numClasses;
        
        //allows us to deal with it if the class attribute is numeric.
        if(theNode.dataSet.instance(0).classAttribute().isNumeric())
        {
            numClasses = theNode.dataSet.numDistinctValues(theNode.dataSet.instance(0).classAttribute());
        }
        else
        {
            numClasses = dataSet.numClasses();
        }
        //initialize the array

        double[] array = new double[(int)numClasses];
        
        double entropy = 0.0;
        double numInstances = theNode.dataSet.numInstances();

        if(numInstances == 1 || theNode.allTheSame())
        {
            return 0;
        }

        //Counts the number of instances of each class in the data set.
        for(int i = 0; i < numInstances; i++)
        {
            array[(int) theNode.dataSet.instance(i).classValue()] += 1;
        }

        for(int i = 0; i < numClasses; i++)
        {
            if(array[i] != 0 && numInstances != 0)
            {
                entropy += ((array[i] / numInstances) * (Math.log((array[i] / numInstances)) / Math.log(2)));
            }
        }
        
        entropy *= -1;
        return entropy;
    }
    
    public double findRange(int attIndex, Node theNode)
    {
        double range = 0;
        double highest = Double.NEGATIVE_INFINITY;
        double lowest = Double.POSITIVE_INFINITY;
        
        int numInstances = theNode.dataSet.numInstances();
        
        //Find the highest value in the node parameter...
        for(int i = 0; i < numInstances; i++)
        {
           if(theNode.dataSet.instance(i).value(attIndex) > highest)
           {
               highest = theNode.dataSet.instance(i).value(attIndex);
           }
           if(theNode.dataSet.instance(i).value(attIndex) < lowest)
           {
               lowest = theNode.dataSet.instance(i).value(attIndex);
           }
        }
        range = highest - lowest;

        return range;
    }
    
    public double findLowest(int attIndex, Node theNode)
    {
        double lowest = Double.POSITIVE_INFINITY;
        int numInstances = theNode.dataSet.numInstances();
        
        for(int i = 0; i < numInstances; i++)
        {
           if(theNode.dataSet.instance(i).value(attIndex) < lowest)
           {
               lowest = theNode.dataSet.instance(i).value(attIndex);
           }
        }  
        return lowest;
    }
    
    public void initializeList(List<Double> theList, int numItems)
    {
        for(int i = 0; i < numItems; i++)
        {
            theList.add(0.00); 
        }
    }
    
    public double findNodeScore(double range, int attIndex, double totalEntropy, Node theNode)
    {
        double score = 0;
        double increment = range / binNum;
        double lowest = findLowest(attIndex, theNode);
        double grandTotal = 0.00;
        int numClasses = theNode.dataSet.numClasses();
        double numInstances = theNode.dataSet.numInstances();  
        
        List<double[]> ranges = new ArrayList<double[]>();
        List<Double> entropies = new ArrayList<Double>();
        List<Double> totals = new ArrayList<Double>();
        
        for(int i = 0; i < binNum; i++)
        {
            double[] rangeArray =  new double[numClasses]; 
            ranges.add(rangeArray);
        }
        
        //count the number that will fall into each range...
        for(int i = 0; i < numInstances; i++)
        {
            for(int j = 0; j < numClasses; j++)
            {
                if(theNode.dataSet.instance(i).value(attIndex) <= (lowest + (increment * (j + 1))))
                {
                    ranges.get(j)[(int) theNode.dataSet.instance(i).classValue()] += 1;
                    break;
                }
            }
        }
        
        initializeList(entropies, binNum);
        initializeList(totals, binNum);
        
        //Calculates the entropies for each bin...
        for(int i = 0; i < binNum; i++)
        {
            for(int j = 0; j < numClasses; j++)
            {
                double temp = (totals.get(i)) + (ranges.get(i)[j]);
                totals.set(i, temp);
                if(ranges.get(i)[j] != 0)
                {
                    double temp2 = entropies.get(i) + ((ranges.get(i)[j] / numInstances) * (Math.log((ranges.get(i)[j] / numInstances)) / Math.log(2)));
                    entropies.set(i, temp2);
                }
            }
        }
        
        for(int i = 0; i < binNum; i++)
        {
            double temp = (entropies.get(i) * -1);
            entropies.set(i, temp);
            grandTotal += totals.get(i);
        }
        
        //Find the average of the results
        for(int i = 0; i < binNum; i++)
        {
            score += (totals.get(i) / grandTotal) * entropies.get(i);
        }


        return score;
    }
    
    public int scoreForNode(Node theNode)
    {      
        //Find the Entropy for the whole thing...

        double score = 0;
        double entropy = findEntropy(theNode);
        
        
        int numAttributes = theNode.dataSet.numAttributes() - 1;
        
        double bestAttributeScore = Double.NEGATIVE_INFINITY;
        int bestAttribute = 0;
        
        for(int i = 0; i < numAttributes; i++)
        {
            if(!theNode.usedFeatures.contains(i))
            {           
                double range = findRange(i, theNode);
                score = findNodeScore(range, i, entropy, theNode);
            
                if(score > bestAttributeScore)
                {
                    bestAttributeScore = score;
                    bestAttribute = i;
                }
            
            }
        }
        
        return bestAttribute;
    }
    
    public ArrayList copyArrayList(Node root)
    {
        ArrayList copiedList = new ArrayList<Integer>();
       
        ListIterator<Integer> listIterator = root.usedFeatures.listIterator();
        ArrayList temp = new ArrayList<Integer>();
        temp = root.usedFeatures;
        
        while(listIterator.hasNext())
        {
            int i = listIterator.next();
            copiedList.add(listIterator.next());
        }
        
        return copiedList;
    
    }
    
    public void addTreeNode(int attIndex, Node theNode)
    {
        double range = findRange(attIndex, theNode);
        double increment = range / binNum;
        double lowest = findLowest(attIndex, theNode);
        int numInstances = theNode.dataSet.numInstances();
        
        for(int i = 0; i < binNum; i++)
        {
            Node child;
            Instances InstCopy = new Instances(theNode.dataSet);
            ArrayList UFcopy = (ArrayList<Integer>) theNode.usedFeatures.clone();
            if(i == 0)
            {
                child = new Node(InstCopy, UFcopy, 0, Double.NEGATIVE_INFINITY, (lowest+increment), attIndex, iTree.root);
            }
            else if(i == (binNum - 1))
            {
                child = new Node(InstCopy, UFcopy, i, (lowest + (i*increment)), Double.POSITIVE_INFINITY, attIndex, iTree.root);
            }
            else
            {
                child = new Node(InstCopy, UFcopy, i, (lowest + (increment * i)), (lowest + (increment * (i+1))), attIndex, iTree.root);
            }
            theNode.addChild(child);
        }
        
        filterInstances(theNode, attIndex);
    }
    
    
    public void filterInstances(Node theNode, int attIndex)
    {
        int numInstances;
        for(int i = 0; i < theNode.numChildren; i++)
        {
            numInstances = theNode.getChildAt(i).dataSet.numInstances();
            for(int j = 0; j < numInstances; j++)
            {
                if(!(theNode.getChildAt(i).withinRange(theNode.getChildAt(i).dataSet.instance(j).value(attIndex), attIndex)))
                {
                    theNode.getChildAt(i).dataSet.delete(j);
                    j -= 1;
                    numInstances -= 1;
                }
            }
        }
    }
    public void buildTree(Node root)
    {
        //create the root, this should contain all the instances.
        int attribute2Split;
        Node nodeOfInterest = iTree.root;
        Boolean done = false;
        
        while(!done)
        {
            if(!nodeOfInterest.allFeaturesUsed() && !nodeOfInterest.allTheSame() && nodeOfInterest.numChildren != binNum)
            {
                attribute2Split = scoreForNode(nodeOfInterest);

                if(nodeOfInterest == iTree.root)
                {
                    nodeOfInterest.splitOn = attribute2Split;
                }
                nodeOfInterest.usedFeatures.add(attribute2Split);
                addTreeNode(attribute2Split, nodeOfInterest);
                nodeOfInterest = nodeOfInterest.getChildAt(0);
            }
            else
            {
                if(nodeOfInterest.index != (binNum - 1))
                {
                    nodeOfInterest = nodeOfInterest.getSibling();
                }
                else
                {
                    while(nodeOfInterest.index == (binNum - 1))
                    {
                        nodeOfInterest = nodeOfInterest.getParent();
                        if(nodeOfInterest == iTree.root)
                        {
                            done = true;
                            break;
                        }
                    }
                }
            }
        }       
    }
    
    public void runTreeStuff()
    {
        iTree = new InstanceTree(dataSet);
        buildTree(iTree.root);
    }
    
    public Instances fixMissingData(Instances iToFix)
    {
        for(int i = 0; i < iToFix.numInstances(); i++)
        {
            for(int j = 0; j < iToFix.numAttributes(); j++)
            {
                if(iToFix.instance(i).isMissing(j))
                {
                    iToFix.instance(i).setValue(j, 0.0);
                }
            }
        }
        return iToFix;
    }
    
    /********************************************************************
     * Builds the classifier using the instances. 
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        dataSet = fixMissingData(i);
        iTree = new InstanceTree(dataSet);
        buildTree(iTree.root);
        //runTreeStuff();

        
        iTree.printTree();
    }
   
    /********************************************************************
     * Classifies the instance passed to it. 
     ********************************************************************/
    @Override
    public double classifyInstance(Instance inst) throws Exception
    {
        Node nodeBeingChecked = iTree.root;
        Node nextNode;
        while(nodeBeingChecked.hasChildren())
        {         
            nextNode = nodeBeingChecked.findChild(inst);
            
            if(nextNode == null)
            {
                return 1.00;
            }
                nodeBeingChecked = nextNode;
        }
        
        if(nodeBeingChecked.dataSet.numInstances() != 0)
        {
            return (nodeBeingChecked.makeGuess());
        }
        else
        {
            return (nodeBeingChecked.getParent().makeGuess());

        }
    }
}
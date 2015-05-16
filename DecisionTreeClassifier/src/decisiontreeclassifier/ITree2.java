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
    List attributeSelection;
    int binNum;
    Node MrSpecial;
    /********************************************************************
     * Default constructor. 
     ********************************************************************/
    public ITree2(int bn)
    {
        attributeSelection = new ArrayList();
        binNum = bn;
    }
    
    public double findEntropy(Node theNode)
    {
        //initialize the array
        double numClasses = theNode.dataSet.numClasses();
        double[] array = new double[(int)numClasses];
        double entropy = 0.0;
        double numInstances = theNode.dataSet.numInstances();
        
        if(numInstances == 1)
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
            //System.out.println("------------");
            //System.out.println("Entropy: " + entropy);
            if(array[i] != 0 && numInstances != 0)
            {
                entropy += ((array[i] / numInstances) * (Math.log((array[i] / numInstances)) / Math.log(2)));
            }
        }
        
        entropy *= -1;
        
        if(Double.isNaN(entropy))
        {
            System.out.println("Error!");
            System.exit(1);
        
        }
        return entropy;
    }
    
    public double findRange(int attIndex, Node theNode)
    {
        //System.out.println("        Find Range Called");
        double range = 0;
        double highest = Double.NEGATIVE_INFINITY;
        double lowest = Double.POSITIVE_INFINITY;
        //System.out.println(theNode.dataSet.instance(0).attribute(attIndex));
        
        int numInstances = theNode.dataSet.numInstances();
        //System.out.println("        Find Range Setup Finished");
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
        //System.out.println("Range found: " + range);
        //System.out.println("Increment will be: " + range/binNum);
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
        //System.out.println("            In Find Node Score");
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
        //System.out.println("Score For Node Called");
        //System.out.println("    Debugging info----");
        
        //Find the Entropy for the whole thing...
        double score = 0;
        double entropy = findEntropy(theNode);
        
        //System.out.println("        Entropy: " + entropy);
        
        int numAttributes = theNode.dataSet.numAttributes() - 1;
        
        double bestAttributeScore = Double.NEGATIVE_INFINITY;
        int bestAttribute = 0;
        
        for(int i = 0; i < numAttributes; i++)
        {
            if(!theNode.usedFeatures.contains(i))
            {           
          //      System.out.println("        Finding range for attribute " + i);
                double range = findRange(i, theNode);
          //      System.out.println("        Range: " + range);
                score = findNodeScore(range, i, entropy, theNode);
            
                if(score > bestAttributeScore)
                {
                    bestAttributeScore = score;
                    bestAttribute = i;
                }
            
            }
        }
       // System.out.println("        Score: " + bestAttribute);
        //System.out.println("    End Debugging----");
        //System.out.println("Score For Node Completed");
        
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
        
        //System.out.println("    check 1");
        for(int i = 0; i < binNum; i++)
        {
            //System.out.println("    test... " + binNum);
            Node child;
            Instances InstCopy = new Instances(theNode.dataSet);
            ArrayList UFcopy = (ArrayList<Integer>) theNode.usedFeatures.clone();
            if(i == 0)
            {
     //           System.out.println("Adding attIndex " + attIndex);
                child = new Node(InstCopy, UFcopy, 0, Double.NEGATIVE_INFINITY, (lowest+increment), attIndex, iTree.root);
            }
            else if(i == (binNum - 1))
            {
       //         System.out.println("Adding attIndex " + attIndex);
                child = new Node(InstCopy, UFcopy, i, (lowest + (i*increment)), Double.POSITIVE_INFINITY, attIndex, iTree.root);
            }
            else
            {
         //       System.out.println("Adding attIndex " + attIndex);
                child = new Node(InstCopy, UFcopy, i, (lowest + (increment * i)), (lowest + (increment * (i+1))), attIndex, iTree.root);
            }
            theNode.addChild(child);
        }
        
        //System.out.println("    check 2");

        List offsets = new ArrayList<Integer>();

        for(int i = 0; i < binNum; i++)
        {
            offsets.add(0);
        }
        
        //System.out.println("    check 3");
        filterInstances(theNode, attIndex);
        //System.out.println("Add Tree Node Completed");
    }
    
    
    public void filterInstances(Node theNode, int attIndex)
    {
        int numInstances;
        for(int i = 0; i < theNode.numChildren; i++)
        {
          //  System.out.println("Check 1");
            numInstances = theNode.getChildAt(i).dataSet.numInstances();
            for(int j = 0; j < numInstances; j++)
            {
            //    System.out.println("Check 2");
            //    System.out.println("    numInst: " + numInstances + " j: " + j);
                if(!(theNode.getChildAt(i).withinRange(theNode.getChildAt(i).dataSet.instance(j).value(attIndex), attIndex)))
                {
                    theNode.getChildAt(i).dataSet.delete(j);
                    j -= 1;
                    numInstances -= 1;
                    
                }
            //    else
            //    {
             //       System.out.println("Keeping it.");
            //    }
            }
        }
    }
    public void buildTree(Node root)
    {
        //System.out.println("Build Tree Called");
        //create the root, this should contain all the instances.
        int attribute2Split;
        Node nodeOfInterest = iTree.root;
        Boolean done = false;
        
        //System.out.println("About to start the while loop"); -- Removed, it output this statement.
        while(!done)
        {
          //  System.out.println(nodeOfInterest.dataSet.numInstances());
            if(nodeOfInterest.allFeaturesUsed() || nodeOfInterest.dataSet.numInstances() == 1)
            {
               // System.out.println("here1 index: " + nodeOfInterest.index + " inst: " + nodeOfInterest.dataSet.numInstances() + 
                //        " Parent: " + nodeOfInterest.getParent().dataSet.numInstances());
                
                if(nodeOfInterest.index != (binNum - 1))
                {
                    //System.out.println("test");
                    nodeOfInterest = nodeOfInterest.getSibling();
                }
                else
                {   
                    //System.out.println("In the else statement");
                    while((nodeOfInterest.allFeaturesUsed() || nodeOfInterest.dataSet.numInstances() == 1) && !done || nodeOfInterest.done)
                    {                       
                        //System.out.println("while check");
                        
                        if(nodeOfInterest.index == (binNum - 1))
                        {
                            nodeOfInterest.getParent().done = true;
                        }
                        
                        if(nodeOfInterest != iTree.root)
                        {
                            //System.out.println("if one");
                            nodeOfInterest = nodeOfInterest.getParent();
                            //System.out.println("if one done");
                        }
                        
                        //System.out.println("here1 index: " + nodeOfInterest.index + " inst: " + nodeOfInterest.dataSet.numInstances() + 
                        //" Parent: " + nodeOfInterest.getParent().dataSet.numInstances());
                        
                        if(nodeOfInterest.index != (binNum - 1) && nodeOfInterest != iTree.root)
                        {
                            //System.out.println("if two");
                            nodeOfInterest = nodeOfInterest.getSibling();   
                        }
                        

                        
                        //System.out.println("here2 index: " + nodeOfInterest.index + " inst: " + nodeOfInterest.dataSet.numInstances() + 
                        //" Parent: " + nodeOfInterest.getParent().dataSet.numInstances());
                        if((nodeOfInterest.getParent() == iTree.root && nodeOfInterest.index == (binNum - 1) && nodeOfInterest.dataSet.numInstances() == 1)
                                || nodeOfInterest == iTree.root)
                        {
                            //System.out.println("In Done");
                            done = true;
                            break;
                        }
                    }
                }
            }
            else
            {       
            //    System.out.println("in the else");
                if(nodeOfInterest.dataSet.numInstances() > 0)
                {
              //      System.out.println("in the first if");
                    //works fine up to here
                    attribute2Split = scoreForNode(nodeOfInterest);
                //    System.out.println("here1");
                    if(nodeOfInterest == iTree.root)
                    {
                        nodeOfInterest.splitOn = attribute2Split;
                    }
                    //start debugging here...
                    nodeOfInterest.usedFeatures.add(attribute2Split);
                    addTreeNode(attribute2Split, nodeOfInterest);
                    nodeOfInterest = nodeOfInterest.getChildAt(0);
                  //  System.out.println("ending the first if.");
                }
                else
                {
                    if(nodeOfInterest.index != (binNum - 1))
                    {
                        nodeOfInterest = nodeOfInterest.getSibling();
                    }
                    else
                    {   
                        while(nodeOfInterest.allFeaturesUsed() || (nodeOfInterest.dataSet.numInstances() == 1) || 
                                (nodeOfInterest.dataSet.numInstances() == 0))
                        {                            
                            nodeOfInterest = nodeOfInterest.getParent();
                            
                            if((nodeOfInterest.index != (binNum - 1)) && (nodeOfInterest != iTree.root))
                            {
                                nodeOfInterest = nodeOfInterest.getSibling();                       
                            }
                            
                            if (nodeOfInterest.getParent() == iTree.root)
                            {
                                done = true;
                            }
                        }
                    }
                }
            }
        //System.out.println();
       // System.out.println();
       // iTree.printTree();
       // System.out.println();
       // System.out.println();
        }
        //iTree.printTree();
        //System.out.println("Build Tree Completed");
    }
    
    public void runTreeStuff()
    {
        //System.out.println("Run Tree Stuff Called");
        iTree = new InstanceTree(dataSet);
        //System.out.println("Instance Tree Successfully Created");
        buildTree(iTree.root);
        //System.out.println("Run Tree Stuff Completed");
    }
    
    /********************************************************************
     * Builds the classifier using the instances. 
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        for(int j = 0; j < i.numInstances(); j++)
        {
            System.out.println(i.instance(j).classValue());
        }
        System.out.println("ASDF: " + i.enumerateAttributes());
//System.out.println("Build Classifier Called");
        dataSet = i;
        //System.out.println(dataSet);
        runTreeStuff();
        //System.out.println("Build Classifier Completed");
        iTree.printTree();
    }
   
    /********************************************************************
     * Classifies the instance passed to it. 
     ********************************************************************/
    @Override
    public double classifyInstance(Instance inst) throws Exception
    {
        //System.out.println("-------------");
        //System.out.println("Inst: " + inst);
        Node nodeBeingChecked = iTree.root;
        
        while(nodeBeingChecked.hasChildren())
        {
            //System.out.println("Node: " + nodeBeingChecked.dataSet.numInstances());
            nodeBeingChecked = nodeBeingChecked.findChild(inst);
        }

        //System.out.println(nodeBeingChecked.dataSet);
        
        if(nodeBeingChecked.dataSet.numInstances() != 0)
        {

            return (nodeBeingChecked.makeGuess());
            
            //return (nodeBeingChecked.dataSet.instance(0).classValue());
        }
        else
        {
          //  System.out.println("Guessing a " + nodeBeingChecked.getParent().dataSet.instance(0).classValue());
          //  System.out.println("-----------");
            return (nodeBeingChecked.getParent().makeGuess());
            //return (nodeBeingChecked.getParent().dataSet.instance(0).classValue());
        }
    }
}
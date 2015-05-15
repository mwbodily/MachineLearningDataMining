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
public class ID3Classifier extends Classifier{
    private Instances dataSet;
    InstanceTree iTree;
    
    /********************************************************************
     * Default constructor. 
     ********************************************************************/
    public ID3Classifier()
    {

    }
    
    public double findEntropy()
    {
        int numClasses = dataSet.numClasses();
        
        //initialize the array
        double[] array = new double[numClasses];
        int numInstances = dataSet.numInstances();
        
        for(int i = 0; i < numInstances; i++)
        {
            array[(int) dataSet.instance(i).classValue()] += 1;
            //System.out.println(dataSet.instance(i).classValue());
        }
        
        if(false)
        {
            for(int i = 0; i < numClasses; i++)
            {
             System.out.println(array[i]);
            }
        }
        
        double entropy = 0.0;
        
        for(int i = 0; i < numClasses; i++)
        {
            //double test = Math.log((array[i] / numInstances)) / Math.log(2);
            //System.out.println("Adding: " + ((array[i] / numInstances) * (Math.log((array[i] / numInstances)) / Math.log(2))));
            entropy += ((array[i] / numInstances) * (Math.log((array[i] / numInstances)) / Math.log(2)));
            //System.out.println(entropy);
        }
        
        entropy *= -1;
        //System.out.println(entropy);
        return entropy;
    }
    
    public double findRange(int attIndex)
    {
        double range = 0;
        double highest = Double.NEGATIVE_INFINITY;
        double lowest = Double.POSITIVE_INFINITY;
        
        
        for(int i = 0; i < dataSet.numInstances(); i++)
        {
           if(dataSet.instance(i).value(attIndex) < lowest)
           {
               lowest = dataSet.instance(i).value(attIndex);
           }
           else if(dataSet.instance(i).value(attIndex) > highest)
           {
               highest = dataSet.instance(i).value(attIndex);
           }
        }
        
        //System.out.println(highest + "-(" + lowest + ")=" + (highest - lowest)); 
        range = highest - lowest;
        return range;
    }
    
    public double findNodeScore(double range, int attIndex, double totalEntropy)
    {
        double score = 0;
        double increment = range / 3;
        double lowest = dataSet.kthSmallestValue(attIndex, 1);
      
        int numClasses = dataSet.numClasses();
        
        double[] range1 =  new double[numClasses]; //neg infinity to range/3
        double[] range2 =  new double[numClasses]; //range/3 to range*2
        double[] range3 =  new double[numClasses]; //range*2 to pos infinity
        
        double numInstances = dataSet.numInstances();
        //System.out.println("Your increment is: " + increment);
        
        
        for(int i = 0; i < numInstances; i++)
        {
            if(dataSet.instance(i).value(attIndex) < (lowest + increment))
            {
                range1[(int) dataSet.instance(i).classValue()] += 1;
            }
            else if(dataSet.instance(i).value(attIndex) >= (lowest + increment) && dataSet.instance(i).value(attIndex) < (lowest + (2*increment)))
            {
                range2[(int) dataSet.instance(i).classValue()] += 1;
            }
            else
            {
                range3[(int) dataSet.instance(i).classValue()] += 1;
            }
        }
        
        //Find the entropy of each new node...
        double entropy1 = 0.0;
        double entropy2 = 0.0;
        double entropy3 = 0.0;
        
        double total1 = 0;
        double total2 = 0;
        double total3 = 0;
        
       // System.out.println("Range 1");
        for(int i = 0; i < numClasses; i++)
        {
            total1 += range1[i];
          //  System.out.println(range1[i]);
            
            if(range1[i] != 0)
            {
                entropy1 += ((range1[i] / numInstances) * (Math.log((range1[i] / numInstances)) / Math.log(2)));
            }
            //System.out.print("ENT:  " + entropy1);
        }
        
        //System.out.println("Range 2");
        for(int i = 0; i < numClasses; i++)
        {
            total2 += range2[i];
          //  System.out.println(range2[i]);
            if(range2[i] != 0)
            {
                entropy2 += ((range2[i] / numInstances) * (Math.log((range2[i] / numInstances)) / Math.log(2)));
            }
        }
        
        //System.out.println("Range 3");
        for(int i = 0; i < numClasses; i++)
        {
            total3 += range3[i];
          //  System.out.println(range3[i]);
            
            if(range3[i] != 0)
            {
                entropy3 += ((range3[i] / numInstances) * (Math.log((range3[i] / numInstances)) / Math.log(2)));
            }
        }
        
        //System.out.println("TOTALS: " + total1 + " " + total2 + " " + total3);
        
        //Find the average of the results
        double grandTotal = total1 + total2 + total3;
        
        entropy1 *= -1;
        entropy2 *= -1;
        entropy3 *= -1;
        
        //System.out.println("ent1:  " + entropy1);
        //System.out.println("ent2:  " + entropy2);
        //System.out.println("ent3:  " + entropy3);
        
        score = ((total1 / grandTotal) * entropy1) + ((total2 / grandTotal) * entropy2) + ((total3 / grandTotal) * entropy3);
        
        //System.out.println("The score I got was: " + score);

        return score;
    }
    
    public int scoreForNode(Node theNode)
    {
        //Find the Entropy for the whole thing...
        double score = 0;
        double entropy = findEntropy();
        System.out.println("Entropy: " + entropy);
        
        int numAttributes = dataSet.numAttributes() - 1;
        
        //find the Info Gain for each attribute...
        //divide the feature's values into categories.. This should only be done if it's numeric...
        
        double bestAttributeScore = Double.NEGATIVE_INFINITY;
        int bestAttribute = 0;
        
        for(int i = 0; i < numAttributes; i++)
        {
            if(!theNode.usedFeatures.contains(i))
            {           
                //System.out.println("Attribute " + i + "----------");
                double range = findRange(i);
                score = findNodeScore(range, i, entropy);
            
                if(score > bestAttributeScore)
                {
                
                    bestAttributeScore = score;
                    bestAttribute = i;
                }
            
               //System.out.println("");
            }
        }
        
        System.out.println("Splitting on " + bestAttribute + " with a score of " + bestAttributeScore);
        
        return bestAttribute;
    }
    
    public ArrayList copyArrayList(Node root)
    {
        ArrayList copiedList = new ArrayList<Integer>();
       
        ListIterator<Integer> listIterator = root.usedFeatures.listIterator();
        ArrayList temp = new ArrayList<Integer>();
        temp = root.usedFeatures;
        
        //for(int i : temp)
       // {
            
       // }
        while(listIterator.hasNext())
        {
            int i = listIterator.next();
            copiedList.add(listIterator.next());
        }
        
        return copiedList;
    
    }
    
    public void addTreeNode(int attIndex, Node root)
    {
        double range = findRange(attIndex);
        double increment = range / 3;
        double lowest = root.dataSet.kthSmallestValue(attIndex, 1);
        
        Instances copy1 = new Instances(root.dataSet);
        Instances copy2 = new Instances(root.dataSet);
        Instances copy3 = new Instances(root.dataSet);
        
        ArrayList lCopy1 = (ArrayList<Integer>) root.usedFeatures.clone();//new ArrayList(root.usedFeatures);
        ArrayList lCopy2 = (ArrayList<Integer>) root.usedFeatures.clone();//copyArrayList(root);//new ArrayList(root.usedFeatures);
        ArrayList lCopy3 = (ArrayList<Integer>) root.usedFeatures.clone();//copyArrayList(root);//new ArrayList(root.usedFeatures);
        
        Node child1 = new Node(copy1, lCopy1, 0);
        Node child2 = new Node(copy2, lCopy2, 1);
        Node child3 = new Node(copy3, lCopy3, 2);
       
        int numInstances = root.dataSet.numInstances();
        
        int offset1 = 0;
        int offset2 = 0;
        int offset3 = 0;
        
        System.out.println("here "  + root.dataSet.numInstances());
        
        for(int i = 0; i < numInstances; i++)
        {
            if(root.dataSet.instance(i).value(attIndex) < (lowest + increment))
            {
                child2.dataSet.delete((i - offset2));
                child3.dataSet.delete((i - offset3));
                
                offset2 += 1;
                offset3 += 1;
            }
            else if(dataSet.instance(i).value(attIndex) >= (lowest + increment) && dataSet.instance(i).value(attIndex) < (lowest + (2*increment)))
            {
                child1.dataSet.delete((i - offset1));
                child3.dataSet.delete((i - offset3));
                
                offset1 += 1;
                offset3 += 1;
            }
            else
            {
                child1.dataSet.delete((i - offset1));
                child2.dataSet.delete((i - offset2));
                
                offset1 += 1;
                offset2 += 1;
            }
        }
        root.addChild(child1);
        root.addChild(child2);
        root.addChild(child3);
        
        System.out.println("root "  + root.dataSet.numInstances());
        System.out.println("c1 "  + child1.dataSet.numInstances());
        System.out.println("c2 "  + child2.dataSet.numInstances());
        System.out.println("c3 "  + child3.dataSet.numInstances());
        //System.out.println("c4 "  + child4.dataSet.numInstances());
    }
    
    public void buildTree(Node root)
    {
        //create the root, this should contain all the instances.
        int attribute2Split;
        Node nodeOfInterest = iTree.root;
        Boolean done = false;
        
        while(!done)
        {
            //System.out.println("In the main while loop");
            if(nodeOfInterest.allFeaturesUsed())
            {
                System.out.println("All features used...");
                //nodeOfInterest.outputUsedFeatures();
                //System.out.println("I'm in the if statement lv.1");
                if(nodeOfInterest.index != 2)
                {
                    //System.out.println("I'm in the if statement lv.2");
                    System.out.println("1 Switching Sibling... Ind: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                    nodeOfInterest = nodeOfInterest.getSibling();
                    System.out.println("1 We're now on sibling: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                }
                else
                {   
                    //System.out.println("I'm in the 2nd level else statement");
                    //while(nodeOfInterest.index == 2 && nodeOfInterest.allFeaturesUsed())
                    while(nodeOfInterest.allFeaturesUsed())
                    {
                        System.out.println("1 Going up a level " + + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                        
                        if(nodeOfInterest != iTree.root)
                        {
                            nodeOfInterest = nodeOfInterest.getParent();
                        }
                        
                        if(nodeOfInterest.index != 2 && nodeOfInterest != iTree.root)
                        {
                            System.out.println("2 Switching Sibling... Ind: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                            nodeOfInterest = nodeOfInterest.getSibling();
                            System.out.println("2 We're now on sibling: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                        
                        }
                        if(nodeOfInterest.getParent() == iTree.root && nodeOfInterest.index == 3)
                        {
                            done = true;
                        
                        }
                    }
                }
            }
            else
            {                
                System.out.println("Splitting Node.");
                //Get the higest scoring attribute 
                if(nodeOfInterest.dataSet.numInstances() > 0)
                {
                    attribute2Split = scoreForNode(nodeOfInterest);
            
                    //make sure we don't just keep splitting on that attribute.
                    nodeOfInterest.usedFeatures.add(attribute2Split);
                
                    //add the node to the tree
                    addTreeNode(attribute2Split, nodeOfInterest);
        
                    nodeOfInterest = nodeOfInterest.getChildAt(0);
                    
                    
                }
                else
                {
                    System.out.println("No Instances. switching items" + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numAttributes());
                    if(nodeOfInterest.index != 2)
                    {
                        //System.out.println("I'm in the if statement lv.2");
                        System.out.println("3 Switching Sibling... Ind: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                        nodeOfInterest = nodeOfInterest.getSibling();
                        System.out.println("3 We're now on sibling: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                    }
                    else
                    {   
                        //System.out.println("I'm in the 2nd level else statement");
                        
                        //while(nodeOfInterest.index == 2 && nodeOfInterest.allFeaturesUsed())
                        while(nodeOfInterest.allFeaturesUsed() || (nodeOfInterest.dataSet.numInstances() == 0))
                        {
                            System.out.println("loop!");
                            System.out.println("2 Going up a level " + + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                            
                            nodeOfInterest = nodeOfInterest.getParent();
                            
                            if((nodeOfInterest.index != 2) && (nodeOfInterest != iTree.root))
                            {
                                System.out.println("4 Switching Sibling... Ind: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                                nodeOfInterest = nodeOfInterest.getSibling();
                                System.out.println("4 We're now on sibling: " + nodeOfInterest.index + " " + nodeOfInterest.dataSet.numInstances());
                        
                            }
                            
                            if (nodeOfInterest.getParent() == iTree.root)
                            {
                                done = true;
                            }
                        }
                    }
                }
            }
        }
        iTree.printTree();
    }
    
    public void runTreeStuff()
    {
        //System.out.println("This is a test...");
        iTree = new InstanceTree(dataSet);
        buildTree(iTree.root);
    }
    
    /********************************************************************
     * Builds the classifier using the instances. 
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        dataSet = i;
        //System.out.println(i);
        runTreeStuff();
        
    }
   
    /********************************************************************
     * Classifies the instance passed to it. 
     ********************************************************************/
    @Override
    public double classifyInstance(Instance inst) throws Exception
    {
        return 0;   
    }
}

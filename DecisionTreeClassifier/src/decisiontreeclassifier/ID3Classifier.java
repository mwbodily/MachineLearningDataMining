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
        double lowest = dataSet.kthSmallestValue(attIndex, 1);
        
        Instances copy1 = new Instances(dataSet);
        Instances copy2 = new Instances(dataSet);
        Instances copy3 = new Instances(dataSet);
       System.out.println("-------------------------------------"); 
        
        
        ArrayList lCopy1 = (ArrayList<Integer>) root.usedFeatures.clone();//new ArrayList(root.usedFeatures);
       // lCopy1.add(7);
        //System.out.println("Root:");
        //root.outputUsedFeatures();
       /*System.out.println("Child:");
            ListIterator<Integer> listIterator = lCopy1.listIterator();
            while(listIterator.hasNext())
            {
                System.out.println(listIterator.next());
            }
            
            System.out.println("Root again");
            root.outputUsedFeatures();
       System.out.println("-------------------------------------"); 
*/
        
//System.out.println("")
        
        ArrayList lCopy2 = (ArrayList<Integer>) root.usedFeatures.clone();//copyArrayList(root);//new ArrayList(root.usedFeatures);
        ArrayList lCopy3 = (ArrayList<Integer>) root.usedFeatures.clone();//copyArrayList(root);//new ArrayList(root.usedFeatures);
        
        Node child1 = new Node(copy1, lCopy1, 0);
        
        //System.out.println("Child1 UF:----");
        //child1.outputUsedFeatures();
        //System.out.println("Child1 UF:----"); 
        Node child2 = new Node(copy2, lCopy2, 1);
        Node child3 = new Node(copy3, lCopy3, 2);
       
        int numInstances = dataSet.numInstances();
        
        int offset1 = 0;
        int offset2 = 0;
        int offset3 = 0;
        
        for(int i = 0; i < numInstances; i++)
        {
            if(dataSet.instance(i).value(attIndex) < (lowest + increment))
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
        
        //System.out.println("TEST: " + iTree.root.numChildren());
        
    }
    
    public void buildTree(Node root)
    {
        //System.out.println("HERE 0");
        //create the root, this should contain all the instances.
        int attribute2Split;
        Node nodeOfInterest = iTree.root;
        Boolean done = false;
        
        while(!done)
        {
            //System.out.println("In the main while loop");
            if(nodeOfInterest.allFeaturesUsed())
            {
                //nodeOfInterest.outputUsedFeatures();
                //System.out.println("I'm in the if statement lv.1");
                if(nodeOfInterest.index != 2)
                {
                    //System.out.println("I'm in the if statement lv.2");
                    System.out.println("Switching Sibling... Ind: " + nodeOfInterest.index);
                    nodeOfInterest = nodeOfInterest.getSibling();
                    System.out.println("We're now on sibling: " + nodeOfInterest.index);
                }
                else
                {   
                    //System.out.println("I'm in the 2nd level else statement");
                    while(nodeOfInterest.index == 2)
                    {
                        System.out.println("Going up a level " + nodeOfInterest.index);
                        nodeOfInterest = nodeOfInterest.getParent();
                        if(nodeOfInterest.index != 2)
                        {
                            System.out.println("Switching Sibling... Ind: " + nodeOfInterest.index);
                            nodeOfInterest = nodeOfInterest.getSibling();
                        
                        }
                        if(nodeOfInterest.getParent() == iTree.root && nodeOfInterest.index == 3)
                        {
                            done = true;
                        
                        }
                    }
                   
                    nodeOfInterest = nodeOfInterest.getSibling();
                }
            }
            else
            {
        //for(int i = 0; i < 2; i++)
        //{
                
                //Get the higest scoring attribute 
                attribute2Split = scoreForNode(nodeOfInterest);
            
                                //make sure we don't just keep splitting on that attribute.
                                nodeOfInterest.usedFeatures.add(attribute2Split);
                //add the node to the tree
                addTreeNode(attribute2Split, nodeOfInterest);
        

                //System.out.println("UF------------");
                //nodeOfInterest.outputUsedFeatures();
                //System.out.println("UF------------ END");
                //System.out.println("THINGER:: " + nodeOfInterest.usedFeatures.get(0));
                //get the first child of the node of interest.
                //System.out.println("here 3");
                nodeOfInterest = nodeOfInterest.getChildAt(0);
                //System.out.println("here 4");
            }
            //System.out.println("This is a test here too.");
        }
        //}
        
        iTree.printTree();
        //System.out.println(allFeaturesUsed());
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

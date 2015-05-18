/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Mackenzie
 */
public class InstanceTree<T> {
    public Node<T> root;

    /********************************************************************
     * Constructs a tree with a root node containing the data set passed
     * in.
     ********************************************************************/
    public InstanceTree(Instances rootData) {
        root = new Node<T>(rootData);
        root.children = new ArrayList<Node<T>>();
    }
    
    /********************************************************************
    * Calls the recursive function and does necessary setup to print the
    * tree.
    ********************************************************************/
    public void printTree()
    {
        System.out.println("-----------------------------------------------");
        System.out.println("Root: Instances: " + root.dataSet.numInstances());
        printTreeOutput(root, 1);
        System.out.println("-----------------------------------------------");
    }
    
    /********************************************************************
     * Recursively output the tree.
     ********************************************************************/
    private void printTreeOutput(Node theRoot, int tabs)
    {   
        for(int i = 0; i < theRoot.numChildren(); i++)
        {
            for(int j = 0; j < tabs; j++)
            {
                System.out.print("   ");
            }
            System.out.print("Child " + (i+1) + ": Instances: " + theRoot.getChildAt(i).dataSet.numInstances());
            System.out.println(" Split on: " + theRoot.getChildAt(i).splitOn);
            if(theRoot.getChildAt(i) != null)
            {
                tabs++;
                printTreeOutput(theRoot.getChildAt(i), tabs);
            }
            tabs --;
        }
    }
    
    public static class Node<T> {
        private Node<T> parent;
        private List<Node<T>> children;
        public Instances dataSet;
        public ArrayList usedFeatures;
        int index;
        double lowerRange;
        double upperRange;
        public int splitOn;
        public int numChildren;
        public Boolean done;
        Node<T> iRoot;
        
        /********************************************************************
        * Constructor for a node, only needs the instances.
        ********************************************************************/
        public Node(Instances inst)
        {
            numChildren = 0;
            dataSet = inst;
            usedFeatures = new ArrayList<Integer>();
            children = new ArrayList<Node<T>>();
            done = false;
        }
        
        /********************************************************************
        * Constructor for a node.
        * 
        * TODO: cut down on the data needed!
        ********************************************************************/
        public Node(Instances inst, ArrayList uF, int ind, double lR, double uR, int pSplitOn, Node pIRoot)
        {
            numChildren = 0;
            dataSet = inst;
            usedFeatures = uF;
            index = ind;
            children = new ArrayList<Node<T>>();
            lowerRange = lR;
            upperRange = uR;
            done = false;
            iRoot = pIRoot;
            splitOn = pSplitOn;
        }
        
        /********************************************************************
        * Returns the number of children that the node has. 
        ********************************************************************/
        public int numChildren()
        {
            return children.size();
        }
        
        /********************************************************************
        * Sees if the node has used all possible features to split on.
        ********************************************************************/
        public Boolean allFeaturesUsed()
        {
            return((dataSet.numAttributes() - 1) == usedFeatures.size());               
        }
        
        /********************************************************************
        * Adds a child to the node.
        ********************************************************************/
        public void addChild(Node<T> child)
        {
            child.parent = this;
            children.add(child);
            numChildren += 1;
        }
        
        /********************************************************************
        * Checks if the node has children. 
        ********************************************************************/
        public Boolean hasChildren()
        {
            if(numChildren == 0)
            {
                return false;
            }
            return true;
        }
        
        /********************************************************************
        * Makes a guess based on how many instances of each type are in the
        * data set for the leaf node.
        ********************************************************************/
        public double makeGuess()
        {
            double guess = 0.00;
            ArrayList<Integer> numOfEach = new ArrayList<>();

            for(int i = 0; i < iRoot.dataSet.numClasses(); i++)
            {
                numOfEach.add(0);
            }

            for(int i = 0; i < dataSet.numInstances(); i++)
            {
                int temp = numOfEach.get((int) dataSet.instance(i).classValue());
                temp += 1;
                numOfEach.set((int) dataSet.instance(i).classValue(), temp);
            }
            
            for(int i = 0; i < iRoot.dataSet.numClasses(); i++)
            {
                if((double) numOfEach.get(i) > guess)
                {
                    guess = (double) i;
                }
            }
            return guess;
        }
        
        /********************************************************************
        * Finds the proper child for the instance based on what attribute is
        * being tested.
        * 
        * TODO: I don't think this is working correctly. Check on it more.
        ********************************************************************/
        public Node<T> findChild(Instance inst)
        {
            for(int i = 0; i < numChildren; i++)
            {
                if(children.get(i).withinRange(inst.value(children.get(i).splitOn), i))
                {
                    return children.get(i);
                }
            }
            return null;
        }
        
        /********************************************************************
        * Tests to see if every item in the dataSet is the same.
        ********************************************************************/
        public Boolean allTheSame()
        {
            for(int i = 0; i < dataSet.numInstances(); i++)
            {
                if(dataSet.instance(0).classValue() != dataSet.instance(i).classValue())
                {
                    return false;
                }
            }
            return true;
        }
        
        /********************************************************************
        * Tests to see if a value is within the node's range.
        ********************************************************************/
        public Boolean withinRange(double value, int childIndex)
        {
            return(value >= lowerRange && value < upperRange);
        }
        
        /********************************************************************
        * Gets the node's child at index. (The first child is 0.)
        ********************************************************************/
        public Node<T> getChildAt(int index)
        {
            return children.get(index);
        }

       /********************************************************************
        * Gets the node's parent 
        ********************************************************************/
        public Node getParent()
        {
            return parent;
        }
        
       /********************************************************************
        * Gets the node's sibling. 
        ********************************************************************/
        public Node getSibling()
        {
            return parent.getChildAt((index + 1));
        }
    }
}


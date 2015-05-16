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

    public InstanceTree(Instances rootData) {
        root = new Node<T>(rootData);
        //root.dataSet = rootData;
        root.children = new ArrayList<Node<T>>();
        
    }
    
    public void printTree()
    {
        System.out.println("-----------------------------------------------");
        System.out.println("Root: Instances: " + root.dataSet.numInstances());
        printTreeOutput(root, 1);
        System.out.println("-----------------------------------------------");
    }
    
    public void printTreeOutput(Node theRoot, int tabs)
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
        
        public Node(Instances inst)
        {
            numChildren = 0;
            dataSet = inst;
            usedFeatures = new ArrayList<Integer>();
            children = new ArrayList<Node<T>>();
            done = false;
        }
        
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
        
        public Boolean isInNode(double item)
        {
            return (item >= lowerRange && item < upperRange);
        }
        
        public int numChildren()
        {
            return children.size();
        }
        
        public Boolean allFeaturesUsed()
        {
            return((dataSet.numAttributes() - 1) == usedFeatures.size());               
        }
        
        public void addChild(Node<T> child)
        {
            //System.out.println("addChild::: " + child.dataSet.numInstances() + " to " + dataSet.numInstances());
            child.parent = this;
            children.add(child);
            numChildren += 1;
        }
        
        public Boolean hasChildren()
        {
            if(numChildren == 0)
            {
                return false;
            }
            return true;
        }
        
        public double makeGuess()
        {
            double guess = 0.00;
            ArrayList<Integer> numOfEach = new ArrayList<>();
            
            for(int i = 0; i < iRoot.dataSet.numClasses(); i++)
            {
                numOfEach.add(0);
            }
            
            System.out.println("Guessing 1");
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
            System.out.println("Guessing actually finished, guessed a " + guess);
            System.out.println("-----------");
            return guess;
        }
        
        public Node<T> findChild(Instance inst)
        {
            System.out.println("I am child: " + dataSet.numInstances());
            System.out.println("I am split on: " + splitOn);
            //System.out.println(inst);
            for(int i = 0; i < numChildren; i++)
            {
                System.out.println("Instance splitting on: " + splitOn);
                if(children.get(i).withinRange(inst.value(children.get(i).splitOn), i))
                {
                    System.out.println("I'm sending us down child " + children.get(i).dataSet.numInstances());
                    
                    return children.get(i);
                    
                }
                System.out.println("Switching sibling...");
            }
            return null;
        }
        
        public Boolean withinRange(double value, int childIndex)
        {
            //System.out.println("This is where the problem is");
            System.out.println("            Num Instances: " + dataSet.numInstances());
            System.out.println("            LR: " + lowerRange);
            System.out.println("            UR: " + upperRange);
            System.out.println("            Value: " + value);
            Boolean truth = value >= lowerRange && value < upperRange;
            System.out.println("            In Range: " + truth);
            return(value >= lowerRange && value < upperRange);
        }
        
        public Node<T> getChildAt(int index)
        {
            return children.get(index);
        }
        
        public Node getParent()
        {
            return parent;
        }
        
        public void outputUsedFeatures()
        {
            System.out.println("------UF-------");
            ListIterator<Integer> listIterator = usedFeatures.listIterator();
            while(listIterator.hasNext())
            {
                System.out.println(listIterator.next());
            }
            System.out.println("------UF-------");
        
        }
        
        public Node getSibling()
        {
            return parent.getChildAt((index + 1));
        }
    }
}


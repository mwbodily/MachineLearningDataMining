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
            System.out.println("Child " + (i+1) + ": Instances: " + theRoot.getChildAt(i).dataSet.numInstances());
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
        
        public Node(Instances inst)
        {
            numChildren = 0;
            dataSet = inst;
            usedFeatures = new ArrayList<Integer>();
            children = new ArrayList<Node<T>>();
        }
        
        public Node(Instances inst, ArrayList uF, int ind, double lR, double uR, int splitOn)
        {
            numChildren = 0;
            dataSet = inst;
            usedFeatures = uF;
            index = ind;
            children = new ArrayList<Node<T>>();
            lowerRange = lR;
            upperRange = uR;
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
        
        public Node<T> findChild(Instance inst)
        {
            for(int i = 0; i < 3; i++)
            {
                if(children.get(i).withinRange(inst.value(splitOn), i))
                {
                    return children.get(i);
                    
                }
            }
            return null;
        }
        
        public Boolean withinRange(double value, int childIndex)
        {
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


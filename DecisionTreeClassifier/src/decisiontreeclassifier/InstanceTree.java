/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
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
        System.out.println("This functionality is not yet complete.");
        System.out.println("Root: Instances: " + root.dataSet.numInstances());
        
        for(int i = 0; i < root.numChildren(); i++)
        {
            Node temp = root.getChildAt(i);
            System.out.println("Child " + i + ": Instances: " + temp.dataSet.numInstances());
        }
    }
    
    public static class Node<T> {
        private Node<T> parent;
        private List<Node<T>> children;
        public Instances dataSet;
        public ArrayList usedFeatures;
        int index;
        
        
        public Node(Instances inst)
        {
            dataSet = inst;
            usedFeatures = new ArrayList<Integer>();
            children = new ArrayList<Node<T>>();
        }
        
        public Node(Instances inst, ArrayList uF, int ind)
        {
            dataSet = inst;
            usedFeatures = uF;
            index = ind;
            children = new ArrayList<Node<T>>();
        }
        
        public int numChildren()
        {
            return children.size();
        }
        
        public Boolean allFeaturesUsed()
        {
            System.out.println(dataSet.numAttributes());
            return((dataSet.numAttributes() - 1) == usedFeatures.size());
                
        }
        
        public void addChild(Node<T> child)
        {
            //System.out.println("I'm here!!!" + child.index);
            child.parent = this;
            children.add(child);
            //System.out.println(children.get(0).dataSet.numInstances());
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


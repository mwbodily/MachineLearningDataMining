/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighbor;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Mackenzie
 */



public class KNNClassifier extends Classifier{
    private Instances dataSet;
    private int k;


    
    /*******************************************************************
     * Default constructor, currently empty. 
     * @param k = number of neighbors to search for
     ********************************************************************/
    public KNNClassifier(int kVal)
    {
        k = kVal;
    }
    
    /********************************************************************
     * Builds the classifier using the instances. For this iteration, 
     * this too is empty.
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        dataSet = i;
    }
   
     /********************************************************************
     * Finds the Euclidean distance between two points. Note that we 
     * aren't bothering to implement the square root function. Since we are
     * only looking for a general number and we are using the same 
     * measure for all points it will not effect the results of the
     * program. The power of two must stay to account for negative
     * numbers.
     * 
     * This program takes two parameters and returns the distance between
     * them. Using the Euclidean distance formula as follows:
     * 
     *  [insert formula]
     * 
     * @param a - point a
     * @param b - point b
     * @return - Euclidean distance between two points
     ********************************************************************/
    public double distanceEuclid(Instance a, Instance b)
    {
        int numAttributes = (a.numAttributes() - 1);
        double distance = 0;

        for(int i = 0; i < (numAttributes); i++)
        {
            distance += Math.pow(a.value(i) - b.value(i), 2);
            
        }
        return distance;
    }
    
     /********************************************************************
     * This function finds the Manhattan distance between two points 
     * according to the following formula:
     * 
     *  [insert formula]
     * 
     * It requires two instances to find the distance between.
     * @param a - point a
     * @param b = point b
     ********************************************************************/
    public double distanceManhattan(Instance a, Instance b)
    {
        int numAttributes = (a.numAttributes() - 1);
        double distance = 0;
                
        for(int i = 0; i < (numAttributes); i++)
        {
            distance += Math.abs(a.value(i) - b.value(i));
        }

        return distance;
    }
    
    private List initializeNeighbors()
    {
        List neighbors = new ArrayList();
        
        for(int i = 0; i < k; i++)
        {
            Neighbor temp = new Neighbor();
            neighbors.add(temp);
        }
        
        return neighbors;
    }
    
    public Neighbor findNewHighest(List neighbors)
    {
        Neighbor highest;
        highest = (Neighbor)neighbors.get(0);
        Neighbor temp;
        
        for(int i = 1; i < k; i++)
        {
            temp = (Neighbor)neighbors.get(i);
            if(temp.distance > highest.distance)
            {
                highest = temp;
            }
        }
        
        return highest;
        
    }
    
    public double makeGuess(List neighbors)
    {
        //System.out.println("Result Size: " + neighbors.size());
        //System.out.println("NumClasses: " + dataSet.numClasses());
        List<Integer> results = new ArrayList<Integer>();
        for(int i = 0; i <= dataSet.numClasses(); i++)
        {
            results.add(0);
        }
        
        Neighbor temp =(Neighbor)neighbors.get(0);
        //System.out.println("test: " + dataSet.instance(temp.index).attribute(0));
        int value;
        
        for(int i = 0; i < neighbors.size(); i++)
        {
            Neighbor item = (Neighbor) neighbors.get(i);
            int theIndex = item.index;
            
            //gets the current value in that node.
            int current = results.get((int)dataSet.instance(item.index).classValue());
            current += 1;
            results.set((int)dataSet.instance(item.index).classValue(), current);
         
        }
        
        System.out.println("testing...");
        for(int i : results)
        {
            System.out.println("::: " + i);
        }
        
        int highest = 0;
        for(int i =  0; i < results.size(); i++)
        {
            if(results.get(i) > results.get(highest))
            {
                highest = i;
            }
        }
        //System.out.println("highest: " + highest);
        return highest;
    }
    //TODO: allow the user to specify any k that they want. For now
    // k can only equal 1.
    /********************************************************************
     * Classifies the instance passed to it. Currently, this just returns
     * the first instance. 
     * 
     * @param inst
     * @return double - the correct classification.
     * @throws java.lang.Exception
     ********************************************************************/
    @Override
    public double classifyInstance(Instance inst) throws Exception
    {
        //System.out.println("does?");
        List neighbors = new ArrayList();
        
        for(int i = 0; i < k; i++)
        {
            Neighbor temp = new Neighbor();
            neighbors.add(temp);
        }
        //System.out.println("does 2?");
        //List neighbors = initializeNeighbors();
        Neighbor max = (Neighbor) neighbors.get(0);
        int highest = 1;
        double tempDistance;
        //System.out.println("does 3?");
        for(int i = 0; i < dataSet.numInstances(); i++)
        {
            tempDistance = distanceEuclid(inst, dataSet.instance(i));
           // System.out.println("does 4?");
            if(tempDistance < max.distance)
            {
                //System.out.println("does 5?");
                Neighbor temp = new Neighbor(i, tempDistance);
                max = findNewHighest(neighbors);
                //System.out.println("does 6?");
                neighbors.remove(max);
                neighbors.add(temp);
            }
        }
        System.out.println("making guess...");
        return makeGuess(neighbors);
        //return makeGuess(neighbors);
        //Neighbor currentHighest = neighbors.get(0);
       /* double tempDistance;
        
        for(int i = 0; i < dataSet.numInstances(); i++)
        {
            if (k == 1)
            {
                tempDistance = distanceEuclid(inst, dataSet.instance(i));
                if(tempDistance < currentHighest.getDistance())
                {   
                    closestNeighbor = i;
                    currentHighest = tempDistance;
               }   
            }
            else
            {
                tempDistance = distanceEuclid(inst, dataSet.instance(i));
                if(tempDistance < neighbors.get(highest).getDistance())
                {
                    
                }
            }*/
        
        //return 0;//(dataSet.instance(closestNeighbor)).value(4);   
    }
}

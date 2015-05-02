/*
 * This class implements a KNN classification algorithm. It utilizes the weka
 * library 
 */
package nearestneighbor;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Mackenzie Bodily
 */

public class KNNClassifier extends Classifier{
    private Instances dataSet;
    private final int k;

    /*******************************************************************
     * Default constructor, currently empty. 
     * @param kVal = number of neighbors to search for
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
     *  D = sqrt((a1 - a2)^2 + (b1 - b2)^2 + (c1 - c2)^2 + ...)
     * 
     * @param a - point a
     * @param b - point b
     * @return - Euclidean distance between two points
     ********************************************************************/
    private double distanceEuclid(Instance a, Instance b)
    {
        int numAttributes = (a.numAttributes() - 1);
        double distance = 0;

        for(int i = 0; i < numAttributes; i++)
        {
            distance += Math.pow(a.value(i) - b.value(i), 2);
            
        }
        return distance;
    }
    
     /********************************************************************
     * This function finds the Manhattan distance between two points 
     * according to the following formula:
     * 
     *  D = |(a1 - a2)| + |(b1 - b2)| + |(c1 - c2)| + ... + |...|
     * 
     * It requires two instances to find the distance between.
     * @param a - point a
     * @param b = point b
     ********************************************************************/
    private double distanceManhattan(Instance a, Instance b)
    {
        //System.out.println("A: " + a + "\nB: " + b + "\n");
        int numAttributes = (a.numAttributes() - 1);
        double distance = 0;
                
        for(int i = 0; i < numAttributes; i++)
        {
            //System.out.println(a.value(i) + " - " + b.value(i) + " = " + (a.value(i)-b.value(i)));
            distance += Math.abs(a.value(i) - b.value(i));
        }

        //System.out.println("\n\n");
        return distance;
    }
    
    /********************************************************************
     * Initializes a list to contain k Neighbors. These neighbors are 
     * later used to determine which value to guess when determining
     * the value of the testing instances.
     * 
     * @return 
     ********************************************************************/
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
    
    /********************************************************************
     * Finds the neighbor with the highest distance from the instance
     * being tested and returns that value.
     * 
     * @param neighbors - list of neighbors to find the highest of.
     * @return 
     ********************************************************************/
    private Neighbor findNewHighest(List neighbors)
    {
        Neighbor highest;
        highest = (Neighbor)neighbors.get(0);
        Neighbor temp;
        
        for(int i = 0; i < k; i++)
        {
            temp = (Neighbor)neighbors.get(i);
            if(temp.distance > highest.distance)
            {
                highest = temp;
            }
        }
        
        return highest;
        
    }
    
    /*********************************************************************
     * Creates a list with as many nodes as there are possible classes.
     * The frequency of each class among each of the nearest neighbors
     * (contained in the neighbors list) is then taken. The list with
     * these values is then passed so a guess can be made based on these
     * frequencies.
     * 
     * @param neighbors - the k nearest neighbors
     * @return 
     ********************************************************************/
    private List createFrequencyList(List neighbors)
    {
        List<Integer> results = new ArrayList<>();
        int current;
        Neighbor temp;
        
        for(int i = 0; i <= dataSet.numClasses(); i++)
        {
            results.add(0);
        }
        
        for(int i = 0; i < neighbors.size(); i++)
        {
            temp = (Neighbor) neighbors.get(i);
            
            //gets the current value in that node and increments it.
            current = results.get((int)dataSet.instance(temp.index).classValue());
            current += 1;
            results.set((int)dataSet.instance(temp.index).classValue(), current);
        }
        
        return results;
    }
    
    /********************************************************************
     * Uses the frequency of each class among the k nearest neighbors 
     * to guess a class. The class with the highest frequency is selected
     * and, if there is a tie, the first one with that frequency is 
     * selected. 
     * 
     * Note: This is the simple guesser. If a tie is encountered, it will
     * just select the first class that had that value. For a better 
     * tie breaking scheme, see makeGuessWeighted.
     * 
     * @param neighbors
     * @return 
     ********************************************************************/
    private double makeGuessSimple(List neighbors)
    {
        List<Integer> results = createFrequencyList(neighbors);
        int highest = 0;
       
        for(int i =  1; i < results.size(); i++)
        {
            if(results.get(i) > results.get(highest))
            {
                highest = i;
            }
        }
        return highest;
    }
   /********************************************************************
    * Goes through the neighbors list and adds up the distances between
    * the nodes of both types of classes. The class with the lowest sum
    * is returned as the match.
    * 
    * @param highest - one of the classes that is being compared
    * @param i - another of the classes that is being compared
    * @param neighbors - a list of the k closest neighbors to the point
    *                    we are trying to guess
    * 
    * @return 
    ********************************************************************/
    private int breakTieByWeight(List neighbors, int highest, int challenger)
    {
        Neighbor temp;
        double highestDistance = 0;
        double challengerDistance = 0;
        
        //find the neighbors with those classes and sum those distances
        for (int it = 0; it < neighbors.size(); it++)
        {
            temp = (Neighbor) neighbors.get(it);
            if(dataSet.instance(temp.index).classValue() == highest)
            {
                highestDistance += temp.distance;
            }
            else if(dataSet.instance(temp.index).classValue() == challenger)
            {
                challengerDistance += temp.distance;
            }
        }

       //return the lowest distance
        if(highestDistance > challengerDistance)
        {
            return highest;
        }
        
        return challenger;
    }
    
    /********************************************************************
     * Uses the frequency of each class among the k nearest neighbors 
     * to guess a class. The class with the highest frequency is selected
     * and, if there is a tie, the first one with that frequency is 
     * selected. 
     * 
     * Note: This class uses weights to break ties. If two classes
     * have the same frequency, the class with the two closest values
     * will be selected. For a more arbitrary method, see makeGuessSimple.
     * 
     * @param neighbors
     * @return 
     ********************************************************************/
    private double makeGuessWeighted(List neighbors)
    {
        List<Integer> results = createFrequencyList(neighbors);
        int highest = 0;
       
        for(int i =  1; i < results.size(); i++)
        {
            if(results.get(i) > results.get(highest))
            {
                highest = i;
            }
            else if((results.get(i) == results.get(highest)) && highest != 0)
            {          
                highest = breakTieByWeight(neighbors, highest, i);
            }
        }
        return highest;
    }
    
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
        
        List neighbors = new ArrayList();
        
        for(int i = 0; i < k; i++)
        {
            Neighbor temp = new Neighbor();
            neighbors.add(temp);
        }

        Neighbor max = (Neighbor) neighbors.get(0);
        double tempDistance;

        for(int i = 0; i < dataSet.numInstances(); i++)
        {          
            tempDistance = distanceManhattan(inst, dataSet.instance(i));

            if(tempDistance < max.distance)
            {
                Neighbor temp = new Neighbor(i, tempDistance);
                neighbors.remove(max);
                neighbors.add(temp);
                max = findNewHighest(neighbors);
            }
        }
       
        //make a guess based on one of the two options...
        //return makeGuessSimple(neighbors);
        return makeGuessWeighted(neighbors);       
    }
}

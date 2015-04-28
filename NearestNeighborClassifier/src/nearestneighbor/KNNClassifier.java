/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighbor;

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
    private int iter;
    private Instances dataSet;
    /*******************************************************************
     * Default constructor, currently empty. 
     ********************************************************************/
    public KNNClassifier()
    {
        iter = 0;
    }
    
    /********************************************************************
     * Builds the classifier using the instances. For this iteration, 
     * this too is empty.
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        //This is only called once. So You need to figure out a
        //way so that the data is still stored... As it is, I don't
        //think it's being stored...
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
        int numAttributes = a.numAttributes();
        double distance = 0;
                
        for(int i = 0; i < (numAttributes - 1); i++)
        {
            distance += Math.pow((a.value(i) - b.value(i)), 2);
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
    public void distanceManhattan(Instance a, Instance b)
    {
        
    }
    
    //TODO: allow the user to specify any k that they want. For now
    // k can only equal 1.
    /********************************************************************
     * Classifies the instance passed to it. Currently, this just returns
     * the first instance. 
     * 
     * @param inst
     * @throws java.lang.Exception
     ********************************************************************/
    @Override
    public double classifyInstance(Instance inst) throws Exception
    {
        int closestNeighbor = 0;
        double currentBest = Double.POSITIVE_INFINITY;
        double tempDistance;
        
        for(int i = 0; i < dataSet.numInstances(); i++)
        {
            tempDistance = distanceEuclid(inst, dataSet.instance(i));
            if(tempDistance < currentBest)
            {
                closestNeighbor = i;
                currentBest = tempDistance;
            }
        }
        
        return (dataSet.instance(closestNeighbor)).value(4);   
    }
}

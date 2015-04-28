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
    /*******************************************************************
     * Default constructor, currently empty. 
     ********************************************************************/
    public KNNClassifier()
    {
        
    }
    
    /********************************************************************
     * Builds the classifier using the instances. For this iteration, 
     * this too is empty.
     ********************************************************************/
    @Override
    public void buildClassifier(Instances i) throws Exception {
        
    }
   
    /********************************************************************
     * Classifies the instance passed to it. Currently, this just returns
     * the first instance. 
     ********************************************************************/
    @Override
    public double classifyInstance(Instance inst) throws Exception
    {
        return 0;   
    }
}

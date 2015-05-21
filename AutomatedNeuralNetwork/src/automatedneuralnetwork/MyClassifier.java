/*
 * Contains all the functions needed to run for a VERY basic classifier
 */
package automatedneuralnetwork;

import weka.classifiers.*;
import weka.core.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/** 
 * @author Mackenzie Bodily
 */
public class MyClassifier extends Classifier
{
    /*******************************************************************
     * Default constructor, currently empty. 
     ********************************************************************/
    public MyClassifier()
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

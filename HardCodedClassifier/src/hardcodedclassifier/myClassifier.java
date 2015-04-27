/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hardcodedclassifier;

import weka.classifiers.*;
import weka.core.*;

import weka.classifiers.Classifier;
//import weka.classifiers.AbstractClassifier;
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
public class myClassifier extends Classifier
{
    private int test;
    private Instances inst;
    public myClassifier()
    {
        
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        inst = i;
        //throw new UnsupportedOperationException("Not supported yet."); 
        
    }
   
    
}

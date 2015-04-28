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
public class NearestNeighborMain {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try{
            RunClassifier kNN = new RunClassifier();
            kNN.classify();
            kNN.outputResults();
        }
        catch (Exception er)
        {
            System.out.println("Error! " + er);
        }
        
        System.out.println("Program Complete.\n");
    }
    
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hardcodedclassifier;

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
public class HardCodedClassifier {

    /**
     * Currently, no command line parameters are needed in this
     * program. 
     */
    public static void main(String[] args) throws Exception{
        try{
            basicClassifier basic = new basicClassifier();
        }
        catch (Exception tempHandler)
        {
            System.out.println("Error! " + tempHandler);
        }
        
        System.out.println("Program Complete.\n");
    }
    
}

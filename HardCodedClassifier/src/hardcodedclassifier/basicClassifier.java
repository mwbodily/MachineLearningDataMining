/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hardcodedclassifier;

import weka.core.*;
import weka.classifiers.Classifier;//Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

/**
 *
 * @author Mackenzie
 */
public class basicClassifier {
    
    private int test;
    
    public basicClassifier() throws Exception
    {
        test = 3;
        //get the data Note, this file must exist in the top level folder.
        DataSource source = new DataSource("IrisCSV.csv");
        Instances data = source.getDataSet();

        //Set class attribute if it's not supplied.
        if(data.classIndex() == -1)
        {
            data.setClassIndex(data.numAttributes() -1);
        }
       // System.out.println(data.numAttributes());
       // System.out.println(data.attribute(0).name());
        
        //randomize the data
        data.randomize(new Random(0));
        //System.out.println(data);
        
        //break the data into a training and testing set.
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        
        //note that the 3rd variable is the number of items to 
        //copy after the second, NOT the number element in data.
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        //System.out.println("---train--- " + train);
        //System.out.println("---test--- " + test);
        
        
        //Implement the classifier TODO: make this my own classifier
        //Classifier classy = (Classifier)new NaiveBayes();
        myClassifier classy = new myClassifier();
        classy.buildClassifier(data);
        Evaluation trainEval = new Evaluation(train);
        trainEval.evaluateModel(classy, test);
        
        
        //Output the results
        //System.out.println(trainEval.toSummaryString("\nResults\n-----------\n", false));
        
        /*
        FilteredClassifier model = new FilteredClassifier();
        
        model.buildClassifier(train);
        model.buildClassifier(test);
         
        
        //run the tests
        
        //output the results*/
    }

}

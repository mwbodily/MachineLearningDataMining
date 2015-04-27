/********************************************************************
 * This class reads in data and runs a classifier on that data. For a more
 * detailed description of this process see the header comment in 
 * HardCodedClassifierMain.java. The classifier itself is located in 
 * MyClassifier.java.
 ********************************************************************/
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
 * @author Mackenzie Bodily
 */
public class RunClassifier {    
    DataSource source;
    Instances data;
    Instances training;
    Instances testing;
    Instances validation; //not currently in use, but included for future uses
    Evaluation trainingEval;
    
   /********************************************************************
    * Default constructor. Automatically sets the data to that contained in 
    * the sample file IrisCSV.csv.
    ********************************************************************/
    public RunClassifier() throws Exception
    {
        //get the data. Note: this file must exist in the top level folder.
        source = new DataSource("IrisCSV.csv");
        data = source.getDataSet();
        //Set class attribute if it's not supplied.
        if(data.classIndex() == -1)
        {
            data.setClassIndex(data.numAttributes() -1);
        }
    }
    
    /********************************************************************
     * Constructor that allows the user to specify their own file, rather
     * than using the default file.
     ********************************************************************/
    public RunClassifier(String fileName) throws Exception
    {
        source = new DataSource(fileName);
        data = source.getDataSet();
        
        if (data.classIndex() == -1)
        {
            data.setClassIndex(data.numAttributes() - 1);
        }
    }
    /********************************************************************
     *  This function outputs some test data for debugging. 
     ********************************************************************/
    public void outputDEBUGData(Instance data)
    {
        System.out.println(data.numAttributes());
        System.out.println(data.attribute(0).name()); 
        System.out.println(data);
    }
    
   /********************************************************************
    * Default: splits the data 70/30 and creates a testing and training
    * instance.
    ********************************************************************/
    public void splitTrainAndTest()
    {
        //break the data into a training and testing set.
        int trainSize = (int) Math.round(data.numInstances() * 0.7);
        int testSize = data.numInstances() - trainSize;
        
        //note that the 3rd variable is the number of items to 
        //copy after the second, NOT the number element in data.
        training = new Instances(data, 0, trainSize);
        testing = new Instances(data, trainSize, testSize);
    }
    
    /********************************************************************
     * Splits the data into training and testing based on the parameters
     * passed into the function.
     ********************************************************************/
    public void splitTrainAndTest(int trainSize, int testSize)
    {
        training = new Instances(data, 0, trainSize);
        testing = new Instances(data, trainSize, testSize);
    
    }
    
   /********************************************************************
    * Builds the classifier according to the hard coded classifier. 
    * It then evaluates the training.
    ********************************************************************/
    public void classifyData() throws Exception
    {
        //Run the classifier (a Naive Bayes is included for reference).
        //Classifier classy = (Classifier)new NaiveBayes();
        MyClassifier classy = new MyClassifier();
        classy.buildClassifier(data);
        trainingEval = new Evaluation(training);
        trainingEval.evaluateModel(classy, testing);
    }
    
    /********************************************************************
     * Calls the classifier functions to get the classifier started. This
     * is the default version of the class that uses the Iris CSV provided 
     * for free online. A link is provided in the header function of 
     * HardCodedClassifierMain.java. Note that in this usage, only 
     * default functions are used.
     ********************************************************************/
    public void classify() throws Exception
    {        
        //randomize the data
        data.randomize(new Random(0));    

        //divide the data into training and testing groups.
        splitTrainAndTest();
        
        // Classify the data
        classifyData();
        
        //output the results
        outputResults();
        
    }
    
    /********************************************************************
     * Outputs the results of the classification.
     ********************************************************************/
    public void outputResults()//Evaluation eval)
    {
        System.out.println(trainingEval.toSummaryString("\nResults\n-----------\n", 
                false));
    }

}
/********************************************************************
 * This class reads in data and runs a classifier on that data. For a more
 * detailed description of this process see the header comment in 
 * HardCodedClassifierMain.java. The classifier itself is located in 
 * KNNClassifier.java.
 ********************************************************************/
package automatedneuralnetwork;

import java.util.ArrayList;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;
/**
 * @author Mackenzie Bodily
 */
public class RunClassifier {    
    private final DataSource source;
    private final Instances data;
    private Instances training;
    private Instances testing;
    private Instances validation; //not currently used, to be used in the future
    private Evaluation trainingEval;
    private int numLayers;
    
   /********************************************************************
    * Default constructor. Automatically sets the data to that contained in 
    * the sample file IrisCSV.csv.
    * 
     * @throws java.lang.Exception
    ********************************************************************/
    public RunClassifier() throws Exception
    {
        numLayers = 2;
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
     * 
     * @param fileName - a file to test the data on.
     * @throws java.lang.Exception
     ********************************************************************/
    public RunClassifier(String fileName) throws Exception
    {
        numLayers = 2;
        source = new DataSource(fileName);
        data = source.getDataSet();
        
        if (data.classIndex() == -1)
        {
            data.setClassIndex(data.numAttributes() - 1);
        }
    }
    
    /********************************************************************
     *  This function outputs some test data for debugging.
     * 
     * @param data - the Instances to output for debugging.
     ********************************************************************/
    public void outputDEBUGData(Instances data)
    {
        System.out.println(data.numAttributes());
        System.out.println(data.attribute(0).name()); 
        System.out.println(data);
    }
    
   /********************************************************************
    * Default: splits the data 70/30 and creates a testing and training
    * instance.
    ********************************************************************/
    private void splitTrainAndTest()
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
     * 
     * @param trainSize - The percent of the data to be for training
     * @param testSize - The percent of the data to be for testing
     ********************************************************************/
    public void splitTrainAndTest(int trainSize, int testSize)
    {
        //note that the 3rd variable is the number of items to 
        //copy after the second, NOT the number element in data.
        training = new Instances(data, 0, trainSize);
        testing = new Instances(data, trainSize, testSize);
    }
    
   /********************************************************************
    * Builds the classifier according to the hard coded classifier. 
    * It then evaluates the training.
    ********************************************************************/
    private void setUpClassifier() throws Exception
    {
        //Run the classifier (a Naive Bayes is included for reference).
        //Classifier classy = (Classifier)new NaiveBayes();
        int numAttributes = (data.numAttributes() - 1);
        int numClasses = data.numClasses();
        
        //the list of nodes for each layer should include n-1 items since
        //the output layer should have as many nodes as classes.
        ArrayList<Integer> nodeList = new ArrayList();
        
        //hidden layer 1
        nodeList.add(8);

        //output layer
        nodeList.add(numClasses);
        
        //System.out.println("I will want to be creating: " + numLayers + " layers.");
        
        NeuralClassifier classy = new NeuralClassifier(numLayers, numAttributes, nodeList, numClasses);
        classy.buildClassifier(training);
        trainingEval = new Evaluation(training);
        trainingEval.evaluateModel(classy, testing);
    }
    
    /********************************************************************
    * Uses the Standardize filter to standardize the data for more 
    * reliable results.
    ********************************************************************/
    private void standardizeSets() throws Exception
    {
        Standardize filter = new Standardize();
        filter.setInputFormat(training);
        training = Filter.useFilter(training, filter);
        testing = Filter.useFilter(testing, filter);
    }
    
    /********************************************************************
     * Calls the classifier functions to get the classifier started. This
     * is the default version of the class that uses the Iris CSV provided 
     * for free online. A link is provided in the header function of 
     * HardCodedClassifierMain.java. Note that in this usage, only 
     * default functions are used.
     * 
     * @throws java.lang.Exception
     ********************************************************************/
    public void classify() throws Exception
    {        
        //randomize the data
        data.randomize(new Random(15));   
        
        //divide the data into training and testing groups.
        splitTrainAndTest();
        
        //standardize the data
        standardizeSets();
        
        // Classify the data
        setUpClassifier();      
    }
    
    /********************************************************************
     * This version of classify takes the size of the training data and
     * the size of the testing data as parameters. This allows the user
     * the option of not selecting the default values.
     * 
     * @param trainSize - The size of the training split in percent
     * @param testSize - The size of the test split in percent
     * 
     * @throws java.lang.Exception
     ********************************************************************/
    public void classify(int trainSize, int testSize) throws Exception
    {
        //randomize the data
        data.randomize(new Random(1));    

        //divide the data into training and testing groups.
        splitTrainAndTest(trainSize, testSize);
        
        //standardize the data
        standardizeSets();

        // Classify the data
        setUpClassifier();
    }
    
    /********************************************************************
     * Outputs the results of the classification.
     ********************************************************************/
    public void outputResults()
    {
        System.out.println(trainingEval.toSummaryString("\nResults\n-----------\n", 
                false));
    }
}

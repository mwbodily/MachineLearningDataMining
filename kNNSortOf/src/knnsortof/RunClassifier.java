/********************************************************************
 * This class reads in data and runs a classifier on that data. For a more
 * detailed description of this process see the header comment in 
 * HardCodedClassifierMain.java. The classifier itself is located in 
 * MyClassifier.java.
 ********************************************************************/
package knnsortof;

import java.io.PrintWriter;
import static java.lang.Math.abs;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * @author Mackenzie Bodily
 */
public class RunClassifier {    
    private DataSource source;
    private Instances data;
    private Instances training;
    private Instances testing;
    private Instances validation; //not currently used, to be used in the future
    private Evaluation trainingEval;
    
   /********************************************************************
    * Default constructor. Automatically sets the data to that contained in 
    * the sample file IrisCSV.csv.
    ********************************************************************/
    public RunClassifier() throws Exception
    {
        //get the data. Note: this file must exist in the top level folder.
        source = new DataSource("kNNweatherdata.csv");
        data = source.getDataSet();
        //Set class attribute if it's not supplied.
        if(data.classIndex() == -1)
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
    * Builds the classifier according to the hard coded classifier. 
    * It then evaluates the training.
    ********************************************************************/
    public void classifyData() throws Exception
    {
        //instantiate some variables for use in the loop
        double temp = 0; 
        double averageTemperature = 0;
        double correctTemp = 0;
        double error = 0;
        
        //This is the file where we put our results.
        PrintWriter writer = new PrintWriter("results.txt", "UTF-8");
        writer.println("estimate,actual,error");
        System.out.println(data.instance(0));
        //loop through and guess for all applicable values. This requires
        //getting the data from a week previous as well as the data from 
        //one and two years ago.
        for(int i = 730; i < data.numInstances(); i++)
        {
            for(int j = (i-1); j > (i - 7); j--)
            {
                System.out.println(data.instance(j));
                temp = data.instance(j).classValue();
                System.out.println("Day: " + j + " = " + temp);
                temp /= 9;
                System.out.println("\tDay / 9: " + j + " = " + temp);
                averageTemperature += temp;
            }
            temp = data.instance((i-365)).classValue();
            System.out.println("Day: " + (i-365) + " = " + temp);
            temp /= 9;
            
            averageTemperature += temp;
            
            temp = data.instance((i-730)).classValue();
            System.out.println("Day: " + (i-730) + " = " + temp);
            temp /= 9;
            averageTemperature += temp;
            
            //Now we write the result to a file...
            writer.print(averageTemperature + ",");
            
            //Now we write the correct value to the file...
            correctTemp = data.instance(i).classValue();
            writer.print(correctTemp + ",");
            
            //Now that we have a guess, we determine our error and write it 
            //to our file.
            error = abs(correctTemp - averageTemperature);
            writer.println(error + ",");
            //reset the value when finished...
            averageTemperature = 0;
        }
        
        //Just to be tidy, we clean up after ourselves. Like good people.
        writer.close();
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
        // Classify the data
        classifyData();
        
    }
}

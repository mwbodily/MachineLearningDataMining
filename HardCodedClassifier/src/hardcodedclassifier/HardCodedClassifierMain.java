/*
 * The overall goal of this program is to implement a basic hand written
 * classifier. As of this edition, it always guesses the first option.
 * To accomplish this goal, the Iris CSV files were used and read in as 
 * data. The data was then randomized and divided so 70/30 into training
 * and testing data. This data is output after the completion of the 
 * classification.
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
 * @author Mackenzie Bodily
 */
public class HardCodedClassifierMain {

    /********************************************************************
     * Currently, no command line parameters are needed in this
     * program. 
     ********************************************************************/
    public static void main(String[] args) throws Exception{
        try{
            RunClassifier basic = new RunClassifier();
            basic.classify();
        }
        catch (Exception er)
        {
            System.out.println("Error! " + er);
        }
        
        System.out.println("Program Complete.\n");
    }
}

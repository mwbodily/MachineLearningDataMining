/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package automatedneuralnetwork;

/**
 *
 * @author Mackenzie
 */
public class AutomatedNeuralNetworkMain {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try{
            //Default - runs with the Iris data set.
            //RunClassifier kNN = new RunClassifier();
            
            //Runs with the cars data set.            
            RunClassifier ANN = new RunClassifier("diabetes.csv");
            //RunClassifier ANN = new RunClassifier("trialCSV.csv");
            ANN.classify();
            ANN.outputResults();
        }
        catch (Exception er)
        {
            System.out.println("Error! " + er);
        }
        
        System.out.println("Program Complete.\n");
    }
    
}

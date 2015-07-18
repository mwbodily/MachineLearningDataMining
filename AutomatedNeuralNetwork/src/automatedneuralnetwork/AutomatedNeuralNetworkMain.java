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

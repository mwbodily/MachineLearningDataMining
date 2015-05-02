/*
 * This project's purpose is to implement the kNN data mining algorithm.
 * The kNN algorithm is contained in KNNClassifier.java and the required 
 * methods are contained in RunClassifier.java. For more information, 
 * please refer to those class files.
 */
package nearestneighbor;

/**
 * @author Mackenzie Bodily
 */
public class NearestNeighborMain {

    /********************************************************************
     * The driver function. Initializes the RunClassifier object and 
     * calls the methods necessary to build a kNN classifier.
     * 
     * @param args the command line arguments
     ********************************************************************/
    public static void main(String[] args) {
        try{
            //Default - runs with the Iris data set.
            //RunClassifier kNN = new RunClassifier();
            
            //Runs with the cars data set.            
            RunClassifier kNN = new RunClassifier("cars.csv");
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

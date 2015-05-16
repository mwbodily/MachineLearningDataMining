/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontreeclassifier;

/**
 *
 * @author Mackenzie
 */
public class DecisionTreeClassifierMain {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try{
            RunClassifier basic = new RunClassifier("IrisCSV.csv");
            //RunClassifier basic = new RunClassifier("cars.csv");
            basic.classify();
            basic.outputResults();
        }
        catch (Exception er)
        {
            System.out.println("Error! Message: " + er);
        }
        
        System.out.println("Program Complete.\n");
    }
    
    
}

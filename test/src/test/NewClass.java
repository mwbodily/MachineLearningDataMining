/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

/**
 *
 * @author Mackenzie
 */
public class NewClass {
    double error; 
    double output;
    double target;
    int nodeIndex;
    
    public void NewClass()
    {
        System.out.println("HJ");
        
        output = .599;
        target = 1.00;
        
        calculateOLError();
    }
    
    public void calculateOLError()
    {
        System.out.println("\t(OL) The output for this node is: " + output);
        System.out.println("\t(OL) The target for this node is: " + target);
        error = output * (1 - output) * (output - target);
        System.out.println("\t(OL) The resulting error for this target is: " + error);
        
        System.out.println("w = " + (-.4-(.2*error*-1)));
    }
    
    public void calculateHLError(Layer nextLayer)
    {
        error = output*( 1 -output);
        
        for(int i = 0; i < nextLayer.numberOfNodes; i++)
        {
            double tempError = nextLayer.Nodes.get(i).error;
            double tempWeight = nextLayer.Nodes.get(i).weightList.get(nodeIndex);
            System.out.println("\tError of next layer: " + tempError);
            System.out.println("\tWeight of node to that layer: " + tempWeight);
            System.out.println("\tAdding " + (tempError * tempWeight) + " to " + error);
            System.out.println("\t\tFor the HL node " + nodeIndex + " I will be multiplying: " +
                    tempError + " and " + tempWeight + " to get: " + (tempError*tempWeight));
            error += (tempError * tempWeight);
        }
        //System.out.println("\tThe resulting error is: " + error);
        //System.out.println();
    }
    
    public void callThing()
    {
                System.out.println("HJ");
        
        output = .599;
        target = 1.00;
       
        
        calculateOLError();
        
        output = .387;
        double w = -.31;
        System.out.println((output*(1-output)*(w*error))); 
    }
}

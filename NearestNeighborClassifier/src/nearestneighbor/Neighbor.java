/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nearestneighbor;

/**
 *
 * @author Mackenzie
 */
public class Neighbor {

        public int index;
        public double distance;
        public int count;
        
        public Neighbor()
        {
            index = -1;
            distance = Double.POSITIVE_INFINITY;        
            count = 0;
        }
        
        public Neighbor(int pIndex, double pDistance)
        {
            index = pIndex;
            distance = pDistance;      
            count = 0;
        }
}
